"""CommonTIR: DSL-agnostic Tile Intermediate Representation.

Provides a unified IR for representing tile-based compute kernels from
different DSLs (TileLang, Triton, TVM, etc.) and lifting them to
distributed verification IR.

Architecture:
  DSL Source → DSL Adapter → CommonTIR → TIRLifter → Distributed IR
      │                         │
  TileLang TIR            TIRFunc, TIRBlock,
  Triton kernel           TIRGrid, TIRAxis,
  TVM TensorIR            TIRBuffer, TIRAccess

The CommonTIR captures the ESSENCE of tile-based compute:
  - Multi-dimensional loop nests (grid)
  - Block-level compute with typed axes (spatial/reduce)
  - Buffer access patterns (which loop var indexes which buffer dim)
  - Element-wise and reduction operations

This is intentionally minimal — it only models what's needed for
DISTRIBUTED verification (placement analysis, communication inference),
not full kernel compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Core TIR types
# ═══════════════════════════════════════════════════════════════════════════════

class AxisType(Enum):
    SPATIAL = "S"      # parallel loop, can be distributed
    REDUCE = "R"       # reduction loop, needs AllReduce if sharded


@dataclass(frozen=True)
class TIRVar:
    """A loop variable in the TIR grid."""
    name: str
    dtype: str = "int32"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


@dataclass
class TIRAxis:
    """A block axis binding: which loop var, what type (spatial/reduce), what extent."""
    var: TIRVar
    type: AxisType
    extent: int

    @property
    def is_spatial(self) -> bool:
        return self.type == AxisType.SPATIAL

    @property
    def is_reduce(self) -> bool:
        return self.type == AxisType.REDUCE

    def __repr__(self):
        return f"{self.var.name}[{self.type.value}:{self.extent}]"


@dataclass
class TIRGrid:
    """The loop nest: a multi-dimensional grid of loop variables."""
    axes: List[TIRVar] = field(default_factory=list)

    def __repr__(self):
        return f"grid({', '.join(a.name for a in self.axes)})"


@dataclass
class TIRBuffer:
    """A buffer declaration: name, shape, dtype."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self):
        return f"{self.name}[{', '.join(str(s) for s in self.shape)}]"


@dataclass
class TIRAccess:
    """Describes how a buffer is accessed in a block.

    Maps each buffer dimension to the loop variable that indexes it.
    Example: TIRAccess("X", ["i", "k"]) means X[i, k].
    """
    buffer: str
    indices: List[str]  # variable names per dimension

    def find_var_dim(self, var_name: str) -> Optional[int]:
        """Return which buffer dim is indexed by var_name, or None."""
        for i, idx in enumerate(self.indices):
            if idx == var_name:
                return i
        return None

    def __repr__(self):
        return f"{self.buffer}[{', '.join(self.indices)}]"


@dataclass
class TIRBlock:
    """A single compute block within the grid.

    A block has:
      - Typed axes (which grid vars it uses, spatial/reduce)
      - Read accesses (which buffers it reads, how indexed)
      - Write accesses (which buffers it writes)
      - Optional init (for accumulators)
      - Optional body (for documentation)
    """
    name: str
    axes: List[TIRAxis] = field(default_factory=list)
    reads: List[TIRAccess] = field(default_factory=list)
    writes: List[TIRAccess] = field(default_factory=list)
    init: Optional[str] = None
    body: Optional[str] = None

    @property
    def spatial_axes(self) -> List[TIRAxis]:
        return [a for a in self.axes if a.is_spatial]

    @property
    def reduce_axes(self) -> List[TIRAxis]:
        return [a for a in self.axes if a.is_reduce]

    def find_reduce_var_in_access(self, access: TIRAccess) -> Optional[int]:
        """Find which buffer dimension is indexed by a reduce variable."""
        reduce_vars = {a.var.name for a in self.reduce_axes}
        for i, idx in enumerate(access.indices):
            if idx in reduce_vars:
                return i
        return None

    def __repr__(self):
        axes_str = ", ".join(repr(a) for a in self.axes)
        reads_str = ", ".join(repr(r) for r in self.reads)
        writes_str = ", ".join(repr(w) for w in self.writes)
        return f"TIRBlock({self.name}, axes=[{axes_str}], reads=[{reads_str}], writes=[{writes_str}])"


@dataclass
class TIRFunc:
    """A complete TIR function = buffers + grid + blocks.

    This is the top-level TIR representation. A single function may
    contain multiple blocks (e.g., matmul + element-wise in one kernel).

    Example (Row Parallel Linear):
        TIRFunc(
            name="linear",
            buffers={
                "X": TIRBuffer("X", (B, H)),
                "W": TIRBuffer("W", (H, O)),
                "Y": TIRBuffer("Y", (B, O)),
            },
            grid=TIRGrid(axes=[TIRVar("i"), TIRVar("j"), TIRVar("k")]),
            blocks=[
                TIRBlock(
                    name="matmul",
                    axes=[
                        TIRAxis(TIRVar("i"), AxisType.SPATIAL, B),
                        TIRAxis(TIRVar("j"), AxisType.SPATIAL, O),
                        TIRAxis(TIRVar("k"), AxisType.REDUCE, H),
                    ],
                    reads=[
                        TIRAccess("X", ["i", "k"]),
                        TIRAccess("W", ["k", "j"]),
                    ],
                    writes=[TIRAccess("Y", ["i", "j"])],
                )
            ],
        )
    """
    name: str
    buffers: Dict[str, TIRBuffer] = field(default_factory=dict)
    grid: TIRGrid = field(default_factory=TIRGrid)
    blocks: List[TIRBlock] = field(default_factory=list)

    def get_buffer(self, name: str) -> Optional[TIRBuffer]:
        return self.buffers.get(name)

    def get_buffer_shape(self, name: str) -> Optional[Tuple[int, ...]]:
        buf = self.buffers.get(name)
        return buf.shape if buf else None

    def input_buffers(self) -> List[str]:
        """Buffers that are read but not written (inputs)."""
        written = {a.buffer for b in self.blocks for a in b.writes}
        read = {a.buffer for b in self.blocks for a in b.reads}
        return list(read - written)

    def output_buffers(self) -> List[str]:
        """Buffers that are written but not read (outputs)."""
        written = {a.buffer for b in self.blocks for a in b.writes}
        read = {a.buffer for b in self.blocks for a in b.reads}
        return list(written - read)

    def __repr__(self):
        bufs = ", ".join(repr(b) for b in self.buffers.values())
        blocks_str = "\n  ".join(repr(b) for b in self.blocks)
        return f"TIRFunc({self.name}, buffers=[{bufs}],\n  grid={self.grid},\n  blocks=[\n  {blocks_str}\n])"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Block classifier — recognize compute patterns from TIR structure
# ═══════════════════════════════════════════════════════════════════════════════

class BlockType(Enum):
    MATMUL = "matmul"
    ELEMENTWISE = "elementwise"
    ATTENTION = "attention"
    REDUCTION = "reduction"
    UNKNOWN = "unknown"


class BlockClassifier:
    """Classify TIR blocks by their access pattern structure.

    Pure structural classification — no heuristics, no naming conventions.
    Based entirely on: number of operands, reduce axes, and access patterns.
    """

    def classify(self, block: TIRBlock) -> BlockType:
        n_reads = len(block.reads)
        n_writes = len(block.writes)
        n_reduce = len(block.reduce_axes)
        n_spatial = len(block.spatial_axes)

        # Matmul: 2 reads, 1 write, 1 reduce axis
        # Both reads share the reduce variable on different dimensions
        if n_reads == 2 and n_writes == 1 and n_reduce == 1:
            r0 = block.reads[0]
            r1 = block.reads[1]
            w = block.writes[0]
            reduce_var = block.reduce_axes[0].var.name

            # Check: reduce var appears in both reads but NOT in write
            r0_has_reduce = reduce_var in r0.indices
            r1_has_reduce = reduce_var in r1.indices
            w_has_reduce = reduce_var in w.indices

            if r0_has_reduce and r1_has_reduce and not w_has_reduce:
                return BlockType.MATMUL

        # Elementwise: 1-2 reads, 1 write, 0 reduce axes
        if n_reduce == 0 and n_reads <= 2 and n_writes == 1:
            return BlockType.ELEMENTWISE

        # Reduction: 1 read, 1 write, >=1 reduce axis
        if n_reads == 1 and n_writes == 1 and n_reduce >= 1:
            return BlockType.REDUCTION

        # Attention: 3 reads (Q, K, V), 1 write, potentially reduce axes
        if n_reads == 3 and n_writes == 1:
            return BlockType.ATTENTION

        return BlockType.UNKNOWN

    def classify_func(self, func: TIRFunc) -> Dict[str, BlockType]:
        """Classify all blocks in a function."""
        return {b.name: self.classify(b) for b in func.blocks}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DSL Adapters — convert from DSL-specific IR to CommonTIR
# ═══════════════════════════════════════════════════════════════════════════════

class DSLAdapter(ABC):
    """Base class for DSL-to-CommonTIR adapters.

    Each DSL gets its own adapter that knows how to parse/convert
    the DSL's native IR into our CommonTIR representation.
    """

    @abstractmethod
    def can_handle(self, source) -> bool:
        """Check if this adapter can handle the given source."""
        ...

    @abstractmethod
    def convert(self, source, name: str = "") -> TIRFunc:
        """Convert DSL source to CommonTIR."""
        ...


class TileLangAdapter(DSLAdapter):
    """Convert TileLang TIR to CommonTIR.

    TileLang TIR is structurally very close to our CommonTIR,
    so this is mostly a direct mapping with renaming.

    In production, this would call TileLang's TIR parser to get
    the AST, then walk it. For now, we accept our CommonTIR types
    directly (since they're identical in structure).
    """

    def can_handle(self, source) -> bool:
        # Accept CommonTIR types directly (for now)
        return isinstance(source, TIRFunc)

    def convert(self, source: TIRFunc, name: str = "") -> TIRFunc:
        """Pass-through: TileLang TIR IS CommonTIR."""
        if not isinstance(source, TIRFunc):
            raise TypeError(f"TileLangAdapter expects TIRFunc, got {type(source)}")
        return source


class TritonAdapter(DSLAdapter):
    """Convert Triton kernel patterns to CommonTIR.

    Triton uses a different programming model (block-level programs
    with program_id and tl.load/tl.store). This adapter extracts
    the essential loop structure from Triton's semantics.

    Key mapping:
      - Triton program_id(axis) → TIR spatial axis
      - Triton tl.dot(a, b, c) → TIR matmul block with reduce axis
      - Triton tl.load/tl.store → TIRAccess read/write
    """

    def can_handle(self, source) -> bool:
        """Check if source looks like a Triton kernel description."""
        if isinstance(source, dict):
            return source.get("dialect") == "triton"
        return False

    def convert(self, source: dict, name: str = "triton_kernel") -> TIRFunc:
        """Convert a Triton kernel description dict to CommonTIR."""
        if not self.can_handle(source):
            raise ValueError(f"TritonAdapter cannot handle: {source}")

        ops = source.get("ops", [])
        buffers = {}
        blocks = []
        grid_vars = []

        # Extract buffers from load/store ops
        for i, op in enumerate(ops):
            op_type = op.get("type")
            if op_type == "load":
                buf = op.get("buffer", f"buf_{i}")
                shape = tuple(op.get("shape", ()))
                if buf not in buffers:
                    buffers[buf] = TIRBuffer(buf, shape)
            elif op_type == "store":
                buf = op.get("buffer", f"out_{i}")
                shape = tuple(op.get("shape", ()))
                if buf not in buffers:
                    buffers[buf] = TIRBuffer(buf, shape)

        # Extract grid variables from program_id usage
        grid_dims = source.get("grid", [])
        for dim_name in grid_dims:
            grid_vars.append(TIRVar(dim_name))

        # Extract compute blocks
        for i, op in enumerate(ops):
            if op.get("type") == "dot":
                a_name = op.get("a", "A")
                b_name = op.get("b", "B")
                c_name = op.get("c", "C")
                m = op.get("M", 1)
                n = op.get("N", 1)
                k = op.get("K", 1)

                block_axes = [
                    TIRAxis(TIRVar("pid_m"), AxisType.SPATIAL, m),
                    TIRAxis(TIRVar("pid_n"), AxisType.SPATIAL, n),
                    TIRAxis(TIRVar("rk"), AxisType.REDUCE, k),
                ]
                reads = [
                    TIRAccess(a_name, ["pid_m", "rk"]),
                    TIRAccess(b_name, ["rk", "pid_n"]),
                ]
                writes = [TIRAccess(c_name, ["pid_m", "pid_n"])]

                blocks.append(TIRBlock(
                    name=f"dot_{i}",
                    axes=block_axes,
                    reads=reads,
                    writes=writes,
                    body=f"{c_name}[pid_m, pid_n] += {a_name}[pid_m, rk] * {b_name}[rk, pid_n]",
                ))
            elif op.get("type") == "elementwise":
                a_name = op.get("a", "A")
                b_name = op.get("b", "")
                out_name = op.get("out", "O")

                spatial_axes = [
                    TIRAxis(TIRVar(f"pid_{d}"), AxisType.SPATIAL, 1)
                    for d in range(op.get("ndim", 1))
                ]
                reads = [TIRAccess(a_name, [a.var.name for a in spatial_axes])]
                if b_name:
                    reads.append(TIRAccess(b_name, [a.var.name for a in spatial_axes]))
                writes = [TIRAccess(out_name, [a.var.name for a in spatial_axes])]

                blocks.append(TIRBlock(
                    name=f"ew_{i}",
                    axes=spatial_axes,
                    reads=reads,
                    writes=writes,
                    body=f"{out_name} = {a_name}" + (f" {op.get('op', '+')} {b_name}" if b_name else ""),
                ))

        return TIRFunc(
            name=name,
            buffers={name: buf for name, buf in buffers.items()},
            grid=TIRGrid(axes=grid_vars),
            blocks=blocks,
        )


class TVMAdapter(DSLAdapter):
    """Convert TVM TensorIR to CommonTIR.

    TVM TensorIR has:
      - T.block with iter_vars (spatial/reduce)
      - T.match_buffer for buffer binding
      - T.BufferStore / T.BufferLoad for access patterns

    This maps directly to our CommonTIR.
    """

    def can_handle(self, source) -> bool:
        if isinstance(source, dict):
            return source.get("dialect") == "tvm"
        return False

    def convert(self, source: dict, name: str = "tvm_kernel") -> TIRFunc:
        """Convert TVM TensorIR dict to CommonTIR."""
        buffers = {}
        grid_vars = []
        blocks = []

        for buf_name, buf_info in source.get("buffers", {}).items():
            buffers[buf_name] = TIRBuffer(
                buf_name, tuple(buf_info.get("shape", ())),
                buf_info.get("dtype", "float32"),
            )

        for tvm_block in source.get("blocks", []):
            axes = []
            for ax in tvm_block.get("iter_vars", []):
                var = TIRVar(ax.get("var", "v"))
                grid_vars.append(var)
                ax_type = AxisType.SPATIAL if ax.get("kind") == "spatial" else AxisType.REDUCE
                axes.append(TIRAxis(var, ax_type, ax.get("extent", 1)))

            reads = []
            for r in tvm_block.get("reads", []):
                reads.append(TIRAccess(r["buffer"], r.get("indices", [])))

            writes = []
            for w in tvm_block.get("writes", []):
                writes.append(TIRAccess(w["buffer"], w.get("indices", [])))

            blocks.append(TIRBlock(
                name=tvm_block.get("name", "block"),
                axes=axes,
                reads=reads,
                writes=writes,
                init=tvm_block.get("init"),
                body=tvm_block.get("body"),
            ))

        return TIRFunc(
            name=name,
            buffers=buffers,
            grid=TIRGrid(axes=grid_vars),
            blocks=blocks,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DSL Registry — auto-detect and convert any supported DSL
# ═══════════════════════════════════════════════════════════════════════════════

class DSLRegistry:
    """Registry of DSL adapters with auto-detection.

    Usage:
        registry = DSLRegistry()
        registry.register(TileLangAdapter())
        registry.register(TritonAdapter())
        registry.register(TVMAdapter())

        tir = registry.convert(source)  # auto-detects DSL type
    """

    def __init__(self):
        self.adapters: List[DSLAdapter] = []

    def register(self, adapter: DSLAdapter):
        self.adapters.append(adapter)

    def convert(self, source, name: str = "") -> TIRFunc:
        """Auto-detect DSL and convert to CommonTIR.

        Raises ValueError if no adapter can handle the source.
        """
        for adapter in self.adapters:
            if adapter.can_handle(source):
                return adapter.convert(source, name)
        raise ValueError(
            f"No DSL adapter found for source type {type(source).__name__}. "
            f"Registered adapters: {[type(a).__name__ for a in self.adapters]}"
        )

    def convert_all(self, sources: List, name: str = "") -> List[TIRFunc]:
        """Convert multiple sources."""
        return [self.convert(s, name) for s in sources]


# Default registry with all built-in adapters
default_registry = DSLRegistry()
default_registry.register(TileLangAdapter())
default_registry.register(TritonAdapter())
default_registry.register(TVMAdapter())
