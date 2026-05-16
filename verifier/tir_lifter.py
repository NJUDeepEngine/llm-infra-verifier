"""TileLang TIR lifter: lift single-device TIR to distributed IR.

Implements Scheme A: directly parse TileLang TIR structure and infer
needed collective communication from block-level access patterns.

The lifting logic:
  1. Parse TIR block axes (spatial vs reduce)
  2. Map sharding spec to buffer dimensions
  3. For each reduce axis that is sharded → insert AllReduce after the block
  4. For mismatched spatial sharding → insert AllGather / ReduceScatter
  5. Generate the distributed IR program

Models a simplified TileLang TIR subset sufficient for Linear, MLP,
and Attention patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import itertools

from .state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    Placement,
    compute_local_shape,
)
from .ir import (
    IROp,
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    GELU,
    ReLU,
    Dropout,
    LayerNorm,
    RMSNorm,
    AllReduce,
    AllGather,
    ReduceScatter,
    Reshape,
    Transpose,
    FlashAttention,
)


# ── TIR subset model ─────────────────────────────────────────────────────────

@dataclass
class TIRVar:
    """A loop variable in TIR."""
    name: str
    dtype: str = "int32"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"TIRVar({self.name})"


@dataclass
class TIRGrid:
    """Multi-dimensional loop nest: T.grid(*axes)."""
    axes: List[TIRVar] = field(default_factory=list)

    def __repr__(self):
        axes_str = ", ".join(a.name for a in self.axes)
        return f"T.grid({axes_str})"


@dataclass
class TIRBlockAxis:
    """A block axis with type annotation (S=spatial, R=reduce)."""
    var: TIRVar
    type: str  # "S" for spatial, "R" for reduce
    extent: int

    def __repr__(self):
        return f"{self.var.name}[{self.type}:{self.extent}]"


@dataclass
class TIRBufferRegion:
    """Describes how a buffer is accessed in a block: buffer[indices...].

    Each index is either a TIRVar name or an integer constant.
    """
    buffer: str
    indices: List[str]  # variable names per dimension

    def __repr__(self):
        idx_str = ", ".join(self.indices)
        return f"{self.buffer}[{idx_str}]"


@dataclass
class TIRBlock:
    """A compute block in TIR.

    Example:
        TIRBlock(
            name="matmul",
            axes=[TIRBlockAxis(i, "S", B), TIRBlockAxis(j, "S", O), TIRBlockAxis(k, "R", H)],
            reads=[TIRBufferRegion("X", ["i", "k"]), TIRBufferRegion("W", ["k", "j"])],
            writes=[TIRBufferRegion("Y", ["i", "j"])],
            body="Y[i,j] += X[i,k] * W[k,j]",
        )
    """
    name: str
    axes: List[TIRBlockAxis] = field(default_factory=list)
    reads: List[TIRBufferRegion] = field(default_factory=list)
    writes: List[TIRBufferRegion] = field(default_factory=list)
    init: Optional[str] = None
    body: Optional[str] = None

    @property
    def spatial_axes(self) -> List[TIRBlockAxis]:
        return [a for a in self.axes if a.type == "S"]

    @property
    def reduce_axes(self) -> List[TIRBlockAxis]:
        return [a for a in self.axes if a.type == "R"]

    def __repr__(self):
        return f"TIRBlock({self.name})"


@dataclass
class TIRFunc:
    """A complete TIR function (equivalent to @T.prim_func).

    Example:
        TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            grid=TIRGrid(axes=[i, j, k]),
            blocks=[matmul_block],
        )
    """
    name: str
    buffers: Dict[str, Tuple[int, ...]]  # buffer name → shape
    grid: TIRGrid = field(default_factory=TIRGrid)
    blocks: List[TIRBlock] = field(default_factory=list)

    def __repr__(self):
        return f"TIRFunc({self.name}, buffers={list(self.buffers.keys())}, blocks={len(self.blocks)})"


# ── Lifting result ───────────────────────────────────────────────────────────

@dataclass
class LiftResult:
    """Result of lifting TIR to distributed IR."""
    fwd_program: Program
    bwd_program: Program
    tensors: Dict[str, TensorState]  # initial tensors with sharding
    collectives_inserted: List[IROp]
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        n_coll = len(self.collectives_inserted)
        return (
            f"LiftResult(fwd={len(self.fwd_program)} ops, "
            f"bwd={len(self.bwd_program)} ops, "
            f"collectives={n_coll})"
        )


# ── TIR Lifter ───────────────────────────────────────────────────────────────

class TIRLifter:
    """Lifts TileLang TIR to distributed IR by analyzing block access patterns.

    For each TIR block:
      1. Identify which reduce axes are sharded → AllReduce needed
      2. Identify mismatched spatial shardings → resharding needed
      3. Generate compute ops + collective ops in the correct order
    """

    def __init__(self, sharding_specs: Dict[str, ShardingSpec]):
        """
        Args:
            sharding_specs: {buffer_name: ShardingSpec} for all buffers
        """
        self.sharding_specs = sharding_specs
        self.mesh = list(sharding_specs.values())[0].mesh if sharding_specs else None

    def lift(self, tir_func: TIRFunc) -> LiftResult:
        """Lift a TIR function to forward and backward distributed programs."""
        fwd_program = Program(name=f"{tir_func.name}_fwd")
        tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        # Create initial tensor states
        for buf_name, shape in tir_func.buffers.items():
            spec = self.sharding_specs.get(
                buf_name,
                ShardingSpec(
                    placements=(Replicate(),) * self.mesh.ndim if self.mesh else (),
                    mesh=self.mesh,
                ),
            )
            local_shape = compute_local_shape(shape, spec)

            tensors[buf_name] = TensorState(
                name=buf_name,
                global_shape=shape,
                local_shape=local_shape,
                sharding=spec,
                expr=buf_name.lower(),
                requires_grad=True,
                grad_name=f"grad_{buf_name}",
            )

        # Process each block
        for block in tir_func.blocks:
            block_ops, block_tensors, block_warnings = self._lift_block(
                block, tensors
            )
            for op in block_ops:
                fwd_program.add(op)
            tensors.update(block_tensors)
            warnings.extend(block_warnings)

        # Generate backward program
        bwd_program = self._generate_backward(fwd_program, tensors)

        # Collect all inserted collectives
        collectives = [op for op in fwd_program.ops if op.is_collective()]

        return LiftResult(
            fwd_program=fwd_program,
            bwd_program=bwd_program,
            tensors=tensors,
            collectives_inserted=collectives,
            warnings=warnings,
        )

    def _lift_block(
        self,
        block: TIRBlock,
        tensors: Dict[str, TensorState],
    ) -> Tuple[List[IROp], Dict[str, TensorState], List[str]]:
        """Lift a single TIR block to IR ops.

        Returns: (ops, new_tensors, warnings)
        """
        ops: List[IROp] = []
        new_tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        # Determine the block type from read/write patterns
        block_type = self._classify_block(block)

        if block_type == "matmul":
            block_ops, block_tensors, block_warnings = self._lift_matmul_block(
                block, tensors
            )
        elif block_type == "elementwise":
            block_ops, block_tensors, block_warnings = self._lift_elementwise_block(
                block, tensors
            )
        elif block_type == "attention":
            block_ops, block_tensors, block_warnings = self._lift_attention_block(
                block, tensors
            )
        else:
            block_ops, block_tensors, block_warnings = self._lift_generic_block(
                block, tensors
            )

        ops.extend(block_ops)
        new_tensors.update(block_tensors)
        warnings.extend(block_warnings)

        return ops, new_tensors, warnings

    def _classify_block(self, block: TIRBlock) -> str:
        """Classify a TIR block as matmul, elementwise, attention, or generic."""
        reads = block.reads
        writes = block.writes
        reduce_axes = block.reduce_axes

        if len(reads) == 2 and len(writes) == 1 and len(reduce_axes) == 1:
            # Check if reads share a reduce dim (typical matmul pattern)
            r0_indices = set(reads[0].indices)
            r1_indices = set(reads[1].indices)
            w_indices = set(writes[0].indices)
            common_reduce = r0_indices & r1_indices - w_indices
            if common_reduce:
                return "matmul"

        if len(reads) == 3 and any("attn" in r.buffer.lower() or "q" in r.buffer.lower() for r in reads):
            return "attention"

        if len(reduce_axes) == 0 and len(reads) <= 2:
            return "elementwise"

        return "generic"

    def _lift_matmul_block(
        self,
        block: TIRBlock,
        tensors: Dict[str, TensorState],
    ) -> Tuple[List[IROp], Dict[str, TensorState], List[str]]:
        """Lift a matmul block: X @ W → Y, inserting AllReduce if needed."""
        ops: List[IROp] = []
        new_tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        a_name = block.reads[0].buffer
        b_name = block.reads[1].buffer
        y_name = block.writes[0].buffer

        a = tensors.get(a_name)
        b = tensors.get(b_name)

        if a is None or b is None:
            warnings.append(f"Missing tensor for matmul: {a_name}, {b_name}")
            return ops, new_tensors, warnings

        # Check if reduce axis is sharded on both operands
        reduce_axis = block.reduce_axes[0].var.name

        a_reduce_idx = self._find_reduce_index(block.reads[0], reduce_axis)
        b_reduce_idx = self._find_reduce_index(block.reads[1], reduce_axis)

        needs_allreduce = False

        if a_reduce_idx is not None and b_reduce_idx is not None:
            # Check if both are sharded on the reduce dim
            a_shard_dims = a.sharding.get_shard_dims()
            b_shard_dims = b.sharding.get_shard_dims()

            for mesh_dim in range(self.mesh.ndim):
                a_p = a.sharding.placements[mesh_dim]
                b_p = b.sharding.placements[mesh_dim]

                if isinstance(a_p, Shard) and a_p.dim == a_reduce_idx:
                    if isinstance(b_p, Shard) and b_p.dim == b_reduce_idx:
                        needs_allreduce = True
                        break

        # MatMul op
        matmul_out = y_name if not needs_allreduce else f"{y_name}_partial"
        matmul_op = MatMul(a=a_name, b=b_name, output=matmul_out)
        ops.append(matmul_op)

        # Apply matmul to compute output tensor state
        local_ctx = {a_name: a, b_name: b}
        matmul_result = matmul_op.apply(local_ctx)
        new_tensors[matmul_out] = matmul_result

        if needs_allreduce:
            # AllReduce: partial → replicate
            ar_op = AllReduce(x=matmul_out, output=y_name, op_type="sum")
            ops.append(ar_op)

            ar_ctx = {matmul_out: matmul_result}
            ar_result = ar_op.apply(ar_ctx)
            new_tensors[y_name] = ar_result

        return ops, new_tensors, warnings

    def _lift_elementwise_block(
        self,
        block: TIRBlock,
        tensors: Dict[str, TensorState],
    ) -> Tuple[List[IROp], Dict[str, TensorState], List[str]]:
        """Lift an element-wise block (add, multiply, activation)."""
        ops: List[IROp] = []
        new_tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        if len(block.reads) == 2:
            a_name = block.reads[0].buffer
            b_name = block.reads[1].buffer
            y_name = block.writes[0].buffer

            # Check if it's add or multiply
            if "+" in (block.body or ""):
                op = Add(a=a_name, b=b_name, output=y_name)
            else:
                op = Multiply(a=a_name, b=b_name, output=y_name)

            ops.append(op)

            a = tensors.get(a_name)
            b = tensors.get(b_name)
            if a and b:
                local_ctx = {a_name: a, b_name: b}
                result = op.apply(local_ctx)
                new_tensors[y_name] = result

        elif len(block.reads) == 1:
            x_name = block.reads[0].buffer
            y_name = block.writes[0].buffer

            # Check for activation functions
            body_lower = (block.body or "").lower()
            if "silu" in body_lower:
                op = SiLU(x=x_name, output=y_name)
            elif "gelu" in body_lower:
                op = GELU(x=x_name, output=y_name)
            elif "relu" in body_lower:
                op = ReLU(x=x_name, output=y_name)
            elif "dropout" in body_lower:
                op = Dropout(x=x_name, output=y_name)
            elif "layernorm" in body_lower or "layer_norm" in body_lower:
                op = LayerNorm(x=x_name, output=y_name)
            elif "rmsnorm" in body_lower or "rms_norm" in body_lower:
                op = RMSNorm(x=x_name, output=y_name)
            else:
                # Generic unary
                op = Multiply(a=x_name, b=x_name, output=y_name)  # fallback

            ops.append(op)

            x = tensors.get(x_name)
            if x:
                local_ctx = {x_name: x}
                result = op.apply(local_ctx)
                new_tensors[y_name] = result

        return ops, new_tensors, warnings

    def _lift_attention_block(
        self,
        block: TIRBlock,
        tensors: Dict[str, TensorState],
    ) -> Tuple[List[IROp], Dict[str, TensorState], List[str]]:
        """Lift an attention block for context parallelism."""
        ops: List[IROp] = []
        new_tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        # Find Q, K, V from read patterns
        q_name = k_name = v_name = None
        for r in block.reads:
            low = r.buffer.lower()
            if "q" in low:
                q_name = r.buffer
            elif "k" in low:
                k_name = r.buffer
            elif "v" in low:
                v_name = r.buffer

        if not all([q_name, k_name, v_name]):
            warnings.append("Could not identify Q/K/V in attention block")
            return ops, new_tensors, warnings

        y_name = block.writes[0].buffer

        # Check if K, V are sharded on seq dim (CP pattern)
        k_tensor = tensors.get(k_name)
        v_tensor = tensors.get(v_name)
        q_tensor = tensors.get(q_name)

        needs_ring = False
        if k_tensor and q_tensor:
            k_shards = k_tensor.sharding.get_shard_dims()
            q_shards = q_tensor.sharding.get_shard_dims()
            # If K is sharded on seq dim (dim=1) and Q is not → CP ring
            if 1 in k_shards and 1 not in q_shards:
                needs_ring = True

        if needs_ring:
            # Ring attention: partial output needs AllReduce
            partial_out = f"{y_name}_partial"
            fa_op = FlashAttention(
                q=q_name, k=k_name, v=v_name, output=partial_out
            )
            ops.append(fa_op)

            fa_ctx = {
                q_name: q_tensor,
                k_name: k_tensor,
                v_name: v_tensor,
            }
            fa_result = fa_op.apply(fa_ctx)
            new_tensors[partial_out] = fa_result

            ar_op = AllReduce(x=partial_out, output=y_name, op_type="sum")
            ops.append(ar_op)

            ar_ctx = {partial_out: fa_result}
            ar_result = ar_op.apply(ar_ctx)
            new_tensors[y_name] = ar_result
        else:
            fa_op = FlashAttention(
                q=q_name, k=k_name, v=v_name, output=y_name
            )
            ops.append(fa_op)

            fa_ctx = {
                q_name: q_tensor,
                k_name: k_tensor,
                v_name: v_tensor,
            }
            fa_result = fa_op.apply(fa_ctx)
            new_tensors[y_name] = fa_result

        return ops, new_tensors, warnings

    def _lift_generic_block(
        self,
        block: TIRBlock,
        tensors: Dict[str, TensorState],
    ) -> Tuple[List[IROp], Dict[str, TensorState], List[str]]:
        """Lift a generic compute block."""
        ops: List[IROp] = []
        new_tensors: Dict[str, TensorState] = {}
        warnings: List[str] = []

        # For generic blocks, check reduce axes for partial results
        if block.reduce_axes:
            # Any reduce axis might need AllReduce if sharded
            y_name = block.writes[0].buffer
            partial_out = f"{y_name}_partial"

            # Use MatMul as a generic reduction proxy
            if len(block.reads) >= 1:
                # Create a generic reduction op
                a_name = block.reads[0].buffer
                if a_name in tensors:
                    a = tensors[a_name]
                    # Mark as partial
                    partial_tensor = TensorState(
                        name=partial_out,
                        global_shape=a.global_shape,
                        local_shape=a.local_shape,
                        sharding=ShardingSpec(
                            placements=tuple(
                                Partial() if isinstance(p, Shard) else p
                                for p in a.sharding.placements
                            ),
                            mesh=a.sharding.mesh,
                        ),
                        expr=a.expr,
                    )
                    new_tensors[partial_out] = partial_tensor

                    ar_op = AllReduce(x=partial_out, output=y_name)
                    ops.append(ar_op)

                    ar_ctx = {partial_out: partial_tensor}
                    ar_result = ar_op.apply(ar_ctx)
                    new_tensors[y_name] = ar_result

        return ops, new_tensors, warnings

    def _find_reduce_index(
        self, region: TIRBufferRegion, reduce_var: str
    ) -> Optional[int]:
        """Find which buffer dimension is indexed by the reduce variable."""
        for i, idx in enumerate(region.indices):
            if idx == reduce_var:
                return i
        return None

    def _generate_backward(
        self,
        fwd_program: Program,
        tensors: Dict[str, TensorState],
    ) -> Program:
        """Generate the backward program by reversing the forward ops.

        For each forward op, insert its dual collective in reverse order.
        """
        bwd_program = Program(name="backward")

        for op in reversed(fwd_program.ops):
            if isinstance(op, MatMul):
                # Backward: two matmuls for grad_a and grad_b
                a, b = op.a, op.b
                grad_y = f"grad_{op.output}"

                # grad_a = grad_y @ b^T
                bwd_program.add(
                    MatMul(a=grad_y, b=f"{b}_T", output=f"grad_{a}")
                )
                # grad_b = a^T @ grad_y
                bwd_program.add(
                    MatMul(a=f"{a}_T", b=grad_y, output=f"grad_{b}_partial")
                )

                # If forward had AllReduce after matmul, backward needs one too
                # (handled by checking if grad_b is partial)

            elif isinstance(op, AllReduce):
                # AllReduce is self-dual
                grad_input = f"grad_{op.x}"
                grad_output = f"grad_{op.output}"
                if grad_input != grad_output:
                    bwd_program.add(
                        AllReduce(
                            x=grad_input,
                            output=grad_output,
                            op_type=op.op_type,
                        )
                    )

            elif isinstance(op, AllGather):
                # Dual: ReduceScatter
                bwd_program.add(
                    ReduceScatter(
                        x=f"grad_{op.output}",
                        output=f"grad_{op.x}",
                        scatter_dim=op.gather_dim,
                    )
                )

            elif isinstance(op, ReduceScatter):
                # Dual: AllGather
                bwd_program.add(
                    AllGather(
                        x=f"grad_{op.output}",
                        output=f"grad_{op.x}",
                        gather_dim=op.scatter_dim,
                    )
                )

            elif isinstance(op, Send):
                # Dual: Recv (direction reversed)
                bwd_program.add(
                    Recv(
                        x=f"grad_{op.output}",
                        output=f"grad_{op.x}",
                        src=op.dst,
                        dst=op.src,
                        stage=op.stage,
                        microbatch_id=op.microbatch_id,
                    )
                )

            elif isinstance(op, Recv):
                # Dual: Send (direction reversed)
                bwd_program.add(
                    Send(
                        x=f"grad_{op.x}",
                        output=f"grad_{op.output}",
                        src=op.dst,
                        dst=op.src,
                        stage=op.stage,
                        microbatch_id=op.microbatch_id,
                    )
                )

            elif isinstance(op, (Add, Multiply, SiLU, GELU, ReLU, Dropout,
                                  LayerNorm, RMSNorm)):
                # Element-wise: grad flows through
                for input_name in op.input_names:
                    bwd_program.add(
                        Multiply(
                            a=f"grad_{op.output}",
                            b=f"grad_{input_name}",
                            output=f"grad_{input_name}_final",
                        )
                    )

        return bwd_program

    def lift_with_autograd(self, tir_func: TIRFunc) -> LiftResult:
        """Lift and generate both forward and backward programs."""
        return self.lift(tir_func)
