"""Core state definitions for distributed tensor verification.

TensorState tracks per-device tensor metadata: placement, sharding spec,
symbolic expression, and autograd/pipeline annotations. The design follows
DTensor semantics with explicit per-dimension sharding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict, List
import math


# ── SPMD Local Type (R/I/V/P) — trust base from meta-pytorch/spmd_types ──────

class LocalSPMDType(Enum):
    """SPMD local types per mesh axis (from meta-pytorch/spmd_types DESIGN.md).

    Four states with fixed forward↔backward duality:

        REPLICATE (R): data same across ranks  →  gradient is PARTIAL
        INVARIANT (I): data same across ranks  →  gradient is INVARIANT (no comm)
        VARYING  (V): data differs per rank   →  gradient is VARYING
        PARTIAL  (P): pending sum across ranks →  gradient is REPLICATE

    Duality: R↔P, I↔I, V↔V. These are fixed by autograd.
    """
    REPLICATE = "R"
    INVARIANT = "I"
    VARYING = "V"
    PARTIAL = "P"

    def gradient_type(self) -> "LocalSPMDType":
        """Return the gradient's local type (backward dual)."""
        dual = {
            LocalSPMDType.REPLICATE: LocalSPMDType.PARTIAL,
            LocalSPMDType.INVARIANT: LocalSPMDType.INVARIANT,
            LocalSPMDType.VARYING: LocalSPMDType.VARYING,
            LocalSPMDType.PARTIAL: LocalSPMDType.REPLICATE,
        }
        return dual[self]


# ── Global Partition Spec (per-dimension sharding) ────────────────────────────

@dataclass(frozen=True)
class Shard:
    """Tensor dimension `dim` is sharded across a mesh axis.

    This is a GLOBAL property — it describes how shards reassemble.
    In SPMD terms: the PartitionSpec entry for this dimension.
    """
    dim: int

    def __repr__(self):
        return f"Shard({self.dim})"


# ── Legacy Placement types (backward compatibility) ───────────────────────────

class PlacementType(Enum):
    REPLICATE = "replicate"
    SHARD = "shard"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Replicate:
    """DEPRECATED: use LocalSPMDType.REPLICATE instead.

    Tensor is replicated across all devices in the mesh dimension.
    """

    def __repr__(self):
        return "Replicate()"


@dataclass(frozen=True)
class Partial:
    """DEPRECATED: use LocalSPMDType.PARTIAL instead.

    Tensor is partially reduced — needs AllReduce to become Replicate.
    """

    def __repr__(self):
        return "Partial()"


Placement = Shard | Replicate | Partial


# ── Device mesh ──────────────────────────────────────────────────────────────

@dataclass
class DeviceMesh:
    """Multi-dimensional device topology.

    Example:
        mesh = DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"))
        # 2 TP groups × 4 DP groups = 8 devices
    """
    shape: Tuple[int, ...]
    dim_names: Tuple[str, ...]

    def __post_init__(self):
        if len(self.shape) != len(self.dim_names):
            raise ValueError(
                f"shape {self.shape} and dim_names {self.dim_names} must have same length"
            )

    @property
    def num_devices(self) -> int:
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_submesh(self, dim_name: str) -> Tuple[int, int]:
        """Return (size, index) for a named mesh dimension."""
        idx = self.dim_names.index(dim_name)
        return self.shape[idx], idx

    def __repr__(self):
        return f"DeviceMesh(shape={self.shape}, dim_names={self.dim_names})"


# ── Sharding specification ───────────────────────────────────────────────────

@dataclass
class ShardingSpec:
    """Full sharding specification for a tensor on a device mesh.

    `placements` has one entry per mesh dimension, describing how the tensor
    is distributed along that dimension.
    """
    placements: Tuple[Placement, ...]
    mesh: DeviceMesh

    def __post_init__(self):
        if len(self.placements) != self.mesh.ndim:
            raise ValueError(
                f"placements length {len(self.placements)} != mesh ndim {self.mesh.ndim}"
            )

    @property
    def partial(self) -> bool:
        return any(isinstance(p, Partial) for p in self.placements)

    def get_shard_dims(self) -> Dict[int, int]:
        """Return {tensor_dim: mesh_dim} for all Shard placements."""
        result = {}
        for mesh_dim, p in enumerate(self.placements):
            if isinstance(p, Shard):
                result[p.dim] = mesh_dim
        return result

    def __repr__(self):
        placements_str = ", ".join(repr(p) for p in self.placements)
        return f"ShardingSpec(({placements_str},), mesh={self.mesh})"


# ── Access pattern (for TIR blocking) ────────────────────────────────────────

@dataclass
class AccessPattern:
    """Describes how a buffer is accessed in a TIR block.

    Maps each buffer dimension to the loop variable that indexes it,
    used to determine how sharding interacts with compute.
    """
    buffer_name: str
    indices: Tuple[str, ...]  # loop variable name per dim, or None for constant


# ── Tensor state ─────────────────────────────────────────────────────────────

@dataclass
class TensorState:
    """Symbolic tensor state on a single device.

    Two-layer type model (SPMD types from meta-pytorch/spmd_types):
      - local_type: per-axis SPMD type (R/I/V/P) — LOCAL view
      - partition_spec: optional ShardingSpec — GLOBAL reassembly info

    For backward compatibility, the `sharding` and `placement` layer is
    preserved and auto-derived from local_type + partition_spec.
    """
    name: str
    global_shape: Tuple[int, ...]          # shape before sharding
    local_shape: Tuple[int, ...]           # shape on this device after sharding
    sharding: ShardingSpec                 # how the tensor is distributed (legacy, auto-derived)
    expr: str = ""                         # symbolic expression, e.g. "(x @ w)"

    # ── SPMD type layer (new) ──
    local_type: LocalSPMDType = field(default=None)  # auto-derived if None

    def __post_init__(self):
        """Auto-derive SPMD local_type from sharding if not explicitly set."""
        if self.local_type is None:
            if self.sharding.partial:
                object.__setattr__(self, 'local_type', LocalSPMDType.PARTIAL)
            elif not any(isinstance(p, Shard) for p in self.sharding.placements):
                object.__setattr__(self, 'local_type', LocalSPMDType.REPLICATE)
            else:
                object.__setattr__(self, 'local_type', LocalSPMDType.VARYING)

    # Autograd
    requires_grad: bool = False
    grad: Optional[TensorState] = None     # gradient tensor (populated by autograd)
    grad_name: str = ""                    # name of the gradient tensor

    # Pipeline parallelism
    stage: Optional[int] = None            # which PP stage this tensor lives on
    microbatch_id: Optional[int] = None    # which micro-batch (for 1F1B)
    is_activation: bool = False            # saved activation (memory management)

    # Context parallelism
    cp_rank: Optional[int] = None          # which rank in the CP ring

    # Async communication tracking
    _async_handle: Optional[str] = None    # handle name if tensor in-flight (AllReduceAsync etc.)

    @property
    def is_async_in_flight(self) -> bool:
        return self._async_handle is not None

    # ── SPMD type queries ──

    @property
    def is_replicate(self) -> bool:
        return self.local_type == LocalSPMDType.REPLICATE

    @property
    def is_invariant(self) -> bool:
        return self.local_type == LocalSPMDType.INVARIANT

    @property
    def is_varying(self) -> bool:
        return self.local_type == LocalSPMDType.VARYING

    @property
    def is_partial(self) -> bool:
        return self.local_type == LocalSPMDType.PARTIAL

    @property
    def gradient_type(self) -> LocalSPMDType:
        """SPMD gradient type (R↔P, I↔I, V↔V)."""
        return self.local_type.gradient_type()

    # ── Legacy accessors (backward compat) ──

    @property
    def partial(self) -> bool:
        """Legacy: true if type is PARTIAL."""
        return self.is_partial

    @property
    def is_replicated(self) -> bool:
        """Legacy: true if type is REPLICATE or INVARIANT."""
        return self.is_replicate or self.is_invariant

    # ── SPMD type helpers ──

    def with_local_type(self, lt: LocalSPMDType) -> "TensorState":
        """Return copy with different local SPMD type."""
        return TensorState(
            name=self.name, global_shape=self.global_shape,
            local_shape=self.local_shape, sharding=self.sharding,
            expr=self.expr, local_type=lt,
            requires_grad=self.requires_grad, grad=self.grad,
            grad_name=self.grad_name, stage=self.stage,
            microbatch_id=self.microbatch_id, is_activation=self.is_activation,
            cp_rank=self.cp_rank, _async_handle=self._async_handle,
        )

    def with_name(self, name: str) -> TensorState:
        """Return a copy with a different name."""
        return TensorState(
            name=name,
            global_shape=self.global_shape,
            local_shape=self.local_shape,
            sharding=self.sharding,
            expr=self.expr,
            requires_grad=self.requires_grad,
            grad=self.grad,
            grad_name=self.grad_name,
            stage=self.stage,
            microbatch_id=self.microbatch_id,
            is_activation=self.is_activation,
            cp_rank=self.cp_rank,
            _async_handle=self._async_handle,
        )

    def grad_tensor(self, name: str = "") -> TensorState:
        """Create a gradient tensor corresponding to this tensor."""
        gname = name or f"grad_{self.name}"
        return TensorState(
            name=gname,
            global_shape=self.global_shape,
            local_shape=self.local_shape,
            sharding=self.sharding,
            expr=f"grad({self.expr})" if self.expr else "",
            requires_grad=False,
            grad_name="",
            stage=self.stage,
            microbatch_id=self.microbatch_id,
            cp_rank=self.cp_rank,
        )

    def __hash__(self):
        return hash((
            self.name,
            self.global_shape,
            self.local_shape,
            self.sharding.placements,
            self.sharding.mesh.shape,
            self.sharding.mesh.dim_names,
        ))

    def __repr__(self):
        placement_str = ", ".join(repr(p) for p in self.sharding.placements)
        partial_str = " PARTIAL" if self.partial else ""
        stage_str = f" stage={self.stage}" if self.stage is not None else ""
        mb_str = f" mb={self.microbatch_id}" if self.microbatch_id is not None else ""
        return (
            f"TensorState({self.name}, shape={self.global_shape}"
            f"→{self.local_shape}, [{placement_str}]{partial_str}"
            f"{stage_str}{mb_str})"
        )


# ── Helper: apply sharding spec to global shape ───────────────────────────────

def compute_local_shape(
    global_shape: Tuple[int, ...],
    spec: ShardingSpec,
) -> Tuple[int, ...]:
    """Compute the local shape on one device given a sharding spec."""
    shape = list(global_shape)
    for mesh_dim, p in enumerate(spec.placements):
        if isinstance(p, Shard):
            mesh_size = spec.mesh.shape[mesh_dim]
            if shape[p.dim] % mesh_size != 0:
                raise ValueError(
                    f"Dimension {p.dim} size {shape[p.dim]} not divisible "
                    f"by mesh size {mesh_size}"
                )
            shape[p.dim] //= mesh_size
    # Partial and Replicate don't change local shape
    return tuple(shape)
