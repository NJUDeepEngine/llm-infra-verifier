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


# ── Placement types ──────────────────────────────────────────────────────────

class PlacementType(Enum):
    REPLICATE = "replicate"
    SHARD = "shard"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Shard:
    """Tensor is sharded along dimension `dim` across a mesh dimension."""
    dim: int

    def __repr__(self):
        return f"Shard({self.dim})"


@dataclass(frozen=True)
class Replicate:
    """Tensor is replicated across all devices in the mesh dimension."""

    def __repr__(self):
        return "Replicate()"


@dataclass(frozen=True)
class Partial:
    """Tensor is partially reduced — needs AllReduce to become Replicate."""

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

    Tracks placement, sharding, shape, symbolic expression, and autograd
    / pipeline metadata.  This is the central data structure that flows
    through the symbolic executor.
    """
    name: str
    global_shape: Tuple[int, ...]          # shape before sharding
    local_shape: Tuple[int, ...]           # shape on this device after sharding
    sharding: ShardingSpec                 # how the tensor is distributed
    expr: str = ""                         # symbolic expression, e.g. "(x @ w)"

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

    @property
    def partial(self) -> bool:
        return self.sharding.partial

    @property
    def is_replicated(self) -> bool:
        return all(isinstance(p, Replicate) for p in self.sharding.placements)

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
