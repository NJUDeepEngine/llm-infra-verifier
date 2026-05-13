"""TensorState: symbolic tensor state on a single device.

Tracks per-device tensor metadata: placement, sharding spec,
symbolic expression, and autograd/pipeline annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .placement import LocalSPMDType, Shard
from .sharding import ShardingSpec


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
    global_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    sharding: ShardingSpec
    expr: str = ""

    local_type: Optional[LocalSPMDType] = field(default=None)

    def __post_init__(self):
        """Auto-derive SPMD local_type from sharding if not explicitly set."""
        if self.local_type is None:
            if self.sharding.partial:
                object.__setattr__(self, 'local_type', LocalSPMDType.PARTIAL)
            elif not any(isinstance(p, Shard) for p in self.sharding.placements):
                object.__setattr__(self, 'local_type', LocalSPMDType.REPLICATE)
            else:
                object.__setattr__(self, 'local_type', LocalSPMDType.VARYING)

    requires_grad: bool = False
    grad: Optional[TensorState] = None
    grad_name: str = ""

    stage: Optional[int] = None
    microbatch_id: Optional[int] = None
    is_activation: bool = False

    cp_rank: Optional[int] = None

    _async_handle: Optional[str] = None

    @property
    def is_async_in_flight(self) -> bool:
        return self._async_handle is not None

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

    @property
    def partial(self) -> bool:
        """Legacy: true if type is PARTIAL."""
        return self.is_partial

    @property
    def is_replicated(self) -> bool:
        """Legacy: true if type is REPLICATE or INVARIANT."""
        return self.is_replicate or self.is_invariant

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
