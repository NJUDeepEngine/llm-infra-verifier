"""TensorState: symbolic tensor state on a single device.

Tracks per-device tensor metadata: placement, sharding spec,
symbolic expression, and autograd/pipeline annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from .placement import LocalSPMDType, Shard
from .sharding import ShardingSpec, compute_local_shape


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
        # INVARIANT 无法从 placement 自动推导（placement 层没有 Invariant 类型）。
        # INVARIANT 只能通过 Reinterpret(R->I, expert_mode=True) 显式设置。
        # 在 placement 层，INVARIANT tensor 的 sharding 为 Replicate()。
        if self.local_type is None:
            # Priority: Partial > Varying > Replicate
            # Partial 优先：表示"有待求和"，是影响 AllReduce 正确性的关键属性。
            # 对于混合 placement 如 (Shard(0), Partial())，Partial 决定语义。
            if self.sharding.partial:
                self.local_type = LocalSPMDType.PARTIAL
            elif not any(isinstance(p, Shard) for p in self.sharding.placements):
                self.local_type = LocalSPMDType.REPLICATE
            else:
                self.local_type = LocalSPMDType.VARYING

        expected = compute_local_shape(self.global_shape, self.sharding)
        if self.local_shape != expected:
            raise ValueError(
                f"TensorState '{self.name}': local_shape {self.local_shape} "
                f"!= computed {expected} from global_shape={self.global_shape}, "
                f"placements={self.sharding.placements}"
            )

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
        """Return copy with different local SPMD type.

        WARNING: local_type may disagree with the placement-derived type.
        Only use in SPMD-specific ops (Reinterpret, Convert).
        """
        return replace(self, local_type=lt)

    def with_name(self, name: str) -> TensorState:
        """Return a copy with a different name."""
        return replace(self, name=name)

    def grad_tensor(self, name: str = "") -> TensorState:
        """Create a gradient tensor corresponding to this tensor."""
        return replace(
            self,
            name=name or f"grad_{self.name}",
            expr=f"grad({self.expr})" if self.expr else "",
            requires_grad=False,
            grad=None,
            grad_name="",
            _async_handle=None,
            is_activation=False,
        )

    def __hash__(self):
        # device_ids 不参与 hash：同一逻辑 mesh (shape + dim_names) 的不同物理映射
        # 应视为等价。device_ids 仅影响执行时的设备分配，不影响语义。
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
