from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from .base import IROp
from .collective import CollectiveOp
from ..state import (
    TensorState,
    LocalSPMDType,
    Shard,
    Replicate,
    Partial,
    Placement,
    ShardingSpec,
    compute_local_shape,
)


@dataclass
class ZeROGatherParam(CollectiveOp):
    """ZeRO Stage 3: AllGather sharded parameters for forward compute.

    Forward:  Shard(dim) -> Replicate
    Backward: ZeROScatterGrad (ReduceScatter dual)
    """
    x: str
    output: str
    gather_dim: int
    zero_stage: int = 3

    def _validate(self, x: TensorState) -> None:
        if self.zero_stage < 3:
            raise ValueError(
                f"ZeROGatherParam requires stage >= 3, got stage={self.zero_stage}. "
                f"Stages 1-2 do not shard parameters."
            )

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        return tuple(
            Replicate() if isinstance(p, Shard) and p.dim == self.gather_dim else p
            for p in placements
        )

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"ZeROScatterGrad(grad({x.expr}), dim={self.gather_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return ZeROGatherParam(
            x=input_map.get(self.x, self.x), output=output_name,
            gather_dim=self.gather_dim, zero_stage=self.zero_stage,
        )

    def __repr__(self):
        return f"ZeROGatherParam({self.x}, dim={self.gather_dim}, stage={self.zero_stage}) -> {self.output}"


@dataclass
class ZeROScatterGrad(CollectiveOp):
    """ZeRO Stage 2+: ReduceScatter gradients across data-parallel group.

    Forward:  Partial/Replicate -> Shard(scatter_dim)
    Backward: ZeROGatherParam (AllGather dual)
    """
    x: str
    output: str
    scatter_dim: int
    zero_stage: int = 2

    def _validate(self, x: TensorState) -> None:
        if self.zero_stage < 2:
            raise ValueError(
                f"ZeROScatterGrad requires stage >= 2, got stage={self.zero_stage}. "
                f"Stage 1 does not shard gradients."
            )

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def _transform_placements(self, placements, x):
        result = []
        found = False
        for p in placements:
            if isinstance(p, (Replicate, Partial)) and not found:
                result.append(Shard(dim=self.scatter_dim))
                found = True
            else:
                result.append(p)
        return tuple(result)

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"ZeROGatherParam(grad({x.expr}), dim={self.scatter_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return ZeROScatterGrad(
            x=input_map.get(self.x, self.x), output=output_name,
            scatter_dim=self.scatter_dim, zero_stage=self.zero_stage,
        )

    def __repr__(self):
        return f"ZeROScatterGrad({self.x}, dim={self.scatter_dim}, stage={self.zero_stage}) -> {self.output}"


@dataclass
class ZeROPartitionOptState(IROp):
    """ZeRO Stage 1+: Mark optimizer state as partitioned (no communication).

    This is a bookkeeping op: marks that each rank only stores a shard
    of the optimizer state. No actual data movement occurs.
    """
    x: str
    output: str
    partition_dim: int
    zero_stage: int = 1

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        new_placements = tuple(
            Shard(dim=self.partition_dim) if isinstance(p, Replicate) else p
            for p in x.sharding.placements
        )
        out_spec = ShardingSpec(placements=new_placements, mesh=x.sharding.mesh)
        out_local = compute_local_shape(x.global_shape, out_spec)

        result = replace(
            x,
            name=self.output,
            local_shape=out_local,
            sharding=out_spec,
            zero_stage=self.zero_stage,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {}

    def clone_with_names(self, input_map, output_name):
        return ZeROPartitionOptState(
            x=input_map.get(self.x, self.x), output=output_name,
            partition_dim=self.partition_dim, zero_stage=self.zero_stage,
        )

    def __repr__(self):
        return f"ZeROPartitionOptState({self.x}, dim={self.partition_dim}, stage={self.zero_stage}) -> {self.output}"
