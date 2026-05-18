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
class TopKGate(IROp):
    """Top-K expert gating: select top_k experts per token.

    Produces two outputs:
      - output:         gate weights  (same shape as x, sparse)
      - indices_output:  expert indices (same batch dim, top_k)
    """
    x: str
    gate_weight: str
    output: str
    indices_output: str
    num_experts: int
    top_k: int = 1
    capacity_factor: float = 1.0

    @property
    def input_names(self) -> List[str]:
        return [self.x, self.gate_weight]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        gw = ctx[self.gate_weight]

        result = replace(
            x,
            name=self.output,
            expr=f"topk_gate({x.expr}, k={self.top_k})" if x.expr else "",
            num_experts=self.num_experts,
        )
        ctx[self.output] = result

        batch_size = x.global_shape[0]
        indices_global = (batch_size, self.top_k)
        indices_local = compute_local_shape(indices_global, x.sharding)
        indices = TensorState(
            name=self.indices_output,
            global_shape=indices_global,
            local_shape=indices_local,
            sharding=x.sharding,
            expr=f"topk_indices({x.expr}, k={self.top_k})" if x.expr else "",
            num_experts=self.num_experts,
        )
        ctx[self.indices_output] = indices

        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"topk_gate_grad({x.expr})" if x.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return TopKGate(
            x=input_map.get(self.x, self.x),
            gate_weight=input_map.get(self.gate_weight, self.gate_weight),
            output=output_name,
            indices_output=self.indices_output,
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
        )

    def __repr__(self):
        return (
            f"TopKGate({self.x}, experts={self.num_experts}, k={self.top_k}) "
            f"-> {self.output}, {self.indices_output}"
        )


@dataclass
class MoEDispatch(CollectiveOp):
    """MoE AllToAll dispatch: route tokens to their assigned experts.

    Forward:  Shard(split_dim) -> Shard(concat_dim)
    Backward: MoECombine (reverse AllToAll)
    """
    x: str
    output: str
    num_experts: int
    split_dim: int
    concat_dim: int
    expert_capacity: Optional[int] = None
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            p = result[self.mesh_dim]
            if isinstance(p, Shard) and p.dim == self.split_dim:
                result[self.mesh_dim] = Shard(dim=self.concat_dim)
            return tuple(result)
        return tuple(
            Shard(dim=self.concat_dim) if isinstance(p, Shard) and p.dim == self.split_dim else p
            for p in placements
        )

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        result = super().apply(ctx)
        result.num_experts = self.num_experts
        result.expert_capacity = self.expert_capacity
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"MoECombine(grad({x.expr}), {self.concat_dim}->{self.split_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return MoEDispatch(
            x=input_map.get(self.x, self.x), output=output_name,
            num_experts=self.num_experts,
            split_dim=self.split_dim, concat_dim=self.concat_dim,
            expert_capacity=self.expert_capacity, mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        cap = f", cap={self.expert_capacity}" if self.expert_capacity else ""
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return (
            f"MoEDispatch({self.x}, experts={self.num_experts}, "
            f"split={self.split_dim}->concat={self.concat_dim}{cap}{dim_str}) -> {self.output}"
        )


@dataclass
class MoECombine(CollectiveOp):
    """MoE AllToAll combine: collect expert outputs back to original token order.

    Forward:  Shard(split_dim) -> Shard(concat_dim)
    Backward: MoEDispatch (reverse AllToAll)
    """
    x: str
    output: str
    num_experts: int
    split_dim: int
    concat_dim: int
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            p = result[self.mesh_dim]
            if isinstance(p, Shard) and p.dim == self.split_dim:
                result[self.mesh_dim] = Shard(dim=self.concat_dim)
            return tuple(result)
        return tuple(
            Shard(dim=self.concat_dim) if isinstance(p, Shard) and p.dim == self.split_dim else p
            for p in placements
        )

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"MoEDispatch(grad({x.expr}), {self.concat_dim}->{self.split_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return MoECombine(
            x=input_map.get(self.x, self.x), output=output_name,
            num_experts=self.num_experts,
            split_dim=self.split_dim, concat_dim=self.concat_dim,
            mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return (
            f"MoECombine({self.x}, experts={self.num_experts}, "
            f"split={self.split_dim}->concat={self.concat_dim}{dim_str}) -> {self.output}"
        )


@dataclass
class ExpertCompute(IROp):
    """Mark a computation region as expert-local (no collective ops allowed).

    Bookkeeping op: annotates the tensor with expert_id metadata.
    Shape and sharding pass through unchanged.
    """
    x: str
    output: str
    expert_id: int
    num_experts: int

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        result = replace(
            x,
            name=self.output,
            expert_id=self.expert_id,
            num_experts=self.num_experts,
            expr=f"expert{self.expert_id}({x.expr})" if x.expr else "",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"expert{self.expert_id}_grad({x.expr})" if x.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return ExpertCompute(
            x=input_map.get(self.x, self.x), output=output_name,
            expert_id=self.expert_id, num_experts=self.num_experts,
        )

    def __repr__(self):
        return f"ExpertCompute({self.x}, expert={self.expert_id}/{self.num_experts}) -> {self.output}"
