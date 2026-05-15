from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import IROp
from ..state import (
    TensorState,
    Shard,
    ShardingSpec,
    compute_local_shape,
)


@dataclass
class Reshape(IROp):
    """Reshape tensor dimensions."""
    x: str
    output: str
    new_shape: Tuple[int, ...]

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
        result = TensorState(
            name=self.output,
            global_shape=self.new_shape,
            local_shape=compute_local_shape(self.new_shape, x.sharding),
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"reshape({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"reshape_inv(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Reshape(
            x=input_map.get(self.x, self.x),
            output=output_name,
            new_shape=self.new_shape,
        )

    def __repr__(self):
        return f"Reshape({self.x}) -> {self.output}"


@dataclass
class Transpose(IROp):
    """Transpose tensor: Y = X^T."""
    x: str
    output: str
    dim0: int = 0
    dim1: int = 1

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
        new_global = list(x.global_shape)
        new_global[self.dim0], new_global[self.dim1] = (
            new_global[self.dim1],
            new_global[self.dim0],
        )
        new_local = list(x.local_shape)
        new_local[self.dim0], new_local[self.dim1] = (
            new_local[self.dim1],
            new_local[self.dim0],
        )

        new_placements = list(x.sharding.placements)
        for i, p in enumerate(new_placements):
            if isinstance(p, Shard):
                if p.dim == self.dim0:
                    new_placements[i] = Shard(dim=self.dim1)
                elif p.dim == self.dim1:
                    new_placements[i] = Shard(dim=self.dim0)

        out_spec = ShardingSpec(
            placements=tuple(new_placements),
            mesh=x.sharding.mesh,
        )

        result = TensorState(
            name=self.output,
            global_shape=tuple(new_global),
            local_shape=tuple(new_local),
            sharding=out_spec,
            dtype=x.dtype,
            expr=f"transpose({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"transpose(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Transpose(
            x=input_map.get(self.x, self.x),
            output=output_name,
            dim0=self.dim0,
            dim1=self.dim1,
        )

    def __repr__(self):
        return f"Transpose({self.x}, {self.dim0}<->{self.dim1}) -> {self.output}"
