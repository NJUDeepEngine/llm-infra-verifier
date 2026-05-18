from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import IROp
from .spmd import SPMDGuard
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


class CollectiveOp(IROp):
    """Base for single-input collective communication ops.

    Subclasses define their placement transformation via _transform_placements()
    and gradient dual via vjp().
    """
    x: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def is_collective(self) -> bool:
        return True

    @abstractmethod
    def _transform_placements(
        self, placements: Tuple[Placement, ...], x: TensorState
    ) -> Tuple[Placement, ...]:
        """Return new placements after this collective."""
        ...

    def _validate_spmd(self, x: TensorState) -> None:
        """SPMD type precondition check. Override in subclasses."""
        pass

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        self._validate(x)
        self._validate_spmd(x)

        new_placements = self._transform_placements(x.sharding.placements, x)
        out_spec = ShardingSpec(placements=new_placements, mesh=x.sharding.mesh)
        out_local = compute_local_shape(x.global_shape, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            dtype=x.dtype,
            expr=x.expr,
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def _validate(self, x: TensorState) -> None:
        """Optional input validation. Override to add precondition checks."""
        pass

    def _make_grad(self, x: TensorState, expr: str) -> TensorState:
        """Helper to build a gradient TensorState with the input's metadata."""
        return TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=expr,
        )


@dataclass
class AllReduce(CollectiveOp):
    """AllReduce: sum partial tensors across a mesh dimension.

    Forward:  PARTIAL -> REPLICATE (after sum)
    Backward: AllReduce is self-dual
    """
    x: str
    output: str
    op_type: str = "sum"
    mesh_dim: Optional[int] = None

    def _validate(self, x: TensorState) -> None:
        if not x.partial:
            raise ValueError(
                f"AllReduce requires PARTIAL input for tensor '{x.name}', "
                f"got placements={x.sharding.placements} (shape={x.global_shape}, "
                f"local_shape={x.local_shape}). "
                f"AllReduce on a non-PARTIAL tensor is a no-op or indicates "
                f"a missing collective. "
                f"Op: AllReduce({self.x}) -> {self.output}"
            )

    def _validate_spmd(self, x):
        SPMDGuard.check_allreduce_input(x)

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            if isinstance(result[self.mesh_dim], Partial):
                result[self.mesh_dim] = Replicate()
            return tuple(result)
        return tuple(Replicate() if isinstance(p, Partial) else p for p in placements)

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"AllReduce(grad({x.expr}))" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return AllReduce(
            x=input_map.get(self.x, self.x), output=output_name,
            op_type=self.op_type, mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"AllReduce({self.x}, {self.op_type}{dim_str}) -> {self.output}"


@dataclass
class AllGather(CollectiveOp):
    """AllGather: gather sharded tensors along a dimension.

    Forward:  Shard(dim) -> Replicate (that dim is gathered)
    Backward: ReduceScatter (dual)
    """
    x: str
    output: str
    gather_dim: int
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            p = result[self.mesh_dim]
            if isinstance(p, Shard) and p.dim == self.gather_dim:
                result[self.mesh_dim] = Replicate()
            return tuple(result)
        return tuple(
            Replicate() if isinstance(p, Shard) and p.dim == self.gather_dim else p
            for p in placements
        )

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"ReduceScatter(grad({x.expr}), dim={self.gather_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return AllGather(
            x=input_map.get(self.x, self.x), output=output_name,
            gather_dim=self.gather_dim, mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"AllGather({self.x}, dim={self.gather_dim}{dim_str}) -> {self.output}"


@dataclass
class ReduceScatter(CollectiveOp):
    """ReduceScatter: reduce then scatter along a dimension.

    Forward:  Replicate/Partial -> Shard(scatter_dim)
    Backward: AllGather (dual)
    """
    x: str
    output: str
    scatter_dim: int
    op_type: str = "sum"
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            p = result[self.mesh_dim]
            if isinstance(p, (Replicate, Partial)):
                result[self.mesh_dim] = Shard(dim=self.scatter_dim)
            return tuple(result)
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
        if self.mesh_dim is not None:
            out_placements = list(x.sharding.placements)
            p = out_placements[self.mesh_dim]
            if isinstance(p, (Replicate, Partial)):
                out_placements[self.mesh_dim] = Replicate()
            out_placements = tuple(out_placements)
        else:
            out_placements = tuple(
                Replicate() if isinstance(p, (Replicate, Partial)) else p
                for p in x.sharding.placements
            )
        grad_spec = ShardingSpec(placements=out_placements, mesh=x.sharding.mesh)
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=compute_local_shape(x.global_shape, grad_spec),
            sharding=grad_spec,
            dtype=x.dtype,
            expr=f"AllGather(grad({x.expr}), dim={self.scatter_dim})" if x.expr else "",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return ReduceScatter(
            x=input_map.get(self.x, self.x), output=output_name,
            scatter_dim=self.scatter_dim, op_type=self.op_type,
            mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"ReduceScatter({self.x}, dim={self.scatter_dim}{dim_str}) -> {self.output}"


@dataclass
class Broadcast(CollectiveOp):
    """Broadcast: root rank sends data to all ranks.

    Forward:  any placement -> Replicate on all ranks
    Backward: Reduce (dual)
    """
    x: str
    output: str
    root: int = 0

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        return tuple(Replicate() for _ in placements)

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"Reduce(grad({x.expr}), root={self.root})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return Broadcast(
            x=input_map.get(self.x, self.x), output=output_name, root=self.root,
        )

    def __repr__(self):
        return f"Broadcast({self.x}, root={self.root}) -> {self.output}"


@dataclass
class Reduce(CollectiveOp):
    """Reduce: all ranks contribute, only root gets the result.

    Forward:  PARTIAL -> Replicate (resolved on root)
    Backward: Broadcast (dual)
    """
    x: str
    output: str
    root: int = 0
    op_type: str = "sum"

    def _validate(self, x: TensorState) -> None:
        if not x.partial:
            raise ValueError(
                f"Reduce requires PARTIAL input for tensor '{x.name}', "
                f"got placements={x.sharding.placements}. "
                f"Op: Reduce({self.x}) -> {self.output}"
            )

    def _validate_spmd(self, x):
        SPMDGuard.check_allreduce_input(x)

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        return tuple(Replicate() if isinstance(p, Partial) else p for p in placements)

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"Broadcast(grad({x.expr}), root={self.root})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return Reduce(
            x=input_map.get(self.x, self.x), output=output_name,
            root=self.root, op_type=self.op_type,
        )

    def __repr__(self):
        return f"Reduce({self.x}, root={self.root}, {self.op_type}) -> {self.output}"


@dataclass
class AllToAll(CollectiveOp):
    """AllToAll: each rank sends different data to each other rank.

    Forward:  Shard(split_dim) -> Shard(concat_dim)
    Backward: AllToAll with reversed dimensions (self-dual with dim swap)

    Primary use case: MoE expert dispatch/combine (token routing).
    """
    x: str
    output: str
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
            x, f"AllToAll(grad({x.expr}), {self.concat_dim}->{self.split_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return AllToAll(
            x=input_map.get(self.x, self.x), output=output_name,
            split_dim=self.split_dim, concat_dim=self.concat_dim,
            mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"AllToAll({self.x}, split={self.split_dim}, concat={self.concat_dim}{dim_str}) -> {self.output}"


@dataclass
class Scatter(CollectiveOp):
    """Scatter: root distributes different chunks to each rank.

    Forward:  Replicate (on root) -> Shard(dim)
    Backward: Gather (dual)
    """
    x: str
    output: str
    scatter_dim: int
    root: int = 0
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.VARYING

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            if isinstance(result[self.mesh_dim], Replicate):
                result[self.mesh_dim] = Shard(dim=self.scatter_dim)
            return tuple(result)
        result = []
        replaced = False
        for p in placements:
            if isinstance(p, Replicate) and not replaced:
                result.append(Shard(dim=self.scatter_dim))
                replaced = True
            else:
                result.append(p)
        return tuple(result)

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"Gather(grad({x.expr}), dim={self.scatter_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return Scatter(
            x=input_map.get(self.x, self.x), output=output_name,
            scatter_dim=self.scatter_dim, root=self.root,
            mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"Scatter({self.x}, dim={self.scatter_dim}, root={self.root}{dim_str}) -> {self.output}"


@dataclass
class Gather(CollectiveOp):
    """Gather: each rank sends to root, root concatenates.

    Forward:  Shard(dim) -> Replicate (on root)
    Backward: Scatter (dual)
    """
    x: str
    output: str
    gather_dim: int
    root: int = 0
    mesh_dim: Optional[int] = None

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def _transform_placements(self, placements, x):
        if self.mesh_dim is not None:
            result = list(placements)
            p = result[self.mesh_dim]
            if isinstance(p, Shard) and p.dim == self.gather_dim:
                result[self.mesh_dim] = Replicate()
            return tuple(result)
        return tuple(
            Replicate() if isinstance(p, Shard) and p.dim == self.gather_dim else p
            for p in placements
        )

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: self._make_grad(
            x, f"Scatter(grad({x.expr}), dim={self.gather_dim})" if x.expr else ""
        )}

    def clone_with_names(self, input_map, output_name):
        return Gather(
            x=input_map.get(self.x, self.x), output=output_name,
            gather_dim=self.gather_dim, root=self.root,
            mesh_dim=self.mesh_dim,
        )

    def __repr__(self):
        dim_str = f", mesh_dim={self.mesh_dim}" if self.mesh_dim is not None else ""
        return f"Gather({self.x}, dim={self.gather_dim}, root={self.root}{dim_str}) -> {self.output}"
