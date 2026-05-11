"""Distributed IR operations with forward placement semantics and VJP rules.

Each op encodes:
  - forward:  placement propagation (how sharding flows through compute)
  - vjp:      vector-Jacobian product for backward pass
  - constraints: Z3-encodable legality conditions

The IR is the intermediate representation between TileLang TIR (lifted via
TIRLifter) and the symbolic executor / verifier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy

from .state import (
    TensorState,
    Shard,
    Replicate,
    Partial,
    Placement,
    ShardingSpec,
    DeviceMesh,
    compute_local_shape,
)


# ── Base IR operation ────────────────────────────────────────────────────────

class IROp(ABC):
    """Abstract base for all IR operations."""

    def __init__(self):
        self._id = id(self)

    @abstractmethod
    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        """Execute forward, returning the output TensorState.

        ctx is a mutable dict of {name: TensorState} representing the
        current symbolic state of all tensors.
        """
        ...

    @abstractmethod
    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        """Compute VJP, returning {input_name: grad_tensor}.

        ctx contains the tensor states at the time this op was executed
        (including saved tensors for the backward).
        """
        ...

    @abstractmethod
    def is_collective(self) -> bool:
        """Whether this op involves cross-device communication."""
        ...

    @abstractmethod
    def is_p2p(self) -> bool:
        """Whether this op is point-to-point (Send/Recv)."""
        ...

    def is_async(self) -> bool:
        """Whether this op is asynchronous (returns before completion)."""
        return False

    def is_sync(self) -> bool:
        """Whether this op is a synchronization point (Wait/WaitAll)."""
        return False

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """Names of input tensors."""
        ...

    @property
    @abstractmethod
    def output_name(self) -> str:
        """Name of the output tensor."""
        ...

    @abstractmethod
    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        """Clone this op with renamed inputs/output."""
        ...


# ── Compute ops ──────────────────────────────────────────────────────────────

@dataclass
class MatMul(IROp):
    """Matrix multiplication: Y = A @ B.

    Placement rules:
      - Shard(dim0=A_rows) @ Shard(dim1=B_cols) → PARTIAL on reduce dim
      - Shard(dim0=A_rows) @ Replicate → Shard(dim0=rows) (data parallel)
      - Replicate @ Shard(dim0=B_rows) → Shard(dim0=rows) (no comm needed)
      - Shard(dim1=A_cols) @ Shard(dim0=B_rows) → Replicate (column parallel)
    """
    a: str
    b: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.a, self.b]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a = ctx[self.a]
        b = ctx[self.b]

        # Determine output shape
        out_global = (a.global_shape[0], b.global_shape[1])

        # Determine output placement based on input placements
        a_spec = a.sharding
        b_spec = b.sharding

        # Check for row-parallel pattern: both sharded on the reduce dim
        a_shard_dims = a_spec.get_shard_dims()
        b_shard_dims = b_spec.get_shard_dims()

        is_partial = False
        out_placements = list(a_spec.placements)  # start from a's placements

        for mesh_dim, p in enumerate(b_spec.placements):
            if isinstance(p, Shard):
                if p.dim == 0:  # b is sharded on rows (= reduce dim)
                    for a_mesh_dim, a_p in enumerate(a_spec.placements):
                        if isinstance(a_p, Shard) and a_p.dim == 1:  # a sharded on cols
                            if a_mesh_dim == mesh_dim:
                                # Both sharded on reduce dim → PARTIAL
                                is_partial = True
                                out_placements[mesh_dim] = Partial()
                elif p.dim == 1:  # b is sharded on cols
                    out_placements[mesh_dim] = Shard(dim=1)

        out_spec = ShardingSpec(
            placements=tuple(out_placements),
            mesh=a_spec.mesh,
        )
        out_local = compute_local_shape(out_global, out_spec)

        expr = f"({a.expr} @ {b.expr})" if a.expr and b.expr else ""

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            expr=expr,
            requires_grad=a.requires_grad or b.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        a = ctx[self.a]
        b = ctx[self.b]

        grad_a = TensorState(
            name=f"grad_{self.a}",
            global_shape=a.global_shape,
            local_shape=a.local_shape,
            sharding=a.sharding,
            expr=f"grad({a.expr})" if a.expr else "",
            requires_grad=False,
        )
        grad_b = TensorState(
            name=f"grad_{self.b}",
            global_shape=b.global_shape,
            local_shape=b.local_shape,
            sharding=b.sharding,
            expr=f"grad({b.expr})" if b.expr else "",
            requires_grad=False,
        )

        # grad_a = grad_Y @ B^T  (shape: A.shape)
        # grad_b = A^T @ grad_Y  (shape: B.shape)

        return {self.a: grad_a, self.b: grad_b}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return MatMul(
            a=input_map.get(self.a, self.a),
            b=input_map.get(self.b, self.b),
            output=output_name,
        )

    def __repr__(self):
        return f"MatMul({self.a}, {self.b}) -> {self.output}"


@dataclass
class Add(IROp):
    """Element-wise addition: Y = A + B.

    Both inputs must have compatible placements. Output inherits placement.
    """
    a: str
    b: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.a, self.b]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a = ctx[self.a]
        b = ctx[self.b]

        if a.global_shape != b.global_shape:
            raise ValueError(
                f"Add: shape mismatch between '{self.a}' {a.global_shape} and "
                f"'{self.b}' {b.global_shape}. "
                f"Element-wise ops require both inputs to have the same global shape. "
                f"Op: Add({self.a}, {self.b}) -> {self.output}"
            )

        if a.sharding.mesh != b.sharding.mesh:
            raise ValueError(
                f"Add: mesh mismatch between '{self.a}' {a.sharding.mesh} and "
                f"'{self.b}' {b.sharding.mesh}. "
                f"Op: Add({self.a}, {self.b}) -> {self.output}"
            )

        out_placements = []
        for mesh_dim, (pa, pb) in enumerate(zip(a.sharding.placements, b.sharding.placements)):
            if isinstance(pa, Replicate):
                out_placements.append(pb)
            elif isinstance(pb, Replicate):
                out_placements.append(pa)
            elif type(pa) is type(pb) and isinstance(pa, Shard) and pa.dim == pb.dim:
                out_placements.append(pa)
            elif type(pa) is type(pb) and isinstance(pa, Partial):
                out_placements.append(pa)
            else:
                raise ValueError(
                    f"Add: incompatible placements at mesh dim {mesh_dim}: "
                    f"'{self.a}' has {pa}, '{self.b}' has {pb}. "
                    f"Element-wise ops require matching sharding, or one input to be Replicate. "
                    f"Op: Add({self.a}, {self.b}) -> {self.output}"
                )

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=a.sharding.mesh)
        out_local = compute_local_shape(a.global_shape, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=a.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            expr=f"({a.expr} + {b.expr})" if a.expr and b.expr else "",
            requires_grad=a.requires_grad or b.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        # Element-wise: gradient passes through
        a = ctx[self.a]
        b = ctx[self.b]
        grad_a = grad_output.with_name(f"grad_{self.a}")
        grad_b = grad_output.with_name(f"grad_{self.b}")
        return {self.a: grad_a, self.b: grad_b}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Add(
            a=input_map.get(self.a, self.a),
            b=input_map.get(self.b, self.b),
            output=output_name,
        )

    def __repr__(self):
        return f"Add({self.a}, {self.b}) -> {self.output}"


@dataclass
class Multiply(IROp):
    """Element-wise multiplication: Y = A * B."""
    a: str
    b: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.a, self.b]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a = ctx[self.a]
        b = ctx[self.b]

        if a.global_shape != b.global_shape:
            raise ValueError(
                f"Multiply: shape mismatch between '{self.a}' {a.global_shape} and "
                f"'{self.b}' {b.global_shape}. "
                f"Element-wise ops require both inputs to have the same global shape. "
                f"Op: Multiply({self.a}, {self.b}) -> {self.output}"
            )

        if a.sharding.mesh != b.sharding.mesh:
            raise ValueError(
                f"Multiply: mesh mismatch between '{self.a}' {a.sharding.mesh} and "
                f"'{self.b}' {b.sharding.mesh}. "
                f"Op: Multiply({self.a}, {self.b}) -> {self.output}"
            )

        out_placements = []
        for mesh_dim, (pa, pb) in enumerate(zip(a.sharding.placements, b.sharding.placements)):
            if isinstance(pa, Replicate):
                out_placements.append(pb)
            elif isinstance(pb, Replicate):
                out_placements.append(pa)
            elif type(pa) is type(pb) and isinstance(pa, Shard) and pa.dim == pb.dim:
                out_placements.append(pa)
            elif type(pa) is type(pb) and isinstance(pa, Partial):
                out_placements.append(pa)
            else:
                raise ValueError(
                    f"Multiply: incompatible placements at mesh dim {mesh_dim}: "
                    f"'{self.a}' has {pa}, '{self.b}' has {pb}. "
                    f"Element-wise ops require matching sharding, or one input to be Replicate. "
                    f"Op: Multiply({self.a}, {self.b}) -> {self.output}"
                )

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=a.sharding.mesh)
        out_local = compute_local_shape(a.global_shape, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=a.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            expr=f"({a.expr} * {b.expr})" if a.expr and b.expr else "",
            requires_grad=a.requires_grad or b.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        a = ctx[self.a]
        b = ctx[self.b]
        # grad_a = grad_Y * b, grad_b = grad_Y * a
        grad_a = TensorState(
            name=f"grad_{self.a}",
            global_shape=a.global_shape,
            local_shape=a.local_shape,
            sharding=a.sharding,
            expr=f"grad({a.expr}) * {b.expr}" if a.expr and b.expr else "",
        )
        grad_b = TensorState(
            name=f"grad_{self.b}",
            global_shape=b.global_shape,
            local_shape=b.local_shape,
            sharding=b.sharding,
            expr=f"grad({a.expr}) * {a.expr}" if a.expr and b.expr else "",
        )
        return {self.a: grad_a, self.b: grad_b}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Multiply(
            a=input_map.get(self.a, self.a),
            b=input_map.get(self.b, self.b),
            output=output_name,
        )

    def __repr__(self):
        return f"Multiply({self.a}, {self.b}) -> {self.output}"


@dataclass
class SiLU(IROp):
    """SiLU activation: Y = X * sigmoid(X)."""
    x: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"silu({x.expr})" if x.expr else "",
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
            expr=f"silu_grad({x.expr})",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return SiLU(
            x=input_map.get(self.x, self.x),
            output=output_name,
        )

    def __repr__(self):
        return f"SiLU({self.x}) -> {self.output}"


# ── Collective communication ops ─────────────────────────────────────────────

@dataclass
class AllReduce(IROp):
    """AllReduce: sum partial tensors across a mesh dimension.

    Forward:  PARTIAL → REPLICATE (after sum)
    Backward: REPLICATE → REPLICATE (AllReduce is self-dual)
    """
    x: str
    output: str
    op_type: str = "sum"  # "sum" | "avg" | "max"

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        if not x.partial:
            raise ValueError(
                f"AllReduce requires PARTIAL input for tensor '{x.name}', "
                f"got placements={x.sharding.placements} (shape={x.global_shape}, "
                f"local_shape={x.local_shape}). "
                f"AllReduce on a non-PARTIAL tensor is a no-op or indicates "
                f"a missing collective. "
                f"Op: AllReduce({self.x}) -> {self.output}"
            )

        # AllReduce converts Partial → Replicate (on the partial mesh dim)
        new_placements = []
        for p in x.sharding.placements:
            if isinstance(p, Partial):
                new_placements.append(Replicate())
            else:
                new_placements.append(p)

        out_spec = ShardingSpec(
            placements=tuple(new_placements),
            mesh=x.sharding.mesh,
        )
        out_local = x.global_shape  # Replicated → local == global

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            expr=x.expr,  # AllReduce preserves the expression
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
        # AllReduce(sum) is self-dual: grad_x = AllReduce(grad_y)
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"AllReduce(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return AllReduce(
            x=input_map.get(self.x, self.x),
            output=output_name,
            op_type=self.op_type,
        )

    def __repr__(self):
        return f"AllReduce({self.x}, {self.op_type}) -> {self.output}"


@dataclass
class AllGather(IROp):
    """AllGather: gather sharded tensors along a dimension.

    Forward:  Shard(dim) → Replicate (that dim is gathered)
    Backward: ReduceScatter (dual of AllGather)
    """
    x: str
    output: str
    gather_dim: int

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]

        # Find the Shard placement on gather_dim
        new_placements = []
        for p in x.sharding.placements:
            if isinstance(p, Shard) and p.dim == self.gather_dim:
                new_placements.append(Replicate())
            else:
                new_placements.append(p)

        out_spec = ShardingSpec(
            placements=tuple(new_placements),
            mesh=x.sharding.mesh,
        )

        # Full shape after gather
        out_global = x.global_shape
        out_local = out_global  # replicated → local is full

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            expr=x.expr,
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
        # VJP of AllGather is ReduceScatter on the same dim
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"ReduceScatter(grad({x.expr}), dim={self.gather_dim})" if x.expr else "",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return AllGather(
            x=input_map.get(self.x, self.x),
            output=output_name,
            gather_dim=self.gather_dim,
        )

    def __repr__(self):
        return f"AllGather({self.x}, dim={self.gather_dim}) -> {self.output}"


@dataclass
class ReduceScatter(IROp):
    """ReduceScatter: reduce then scatter along a dimension.

    Forward:  Replicate → Shard(scatter_dim) with partial sum
    Backward: AllGather (dual)
    """
    x: str
    output: str
    scatter_dim: int
    op_type: str = "sum"

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]

        # Replace Replicate with Shard(scatter_dim)
        new_placements = []
        found = False
        for p in x.sharding.placements:
            if isinstance(p, Replicate) and not found:
                new_placements.append(Shard(dim=self.scatter_dim))
                found = True
            else:
                new_placements.append(p)

        out_spec = ShardingSpec(
            placements=tuple(new_placements),
            mesh=x.sharding.mesh,
        )
        out_local = compute_local_shape(x.global_shape, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            expr=x.expr,
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
        # VJP of ReduceScatter is AllGather
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.global_shape,
            sharding=ShardingSpec(
                placements=tuple(
                    Replicate() if isinstance(p, Replicate) else p
                    for p in x.sharding.placements
                ),
                mesh=x.sharding.mesh,
            ),
            expr=f"AllGather(grad({x.expr}), dim={self.scatter_dim})" if x.expr else "",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return ReduceScatter(
            x=input_map.get(self.x, self.x),
            output=output_name,
            scatter_dim=self.scatter_dim,
            op_type=self.op_type,
        )

    def __repr__(self):
        return f"ReduceScatter({self.x}, dim={self.scatter_dim}) -> {self.output}"


# ── Point-to-point communication ops ─────────────────────────────────────────

@dataclass
class Send(IROp):
    """Send tensor to another device (PP).

    Forward:  data flows src → dst
    Backward: Recv(grad, from=dst) — direction reversed
    """
    x: str
    output: str
    src: int
    dst: int
    stage: int
    microbatch_id: int

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        # Send copies the tensor to the destination device
        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=x.expr,
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
            stage=self.dst,
            microbatch_id=self.microbatch_id,
            is_activation=True,  # sent activations need saving for bwd
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        # VJP of Send is Recv (direction reversed)
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"Recv(grad({x.expr}), src={self.dst})" if x.expr else "",
            stage=self.src,
            microbatch_id=self.microbatch_id,
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return True

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Send(
            x=input_map.get(self.x, self.x),
            output=output_name,
            src=self.src,
            dst=self.dst,
            stage=self.stage,
            microbatch_id=self.microbatch_id,
        )

    def __repr__(self):
        return (
            f"Send({self.x}, {self.src}→{self.dst}, "
            f"stage={self.stage}, mb={self.microbatch_id}) -> {self.output}"
        )


@dataclass
class Recv(IROp):
    """Receive tensor from another device (PP).

    Forward:  data flows src → dst
    Backward: Send(grad, to=src) — direction reversed
    """
    x: str       # placeholder name on the receiving side
    output: str
    src: int
    dst: int
    stage: int
    microbatch_id: int

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx.get(self.x)
        if x is None:
            raise ValueError(f"Recv: source tensor '{self.x}' not found in context")

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=x.expr,
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
            stage=self.dst,
            microbatch_id=self.microbatch_id,
            is_activation=True,
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        # VJP of Recv is Send (direction reversed)
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"Send(grad({x.expr}), dst={self.src})" if x.expr else "",
            stage=self.dst,
            microbatch_id=self.microbatch_id,
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return True

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Recv(
            x=input_map.get(self.x, self.x),
            output=output_name,
            src=self.src,
            dst=self.dst,
            stage=self.stage,
            microbatch_id=self.microbatch_id,
        )

    def __repr__(self):
        return (
            f"Recv({self.x}, {self.src}→{self.dst}, "
            f"stage={self.stage}, mb={self.microbatch_id}) -> {self.output}"
        )


# ── Shape manipulation ops ───────────────────────────────────────────────────

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

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        result = TensorState(
            name=self.output,
            global_shape=self.new_shape,
            local_shape=self.new_shape,  # simplification
            sharding=x.sharding,
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
            expr=f"reshape_inv(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

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

        # Shard dims may need to swap too
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
        # VJP of transpose is transpose back
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            expr=f"transpose(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad_x}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Transpose(
            x=input_map.get(self.x, self.x),
            output=output_name,
            dim0=self.dim0,
            dim1=self.dim1,
        )

    def __repr__(self):
        return f"Transpose({self.x}, {self.dim0}↔{self.dim1}) -> {self.output}"


# ── Flash Attention (for CP) ─────────────────────────────────────────────────

@dataclass
class FlashAttention(IROp):
    """Block-wise flash attention: O = softmax(Q @ K^T / sqrt(d)) @ V.

    For context parallelism: Q is full per device, K and V are sharded on
    seq_len dim and rotated via ring communication.

    This op tracks:
      - Q: (B, S_local, H, D) or (B, S, H, D) depending on setup
      - K, V: (B, S_shard, H, D) — sharded on seq dim
      - O: (B, S, H, D) — partial, needs reduction across CP ranks
    """
    q: str
    k: str
    v: str
    output: str
    softmax_scale: float = 1.0
    causal: bool = False

    @property
    def input_names(self) -> List[str]:
        return [self.q, self.k, self.v]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        q = ctx[self.q]
        k = ctx[self.k]
        v = ctx[self.v]

        # O shape = Q shape
        out_global = q.global_shape

        # If K, V are sharded on seq dim and Q is not → output is PARTIAL
        is_partial = False
        out_placements = list(q.sharding.placements)

        for mesh_dim, p in enumerate(k.sharding.placements):
            if isinstance(p, Shard) and p.dim == 1:  # K sharded on seq dim
                if not any(
                    isinstance(qp, Shard) and qp.dim == 1
                    for qp in q.sharding.placements
                ):
                    is_partial = True
                    out_placements[mesh_dim] = Partial()

        out_spec = ShardingSpec(
            placements=tuple(out_placements),
            mesh=q.sharding.mesh,
        )
        out_local = compute_local_shape(out_global, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            expr=f"attn({q.expr}, {k.expr}, {v.expr})" if q.expr else "",
            requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        q = ctx[self.q]
        k = ctx[self.k]
        v = ctx[self.v]

        # FlashAttention backward computes grads for Q, K, V
        # Symbolically: grad_Q, grad_K, grad_V from standard attention VJP
        grad_q = TensorState(
            name=f"grad_{self.q}",
            global_shape=q.global_shape,
            local_shape=q.local_shape,
            sharding=q.sharding,
            expr=f"attn_grad_q({q.expr})",
        )
        grad_k = TensorState(
            name=f"grad_{self.k}",
            global_shape=k.global_shape,
            local_shape=k.local_shape,
            sharding=k.sharding,
            expr=f"attn_grad_k({k.expr})",
        )
        grad_v = TensorState(
            name=f"grad_{self.v}",
            global_shape=v.global_shape,
            local_shape=v.local_shape,
            sharding=v.sharding,
            expr=f"attn_grad_v({v.expr})",
        )
        return {self.q: grad_q, self.k: grad_k, self.v: grad_v}

    def is_collective(self) -> bool:
        return False  # FA itself is compute, not collective

    def is_p2p(self) -> bool:
        return False

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return FlashAttention(
            q=input_map.get(self.q, self.q),
            k=input_map.get(self.k, self.k),
            v=input_map.get(self.v, self.v),
            output=output_name,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )

    def __repr__(self):
        return f"FlashAttention({self.q}, {self.k}, {self.v}) -> {self.output}"


# ── Async communication ops ──────────────────────────────────────────────────

@dataclass
class Handle:
    """A handle to an in-flight asynchronous operation.

    Tracks the async op's output buffer, creation time (issue),
    and expected completion time (after Wait).
    """
    name: str
    op_name: str = ""           # which op created this handle
    buffer_name: str = ""       # tensor being written asynchronously
    is_active: bool = True      # True until Wait completes


@dataclass(frozen=True)
class Stream:
    """A CUDA stream abstraction.

    Ops on the same stream execute sequentially (program order).
    Ops on different streams can execute concurrently.
    """
    name: str
    device_id: int = 0

    def __hash__(self):
        return hash((self.name, self.device_id))


# Default stream
DEFAULT_STREAM = Stream("default", 0)
COMM_STREAM = Stream("comm", 0)     # communication stream for overlap
COMPUTE_STREAM = Stream("compute", 0)  # compute stream


@dataclass
class AllReduceAsync(IROp):
    """Asynchronous AllReduce: launches and returns immediately.

    The output buffer is being written asynchronously. Must be
    followed by a Wait(handle) before the output is read.

    Forward: PARTIAL → REPLICATE (after Wait)
    """
    x: str
    output: str
    handle: str
    op_type: str = "sum"
    stream: Stream = COMM_STREAM

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        if not x.partial:
            raise ValueError(
                f"AllReduceAsync requires PARTIAL input for tensor '{x.name}', "
                f"got placements={x.sharding.placements} (shape={x.global_shape}, "
                f"local_shape={x.local_shape}). "
                f"Op: AllReduceAsync({self.x}) -> {self.output}"
            )
        new_placements = []
        for p in x.sharding.placements:
            if isinstance(p, Partial):
                new_placements.append(Replicate())
            else:
                new_placements.append(p)
        out_spec = ShardingSpec(placements=tuple(new_placements), mesh=x.sharding.mesh)
        result = TensorState(
            name=self.output, global_shape=x.global_shape,
            local_shape=x.global_shape, sharding=out_spec,
            expr=x.expr, requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
            _async_handle=self.handle,  # marks this as in-flight
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: TensorState(
            name=f"grad_{self.x}", global_shape=x.global_shape,
            local_shape=x.local_shape, sharding=x.sharding,
        )}

    def is_collective(self) -> bool:
        return True

    def is_p2p(self) -> bool:
        return False

    def is_async(self) -> bool:
        return True

    def clone_with_names(self, input_map, output_name):
        return AllReduceAsync(
            x=input_map.get(self.x, self.x), output=output_name,
            handle=self.handle, op_type=self.op_type, stream=self.stream,
        )

    def __repr__(self):
        return f"AllReduceAsync({self.x}, {self.op_type}) -> {self.output} [handle={self.handle}]"


@dataclass
class Wait(IROp):
    """Wait for an asynchronous operation to complete.

    After Wait, the output buffer of the async op is safe to read.
    Converts the tensor from async-in-flight to fully materialized.
    """
    handle: str
    tensor: str       # tensor being waited on (same as async op's output)
    output: str        # tensor after wait (safe to read)

    @property
    def input_names(self) -> List[str]:
        return [self.tensor]

    @property
    def output_name(self) -> str:
        return self.output

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        ts = ctx.get(self.tensor)
        if ts is None:
            raise ValueError(f"Wait: tensor '{self.tensor}' not found")
        # Strip the async marker
        result = TensorState(
            name=self.output, global_shape=ts.global_shape,
            local_shape=ts.local_shape, sharding=ts.sharding,
            expr=ts.expr, requires_grad=ts.requires_grad,
            grad_name=ts.grad_name,
            _async_handle=None,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def is_async(self) -> bool:
        return False

    def is_sync(self) -> bool:
        return True

    def clone_with_names(self, input_map, output_name):
        return Wait(
            handle=self.handle,
            tensor=input_map.get(self.tensor, self.tensor),
            output=output_name,
        )

    def __repr__(self):
        return f"Wait({self.handle}) -> {self.output}"


@dataclass
class WaitAll(IROp):
    """Wait for multiple async handles simultaneously."""
    handles: Tuple[str, ...]
    tensors: Tuple[str, ...]   # corresponding tensors
    outputs: Tuple[str, ...]   # safe-to-read tensor names

    @property
    def input_names(self) -> List[str]:
        return list(self.tensors)

    @property
    def output_name(self) -> str:
        return self.outputs[0] if self.outputs else ""

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        for tensor, output in zip(self.tensors, self.outputs):
            ts = ctx.get(tensor)
            if ts is None:
                continue
            result = TensorState(
                name=output, global_shape=ts.global_shape,
                local_shape=ts.local_shape, sharding=ts.sharding,
                expr=ts.expr, requires_grad=ts.requires_grad,
                _async_handle=None,
            )
            ctx[output] = result
        return ctx.get(self.outputs[0]) if self.outputs else None

    def vjp(self, ctx, grad_output):
        return {}

    def is_collective(self) -> bool:
        return False

    def is_p2p(self) -> bool:
        return False

    def is_async(self) -> bool:
        return False

    def is_sync(self) -> bool:
        return True

    def clone_with_names(self, input_map, output_name):
        return WaitAll(
            handles=self.handles,
            tensors=tuple(input_map.get(t, t) for t in self.tensors),
            outputs=self.outputs,
        )

    def __repr__(self):
        return f"WaitAll({list(self.handles)}) -> {list(self.outputs)}"


@dataclass
class OverlapRegion(IROp):
    """Marks a region where compute and communication intentionally overlap.

    Contains compute_ops that run concurrently with comm_ops.
    The verifier checks that no races exist within this region.
    """
    compute_ops: List[IROp]
    comm_ops: List[IROp]
    output: str = ""  # synthetic output name

    @property
    def input_names(self) -> List[str]:
        inputs = []
        for op in self.compute_ops + self.comm_ops:
            inputs.extend(op.input_names)
        return list(set(inputs))

    @property
    def output_name(self) -> str:
        return self.output or "overlap_output"

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        # Execute compute and comm ops — they race by design
        for op in self.compute_ops:
            op.apply(ctx)
        for op in self.comm_ops:
            op.apply(ctx)
        return ctx.get(self.output_name)

    def vjp(self, ctx, grad_output):
        return {}

    def is_collective(self) -> bool:
        return any(op.is_collective() for op in self.comm_ops)

    def is_p2p(self) -> bool:
        return any(op.is_p2p() for op in self.comm_ops)

    def is_async(self) -> bool:
        return True

    def clone_with_names(self, input_map, output_name):
        return OverlapRegion(
            compute_ops=[op.clone_with_names(input_map, op.output_name)
                         for op in self.compute_ops],
            comm_ops=[op.clone_with_names(input_map, op.output_name)
                      for op in self.comm_ops],
            output=output_name,
        )

    def __repr__(self):
        comp_str = ", ".join(type(op).__name__ for op in self.compute_ops)
        comm_str = ", ".join(type(op).__name__ for op in self.comm_ops)
        return f"OverlapRegion(compute=[{comp_str}], comm=[{comm_str}])"


@dataclass
class SendAsync(IROp):
    """Asynchronous Send."""
    x: str
    output: str
    handle: str
    src: int
    dst: int
    stage: int
    microbatch_id: int
    stream: Stream = COMM_STREAM

    @property
    def input_names(self): return [self.x]
    @property
    def output_name(self): return self.output

    def apply(self, ctx):
        x = ctx[self.x]
        result = TensorState(
            name=self.output, global_shape=x.global_shape,
            local_shape=x.local_shape, sharding=x.sharding,
            expr=x.expr, stage=self.dst, microbatch_id=self.microbatch_id,
            _async_handle=self.handle,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {self.x: TensorState(
            name=f"grad_{self.x}", global_shape=ctx[self.x].global_shape,
            local_shape=ctx[self.x].local_shape,
            sharding=ctx[self.x].sharding,
        )}
    def is_collective(self): return True
    def is_p2p(self): return True
    def is_async(self): return True
    def clone_with_names(self, im, on): return SendAsync(
        x=im.get(self.x, self.x), output=on, handle=self.handle,
        src=self.src, dst=self.dst, stage=self.stage,
        microbatch_id=self.microbatch_id, stream=self.stream)
    def __repr__(self):
        return f"SendAsync({self.x}, {self.src}->{self.dst}) [{self.handle}]"


@dataclass
class RecvAsync(IROp):
    """Asynchronous Receive."""
    x: str
    output: str
    handle: str
    src: int
    dst: int
    stage: int
    microbatch_id: int
    stream: Stream = COMM_STREAM

    @property
    def input_names(self): return [self.x]
    @property
    def output_name(self): return self.output

    def apply(self, ctx):
        x = ctx.get(self.x)
        result = TensorState(
            name=self.output, global_shape=x.global_shape if x else (),
            local_shape=x.local_shape if x else (),
            sharding=x.sharding if x else None,
            stage=self.dst, microbatch_id=self.microbatch_id,
            _async_handle=self.handle,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {self.x: TensorState(name=f"grad_{self.x}")}
    def is_collective(self): return True
    def is_p2p(self): return True
    def is_async(self): return True
    def clone_with_names(self, im, on): return RecvAsync(
        x=im.get(self.x, self.x), output=on, handle=self.handle,
        src=self.src, dst=self.dst, stage=self.stage,
        microbatch_id=self.microbatch_id, stream=self.stream)
    def __repr__(self):
        return f"RecvAsync({self.x}, {self.src}->{self.dst}) [{self.handle}]"


# ── Program container ────────────────────────────────────────────────────────

@dataclass
class Program:
    """Container for a sequence of IR operations (forward or backward)."""
    name: str = ""
    ops: List[IROp] = field(default_factory=list)

    def add(self, op: IROp) -> Program:
        self.ops.append(op)
        return self

    @property
    def collectives(self) -> List[IROp]:
        return [op for op in self.ops if op.is_collective()]

    @property
    def p2p_ops(self) -> List[IROp]:
        return [op for op in self.ops if op.is_p2p()]

    @property
    def compute_ops(self) -> List[IROp]:
        return [op for op in self.ops if not op.is_collective()]

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, idx):
        return self.ops[idx]

    def validate_names(self) -> List[str]:
        """Check for duplicate output names in the op sequence."""
        errors: List[str] = []
        seen: Dict[str, int] = {}
        for i, op in enumerate(self.ops):
            oname = op.output_name
            if oname in seen:
                errors.append(
                    f"Duplicate output name '{oname}': "
                    f"op[{i}] ({type(op).__name__}) conflicts with "
                    f"op[{seen[oname]}] ({type(self.ops[seen[oname]]).__name__})"
                )
            seen[oname] = i
        return errors

    def __repr__(self):
        ops_str = "\n  ".join(repr(op) for op in self.ops)
        return f"Program({self.name}, {len(self.ops)} ops):\n  {ops_str}"


def ir_to_str(program: Program) -> str:
    """Pretty-print a program."""
    lines = [f"Program: {program.name}"]
    for i, op in enumerate(program.ops):
        lines.append(f"  [{i}] {op}")
    return "\n".join(lines)
