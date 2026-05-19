from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import IROp
from .spmd import SPMDGuard
from ..state import (
    TensorState,
    LocalSPMDType,
    Shard,
    Replicate,
    Partial,
    ShardingSpec,
    compute_local_shape,
)

R, I, V, P = LocalSPMDType.REPLICATE, LocalSPMDType.INVARIANT, LocalSPMDType.VARYING, LocalSPMDType.PARTIAL


def _merge_spmd_elementwise(a: LocalSPMDType, b: LocalSPMDType) -> LocalSPMDType:
    """SPMD type merge for element-wise binary ops."""
    if a in (R, I):
        return b if b not in (I,) else R
    if b in (R, I):
        return a if a not in (I,) else R
    if a == b:
        return a
    # P + V or V + P -> P (partial absorbs)
    if P in (a, b):
        return P
    return a


class ElementWiseBinaryOp(IROp):
    """Base for element-wise binary ops (Add, Multiply, etc.).

    Handles shared shape/mesh validation and placement merging.
    Subclasses provide the expression operator and VJP logic.
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

    @abstractmethod
    def _make_expr(self, a_expr: str, b_expr: str) -> str:
        """Return the symbolic expression string for this op."""
        ...

    def _validate_and_merge_placements(
        self, ctx: Dict[str, TensorState]
    ) -> Tuple[TensorState, TensorState, ShardingSpec]:
        """Validate shapes/meshes and merge placements. Returns (a, b, out_spec)."""
        a = ctx[self.a]
        b = ctx[self.b]
        op_name = type(self).__name__

        if a.global_shape != b.global_shape:
            raise ValueError(
                f"{op_name}: shape mismatch between '{self.a}' {a.global_shape} and "
                f"'{self.b}' {b.global_shape}. "
                f"Element-wise ops require both inputs to have the same global shape. "
                f"Op: {op_name}({self.a}, {self.b}) -> {self.output}"
            )

        if a.sharding.mesh != b.sharding.mesh:
            raise ValueError(
                f"{op_name}: mesh mismatch between '{self.a}' {a.sharding.mesh} and "
                f"'{self.b}' {b.sharding.mesh}. "
                f"Op: {op_name}({self.a}, {self.b}) -> {self.output}"
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
                    f"{op_name}: incompatible placements at mesh dim {mesh_dim}: "
                    f"'{self.a}' has {pa}, '{self.b}' has {pb}. "
                    f"Element-wise ops require matching sharding, or one input to be Replicate. "
                    f"Op: {op_name}({self.a}, {self.b}) -> {self.output}"
                )

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=a.sharding.mesh)
        return a, b, out_spec

    def propagate_spmd_type(self, input_types):
        a_type = input_types.get(self.a)
        b_type = input_types.get(self.b)
        if a_type is None or b_type is None:
            return None
        return _merge_spmd_elementwise(a_type, b_type)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a, b, out_spec = self._validate_and_merge_placements(ctx)
        out_local = compute_local_shape(a.global_shape, out_spec)

        expr = self._make_expr(a.expr, b.expr) if a.expr and b.expr else ""

        result = TensorState(
            name=self.output,
            global_shape=a.global_shape,
            local_shape=out_local,
            sharding=out_spec,
            dtype=a.dtype,
            expr=expr,
            requires_grad=a.requires_grad or b.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result


@dataclass
class MatMul(IROp):
    """Matrix multiplication: Y = A @ B.

    Placement rules:
      - Shard(dim0=A_rows) @ Shard(dim1=B_cols) -> PARTIAL on reduce dim
      - Shard(dim0=A_rows) @ Replicate -> Shard(dim0=rows) (data parallel)
      - Replicate @ Shard(dim0=B_rows) -> Shard(dim0=rows) (no comm needed)
      - Shard(dim1=A_cols) @ Shard(dim0=B_rows) -> Replicate (column parallel)
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

    def propagate_spmd_type(self, input_types):
        a_type = input_types.get(self.a)
        b_type = input_types.get(self.b)
        if a_type is None or b_type is None:
            return None
        if a_type == R and b_type == R:
            return R
        if a_type == V and b_type == V:
            return None  # dimension-dependent, defer to placement
        if P in (a_type, b_type):
            return P
        if a_type in (R, I):
            return b_type if b_type not in (I,) else R
        if b_type in (R, I):
            return a_type if a_type not in (I,) else R
        return None

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a = ctx[self.a]
        b = ctx[self.b]

        if a.global_shape[1] != b.global_shape[0]:
            raise ValueError(
                f"MatMul: contraction dimension mismatch: "
                f"'{self.a}' has shape {a.global_shape} (cols={a.global_shape[1]}) "
                f"but '{self.b}' has shape {b.global_shape} (rows={b.global_shape[0]}). "
                f"MatMul requires A.shape[1] == B.shape[0]. "
                f"Op: MatMul({self.a}, {self.b}) -> {self.output}"
            )

        out_global = (a.global_shape[0], b.global_shape[1])

        a_spec = a.sharding
        b_spec = b.sharding

        out_placements = list(a_spec.placements)

        for mesh_dim, p in enumerate(b_spec.placements):
            if isinstance(p, Shard):
                if p.dim == 0:
                    for a_mesh_dim, a_p in enumerate(a_spec.placements):
                        if isinstance(a_p, Shard) and a_p.dim == 1:
                            if a_mesh_dim == mesh_dim:
                                out_placements[mesh_dim] = Partial()
                elif p.dim == 1:
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
            dtype=a.dtype,
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
            dtype=a.dtype,
            expr=f"grad({a.expr})" if a.expr else "",
            requires_grad=False,
        )
        grad_b = TensorState(
            name=f"grad_{self.b}",
            global_shape=b.global_shape,
            local_shape=b.local_shape,
            sharding=b.sharding,
            dtype=b.dtype,
            expr=f"grad({b.expr})" if b.expr else "",
            requires_grad=False,
        )

        return {self.a: grad_a, self.b: grad_b}

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return MatMul(
            a=input_map.get(self.a, self.a),
            b=input_map.get(self.b, self.b),
            output=output_name,
        )

    def __repr__(self):
        return f"MatMul({self.a}, {self.b}) -> {self.output}"


@dataclass
class Add(ElementWiseBinaryOp):
    """Element-wise addition: Y = A + B."""
    a: str
    b: str
    output: str

    def _make_expr(self, a_expr: str, b_expr: str) -> str:
        return f"({a_expr} + {b_expr})"

    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        grad_a = grad_output.with_name(f"grad_{self.a}")
        grad_b = grad_output.with_name(f"grad_{self.b}")
        return {self.a: grad_a, self.b: grad_b}

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return Add(
            a=input_map.get(self.a, self.a),
            b=input_map.get(self.b, self.b),
            output=output_name,
        )

    def __repr__(self):
        return f"Add({self.a}, {self.b}) -> {self.output}"


@dataclass
class Multiply(ElementWiseBinaryOp):
    """Element-wise multiplication: Y = A * B."""
    a: str
    b: str
    output: str

    def _make_expr(self, a_expr: str, b_expr: str) -> str:
        return f"({a_expr} * {b_expr})"

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        a = ctx[self.a]
        b = ctx[self.b]
        SPMDGuard.check_multiply(a, b, "Multiply")
        return super().apply(ctx)

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
            dtype=a.dtype,
            expr=f"grad({a.expr}) * {b.expr}" if a.expr and b.expr else "",
        )
        grad_b = TensorState(
            name=f"grad_{self.b}",
            global_shape=b.global_shape,
            local_shape=b.local_shape,
            sharding=b.sharding,
            dtype=b.dtype,
            expr=f"grad({self.output}) * {a.expr}" if a.expr and b.expr else "",
        )
        return {self.a: grad_a, self.b: grad_b}

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

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
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
            dtype=x.dtype,
            expr=f"silu_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        return SiLU(
            x=input_map.get(self.x, self.x),
            output=output_name,
        )

    def __repr__(self):
        return f"SiLU({self.x}) -> {self.output}"


@dataclass
class FlashAttention(IROp):
    """Block-wise flash attention: O = softmax(Q @ K^T / sqrt(d)) @ V.

    For context parallelism: Q is full per device, K and V are sharded on
    seq_len dim and rotated via ring communication.
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

    def propagate_spmd_type(self, input_types):
        types = [input_types.get(n) for n in self.input_names]
        if any(t is None for t in types):
            return None
        if all(t == types[0] for t in types):
            return types[0]
        return None  # mixed types, defer to placement

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        q = ctx[self.q]
        k = ctx[self.k]
        v = ctx[self.v]

        out_global = q.global_shape

        out_placements = list(q.sharding.placements)

        for mesh_dim, p in enumerate(k.sharding.placements):
            if isinstance(p, Shard) and p.dim == 1:
                if not any(
                    isinstance(qp, Shard) and qp.dim == 1
                    for qp in q.sharding.placements
                ):
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
            dtype=q.dtype,
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

        grad_q = TensorState(
            name=f"grad_{self.q}",
            global_shape=q.global_shape,
            local_shape=q.local_shape,
            sharding=q.sharding,
            dtype=q.dtype,
            expr=f"attn_grad_q({q.expr})",
        )
        grad_k = TensorState(
            name=f"grad_{self.k}",
            global_shape=k.global_shape,
            local_shape=k.local_shape,
            sharding=k.sharding,
            dtype=k.dtype,
            expr=f"attn_grad_k({k.expr})",
        )
        grad_v = TensorState(
            name=f"grad_{self.v}",
            global_shape=v.global_shape,
            local_shape=v.local_shape,
            sharding=v.sharding,
            dtype=v.dtype,
            expr=f"attn_grad_v({v.expr})",
        )
        return {self.q: grad_q, self.k: grad_k, self.v: grad_v}

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


@dataclass
class GELU(IROp):
    """GELU activation: Y = X * Phi(X)."""
    x: str
    output: str

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
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"gelu({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"gelu_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return GELU(x=input_map.get(self.x, self.x), output=output_name)

    def __repr__(self):
        return f"GELU({self.x}) -> {self.output}"


@dataclass
class ReLU(IROp):
    """ReLU activation: Y = max(0, X)."""
    x: str
    output: str

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
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"relu({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"relu_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return ReLU(x=input_map.get(self.x, self.x), output=output_name)

    def __repr__(self):
        return f"ReLU({self.x}) -> {self.output}"


@dataclass
class Dropout(IROp):
    """Dropout: Y = X * mask / (1 - p) during training."""
    x: str
    output: str
    p: float = 0.1

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
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"dropout({x.expr}, p={self.p})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"dropout_grad({x.expr}, p={self.p})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return Dropout(x=input_map.get(self.x, self.x), output=output_name, p=self.p)

    def __repr__(self):
        return f"Dropout({self.x}, p={self.p}) -> {self.output}"


@dataclass
class LayerNorm(IROp):
    """LayerNorm: normalize across norm_dim.

    If the input is sharded along norm_dim, raises an error because
    normalization requires the full dimension for correct statistics.
    """
    x: str
    output: str
    norm_dim: int = -1

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
        effective_dim = self.norm_dim if self.norm_dim >= 0 else len(x.global_shape) + self.norm_dim

        for p in x.sharding.placements:
            if isinstance(p, Shard) and p.dim == effective_dim:
                raise ValueError(
                    f"LayerNorm: input '{x.name}' is sharded on norm_dim={effective_dim}. "
                    f"Normalization requires the full dimension for correct statistics. "
                    f"Insert AllGather before LayerNorm or use a different sharding. "
                    f"Op: LayerNorm({self.x}) -> {self.output}"
                )

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"layernorm({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"layernorm_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return LayerNorm(
            x=input_map.get(self.x, self.x), output=output_name, norm_dim=self.norm_dim,
        )

    def __repr__(self):
        return f"LayerNorm({self.x}, dim={self.norm_dim}) -> {self.output}"


@dataclass
class RMSNorm(IROp):
    """RMSNorm: root-mean-square normalization across norm_dim.

    Same sharding constraint as LayerNorm: norm_dim must not be sharded.
    """
    x: str
    output: str
    norm_dim: int = -1

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
        effective_dim = self.norm_dim if self.norm_dim >= 0 else len(x.global_shape) + self.norm_dim

        for p in x.sharding.placements:
            if isinstance(p, Shard) and p.dim == effective_dim:
                raise ValueError(
                    f"RMSNorm: input '{x.name}' is sharded on norm_dim={effective_dim}. "
                    f"Normalization requires the full dimension for correct statistics. "
                    f"Insert AllGather before RMSNorm or use a different sharding. "
                    f"Op: RMSNorm({self.x}) -> {self.output}"
                )

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"rmsnorm({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"rmsnorm_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return RMSNorm(
            x=input_map.get(self.x, self.x), output=output_name, norm_dim=self.norm_dim,
        )

    def __repr__(self):
        return f"RMSNorm({self.x}, dim={self.norm_dim}) -> {self.output}"


@dataclass
class Softmax(IROp):
    """Softmax: normalize along reduction_dim.

    Same constraint as LayerNorm: reduction_dim must not be sharded.
    """
    x: str
    output: str
    dim: int = -1

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
        effective_dim = self.dim if self.dim >= 0 else len(x.global_shape) + self.dim

        for p in x.sharding.placements:
            if isinstance(p, Shard) and p.dim == effective_dim:
                raise ValueError(
                    f"Softmax: input '{x.name}' is sharded on dim={effective_dim}. "
                    f"Softmax requires the full dimension for correct normalization. "
                    f"Insert AllGather before Softmax or use a different sharding. "
                    f"Op: Softmax({self.x}) -> {self.output}"
                )

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"softmax({x.expr})" if x.expr else "",
            requires_grad=x.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
            expr=f"softmax_grad({x.expr})",
        )
        return {self.x: grad_x}

    def clone_with_names(self, input_map, output_name):
        return Softmax(x=input_map.get(self.x, self.x), output=output_name, dim=self.dim)

    def __repr__(self):
        return f"Softmax({self.x}, dim={self.dim}) -> {self.output}"


@dataclass
class Embedding(IROp):
    """Embedding lookup: Y = W[indices].

    Vocab-parallel sharding: when weight is Shard(0) on vocab dim,
    each rank holds a slice of the vocabulary. After lookup, output
    is Partial (needs AllReduce to combine rows from all ranks).
    """
    indices: str
    weight: str
    output: str
    vocab_dim: int = 0

    @property
    def input_names(self) -> List[str]:
        return [self.indices, self.weight]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        w_type = input_types.get(self.weight)
        if w_type == LocalSPMDType.VARYING:
            return LocalSPMDType.PARTIAL
        return w_type

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        indices = ctx[self.indices]
        weight = ctx[self.weight]

        out_global = indices.global_shape + (weight.global_shape[1],)

        out_placements = []
        for p in weight.sharding.placements:
            if isinstance(p, Shard) and p.dim == self.vocab_dim:
                out_placements.append(Partial())
            else:
                out_placements.append(p)

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=weight.sharding.mesh)
        out_local = compute_local_shape(out_global, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            dtype=weight.dtype,
            expr=f"embed({weight.expr}[{indices.expr}])" if weight.expr else "",
            requires_grad=weight.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        weight = ctx[self.weight]
        grad_w = TensorState(
            name=f"grad_{self.weight}",
            global_shape=weight.global_shape,
            local_shape=weight.local_shape,
            sharding=weight.sharding,
            dtype=weight.dtype,
            expr=f"embed_grad({weight.expr})",
        )
        return {self.weight: grad_w}

    def clone_with_names(self, input_map, output_name):
        return Embedding(
            indices=input_map.get(self.indices, self.indices),
            weight=input_map.get(self.weight, self.weight),
            output=output_name,
            vocab_dim=self.vocab_dim,
        )

    def __repr__(self):
        return f"Embedding({self.indices}, {self.weight}) -> {self.output}"


@dataclass
class CrossEntropyLoss(IROp):
    """Cross-entropy loss: L = -sum(targets * log(softmax(logits))).

    Vocab-parallel: when logits are Shard on vocab dim, the loss
    computation requires communication. Output is a scalar with
    Partial placement (needs AllReduce to get the full loss).
    """
    logits: str
    targets: str
    output: str
    vocab_dim: int = -1

    @property
    def input_names(self) -> List[str]:
        return [self.logits, self.targets]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        l_type = input_types.get(self.logits)
        if l_type == LocalSPMDType.VARYING:
            return LocalSPMDType.PARTIAL
        return l_type

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        logits = ctx[self.logits]
        effective_dim = self.vocab_dim if self.vocab_dim >= 0 else len(logits.global_shape) + self.vocab_dim

        out_global = (1,)

        out_placements = []
        vocab_sharded = False
        for p in logits.sharding.placements:
            if isinstance(p, Shard) and p.dim == effective_dim:
                out_placements.append(Partial())
                vocab_sharded = True
            elif isinstance(p, Partial):
                out_placements.append(Partial())
            else:
                out_placements.append(Replicate())

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=logits.sharding.mesh)

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_global,
            sharding=out_spec,
            dtype=logits.dtype,
            expr=f"cross_entropy({logits.expr})" if logits.expr else "",
            requires_grad=logits.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        logits = ctx[self.logits]
        grad_logits = TensorState(
            name=f"grad_{self.logits}",
            global_shape=logits.global_shape,
            local_shape=logits.local_shape,
            sharding=logits.sharding,
            dtype=logits.dtype,
            expr=f"cross_entropy_grad({logits.expr})",
        )
        return {self.logits: grad_logits}

    def clone_with_names(self, input_map, output_name):
        return CrossEntropyLoss(
            logits=input_map.get(self.logits, self.logits),
            targets=input_map.get(self.targets, self.targets),
            output=output_name,
            vocab_dim=self.vocab_dim,
        )

    def __repr__(self):
        return f"CrossEntropyLoss({self.logits}, {self.targets}) -> {self.output}"
