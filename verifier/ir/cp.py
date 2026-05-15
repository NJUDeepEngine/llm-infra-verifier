from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .base import IROp
from .compute import FlashAttention, Add
from .collective import AllReduce
from ..state import (
    TensorState,
    LocalSPMDType,
    Shard,
    Replicate,
    Partial,
    ShardingSpec,
    compute_local_shape,
)


@dataclass
class RingRotate(IROp):
    """Ring P2P rotation: send to next rank, receive from previous.

    Each device sends its tensor to (rank+1) % ring_size and receives
    from (rank-1) % ring_size. Shape and sharding are preserved;
    ring_step is incremented.
    """
    x: str
    output: str
    ring_size: int
    ring_dim: int = 0
    handle: Optional[str] = None

    @property
    def input_names(self) -> List[str]:
        return [self.x]

    @property
    def output_name(self) -> str:
        return self.output

    def is_collective(self) -> bool:
        return True

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx[self.x]
        new_step = (x.ring_step or 0) + 1

        result = replace(
            x,
            name=self.output,
            ring_step=new_step,
            _async_handle=self.handle,
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
            expr=f"RingRotateReverse(grad({x.expr}))" if x.expr else "",
        )
        return {self.x: grad}

    def clone_with_names(self, input_map, output_name):
        return RingRotate(
            x=input_map.get(self.x, self.x), output=output_name,
            ring_size=self.ring_size, ring_dim=self.ring_dim, handle=self.handle,
        )

    def __repr__(self):
        h = f", handle={self.handle}" if self.handle else ""
        return f"RingRotate({self.x}, ring={self.ring_size}, dim={self.ring_dim}{h}) -> {self.output}"


@dataclass
class RingAttentionStep(IROp):
    """Single step of ring attention: local FlashAttention with ring metadata.

    Q is local, K/V come from ring rotation. Output is Partial
    (needs accumulation across all ring steps).
    """
    q: str
    k: str
    v: str
    output: str
    ring_step: int
    ring_size: int
    causal: bool = False
    softmax_scale: float = 1.0

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
        return None

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

        out_spec = ShardingSpec(placements=tuple(out_placements), mesh=q.sharding.mesh)
        out_local = compute_local_shape(out_global, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            dtype=q.dtype,
            expr=f"ring_attn_step{self.ring_step}({q.expr}, {k.expr}, {v.expr})" if q.expr else "",
            requires_grad=q.requires_grad or k.requires_grad or v.requires_grad,
            grad_name=f"grad_{self.output}",
            ring_step=self.ring_step,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        q = ctx[self.q]
        k = ctx[self.k]
        v = ctx[self.v]

        def _make_grad(name, ts, expr_fn):
            return TensorState(
                name=f"grad_{name}",
                global_shape=ts.global_shape,
                local_shape=ts.local_shape,
                sharding=ts.sharding,
                dtype=ts.dtype,
                expr=expr_fn(ts.expr) if ts.expr else "",
                ring_step=self.ring_step,
            )

        return {
            self.q: _make_grad(self.q, q, lambda e: f"ring_attn_grad_q_step{self.ring_step}({e})"),
            self.k: _make_grad(self.k, k, lambda e: f"ring_attn_grad_k_step{self.ring_step}({e})"),
            self.v: _make_grad(self.v, v, lambda e: f"ring_attn_grad_v_step{self.ring_step}({e})"),
        }

    def clone_with_names(self, input_map, output_name):
        return RingAttentionStep(
            q=input_map.get(self.q, self.q),
            k=input_map.get(self.k, self.k),
            v=input_map.get(self.v, self.v),
            output=output_name,
            ring_step=self.ring_step,
            ring_size=self.ring_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )

    def __repr__(self):
        return (
            f"RingAttentionStep({self.q}, {self.k}, {self.v}, "
            f"step={self.ring_step}/{self.ring_size}) -> {self.output}"
        )


@dataclass
class RingAttention(IROp):
    """Composite ring attention op (expandable to primitive sequence).

    High-level: Q is local, K/V rotate through the ring. Each step
    computes local FlashAttention and accumulates partial results.
    Final AllReduce produces replicated output.

    Use apply() for quick verification, expand() for temporal analysis.
    """
    q: str
    k: str
    v: str
    output: str
    ring_size: int
    causal: bool = False
    softmax_scale: float = 1.0

    @property
    def input_names(self) -> List[str]:
        return [self.q, self.k, self.v]

    @property
    def output_name(self) -> str:
        return self.output

    def is_collective(self) -> bool:
        return True

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        q = ctx[self.q]
        out_global = q.global_shape

        out_placements = tuple(
            Replicate() if isinstance(p, Partial) else p
            for p in q.sharding.placements
        )
        out_spec = ShardingSpec(placements=out_placements, mesh=q.sharding.mesh)
        out_local = compute_local_shape(out_global, out_spec)

        result = TensorState(
            name=self.output,
            global_shape=out_global,
            local_shape=out_local,
            sharding=out_spec,
            dtype=q.dtype,
            expr=f"ring_attn({q.expr})" if q.expr else "",
            requires_grad=q.requires_grad,
            grad_name=f"grad_{self.output}",
        )
        ctx[self.output] = result
        return result

    def expand(self) -> List[IROp]:
        """Expand to primitive ops: [Step_0, Rotate, Step_1, ..., Add, AllReduce]."""
        ops: List[IROp] = []
        k_name = self.k
        v_name = self.v

        for step in range(self.ring_size):
            step_out = f"_ring_step{step}_{self.output}"
            ops.append(RingAttentionStep(
                q=self.q, k=k_name, v=v_name, output=step_out,
                ring_step=step, ring_size=self.ring_size,
                causal=self.causal, softmax_scale=self.softmax_scale,
            ))

            if step == 0:
                accum_name = step_out
            else:
                new_accum = f"_ring_accum{step}_{self.output}"
                ops.append(Add(a=accum_name, b=step_out, output=new_accum))
                accum_name = new_accum

            if step < self.ring_size - 1:
                new_k = f"_ring_k_rot{step + 1}"
                new_v = f"_ring_v_rot{step + 1}"
                ops.append(RingRotate(x=k_name, output=new_k, ring_size=self.ring_size))
                ops.append(RingRotate(x=v_name, output=new_v, ring_size=self.ring_size))
                k_name = new_k
                v_name = new_v

        ops.append(AllReduce(x=accum_name, output=self.output))
        return ops

    def vjp(self, ctx, grad_output):
        q = ctx[self.q]
        k = ctx[self.k]
        v = ctx[self.v]

        grad_q = TensorState(
            name=f"grad_{self.q}",
            global_shape=q.global_shape,
            local_shape=q.local_shape,
            sharding=q.sharding,
            dtype=q.dtype,
            expr=f"ring_attn_grad_q({q.expr})" if q.expr else "",
        )
        grad_k = TensorState(
            name=f"grad_{self.k}",
            global_shape=k.global_shape,
            local_shape=k.local_shape,
            sharding=k.sharding,
            dtype=k.dtype,
            expr=f"ring_attn_grad_k({k.expr})" if k.expr else "",
        )
        grad_v = TensorState(
            name=f"grad_{self.v}",
            global_shape=v.global_shape,
            local_shape=v.local_shape,
            sharding=v.sharding,
            dtype=v.dtype,
            expr=f"ring_attn_grad_v({v.expr})" if v.expr else "",
        )
        return {self.q: grad_q, self.k: grad_k, self.v: grad_v}

    def clone_with_names(self, input_map, output_name):
        return RingAttention(
            q=input_map.get(self.q, self.q),
            k=input_map.get(self.k, self.k),
            v=input_map.get(self.v, self.v),
            output=output_name,
            ring_size=self.ring_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )

    def __repr__(self):
        return f"RingAttention({self.q}, {self.k}, {self.v}, ring={self.ring_size}) -> {self.output}"
