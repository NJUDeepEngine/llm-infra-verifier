from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .base import IROp
from .async_ops import Stream, COMM_STREAM
from ..state import TensorState, LocalSPMDType


@dataclass
class Send(IROp):
    """Send tensor to another device (PP).

    Forward:  data flows src -> dst
    Backward: Recv(grad, from=dst) -- direction reversed
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
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
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
            f"Send({self.x}, {self.src}->{self.dst}, "
            f"stage={self.stage}, mb={self.microbatch_id}) -> {self.output}"
        )


@dataclass
class Recv(IROp):
    """Receive tensor from another device (PP).

    Forward:  data flows src -> dst
    Backward: Send(grad, to=src) -- direction reversed
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

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        x = ctx.get(self.x)
        if x is None:
            raise ValueError(f"Recv: source tensor '{self.x}' not found in context")

        result = TensorState(
            name=self.output,
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
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
        x = ctx[self.x]
        grad_x = TensorState(
            name=f"grad_{self.x}",
            global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
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
            f"Recv({self.x}, {self.src}->{self.dst}, "
            f"stage={self.stage}, mb={self.microbatch_id}) -> {self.output}"
        )


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

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx):
        x = ctx[self.x]
        result = TensorState(
            name=self.output, global_shape=x.global_shape,
            local_shape=x.local_shape, sharding=x.sharding,
            dtype=x.dtype,
            expr=x.expr, stage=self.dst, microbatch_id=self.microbatch_id,
            _async_handle=self.handle,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: TensorState(
            name=f"grad_{self.x}", global_shape=x.global_shape,
            local_shape=x.local_shape,
            sharding=x.sharding,
            dtype=x.dtype,
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

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.x)

    def apply(self, ctx):
        x = ctx.get(self.x)
        result = TensorState(
            name=self.output, global_shape=x.global_shape if x else (),
            local_shape=x.local_shape if x else (),
            sharding=x.sharding if x else None,
            dtype=x.dtype if x else None,
            stage=self.dst, microbatch_id=self.microbatch_id,
            _async_handle=self.handle,
        )
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        x = ctx[self.x]
        return {self.x: TensorState(
            name=f"grad_{self.x}", global_shape=x.global_shape,
            local_shape=x.local_shape, sharding=x.sharding,
            dtype=x.dtype,
        )}
    def is_collective(self): return True
    def is_p2p(self): return True
    def is_async(self): return True
    def clone_with_names(self, im, on): return RecvAsync(
        x=im.get(self.x, self.x), output=on, handle=self.handle,
        src=self.src, dst=self.dst, stage=self.stage,
        microbatch_id=self.microbatch_id, stream=self.stream)
    def __repr__(self):
        return f"RecvAsync({self.x}, {self.src}->{self.dst}) [{self.handle}]"
