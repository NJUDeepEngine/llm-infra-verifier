from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import IROp
from ..state import (
    TensorState,
    LocalSPMDType,
    Partial,
    Replicate,
    ShardingSpec,
)


@dataclass
class Handle:
    """A handle to an in-flight asynchronous operation.

    Tracks the async op's output buffer, creation time (issue),
    and expected completion time (after Wait).
    """
    name: str
    op_name: str = ""
    buffer_name: str = ""
    is_active: bool = True


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


DEFAULT_STREAM = Stream("default", 0)
COMM_STREAM = Stream("comm", 0)
COMPUTE_STREAM = Stream("compute", 0)


@dataclass
class AllReduceAsync(IROp):
    """Asynchronous AllReduce: launches and returns immediately.

    The output buffer is being written asynchronously. Must be
    followed by a Wait(handle) before the output is read.

    Forward: PARTIAL -> REPLICATE (after Wait)
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

    def propagate_spmd_type(self, input_types):
        return LocalSPMDType.REPLICATE

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
            _async_handle=self.handle,
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
    """
    handle: str
    tensor: str
    output: str

    @property
    def input_names(self) -> List[str]:
        return [self.tensor]

    @property
    def output_name(self) -> str:
        return self.output

    def propagate_spmd_type(self, input_types):
        return input_types.get(self.tensor)

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        ts = ctx.get(self.tensor)
        if ts is None:
            raise ValueError(f"Wait: tensor '{self.tensor}' not found")
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
    tensors: Tuple[str, ...]
    outputs: Tuple[str, ...]

    @property
    def input_names(self) -> List[str]:
        return list(self.tensors)

    @property
    def output_name(self) -> str:
        return self.outputs[0] if self.outputs else ""

    def propagate_spmd_type(self, input_types):
        if self.tensors:
            return input_types.get(self.tensors[0])
        return None

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
    output: str = ""

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
        for op in self.compute_ops:
            op.apply(ctx)
        for op in self.comm_ops:
            op.apply(ctx)
        return ctx.get(self.output_name)

    def vjp(self, ctx, grad_output):
        return {}

    def is_collective(self) -> bool:
        return any(op.is_collective() for op in self.comm_ops)

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
