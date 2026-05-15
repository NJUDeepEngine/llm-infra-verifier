"""Multi-device symbolic executor for distributed IR programs.

Tracks per-device tensor state as the program executes. Each device has its
own namespace of tensors; collectives update multiple devices atomically.

This is the runtime that drives the symbolic verification — it propagates
placements, shapes, and expressions through the full distributed graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import warnings

from .state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
    TensorSlice,
    compute_tensor_slices,
)
from .ir import (
    IROp,
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    AllReduce,
    AllGather,
    ReduceScatter,
    Send,
    Recv,
    Reshape,
    Transpose,
    FlashAttention,
    AllReduceAsync,
    SendAsync,
    RecvAsync,
    Wait,
    WaitAll,
    OverlapRegion,
    Reinterpret,
    Convert,
    Cast,
    LossScale,
    FP8Quantize,
    FP8Dequantize,
    AmaxUpdate,
    ZeROGatherParam,
    ZeROScatterGrad,
    ZeROPartitionOptState,
    RingRotate,
    RingAttentionStep,
    RingAttention,
    TopKGate,
    MoEDispatch,
    MoECombine,
    ExpertCompute,
)


@dataclass
class DeviceState:
    """State of a single device, including per-tensor slice info."""
    device_id: int
    tensors: Dict[str, TensorState] = field(default_factory=dict)
    slices: Dict[str, TensorSlice] = field(default_factory=dict)

    def get(self, name: str) -> Optional[TensorState]:
        return self.tensors.get(name)

    def get_slice(self, name: str) -> Optional[TensorSlice]:
        return self.slices.get(name)

    def set(self, tensor: TensorState, warn_on_overwrite: bool = True):
        if warn_on_overwrite:
            existing = self.tensors.get(tensor.name)
            if existing is not None and (
                existing.global_shape != tensor.global_shape
                or existing.sharding.placements != tensor.sharding.placements
                or existing.local_shape != tensor.local_shape
            ):
                warnings.warn(
                    f"Overwriting tensor '{tensor.name}' on device {self.device_id}: "
                    f"old (shape={existing.global_shape}, local={existing.local_shape}, "
                    f"placements={existing.sharding.placements}), "
                    f"new (shape={tensor.global_shape}, local={tensor.local_shape}, "
                    f"placements={tensor.sharding.placements})"
                )
        self.tensors[tensor.name] = tensor

    def set_slice(self, name: str, s: TensorSlice):
        self.slices[name] = s

    def has(self, name: str) -> bool:
        return name in self.tensors

    def __repr__(self):
        tensor_names = list(self.tensors.keys())
        return f"Device({self.device_id}): {tensor_names}"


@dataclass
class MultiDeviceExecutor:
    """Symbolic executor over a device mesh.

    Executes IR programs, maintaining per-device tensor states and
    simulating collective communication.
    """
    mesh: DeviceMesh
    spmd_checking: bool = False

    def _apply_op(self, op: IROp, ctx: Dict[str, TensorState]) -> TensorState:
        if self.spmd_checking:
            return op.apply_checked(ctx)
        return op.apply(ctx)

    def __post_init__(self):
        self.devices: Dict[int, DeviceState] = {}
        for i in range(self.mesh.num_devices):
            self.devices[i] = DeviceState(device_id=i)
        self._op_history: List[IROp] = []
        self._intermediate_states: List[Dict[int, Dict[str, TensorState]]] = []

    def register_tensor(
        self,
        tensor: TensorState,
        device_ids: Optional[List[int]] = None,
    ):
        """Register an initial tensor on specified devices (default: all)."""
        if device_ids is None:
            device_ids = list(range(self.mesh.num_devices))

        for did in device_ids:
            existing = self.devices[did].get(tensor.name)
            if existing is not None:
                warnings.warn(
                    f"Re-registering tensor '{tensor.name}' on device {did}: "
                    f"existing tensor will be overwritten. "
                    f"old spec: shape={existing.global_shape}, "
                    f"local={existing.local_shape}, "
                    f"placements={existing.sharding.placements}"
                )
                break

        all_slices = compute_tensor_slices(tensor.global_shape, tensor.sharding)

        for did in device_ids:
            local_t = copy.deepcopy(tensor)
            local_t.local_shape = compute_local_shape(
                tensor.global_shape, tensor.sharding
            )
            self.devices[did].set(local_t, warn_on_overwrite=False)
            if did in all_slices:
                self.devices[did].set_slice(tensor.name, all_slices[did])

    def get_tensor(self, name: str, device_id: int = 0) -> Optional[TensorState]:
        """Get a tensor from a specific device."""
        return self.devices[device_id].get(name)

    def get_all_devices_tensor(self, name: str) -> Dict[int, TensorState]:
        """Get tensor state across all devices."""
        result = {}
        for did, dev in self.devices.items():
            t = dev.get(name)
            if t is not None:
                result[did] = t
        return result

    def run_program(self, program: Program) -> Dict[str, TensorState]:
        """Run a full program on all devices.

        Returns the final tensor state (from device 0) for verification.
        Collectives are broadcast to all participating devices.
        """
        self._op_history.clear()
        self._intermediate_states.clear()

        for op in program.ops:
            self._execute_op(op)

        # Return device 0's view of all tensors
        return dict(self.devices[0].tensors)

    def reset_devices(self):
        """Clear all tensor state from all devices.

        Call this to start fresh without creating a new executor.
        Note: this also clears registered initial tensors.
        """
        for dev in self.devices.values():
            dev.tensors.clear()

    def run_fwd(self, program: Program) -> Dict[str, TensorState]:
        """Run forward pass only."""
        return self.run_program(program)

    def _execute_op(self, op: IROp):
        """Dispatch an op to the appropriate execution method."""
        self._op_history.append(op)

        if isinstance(op, MatMul):
            self._exec_matmul(op)
        elif isinstance(op, Add):
            self._exec_elementwise(op)
        elif isinstance(op, Multiply):
            self._exec_elementwise(op)
        elif isinstance(op, SiLU):
            self._exec_unary(op)
        elif isinstance(op, AllReduce):
            self._exec_allreduce(op)
        elif isinstance(op, AllGather):
            self._exec_allgather(op)
        elif isinstance(op, ReduceScatter):
            self._exec_reducescatter(op)
        elif isinstance(op, Send):
            self._exec_send(op)
        elif isinstance(op, Recv):
            self._exec_recv(op)
        elif isinstance(op, Reshape):
            self._exec_unary(op)
        elif isinstance(op, Transpose):
            self._exec_unary(op)
        elif isinstance(op, FlashAttention):
            self._exec_flash_attn(op)
        elif isinstance(op, AllReduceAsync):
            self._exec_collective_unary(op)
        elif isinstance(op, (Wait, WaitAll)):
            self._exec_sync(op)
        elif isinstance(op, SendAsync):
            self._exec_send_async(op)
        elif isinstance(op, RecvAsync):
            self._exec_recv_async(op)
        elif isinstance(op, OverlapRegion):
            self._exec_overlap(op)
        elif isinstance(op, (Reinterpret, Convert)):
            self._exec_unary(op)
        elif isinstance(op, (Cast, LossScale)):
            self._exec_unary(op)
        elif isinstance(op, (FP8Quantize, FP8Dequantize, AmaxUpdate)):
            self._exec_unary(op)
        elif isinstance(op, ZeROGatherParam):
            self._exec_allgather(op)
        elif isinstance(op, ZeROScatterGrad):
            self._exec_reducescatter(op)
        elif isinstance(op, ZeROPartitionOptState):
            self._exec_unary(op)
        elif isinstance(op, RingRotate):
            self._exec_ring_rotate(op)
        elif isinstance(op, RingAttentionStep):
            self._exec_flash_attn(op)
        elif isinstance(op, RingAttention):
            for sub in op.expand():
                self._execute_op(sub)
        elif isinstance(op, TopKGate):
            self._exec_topk_gate(op)
        elif isinstance(op, (MoEDispatch, MoECombine)):
            self._exec_collective_unary(op)
        elif isinstance(op, ExpertCompute):
            self._exec_unary(op)
        else:
            raise ValueError(
                f"Unknown op type: {type(op).__name__} (repr: {op}). "
                f"inputs={op.input_names}, output={op.output_name}."
            )

        # Save intermediate state snapshot
        self._save_state()

    def _exec_matmul(self, op: MatMul):
        """Execute MatMul on all devices, propagating slices."""
        for did, dev in self.devices.items():
            a = dev.get(op.a)
            b = dev.get(op.b)
            if a is None or b is None:
                if a is None:
                    warnings.warn(
                        f"MatMul: input '{op.a}' not found on device {did}. "
                        f"Skipping op. Available: {list(dev.tensors.keys())}"
                    )
                if b is None:
                    warnings.warn(
                        f"MatMul: input '{op.b}' not found on device {did}. "
                        f"Skipping op. Available: {list(dev.tensors.keys())}"
                    )
                continue
            local_ctx = {op.a: a, op.b: b}
            result = self._apply_op(op, local_ctx)
            dev.set(result)

            # Propagate slices: Y = A @ B → Y rows from A, Y cols from B
            sa = dev.get_slice(op.a)
            sb = dev.get_slice(op.b)
            if sa is not None and sb is not None:
                out_global = result.global_shape
                out_offsets = (sa.offsets[0], sb.offsets[1] if len(sb.offsets) > 1 else 0)
                dev.set_slice(op.output, TensorSlice(
                    device_id=did,
                    global_shape=out_global,
                    local_shape=result.local_shape,
                    offsets=out_offsets,
                ))

    def _exec_elementwise(self, op: Add | Multiply):
        """Execute element-wise op on all devices."""
        for did, dev in self.devices.items():
            a = dev.get(op.a)
            b = dev.get(op.b)
            if a is None or b is None:
                if a is None:
                    warnings.warn(
                        f"{type(op).__name__}: input '{op.a}' not found on device {did}. "
                        f"Skipping op. Available: {list(dev.tensors.keys())}"
                    )
                if b is None:
                    warnings.warn(
                        f"{type(op).__name__}: input '{op.b}' not found on device {did}. "
                        f"Skipping op. Available: {list(dev.tensors.keys())}"
                    )
                continue
            local_ctx = {op.a: a, op.b: b}
            result = self._apply_op(op, local_ctx)
            dev.set(result)
            # Element-wise: inherit slice from whichever input is sharded
            sa = dev.get_slice(op.a)
            sb = dev.get_slice(op.b)
            src = sa or sb
            if src is not None:
                dev.set_slice(op.output, TensorSlice(
                    device_id=did,
                    global_shape=result.global_shape,
                    local_shape=result.local_shape,
                    offsets=src.offsets,
                ))

    def _exec_unary(self, op: SiLU | Reshape | Transpose):
        """Execute unary op on all devices."""
        for did, dev in self.devices.items():
            x = dev.get(op.input_names[0])
            if x is None:
                warnings.warn(
                    f"{type(op).__name__}: input '{op.input_names[0]}' not found on "
                    f"device {did}. Skipping op. Available: {list(dev.tensors.keys())}"
                )
                continue
            local_ctx = {op.input_names[0]: x}
            result = self._apply_op(op, local_ctx)
            dev.set(result)
            # Unary: inherit slice from input
            sx = dev.get_slice(op.input_names[0])
            if sx is not None:
                dev.set_slice(op.output_name, TensorSlice(
                    device_id=did,
                    global_shape=result.global_shape,
                    local_shape=result.local_shape,
                    offsets=sx.offsets,
                ))

    def _exec_allreduce(self, op: AllReduce):
        """AllReduce: each device reduces its partial → replicated."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = self._apply_op(op, local_ctx)
            dev.set(result)
            # After AllReduce, every device holds the full tensor
            dev.set_slice(op.output, TensorSlice(
                device_id=did,
                global_shape=result.global_shape,
                local_shape=result.local_shape,
                offsets=tuple(0 for _ in result.global_shape),
            ))

    def _exec_allgather(self, op: AllGather):
        """AllGather: gather sharded dims across devices."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = self._apply_op(op, local_ctx)
            dev.set(result)

    def _exec_reducescatter(self, op: ReduceScatter):
        """ReduceScatter: reduce then scatter."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = self._apply_op(op, local_ctx)
            dev.set(result)

    def _exec_send(self, op: Send):
        """Send tensor from src to dst device."""
        src_dev = self.devices.get(op.src)
        dst_dev = self.devices.get(op.dst)
        if src_dev is None or dst_dev is None:
            return

        x = src_dev.get(op.x)
        if x is None:
            raise ValueError(
                f"Send: tensor '{op.x}' not found on source device {op.src} "
                f"(dst={op.dst}). Available tensors on device {op.src}: "
                f"{list(src_dev.tensors.keys())}"
            )

        # Copy to dst device
        sent = copy.deepcopy(x)
        sent.name = op.output
        sent.stage = op.stage
        sent.microbatch_id = op.microbatch_id
        dst_dev.set(sent)

    def _exec_recv(self, op: Recv):
        """Receive tensor on dst device from src.

        The corresponding Send has already placed the tensor on the dst
        device (under the Send's output name).  Recv looks for it there
        and renames it.
        """
        dst_dev = self.devices.get(op.dst)
        if dst_dev is None:
            return

        # The Send already wrote the tensor to dst_dev under op.x (the Send's output)
        sent = dst_dev.get(op.x)
        if sent is None:
            raise ValueError(
                f"Recv: no matching Send result '{op.x}' found on device {op.dst}. "
                f"Available tensors: {list(dst_dev.tensors.keys())}"
            )

        received = copy.deepcopy(sent)
        received.name = op.output
        received.stage = op.dst
        received.microbatch_id = op.microbatch_id
        dst_dev.set(received)

    def _exec_flash_attn(self, op: FlashAttention):
        """Execute FlashAttention on all devices (CP semantics)."""
        for did, dev in self.devices.items():
            q = dev.get(op.q)
            k = dev.get(op.k)
            v = dev.get(op.v)
            if q is None or k is None or v is None:
                continue
            local_ctx = {op.q: q, op.k: k, op.v: v}
            result = self._apply_op(op, local_ctx)
            dev.set(result)
            # Output inherits Q's slice
            sq = dev.get_slice(op.q)
            if sq is not None:
                dev.set_slice(op.output, TensorSlice(
                    device_id=did,
                    global_shape=result.global_shape,
                    local_shape=result.local_shape,
                    offsets=sq.offsets,
                ))

    def _exec_collective_unary(self, op):
        """Execute a collective unary op (e.g. AllReduceAsync) on all devices."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = self._apply_op(op, local_ctx)
            dev.set(result)

    def _exec_sync(self, op):
        """Execute Wait/WaitAll on all devices."""
        for did, dev in self.devices.items():
            local_ctx = dict(dev.tensors)
            result = self._apply_op(op, local_ctx)
            if result is not None:
                dev.set(result)
            if isinstance(op, WaitAll):
                for out_name in op.outputs:
                    t = local_ctx.get(out_name)
                    if t is not None:
                        dev.set(t)

    def _exec_send_async(self, op: SendAsync):
        """Execute async Send."""
        src_dev = self.devices.get(op.src)
        dst_dev = self.devices.get(op.dst)
        if src_dev is None or dst_dev is None:
            return
        x = src_dev.get(op.x)
        if x is None:
            return
        local_ctx = {op.x: x}
        result = self._apply_op(op, local_ctx)
        dst_dev.set(result)

    def _exec_recv_async(self, op: RecvAsync):
        """Execute async Recv."""
        dst_dev = self.devices.get(op.dst)
        if dst_dev is None:
            return
        x = dst_dev.get(op.x)
        local_ctx = {op.x: x} if x else {}
        result = self._apply_op(op, local_ctx)
        dst_dev.set(result)

    def _exec_ring_rotate(self, op: RingRotate):
        """Ring rotation: each device gets tensor from (rank-1) % ring_size."""
        current = {}
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is not None:
                current[did] = copy.deepcopy(x)

        for did, dev in self.devices.items():
            src = (did - 1) % op.ring_size
            if src in current:
                rotated = copy.deepcopy(current[src])
                rotated.name = op.output
                rotated.ring_step = (current[src].ring_step or 0) + 1
                if op.handle:
                    rotated._async_handle = op.handle
                dev.set(rotated)

    def _exec_topk_gate(self, op: TopKGate):
        """TopKGate: dual-output (weights + indices) on all devices."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            gw = dev.get(op.gate_weight)
            if x is None or gw is None:
                continue
            local_ctx = {op.x: x, op.gate_weight: gw}
            result = self._apply_op(op, local_ctx)
            dev.set(result)
            indices = local_ctx.get(op.indices_output)
            if indices is not None:
                dev.set(indices)

    def _exec_overlap(self, op: OverlapRegion):
        """Execute an overlap region: run compute and comm ops."""
        for sub in op.compute_ops + op.comm_ops:
            self._execute_op(sub)

    def _save_state(self):
        """Save a snapshot of all device states."""
        snapshot = {}
        for did, dev in self.devices.items():
            snapshot[did] = {
                name: copy.deepcopy(t) for name, t in dev.tensors.items()
            }
        self._intermediate_states.append(snapshot)

    @property
    def op_history(self) -> List[IROp]:
        return list(self._op_history)

    def state_snapshot(self, step: int) -> Dict[int, Dict[str, TensorState]]:
        """Return the state snapshot at a given step."""
        if step < 0 or step >= len(self._intermediate_states):
            raise IndexError(f"Step {step} out of range [0, {len(self._intermediate_states)})")
        return self._intermediate_states[step]

    def final_state(self) -> Dict[int, Dict[str, TensorState]]:
        """Return the final state of all devices."""
        result = {}
        for did, dev in self.devices.items():
            result[did] = dict(dev.tensors)
        return result

    def final_slices(self) -> Dict[int, Dict[str, TensorSlice]]:
        """Return the final per-device slices for all tensors."""
        result = {}
        for did, dev in self.devices.items():
            result[did] = dict(dev.slices)
        return result

    def __repr__(self):
        dev_strs = []
        for did, dev in self.devices.items():
            tensors = list(dev.tensors.keys())
            dev_strs.append(f"  device_{did}: {tensors}")
        return f"MultiDeviceExecutor(mesh={self.mesh}):\n" + "\n".join(dev_strs)
