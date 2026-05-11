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
)


@dataclass
class DeviceState:
    """State of a single device."""
    device_id: int
    tensors: Dict[str, TensorState] = field(default_factory=dict)

    def get(self, name: str) -> Optional[TensorState]:
        return self.tensors.get(name)

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

        for did in device_ids:
            local_t = copy.deepcopy(tensor)
            local_t.local_shape = compute_local_shape(
                tensor.global_shape, tensor.sharding
            )
            self.devices[did].set(local_t, warn_on_overwrite=False)

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
        else:
            raise ValueError(
                f"Unknown op type: {type(op).__name__} (repr: {op}). "
                f"inputs={op.input_names}, output={op.output_name}. "
                f"Supported types: MatMul, Add, Multiply, SiLU, AllReduce, AllGather, "
                f"ReduceScatter, Send, Recv, Reshape, Transpose, FlashAttention"
            )

        # Save intermediate state snapshot
        self._save_state()

    def _exec_matmul(self, op: MatMul):
        """Execute MatMul on all devices."""
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
            # Use the op's apply method with device-local context
            local_ctx = {op.a: a, op.b: b}
            result = op.apply(local_ctx)
            dev.set(result)

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
            result = op.apply(local_ctx)
            dev.set(result)

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
            result = op.apply(local_ctx)
            dev.set(result)

    def _exec_allreduce(self, op: AllReduce):
        """AllReduce: each device reduces its partial → replicated."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = op.apply(local_ctx)
            dev.set(result)

    def _exec_allgather(self, op: AllGather):
        """AllGather: gather sharded dims across devices."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = op.apply(local_ctx)
            dev.set(result)

    def _exec_reducescatter(self, op: ReduceScatter):
        """ReduceScatter: reduce then scatter."""
        for did, dev in self.devices.items():
            x = dev.get(op.x)
            if x is None:
                continue
            local_ctx = {op.x: x}
            result = op.apply(local_ctx)
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
        """Execute FlashAttention on all devices (CP semantics).

        Each device has Q replicated but K, V sharded on seq_len.
        """
        for did, dev in self.devices.items():
            q = dev.get(op.q)
            k = dev.get(op.k)
            v = dev.get(op.v)
            if q is None or k is None or v is None:
                continue
            local_ctx = {op.q: q, op.k: k, op.v: v}
            result = op.apply(local_ctx)
            dev.set(result)

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

    def __repr__(self):
        dev_strs = []
        for did, dev in self.devices.items():
            tensors = list(dev.tensors.keys())
            dev_strs.append(f"  device_{did}: {tensors}")
        return f"MultiDeviceExecutor(mesh={self.mesh}):\n" + "\n".join(dev_strs)
