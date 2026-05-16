"""Multi-device symbolic executor for distributed IR programs.

Tracks per-device tensor state as the program executes. Each device has its
own namespace of tensors; collectives update multiple devices atomically.

This is the runtime that drives the symbolic verification — it propagates
placements, shapes, and expressions through the full distributed graph.

Dispatch Architecture:
    _execute_op() uses a registry (_DISPATCH_TABLE) mapping op types to handler
    methods. Handlers fall into a few categories:
      - per_device: generic loop over all devices, apply op locally
      - p2p: cross-device data movement (Send/Recv)
      - ring: snapshot-then-permute (RingRotate)
      - expand: recursively execute sub-ops (RingAttention, OverlapRegion)
      - sync: Wait/WaitAll with multi-output handling
    New ops only need a registry entry; most map to _exec_per_device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Type
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
    Broadcast,
    Reduce,
    AllToAll,
    Scatter,
    Gather,
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


# ── Slice propagation rules ──────────────────────────────────────────────────

class SliceRule(Enum):
    """How to propagate TensorSlice after an op executes."""
    INHERIT_FIRST = "inherit_first"
    INHERIT_ANY = "inherit_any"
    MATMUL = "matmul"
    ZERO_OFFSET = "zero_offset"
    NONE = "none"


# ── DeviceState ──────────────────────────────────────────────────────────────


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


# ── MultiDeviceExecutor ──────────────────────────────────────────────────────


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

    # ── Public API ───────────────────────────────────────────────────────

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
        """
        self._op_history.clear()
        self._intermediate_states.clear()

        for op in program.ops:
            self._execute_op(op)

        return dict(self.devices[0].tensors)

    def reset_devices(self):
        """Clear all tensor state from all devices."""
        for dev in self.devices.values():
            dev.tensors.clear()

    def run_fwd(self, program: Program) -> Dict[str, TensorState]:
        """Run forward pass only."""
        return self.run_program(program)

    # ── Dispatch ─────────────────────────────────────────────────────────

    def _execute_op(self, op: IROp):
        """Dispatch an op via the registry table."""
        self._op_history.append(op)

        handler = self._find_handler(op)
        handler(op)

        self._save_state()

    def _find_handler(self, op: IROp) -> Callable:
        """Look up the handler for an op type in the dispatch table."""
        for op_type, handler_name in _DISPATCH_TABLE:
            if isinstance(op, op_type):
                return getattr(self, handler_name)
        raise ValueError(
            f"Unknown op type: {type(op).__name__} (repr: {op}). "
            f"inputs={op.input_names}, output={op.output_name}."
        )

    # ── Generic per-device handler ───────────────────────────────────────

    def _exec_per_device(self, op: IROp):
        """Generic handler: apply op independently on each device."""
        input_names = op.input_names
        for did, dev in self.devices.items():
            local_ctx = {}
            missing = False
            for name in input_names:
                t = dev.get(name)
                if t is None:
                    warnings.warn(
                        f"{type(op).__name__}: input '{name}' not found on "
                        f"device {did}. Skipping op. "
                        f"Available: {list(dev.tensors.keys())}"
                    )
                    missing = True
                    break
                local_ctx[name] = t
            if missing:
                continue

            result = self._apply_op(op, local_ctx)
            dev.set(result)
            self._propagate_slice(op, dev, did, result, local_ctx)

    # ── Slice propagation ────────────────────────────────────────────────

    def _propagate_slice(
        self, op: IROp, dev: DeviceState, did: int,
        result: TensorState, local_ctx: Dict[str, TensorState],
    ):
        """Propagate slice info based on the op's slice rule."""
        rule = _SLICE_RULES.get(type(op), SliceRule.INHERIT_FIRST)

        if rule == SliceRule.NONE:
            return

        if rule == SliceRule.ZERO_OFFSET:
            dev.set_slice(op.output_name, TensorSlice(
                device_id=did,
                global_shape=result.global_shape,
                local_shape=result.local_shape,
                offsets=tuple(0 for _ in result.global_shape),
            ))
            return

        if rule == SliceRule.MATMUL:
            sa = dev.get_slice(op.input_names[0])
            sb = dev.get_slice(op.input_names[1])
            if sa is not None and sb is not None:
                out_offsets = (
                    sa.offsets[0],
                    sb.offsets[1] if len(sb.offsets) > 1 else 0,
                )
                dev.set_slice(op.output_name, TensorSlice(
                    device_id=did,
                    global_shape=result.global_shape,
                    local_shape=result.local_shape,
                    offsets=out_offsets,
                ))
            return

        if rule == SliceRule.INHERIT_ANY:
            for name in op.input_names:
                s = dev.get_slice(name)
                if s is not None:
                    dev.set_slice(op.output_name, TensorSlice(
                        device_id=did,
                        global_shape=result.global_shape,
                        local_shape=result.local_shape,
                        offsets=s.offsets,
                    ))
                    return
            return

        # SliceRule.INHERIT_FIRST (default)
        if op.input_names:
            s = dev.get_slice(op.input_names[0])
            if s is not None:
                dev.set_slice(op.output_name, TensorSlice(
                    device_id=did,
                    global_shape=result.global_shape,
                    local_shape=result.local_shape,
                    offsets=s.offsets,
                ))

    # ── Specialized handlers ─────────────────────────────────────────────

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

        sent = copy.deepcopy(x)
        sent.name = op.output
        sent.stage = op.stage
        sent.microbatch_id = op.microbatch_id
        dst_dev.set(sent)

    def _exec_recv(self, op: Recv):
        """Receive tensor on dst device from src."""
        dst_dev = self.devices.get(op.dst)
        if dst_dev is None:
            return

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

    def _exec_send_async(self, op: SendAsync):
        """Execute async Send: apply on src, store result on dst."""
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
        """Execute async Recv: apply on dst device."""
        dst_dev = self.devices.get(op.dst)
        if dst_dev is None:
            return
        x = dst_dev.get(op.x)
        local_ctx = {op.x: x} if x else {}
        result = self._apply_op(op, local_ctx)
        dst_dev.set(result)

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

    def _exec_expand(self, op):
        """Recursively execute sub-ops from expand()."""
        for sub in op.expand():
            self._execute_op(sub)

    def _exec_overlap(self, op: OverlapRegion):
        """Execute an overlap region: run compute and comm ops."""
        for sub in op.compute_ops + op.comm_ops:
            self._execute_op(sub)

    # ── State management ─────────────────────────────────────────────────

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


# ── Dispatch Table ───────────────────────────────────────────────────────────
# Order matters: more specific types must come before their base classes.
# Each entry is (op_type_or_tuple, handler_method_name).

_DISPATCH_TABLE: List[Tuple[type, str]] = [
    # P2P (must check before generic per_device since they have special routing)
    (Send, "_exec_send"),
    (Recv, "_exec_recv"),
    (SendAsync, "_exec_send_async"),
    (RecvAsync, "_exec_recv_async"),
    # Sync
    ((Wait, WaitAll), "_exec_sync"),
    # Ring (snapshot-then-permute)
    (RingRotate, "_exec_ring_rotate"),
    # Expand (composite ops)
    (RingAttention, "_exec_expand"),
    (OverlapRegion, "_exec_overlap"),
    # Dual-output
    (TopKGate, "_exec_topk_gate"),
    # Everything else: generic per-device apply
    (MatMul, "_exec_per_device"),
    ((Add, Multiply), "_exec_per_device"),
    (SiLU, "_exec_per_device"),
    (FlashAttention, "_exec_per_device"),
    (RingAttentionStep, "_exec_per_device"),
    (AllReduce, "_exec_per_device"),
    (AllGather, "_exec_per_device"),
    (ReduceScatter, "_exec_per_device"),
    ((Broadcast, Reduce, AllToAll, Scatter, Gather), "_exec_per_device"),
    (AllReduceAsync, "_exec_per_device"),
    ((MoEDispatch, MoECombine), "_exec_per_device"),
    (ExpertCompute, "_exec_per_device"),
    ((Reshape, Transpose), "_exec_per_device"),
    ((Reinterpret, Convert), "_exec_per_device"),
    ((Cast, LossScale), "_exec_per_device"),
    ((FP8Quantize, FP8Dequantize, AmaxUpdate), "_exec_per_device"),
    (ZeROGatherParam, "_exec_per_device"),
    (ZeROScatterGrad, "_exec_per_device"),
    (ZeROPartitionOptState, "_exec_per_device"),
]

# ── Slice Rules ──────────────────────────────────────────────────────────────
# Maps op type -> SliceRule. Ops not listed default to INHERIT_FIRST.

_SLICE_RULES: Dict[type, SliceRule] = {
    MatMul: SliceRule.MATMUL,
    Add: SliceRule.INHERIT_ANY,
    Multiply: SliceRule.INHERIT_ANY,
    AllReduce: SliceRule.ZERO_OFFSET,
    Broadcast: SliceRule.ZERO_OFFSET,
    Reduce: SliceRule.NONE,
    AllGather: SliceRule.NONE,
    ReduceScatter: SliceRule.NONE,
    AllToAll: SliceRule.NONE,
    Scatter: SliceRule.NONE,
    Gather: SliceRule.NONE,
    AllReduceAsync: SliceRule.NONE,
    MoEDispatch: SliceRule.NONE,
    MoECombine: SliceRule.NONE,
    ZeROGatherParam: SliceRule.NONE,
    ZeROScatterGrad: SliceRule.NONE,
    ZeROPartitionOptState: SliceRule.NONE,
    AmaxUpdate: SliceRule.NONE,
}
