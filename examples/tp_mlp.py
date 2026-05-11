"""
Example: Tensor Parallelism — Megatron-style MLP (Gate + Up + Down).

Megatron TP MLP structure:
  Column Parallel (gate, up):  X(Replicate) @ W(Shard1) → Shard(1), NO fwd comm
  Element-wise:                SiLU(gate_Shard1) * up_Shard1 → h_Shard1
  Row Parallel (down):         h(Shard1) @ W(Shard0) → PARTIAL → AllReduce → Y

Forward:  only 1 AllReduce (after down projection)
Backward: AllReduce for grad_X from gate and up projections

This example demonstrates:
  - Correct placement propagation through Column → Element-wise → Row
  - Autograd identifies where collectives are needed in backward
  - Verification checks gradient duality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    compute_local_shape,
)
from verifier.ir import (
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    AllReduce,
    ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine
from verifier.solver import DistributedVerifier


def build_tp_mlp():
    """Build a Megatron-style TP MLP.

    X:      (B, H)   Replicated
    W_gate: (H, I)   Shard(1) on intermediate dim (Column Parallel)
    W_up:   (H, I)   Shard(1) on intermediate dim (Column Parallel)
    W_down: (I, O)   Shard(0) on intermediate dim (Row Parallel)

    Forward ops:
      gate = SiLU(X @ W_gate)    → Shard(1), no fwd comm
      up   = X @ W_up            → Shard(1), no fwd comm
      h    = gate * up           → Shard(1)
      Y    = h @ W_down          → PARTIAL → AllReduce → Replicate
    """
    B, H, I, O = 8, 16, 64, 16
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    # Column Parallel: X is replicated, W is sharded on output dim
    x = TensorState(
        name="x",
        global_shape=(B, H),
        local_shape=(B, H),  # Replicated → full shape
        sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
        expr="x",
        requires_grad=True,
    )
    w_gate = TensorState(
        name="w_gate",
        global_shape=(H, I),
        local_shape=(H, I // 2),  # Shard on dim=1 (I=64 → 32)
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="w_gate",
        requires_grad=True,
    )
    w_up = TensorState(
        name="w_up",
        global_shape=(H, I),
        local_shape=(H, I // 2),
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="w_up",
        requires_grad=True,
    )
    w_down = TensorState(
        name="w_down",
        global_shape=(I, O),
        local_shape=(I // 2, O),  # Shard on dim=0 (I=64 → 32)
        sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
        expr="w_down",
        requires_grad=True,
    )

    # Forward program
    fwd = Program(name="tp_mlp_fwd")

    # Gate projection (Column Parallel): no AllReduce in fwd
    fwd.add(MatMul(a="x", b="w_gate", output="gate_raw"))
    fwd.add(SiLU(x="gate_raw", output="gate"))

    # Up projection (Column Parallel): no AllReduce in fwd
    fwd.add(MatMul(a="x", b="w_up", output="up"))

    # Element-wise
    fwd.add(Multiply(a="gate", b="up", output="h"))

    # Down projection (Row Parallel): needs AllReduce
    fwd.add(MatMul(a="h", b="w_down", output="y_partial"))
    fwd.add(AllReduce(x="y_partial", output="y", op_type="sum"))

    tensors = {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}

    return fwd, tensors, mesh


def main():
    print("=" * 60)
    print("  DTENSOR-VERIFIER: Megatron TP MLP Demo")
    print("=" * 60)

    fwd, tensors, mesh = build_tp_mlp()

    print("\nForward program:")
    print(ir_to_str(fwd))
    print(f"\n  Collectives in fwd: {len(fwd.collectives)} (expected: 1, only down projection)")

    # Execute
    executor = MultiDeviceExecutor(mesh)
    for name, t in tensors.items():
        executor.register_tensor(t)
    result = executor.run_program(fwd)

    print("\nDevice 0 tensor states after forward:")
    for name, t in executor.devices[0].tensors.items():
        partial_str = " [PARTIAL]" if t.partial else ""
        shard_str = ""
        for p in t.sharding.placements:
            if isinstance(p, Shard):
                shard_str = f" Shard({p.dim})"
        print(f"  {name:15s} shape={str(t.global_shape):12s} → local={str(t.local_shape):12s}{shard_str}{partial_str}")

    # Autograd
    autograd = AutogradEngine()
    for op in fwd.ops:
        tensor_states = {**executor.devices[0].tensors}
        autograd.record(op, tensor_states)

    bwd = autograd.generate_backward("y")

    print("\nBackward program (showing dual collectives):")
    print(ir_to_str(bwd))

    # Verify
    verifier = DistributedVerifier()
    verifier.verify_all(fwd, result, bwd_program=bwd)

    print("\n" + verifier.summary())

    # Manual spot checks
    y = executor.get_tensor("y", device_id=0)
    h = executor.get_tensor("h", device_id=0)
    gate = executor.get_tensor("gate", device_id=0)
    y_partial = executor.get_tensor("y_partial", device_id=0)

    print("\nKey spot checks:")
    checks = [
        ("y.partial == False", not y.partial),
        ("y_partial.partial == True", y_partial.partial),
        ("gate is Shard(1) (Column Parallel output)", any(
            isinstance(p, Shard) and p.dim == 1 for p in gate.sharding.placements
        )),
        ("h is Shard(1) (element-wise preserves placement)", any(
            isinstance(p, Shard) and p.dim == 1 for p in h.sharding.placements
        )),
        ("Only 1 AllReduce in fwd", len(fwd.collectives) == 1),
    ]
    for desc, ok in checks:
        print(f"  {'[PASS]' if ok else '[FAIL]'} {desc}")

    all_ok = all(ok for _, ok in checks)
    print(f"\n  Overall: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    main()
