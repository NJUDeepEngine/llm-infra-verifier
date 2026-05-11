"""
Example: Tensor Parallelism — Row Parallel Linear (fwd + bwd).

Tests two cases:
  1. CORRECT:  MatMul(Shard(H), Shard(H)) → AllReduce → Y(Replicate)
  2. INCORRECT: MatMul without AllReduce → Y is still PARTIAL → verification FAIL

Row Parallel Linear semantics (Megatron-style):
  X: shape (B, H), Shard on dim=1 (the H/reduce dim)
  W: shape (H, O), Shard on dim=0 (the H/reduce dim)
  Both sharded on the reduce dimension → MatMul output is PARTIAL
  AllReduce converts PARTIAL → REPLICATE

This is the minimal end-to-end demo of the verification pipeline:
  TIR → Lift → Execute → Autograd → Verify
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
    AllReduce,
    ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine
from verifier.solver import DistributedVerifier


def demo_tp_linear_correct():
    """Row Parallel Linear: the CORRECT case with AllReduce."""
    print("=" * 60)
    print("TP Linear — CORRECT (with AllReduce)")
    print("=" * 60)

    # Device mesh: 2 GPUs along TP dimension
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    # Row Parallel: both operands sharded on reduce dim H=16
    # X: Shard(dim=1) → local (8, 8), W: Shard(dim=0) → local (8, 32)
    B, H, O = 8, 16, 32
    x = TensorState(
        name="x",
        global_shape=(B, H),
        local_shape=(B, H // 2),
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="x",
        requires_grad=True,
    )
    w = TensorState(
        name="w",
        global_shape=(H, O),
        local_shape=(H // 2, O),
        sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
        expr="w",
        requires_grad=True,
    )

    # IR program: MatMul → AllReduce
    fwd = Program(name="tp_linear_correct")
    fwd.add(MatMul(a="x", b="w", output="y_partial"))
    fwd.add(AllReduce(x="y_partial", output="y", op_type="sum"))

    print("\nForward program:")
    print(ir_to_str(fwd))

    # Execute on multi-device
    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(x)
    executor.register_tensor(w)
    result = executor.run_program(fwd)

    print("\nDevice states after forward:")
    print(executor)

    y = executor.get_tensor("y", device_id=0)
    y_partial = executor.get_tensor("y_partial", device_id=0)
    print(f"\n  y_partial: {y_partial}")
    print(f"  y:         {y}")

    # Autograd: generate backward
    autograd = AutogradEngine()
    for op in fwd.ops:
        tensor_states = executor.devices[0].tensors
        autograd.record(op, tensor_states)

    bwd = autograd.generate_backward("y")
    print("\nBackward program:")
    print(ir_to_str(bwd))

    # Verify
    verifier = DistributedVerifier()
    verifier.verify_all(fwd, result, bwd_program=bwd)

    print("\n" + verifier.summary())
    print()

    return verifier.results


def demo_tp_linear_missing_allreduce():
    """Row Parallel Linear: the INCORRECT case — missing AllReduce."""
    print("=" * 60)
    print("TP Linear — INCORRECT (missing AllReduce)")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    B, H, O = 8, 16, 32
    x = TensorState(
        name="x",
        global_shape=(B, H),
        local_shape=(B, H // 2),
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="x",
        requires_grad=True,
    )
    w = TensorState(
        name="w",
        global_shape=(H, O),
        local_shape=(H // 2, O),
        sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
        expr="w",
        requires_grad=True,
    )

    # BUG: MatMul only — NO AllReduce!
    fwd = Program(name="tp_linear_bug")
    fwd.add(MatMul(a="x", b="w", output="y"))

    print("\nForward program (BUG):")
    print(ir_to_str(fwd))

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(x)
    executor.register_tensor(w)
    result = executor.run_program(fwd)

    y = executor.get_tensor("y", device_id=0)
    print(f"\n  y: {y}")
    print(f"  y.partial = {y.partial}  <- should be False but is True!")

    # Verify
    verifier = DistributedVerifier()
    verifier.verify_postcondition(y, expected_partial=False)
    verifier.verify_communication_legality(fwd)

    print("\n" + verifier.summary())
    print()

    return verifier.results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DTENSOR-VERIFIER: Tensor Parallelism Demo")
    print("=" * 60 + "\n")

    results_correct = demo_tp_linear_correct()
    results_bug = demo_tp_linear_missing_allreduce()

    # Summary
    all_pass = all(r.passed for r in results_correct)
    bug_detected = any(not r.passed for r in results_bug)

    print("=" * 60)
    print("  FINAL VERDICT")
    print("=" * 60)
    print(f"  Correct case:  {'PASSED' if all_pass else 'FAILED (unexpected)'}")
    print(f"  Bug detected:  {'YES (expected)' if bug_detected else 'NO (unexpected)'}")
    print()
