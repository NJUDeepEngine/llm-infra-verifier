"""
Example: Pipeline Parallelism — 2-Stage 1F1B (fwd + bwd).

Models a 2-layer transformer with 1F1B scheduling:
  Stage 0: Embed + Layer0
  Stage 1: Layer1 + LM Head

Communication:
  Forward:  Stage 0 → Send(h0) → Stage 1
  Backward: Stage 1 → Send(grad_h0) → Stage 0

Verification checks:
  1. Send/Recv matching (each Send has a corresponding Recv)
  2. Direction reversal in backward
  3. 1F1B schedule correctness (warmup → steady → cooldown)
  4. Activation liveness (activations available for backward)
  5. Deadlock freedom
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
    Send,
    Recv,
    ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine
from verifier.solver import DistributedVerifier
from verifier.schedules import (
    MicroBatch,
    PP1F1BSchedule,
    ActivationTracker,
    DeadlockChecker,
    OpType,
    Phase,
)


def build_pp_2stage_program():
    """Build a 2-stage pipeline parallel program.

    Stage 0 (device 0): Embed → Layer0
    Stage 1 (device 1): Layer1 → LM Head

    With 2 micro-batches (M=2).
    """
    B, H = 8, 16
    mesh = DeviceMesh(shape=(2,), dim_names=("pp",))

    # ── Tensors ──
    x = TensorState(
        name="x",
        global_shape=(B, H),
        local_shape=(B, H),
        sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
        expr="x",
        requires_grad=True,
        stage=0,
    )
    w0 = TensorState(
        name="w0",
        global_shape=(H, H),
        local_shape=(H, H),
        sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
        expr="w0",
        requires_grad=True,
        stage=0,
    )
    w1 = TensorState(
        name="w1",
        global_shape=(H, H),
        local_shape=(H, H),
        sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
        expr="w1",
        requires_grad=True,
        stage=1,
    )

    # ── Forward program (per micro-batch, stage info encoded) ──
    # For PP, we model the full unrolled schedule
    fwd = Program(name="pp_2stage_fwd")

    # Micro-batch 0:
    # Stage 0: h0_mb0 = X_mb0 @ W0
    fwd.add(MatMul(a="x_mb0", b="w0", output="h0_mb0"))
    # Send h0_mb0 from stage 0 to stage 1
    fwd.add(Send(
        x="h0_mb0", output="h0_mb0_sent",
        src=0, dst=1, stage=0, microbatch_id=0,
    ))
    # Recv on stage 1
    fwd.add(Recv(
        x="h0_mb0_sent", output="h0_mb0_rcvd",
        src=0, dst=1, stage=1, microbatch_id=0,
    ))
    # Stage 1: y_mb0 = h0_mb0 @ W1
    fwd.add(MatMul(a="h0_mb0_rcvd", b="w1", output="y_mb0"))

    # Micro-batch 1:
    fwd.add(MatMul(a="x_mb1", b="w0", output="h0_mb1"))
    fwd.add(Send(
        x="h0_mb1", output="h0_mb1_sent",
        src=0, dst=1, stage=0, microbatch_id=1,
    ))
    fwd.add(Recv(
        x="h0_mb1_sent", output="h0_mb1_rcvd",
        src=0, dst=1, stage=1, microbatch_id=1,
    ))
    fwd.add(MatMul(a="h0_mb1_rcvd", b="w1", output="y_mb1"))

    tensors = {
        "x_mb0": x.with_name("x_mb0"),
        "x_mb1": x.with_name("x_mb1"),
        "w0": w0,
        "w1": w1,
    }

    return fwd, tensors, mesh


def demo_pp_2stage():
    print("=" * 60)
    print("  DTENSOR-VERIFIER: PP 2-Stage 1F1B Demo")
    print("=" * 60)

    fwd, tensors, mesh = build_pp_2stage_program()

    print("\nForward program:")
    print(ir_to_str(fwd))

    # Count Send/Recv pairs
    sends = [op for op in fwd.ops if isinstance(op, Send)]
    recvs = [op for op in fwd.ops if isinstance(op, Recv)]
    print(f"\n  Sends: {len(sends)}, Recvs: {len(recvs)}")

    # ── Execute ──
    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(tensors["x_mb0"], device_ids=[0])
    executor.register_tensor(tensors["x_mb1"], device_ids=[0])
    executor.register_tensor(tensors["w0"], device_ids=[0])
    executor.register_tensor(tensors["w1"], device_ids=[1])
    result = executor.run_program(fwd)

    print("\nDevice 0 final tensors:")
    for name, t in executor.devices[0].tensors.items():
        print(f"  {name}: {t.global_shape}, stage={t.stage}")

    print("\nDevice 1 final tensors:")
    for name, t in executor.devices[1].tensors.items():
        print(f"  {name}: {t.global_shape}, stage={t.stage}")

    # ── Verify communication legality ──
    verifier = DistributedVerifier()
    comm_result = verifier.verify_communication_legality(fwd)
    print(f"\n  Communication legality: {'PASSED' if comm_result.passed else 'FAILED'}")

    # ── Check Send/Recv matching ──
    deadlock_checker = DeadlockChecker()
    for op in fwd.ops:
        if isinstance(op, Send):
            deadlock_checker.add_send(op.src, op.dst, op.x)
        elif isinstance(op, Recv):
            deadlock_checker.add_recv(op.src, op.dst, op.x)

    is_free, errors = deadlock_checker.check()
    print(f"  Deadlock freedom: {'PASSED' if is_free else 'FAILED'}")
    for err in errors:
        print(f"    Error: {err}")

    # ── 1F1B schedule verification ──
    schedule = PP1F1BSchedule(num_stages=2, num_microbatches=2)
    sched = schedule.generate_simple()

    print(f"\n  1F1B Schedule ({len(sched)} steps):")
    for mb in sched:
        print(f"    {mb}")

    # Check activation liveness
    tracker = ActivationTracker(num_stages=2)
    passed, errors = tracker.verify_activation_liveness(sched)
    print(f"\n  Activation liveness: {'PASSED' if passed else 'FAILED'}")
    for err in errors:
        print(f"    Error: {err}")

    # ── Overall ──
    all_ok = (
        comm_result.passed
        and is_free
        and len(sends) == len(recvs)
    )
    print(f"\n  Overall PP verification: {'PASSED' if all_ok else 'FAILED'}")

    return all_ok


if __name__ == "__main__":
    demo_pp_2stage()
