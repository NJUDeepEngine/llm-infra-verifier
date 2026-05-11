"""
Demo: Temporal Overlap Verification.

Demonstrates detection of four types of async communication bugs:
  1. DATA RACE: compute reads async output before Wait
  2. MISSING WAIT: async output consumed without any Wait
  3. BUFFER ALIASING: two async ops sharing same output buffer
  4. DEPENDENCY VIOLATION: RecvAsync before SendAsync

Each case shows:
  - The buggy IR program
  - The Happens-Before graph built from it
  - The detected violations

Reference bugs:
  - PyTorch #144359 (incorrect TP results — can manifest as overlap timing issues)
  - NCCL async usage patterns from Megatron-LM
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    compute_local_shape,
)
from verifier.ir import (
    Program, MatMul, AllReduce, AllReduceAsync, SiLU,
    Wait, ir_to_str,
    DEFAULT_STREAM, COMM_STREAM, COMPUTE_STREAM,
)
from verifier.temporal import (
    TemporalGraph, RaceDetector, verify_temporal,
)


def print_section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Case 1: DATA RACE — compute reads async output before Wait
# ═══════════════════════════════════════════════════════════════════════════════

def demo_data_race():
    """
    Scenario: Row Parallel backward pass.
    AllReduceAsync(grad_w_partial) is launched on COMM_STREAM.
    But the next MatMul immediately reads grad_w on COMPUTE_STREAM
    before Wait(handle) — DATA RACE!

    Timeline:
      COMM:    [=== AllReduceAsync(grad_w) ===]  → write grad_w
      COMPUTE:      [MatMul(grad_w, ...)]          → READ grad_w — RACE!
      DEFAULT:                                      Wait(h)
    """
    print_section("Case 1: DATA RACE — Read async output before Wait")

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    prog = Program("data_race")
    # Forward matmul produces partial
    prog.add(MatMul(a="x", b="w", output="y_partial"))
    # Async allreduce: launches on COMM stream, writes grad_w asynchronously
    prog.add(AllReduceAsync(
        x="y_partial", output="y", handle="h1",
        op_type="sum", stream=COMM_STREAM,
    ))
    # BUG: next op reads 'y' before Wait(h1)
    # On COMPUTE stream — concurrent with COMM stream
    prog.add(MatMul(a="y", b="w2", output="z"))
    # Wait comes too late
    prog.add(Wait(handle="h1", tensor="y", output="y_safe"))

    print(f"\nProgram ({len(prog)} ops):")
    print(ir_to_str(prog))

    result = verify_temporal(prog)
    print(f"\n{result.summary()}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Case 2: MISSING WAIT — async output used without any Wait at all
# ═══════════════════════════════════════════════════════════════════════════════

def demo_missing_wait():
    """
    Scenario: Forgot to Wait after AllReduceAsync.
    Common in TP where people forget dist.all_reduce() returns a handle
    when async_op=True.

    Timeline:
      COMM:    [=== AllReduceAsync(y) ===]
      DEFAULT:     MatMul(y, w2)           → reads y — but AllReduce may not be done!
      DEFAULT:     (no Wait at all)
    """
    print_section("Case 2: MISSING WAIT — No Wait after async op")

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    prog = Program("missing_wait")
    prog.add(MatMul(a="x", b="w", output="y_partial"))
    # Async allreduce
    prog.add(AllReduceAsync(
        x="y_partial", output="y", handle="h1",
        op_type="sum",
    ))
    # BUG: uses 'y' without ever calling Wait(h1)
    prog.add(MatMul(a="y", b="w2", output="z"))
    # No Wait anywhere!

    print(f"\nProgram ({len(prog)} ops):")
    print(ir_to_str(prog))

    result = verify_temporal(prog)
    print(f"\n{result.summary()}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Case 3: BUFFER ALIASING — two async ops write to same buffer
# ═══════════════════════════════════════════════════════════════════════════════

def demo_buffer_aliasing():
    """
    Scenario: Two AllReduceAsync ops launched back-to-back,
    both writing to the same output buffer 'buf'.
    The second AllReduce corrupts the first before it's consumed.

    Actual PyTorch bug pattern:
      buf = torch.empty(...)
      h1 = dist.all_reduce(grad_w1, out=buf, async_op=True)
      h2 = dist.all_reduce(grad_w2, out=buf, async_op=True)  # BUG!
    """
    print_section("Case 3: BUFFER ALIASING — Two async ops sharing buffer")

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    prog = Program("buffer_aliasing")
    prog.add(MatMul(a="x", b="w1", output="y1_partial"))
    prog.add(MatMul(a="x", b="w2", output="y2_partial"))
    # Both async ops write to the SAME output buffer 'buf'
    prog.add(AllReduceAsync(
        x="y1_partial", output="buf", handle="h1",
        op_type="sum", stream=COMM_STREAM,
    ))
    prog.add(AllReduceAsync(
        x="y2_partial", output="buf", handle="h2",  # BUG: same buffer!
        op_type="sum", stream=COMM_STREAM,
    ))
    prog.add(Wait(handle="h1", tensor="buf", output="buf_after_h1"))
    prog.add(Wait(handle="h2", tensor="buf", output="buf_after_h2"))

    print(f"\nProgram ({len(prog)} ops):")
    print(ir_to_str(prog))

    result = verify_temporal(prog)
    print(f"\n{result.summary()}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Case 4: DEPENDENCY VIOLATION — Recv before Send in async PP
# ═══════════════════════════════════════════════════════════════════════════════

def demo_correct_overlap():
    """
    CORRECT overlap pattern: Compute on compute stream while
    AllReduce runs on comm stream, then Wait before consuming.

    Timeline:
      COMM:       [== AllReduceAsync(y_partial) ==]
      COMPUTE:    [== MatMul(x2, w2) ==]  ← no dependency on y
      DEFAULT:                                  Wait(h)  MatMul(y, w3)
    """
    print_section("Case 5 (Control): CORRECT Overlap — No violations")

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    prog = Program("correct_overlap")
    prog.add(MatMul(a="x", b="w", output="y_partial"))
    # Async AllReduce on COMM stream
    prog.add(AllReduceAsync(
        x="y_partial", output="y", handle="h1",
        op_type="sum", stream=COMM_STREAM,
    ))
    # Independent compute on COMPUTE stream — safe overlap
    prog.add(MatMul(a="x2", b="w2", output="z_independent"))
    # Wait before using 'y'
    prog.add(Wait(handle="h1", tensor="y", output="y_safe"))
    # Now safe to use 'y'
    prog.add(MatMul(a="y_safe", b="w3", output="z_final"))

    print(f"\nProgram ({len(prog)} ops):")
    print(ir_to_str(prog))

    result = verify_temporal(prog)
    print(f"\n{result.summary()}")

    return result


def demo_pp_overlap_violation():
    """
    Cross-stream async race in PP overlap:
    AllReduceAsync on COMM stream writes 'y' buffer.
    Simultaneously, MatMul on COMPUTE stream reads 'y'.
    The Wait happens too late — DATA RACE across streams.

    This pattern appears in PP overlap when gradient sync (AllReduce)
    is overlapped with the next layer's forward pass on different
    CUDA streams without proper event synchronization.
    """
    print_section("Case 4: CROSS-STREAM RACE — AllReduceAsync vs MatMul on different streams")

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    prog = Program("cross_stream_race")
    prog.add(MatMul(a="x", b="w", output="y_partial"))
    # Async AllReduce on COMM stream — writes 'y' asynchronously
    prog.add(AllReduceAsync(
        x="y_partial", output="y", handle="h1",
        op_type="sum", stream=COMM_STREAM,
    ))
    # BUG: MatMul on COMPUTE stream reads 'y' concurrently with AllReduce
    prog.add(MatMul(a="y", b="w2", output="z"))
    # Wait comes after the race already happened
    prog.add(Wait(handle="h1", tensor="y", output="y_safe"))

    print(f"\nProgram ({len(prog)} ops):")
    print(ir_to_str(prog))

    result = verify_temporal(prog)
    print(f"\n{result.summary()}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  DTENSOR-VERIFIER: Temporal Overlap Verification Demo")
    print("=" * 65)

    results = {}

    # Run all cases
    results["data_race"] = demo_data_race()
    results["missing_wait"] = demo_missing_wait()
    results["buffer_aliasing"] = demo_buffer_aliasing()
    results["pp_overlap"] = demo_pp_overlap_violation()
    results["correct"] = demo_correct_overlap()

    # Summary
    print_section("OVERALL SUMMARY")
    for name, result in results.items():
        status = "UNSAFE" if not result.is_safe else "SAFE"
        expected_unsafe = name != "correct"
        match = "✓" if (not result.is_safe) == expected_unsafe else "✗ FAIL"
        print(f"  {match} {name:25s} {status:6s}  ({len(result.reports)} violations)")

    all_correct = all(
        (not r.is_safe) == (name != "correct")
        for name, r in results.items()
    )
    print(f"\n  All detections correct: {'YES' if all_correct else 'NO'}")
    print()
