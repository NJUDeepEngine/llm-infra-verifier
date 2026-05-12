"""
SPMD Type System Demo (meta-pytorch/spmd_types integration).

Demonstrates the new capabilities:
  1. LocalSPMDType (R/I/V/P) — the 4 fundamental states
  2. Type duality: R↔P, I↔I, V↔V
  3. Reinterpret: change type WITHOUT communication
  4. Convert: change type WITHOUT communication, WITH data ops
  5. Partial*Partial FORBIDDEN (spmd_types rule)
  6. Invariant gradient: no AllReduce needed
  7. Varying tensor: "different per rank, unknown reassembly"

Verification scenarios:
  S1: Basic type duality (R↔P, I↔I, V↔V)
  S2: Reinterpret transitions (R→V, R→P, V→P, P→V)
  S3: Convert transitions (R→P, P→R, I→R, I→V)
  S4: Partial*Partial guard (REJECTED)
  S5: AllReduce on REPLICATE (REJECTED)
  S6: Invariant gradient — no AllReduce needed
  S7: Full GPT-2 layer with SPMD types
  S8: SPMD type-aware gradient duality
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    LocalSPMDType, compute_local_shape,
)
from verifier.ir import (
    Program, MatMul, Multiply, AllReduce, AllGather, ReduceScatter,
    Reinterpret, Convert, SPMDGuard, ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier
from verifier.autograd import AutogradEngine


def s1_type_duality():
    """SPMD type duality: R↔P, I↔I, V↔V."""
    print("=" * 60)
    print("  S1: SPMD Type Duality (R↔P, I↔I, V↔V)")
    print("=" * 60)

    for lt in LocalSPMDType:
        gt = lt.gradient_type()
        print(f"  {lt.value} (forward) → {gt.value} (gradient)")

    # Verify the fixed pairs
    assert LocalSPMDType.REPLICATE.gradient_type() == LocalSPMDType.PARTIAL
    assert LocalSPMDType.PARTIAL.gradient_type() == LocalSPMDType.REPLICATE
    assert LocalSPMDType.INVARIANT.gradient_type() == LocalSPMDType.INVARIANT
    assert LocalSPMDType.VARYING.gradient_type() == LocalSPMDType.VARYING
    print("  All duality pairs correct ✓")

    return True


def s2_reinterpret_transitions():
    """Valid Reinterpret transitions."""
    print("\n" + "=" * 60)
    print("  S2: Reinterpret (no communication type changes)")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", local_type=LocalSPMDType.REPLICATE)

    valid = [
        (LocalSPMDType.REPLICATE, LocalSPMDType.VARYING, False),
        (LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL, False),
        (LocalSPMDType.VARYING, LocalSPMDType.PARTIAL, False),
        (LocalSPMDType.PARTIAL, LocalSPMDType.VARYING, False),
    ]

    for src, dst, expert in valid:
        prog = Program("test").add(
            Reinterpret("x", "y", src_type=src, dst_type=dst, expert_mode=expert))
        # Apply: should succeed
        ctx = {"x": x.with_local_type(src)}
        result = prog.ops[0].apply(ctx)
        assert result.local_type == dst, f"Expected {dst}, got {result.local_type}"
        print(f"  {src.value} → {dst.value}: OK")

    # Expert-only should raise
    try:
        Reinterpret("x", "y", LocalSPMDType.REPLICATE, LocalSPMDType.INVARIANT)
        print("  R → I without expert_mode: FAIL (should raise)")
        return False
    except ValueError as e:
        print(f"  R → I without expert_mode: BLOCKED ✓")

    # Invalid should raise
    try:
        Reinterpret("x", "y", LocalSPMDType.INVARIANT, LocalSPMDType.REPLICATE)
        print(f"  I → R: FAIL (should raise)")
        return False
    except ValueError:
        print(f"  I → R: BLOCKED ✓")

    return True


def s3_convert_transitions():
    """Valid Convert transitions."""
    print("\n" + "=" * 60)
    print("  S3: Convert (no communication, with data ops)")
    print("=" * 60)

    valid = [
        (LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL),
        (LocalSPMDType.PARTIAL, LocalSPMDType.REPLICATE),
        (LocalSPMDType.INVARIANT, LocalSPMDType.REPLICATE),
        (LocalSPMDType.INVARIANT, LocalSPMDType.VARYING),
        (LocalSPMDType.INVARIANT, LocalSPMDType.PARTIAL),
    ]

    for src, dst in valid:
        op = Convert("x", "y", src_type=src, dst_type=dst)
        print(f"  {src.value} → {dst.value}: OK")

    try:
        Convert("x", "y", LocalSPMDType.REPLICATE, LocalSPMDType.VARYING)
        print(f"  R → V via Convert: FAIL (should use Reinterpret)")
        return False
    except ValueError:
        print(f"  R → V via Convert: BLOCKED (use Reinterpret) ✓")

    return True


def s4_partial_times_partial_rejected():
    """SPMD rule: Partial * Partial is FORBIDDEN."""
    print("\n" + "=" * 60)
    print("  S4: Partial * Partial FORBIDDEN (SPMD rule)")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    a = TensorState("a", (8, 16), (8, 16),
        ShardingSpec((Partial(),), mesh), "a", local_type=LocalSPMDType.PARTIAL)
    b = TensorState("b", (8, 16), (8, 16),
        ShardingSpec((Partial(),), mesh), "b", local_type=LocalSPMDType.PARTIAL)

    ctx = {"a": a, "b": b}
    try:
        Multiply("a", "b", "c").apply(ctx)
        print("  Multiply(P, P): ALLOWED (should be REJECTED)")
        return False
    except ValueError as e:
        print(f"  Multiply(P, P): REJECTED ✓")
        print(f"    {str(e)[:100]}...")

    return True


def s5_allreduce_on_replicate_rejected():
    """AllReduce on REPLICATE is an error."""
    print("\n" + "=" * 60)
    print("  S5: AllReduce on REPLICATE REJECTED")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", local_type=LocalSPMDType.REPLICATE)

    try:
        SPMDGuard.check_allreduce_input(x)
        print("  AllReduce(R): ALLOWED (should be REJECTED)")
        return False
    except ValueError as e:
        print(f"  AllReduce(R): REJECTED ✓")
        print(f"    {str(e)[:100]}")

    return True


def s6_invariant_gradient_no_comm():
    """INVARIANT tensor: gradient is already identical, no AllReduce needed."""
    print("\n" + "=" * 60)
    print("  S6: Invariant Gradient — No AllReduce Needed")
    print("=" * 60)

    # I→I duality means: if forward uses INVARIANT, backward gradient
    # is also INVARIANT — NO communication needed.
    lt = LocalSPMDType.INVARIANT
    gt = lt.gradient_type()
    assert gt == LocalSPMDType.INVARIANT

    print(f"  Forward: {lt.value} (all ranks have same data)")
    print(f"  Backward gradient: {gt.value} (already identical)")
    print(f"  Communication needed: NONE")
    print(f"  vs REPLICATE: gradient is PARTIAL → needs AllReduce")
    print(f"  This saves one AllReduce per INVARIANT activation in TP.")

    # In Megatron with SP=False, inter-block activations are I@tp
    # → no gradient communication needed between layers
    return True


def s7_gpt2_layer_with_spmd_types():
    """Full GPT-2 layer modeled with explicit SPMD types."""
    print("\n" + "=" * 60)
    print("  S7: GPT-2 Layer with SPMD Types")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    BS, H = 8192, 768

    # Input: REPLICATE
    x = TensorState("x", (BS, H), (BS, H),
        ShardingSpec((Replicate(),), mesh), "x",
        local_type=LocalSPMDType.REPLICATE, requires_grad=True)

    # QKV: ColumnParallel → output is VARYING (Shard(1), unknown reassembly local)
    w_qkv = TensorState("W_qkv", (H, 3*H), (H, 3*H//2),
        ShardingSpec((Shard(dim=1),), mesh), "W_qkv",
        local_type=LocalSPMDType.REPLICATE, requires_grad=True)

    # Output proj: RowParallel → needs AllReduce
    w_out = TensorState("W_out", (H, H), (H//2, H),
        ShardingSpec((Shard(dim=0),), mesh), "W_out",
        local_type=LocalSPMDType.REPLICATE, requires_grad=True)

    fwd = Program("spmd_gpt2")
    fwd.add(MatMul("x", "W_qkv", "qkv"))                             # R @ R(Shard1) → V
    fwd.add(MatMul("qkv", "W_out", "attn_partial"))                   # V @ R(Shard0) → P
    fwd.add(AllReduce("attn_partial", "attn_out", op_type="sum"))     # P → R
    fwd.add(Multiply("x", "attn_out", "output"))                      # R * R → R

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(x); executor.register_tensor(w_qkv); executor.register_tensor(w_out)
    state = executor.run_program(fwd)

    # Check SPMD types at each stage
    checks = [
        ("qkv", LocalSPMDType.VARYING),       # ColumnParallel output
        ("attn_partial", LocalSPMDType.PARTIAL),  # RowParallel before AR
        ("attn_out", LocalSPMDType.REPLICATE),    # After AllReduce
        ("output", LocalSPMDType.REPLICATE),       # Element-wise R*R
    ]
    all_ok = True
    for name, expected in checks:
        ts = state.get(name)
        actual = ts.local_type if ts else "MISSING"
        ok = ts and ts.local_type == expected
        if not ok:
            all_ok = False
        print(f"  {name:15s} → {actual.value:5s} (expected {expected.value:5s}) {'✓' if ok else '✗'}")

    print(f"  {'ALL PASSED' if all_ok else 'FAILURES'}")
    return all_ok


def s8_spmd_gradient_duality():
    """SPMD type-aware gradient duality check."""
    print("\n" + "=" * 60)
    print("  S8: SPMD Gradient Duality Check")
    print("=" * 60)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(dim=1),), mesh), "x",
        local_type=LocalSPMDType.VARYING, requires_grad=True)
    w = TensorState("w", (16, 32), (8, 32),
        ShardingSpec((Shard(dim=0),), mesh), "w",
        local_type=LocalSPMDType.REPLICATE, requires_grad=True)

    # Forward: V @ R(Shard0) → P → AllReduce → R
    fwd = Program("spmd_fwd")
    fwd.add(MatMul("x", "w", "y_partial"))       # V @ R → P
    fwd.add(AllReduce("y_partial", "y", "sum"))   # P → R

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(x); executor.register_tensor(w)
    executor.run_program(fwd)

    autograd = AutogradEngine()
    for op in fwd.ops:
        autograd.record(op, executor.devices[0].tensors)
    bwd = autograd.generate_backward("y")

    verifier = DistributedVerifier()
    duality = verifier.verify_gradient_duality(fwd, bwd)

    # Print type flow
    print(f"  Forward type flow:")
    print(f"    x(V) @ w(R) → y_partial(P) → AllReduce → y(R)")
    print(f"  Backward: AllReduce is self-dual")
    print(f"  Gradient duality: {'PASSED' if duality.passed else 'FAILED'}")

    return duality.passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLM-INFRA-VERIFIER: SPMD Type System Demo")
    print("  Trust base: meta-pytorch/spmd_types DESIGN.md")
    print("=" * 60)

    results = {
        "S1": s1_type_duality(),
        "S2": s2_reinterpret_transitions(),
        "S3": s3_convert_transitions(),
        "S4": s4_partial_times_partial_rejected(),
        "S5": s5_allreduce_on_replicate_rejected(),
        "S6": s6_invariant_gradient_no_comm(),
        "S7": s7_gpt2_layer_with_spmd_types(),
        "S8": s8_spmd_gradient_duality(),
    }

    print("\n" + "=" * 60)
    print("  SPMD INTEGRATION SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  {sum(1 for v in results.values() if v)}/{len(results)} passed")
    print()
