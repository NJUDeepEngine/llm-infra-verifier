"""
Consistency Verification: Single-GPU vs Distributed GPT-2 Transformer Layer.

This demo verifies that a distributed (Tensor Parallel) GPT-2 transformer
layer produces outputs EQUIVALENT to the single-GPU version.

The key mechanism: Z3 SMT solver encodes placement propagation rules as
constraints and formally checks whether outputs can ever be non-Replicate.

    UNSAT → Z3 proved equivalence (no counterexample exists)
    SAT   → Z3 found a bug (counterexample shows exactly what breaks)

Four scenarios:
  S1: Correct TP (2 AllReduces)         → Z3 proves UNSAT  (verified)
  S2: Missing AllReduce after attention  → Z3 finds SAT     (bug: output Partial)
  S3: Missing AllReduce after MLP        → Z3 finds SAT     (bug: output Partial)
  S4: Parametric (unknown weight shard)  → Z3 finds SAT     (searches all placements)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    DeviceTopology, TensorSlice,
    compute_local_shape, compute_tensor_slices,
)
from verifier.ir import (
    Program, MatMul, Add, Multiply, SiLU, AllReduce, FlashAttention,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import (
    Z3PlacementSolver, DistributedVerifier, VerifyResult,
    PL_R, PL_S0, PL_S1, PL_P, _PL_NAMES,
)


TP = 2
MESH = DeviceMesh(shape=(TP,), dim_names=("tp",))
BS, H, H3, F2, F = 8192, 768, 2304, 6144, 3072


# ═══════════════════════════════════════════════════════════════════════════════
# Build programs
# ═══════════════════════════════════════════════════════════════════════════════

def single_gpu_program() -> Program:
    """Reference: GPT-2 transformer layer on a single GPU.

    No collectives — every tensor is Replicate.
    """
    p = Program("gpt2_single_gpu")
    # Attention
    p.add(MatMul(a="x", b="W_qkv", output="qkv"))
    p.add(FlashAttention(q="qkv", k="qkv", v="qkv", output="attn"))
    p.add(MatMul(a="attn", b="W_out", output="h_attn"))
    p.add(Add(a="x", b="h_attn", output="h1"))
    # MLP
    p.add(MatMul(a="h1", b="W_gate_up", output="gate_up"))
    p.add(SiLU(x="gate_up", output="gate_act"))
    p.add(Multiply(a="gate_act", b="gate_up", output="h_mlp"))
    p.add(MatMul(a="h_mlp", b="W_down", output="mlp_out"))
    p.add(Add(a="h1", b="mlp_out", output="output"))
    return p


def distributed_program(
    include_attn_ar: bool = True,
    include_mlp_ar: bool = True,
) -> Program:
    """GPT-2 layer with Tensor Parallelism (TP=2).

    Column Parallel (QKV, Gate+Up):  x(R) @ W(S1) → y(S1)   — no AllReduce
    Row Parallel (Out, Down):        x(S1) @ W(S0) → y(P)   — needs AllReduce
    """
    p = Program("gpt2_tp2")
    # Attention
    p.add(MatMul(a="x", b="W_qkv", output="qkv_shard"))
    p.add(FlashAttention(q="qkv_shard", k="qkv_shard", v="qkv_shard",
                          output="attn_shard"))
    p.add(MatMul(a="attn_shard", b="W_out", output="attn_partial"))
    if include_attn_ar:
        p.add(AllReduce(x="attn_partial", output="attn_out", op_type="sum"))
        p.add(Add(a="x", b="attn_out", output="h1"))
    else:
        p.add(Add(a="x", b="attn_partial", output="h1"))
    # MLP
    p.add(MatMul(a="h1", b="W_gate_up", output="gate_up_shard"))
    p.add(SiLU(x="gate_up_shard", output="gate_act"))
    p.add(Multiply(a="gate_act", b="gate_up_shard", output="h_mlp"))
    p.add(MatMul(a="h_mlp", b="W_down", output="mlp_partial"))
    if include_mlp_ar:
        p.add(AllReduce(x="mlp_partial", output="mlp_out", op_type="sum"))
        p.add(Add(a="h1", b="mlp_out", output="output"))
    else:
        p.add(Add(a="h1", b="mlp_partial", output="output"))
    return p


# Input tensor specs for the distributed version
INPUT_PLACEMENTS = {
    "x":          Replicate(),
    "W_qkv":      Shard(dim=1),   # ColumnParallel: shard output dim
    "W_out":      Shard(dim=0),   # RowParallel: shard input dim
    "W_gate_up":  Shard(dim=1),   # ColumnParallel
    "W_down":     Shard(dim=0),   # RowParallel
}


def make_tensors():
    """Create concrete TensorState objects for executor-based verification."""
    return {
        "x": TensorState("x", (BS, H), (BS, H),
            ShardingSpec((Replicate(),), MESH), "x", requires_grad=True),
        "W_qkv": TensorState("W_qkv", (H, H3), (H, H3 // TP),
            ShardingSpec((Shard(dim=1),), MESH), "W_qkv", requires_grad=True),
        "W_out": TensorState("W_out", (H, H), (H // TP, H),
            ShardingSpec((Shard(dim=0),), MESH), "W_out", requires_grad=True),
        "W_gate_up": TensorState("W_gate_up", (H, F2), (H, F2 // TP),
            ShardingSpec((Shard(dim=1),), MESH), "W_gate_up", requires_grad=True),
        "W_down": TensorState("W_down", (F, H), (F // TP, H),
            ShardingSpec((Shard(dim=0),), MESH), "W_down", requires_grad=True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Z3 verification helper
# ═══════════════════════════════════════════════════════════════════════════════

def verify_with_z3(
    program: Program,
    placements: dict,
    output_names: list,
    label: str,
    parametric_vars: list = None,
):
    """Run Z3 placement verification and print results."""
    z3s = Z3PlacementSolver()

    for name, pl in placements.items():
        z3s.add_input(name, pl)

    z3s.encode_program(program)
    n_constraints = z3s.num_constraints

    print(f"\n  Z3 encoding: {len(z3s._vars)} tensor variables, {n_constraints} constraints")

    if parametric_vars:
        print(f"  Parametric (unconstrained): {parametric_vars}")

    # Check output equivalence
    eq_results = z3s.check_output_equivalence(output_names)
    for r in eq_results:
        status = "VERIFIED (UNSAT)" if r.passed else "BUG FOUND (SAT)"
        print(f"\n  [{status}] {r.condition}")
        print(f"    {r.details}")
        if r.counterexample:
            print(f"    Counterexample (tensor → placement):")
            for tn, pl in sorted(r.counterexample.items()):
                marker = " <<<" if pl == "P" else ""
                print(f"      {tn:20s} = {pl}{marker}")

    # Check collective preconditions
    pc_results = z3s.check_collective_preconditions(program)
    for r in pc_results:
        status = "OK" if r.passed else "WARNING"
        print(f"  [{status}] {r.condition}: {r.details}")

    return eq_results + pc_results


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Correct TP
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_1():
    print("=" * 70)
    print("  S1: CORRECT TP — AllReduce after each RowParallel MatMul")
    print("=" * 70)

    ref = single_gpu_program()
    dist = distributed_program(include_attn_ar=True, include_mlp_ar=True)

    print(f"\n  Single-GPU program ({len(ref)} ops): all tensors Replicate")
    print(f"  Distributed program ({len(dist)} ops):")
    for i, op in enumerate(dist.ops):
        coll = " [COLLECTIVE]" if op.is_collective() else ""
        print(f"    [{i:2d}] {op}{coll}")

    # Z3 verification
    results = verify_with_z3(dist, INPUT_PLACEMENTS, ["output"], "correct_tp")

    # Executor verification (concrete, agrees with Z3)
    executor = MultiDeviceExecutor(MESH)
    for ts in make_tensors().values():
        executor.register_tensor(ts)
    state = executor.run_program(dist)
    out = state["output"]
    print(f"\n  Executor confirmation: output.partial={out.partial}, "
          f"placement={out.sharding.placements[0]}")

    return all(r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Missing AllReduce after attention
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_2():
    print("\n" + "=" * 70)
    print("  S2: BUG — Missing AllReduce after attention output projection")
    print("=" * 70)

    dist = distributed_program(include_attn_ar=False, include_mlp_ar=True)

    print(f"\n  Distributed program ({len(dist)} ops) — NO AllReduce after attn:")
    for i, op in enumerate(dist.ops):
        coll = " [COLLECTIVE]" if op.is_collective() else ""
        print(f"    [{i:2d}] {op}{coll}")

    results = verify_with_z3(dist, INPUT_PLACEMENTS, ["output"], "missing_attn_ar")

    return any(not r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Missing AllReduce after MLP
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_3():
    print("\n" + "=" * 70)
    print("  S3: BUG — Missing AllReduce after MLP down projection")
    print("=" * 70)

    dist = distributed_program(include_attn_ar=True, include_mlp_ar=False)

    print(f"\n  Distributed program ({len(dist)} ops) — NO AllReduce after MLP:")
    for i, op in enumerate(dist.ops):
        coll = " [COLLECTIVE]" if op.is_collective() else ""
        print(f"    [{i:2d}] {op}{coll}")

    results = verify_with_z3(dist, INPUT_PLACEMENTS, ["output"], "missing_mlp_ar")

    return any(not r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Parametric — unknown weight placement
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_4():
    print("\n" + "=" * 70)
    print("  S4: PARAMETRIC — W_out placement unknown, Z3 searches all options")
    print("=" * 70)

    dist = distributed_program(include_attn_ar=True, include_mlp_ar=True)

    # Leave W_out UNCONSTRAINED — Z3 will explore all 4 possible placements
    partial_placements = {
        "x":          Replicate(),
        "W_qkv":      Shard(dim=1),
        # W_out: NOT specified — symbolic
        "W_gate_up":  Shard(dim=1),
        "W_down":     Shard(dim=0),
    }

    print(f"\n  All inputs have known placements EXCEPT W_out (symbolic).")
    print(f"  Z3 will search: can ANY placement of W_out cause output != Replicate?")

    results = verify_with_z3(
        dist, partial_placements, ["output"], "parametric",
        parametric_vars=["W_out"],
    )

    return any(not r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Non-fused program for shape/slice verification (S5-S7)
#
# S1-S4 use fused QKV (H→3H) and gate+up (H→2F) — correct for placement
# propagation but shapes don't chain. For L1/L2 we use non-fused weights
# so every MatMul contraction dim is consistent.
# ═══════════════════════════════════════════════════════════════════════════════

def shape_correct_program() -> Program:
    """GPT-2 TP layer with non-fused weights (all shapes chain correctly)."""
    p = Program("gpt2_tp2_unfused")
    # Attention: Q projection only (non-fused)
    p.add(MatMul(a="x", b="W_q", output="q_shard"))
    p.add(FlashAttention(q="q_shard", k="q_shard", v="q_shard",
                          output="attn_shard"))
    p.add(MatMul(a="attn_shard", b="W_out", output="attn_partial"))
    p.add(AllReduce(x="attn_partial", output="attn_out", op_type="sum"))
    p.add(Add(a="x", b="attn_out", output="h1"))
    # MLP: gate projection only (non-fused)
    p.add(MatMul(a="h1", b="W_gate", output="gate_shard"))
    p.add(SiLU(x="gate_shard", output="gate_act"))
    p.add(MatMul(a="gate_act", b="W_down", output="mlp_partial"))
    p.add(AllReduce(x="mlp_partial", output="mlp_out", op_type="sum"))
    p.add(Add(a="h1", b="mlp_out", output="output"))
    return p


SHAPE_PLACEMENTS = {
    "x":      Replicate(),
    "W_q":    Shard(dim=1),   # ColumnParallel
    "W_out":  Shard(dim=0),   # RowParallel
    "W_gate": Shard(dim=1),   # ColumnParallel
    "W_down": Shard(dim=0),   # RowParallel
}

SHAPE_DIMS = {
    "x":      (BS, H),        # (8192, 768)
    "W_q":    (H, H),         # (768, 768)
    "W_out":  (H, H),         # (768, 768)
    "W_gate": (H, F),         # (768, 3072)
    "W_down": (F, H),         # (3072, 768)
}


def make_shape_tensors():
    return {
        "x": TensorState("x", (BS, H), (BS, H),
            ShardingSpec((Replicate(),), MESH), "x", requires_grad=True),
        "W_q": TensorState("W_q", (H, H), (H, H // TP),
            ShardingSpec((Shard(dim=1),), MESH), "W_q", requires_grad=True),
        "W_out": TensorState("W_out", (H, H), (H // TP, H),
            ShardingSpec((Shard(dim=0),), MESH), "W_out", requires_grad=True),
        "W_gate": TensorState("W_gate", (H, F), (H, F // TP),
            ShardingSpec((Shard(dim=1),), MESH), "W_gate", requires_grad=True),
        "W_down": TensorState("W_down", (F, H), (F // TP, H),
            ShardingSpec((Shard(dim=0),), MESH), "W_down", requires_grad=True),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Z3 Shape Verification (L1)
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_5():
    print("\n" + "=" * 70)
    print("  S5: SHAPE VERIFICATION — Z3 proves shapes are consistent")
    print("=" * 70)

    dist = shape_correct_program()

    z3s = Z3PlacementSolver()
    for name, pl in SHAPE_PLACEMENTS.items():
        z3s.add_input(name, pl)
    for name, shape in SHAPE_DIMS.items():
        z3s.add_input_shape(name, shape)

    z3s.encode_program(dist)
    z3s.encode_shape_constraints(dist, tp_size=TP)

    print(f"\n  Program ({len(dist)} ops, non-fused weights for shape correctness):")
    for i, op in enumerate(dist.ops):
        coll = " [COLLECTIVE]" if op.is_collective() else ""
        print(f"    [{i:2d}] {op}{coll}")

    print(f"\n  Z3 encoding: {z3s.num_shape_vars} shape variables, "
          f"{z3s.num_constraints} total constraints")

    results = z3s.check_shape_consistency()

    overall = results[0]
    checks = results[1:]
    print(f"\n  [{('PASS' if overall.passed else 'FAIL')}] {overall.condition}: "
          f"{overall.details}")

    # Show per-MatMul contraction dim checks
    for op in dist.ops:
        if isinstance(op, MatMul):
            a_shape = SHAPE_DIMS.get(op.a)
            b_shape = SHAPE_DIMS.get(op.b)
            if a_shape and b_shape:
                ok = a_shape[1] == b_shape[0]
                status = "VERIFIED" if ok else "MISMATCH"
                print(f"  [{status}] MatMul({op.a}, {op.b}): "
                      f"contraction {op.a}.d1={a_shape[1]} == {op.b}.d0={b_shape[0]}")

    # Show divisibility
    failures = [r for r in checks if not r.passed]
    if failures:
        for r in failures:
            print(f"  [FAIL] {r.condition}: {r.details}")
    else:
        print(f"  [VERIFIED] All {len(checks)} shard-dim divisibility checks passed (TP={TP})")

    return all(r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Z3 Slice Alignment (L2)
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_6():
    print("\n" + "=" * 70)
    print("  S6: SLICE ALIGNMENT — Z3 verifies per-device slices match")
    print("=" * 70)

    dist = shape_correct_program()

    z3s = Z3PlacementSolver()
    for name, pl in SHAPE_PLACEMENTS.items():
        z3s.add_input(name, pl)
    for name, shape in SHAPE_DIMS.items():
        z3s.add_input_shape(name, shape)

    z3s.encode_program(dist)
    z3s.encode_shape_constraints(dist, tp_size=TP)
    z3s.encode_slice_constraints(dist, tp_size=TP)

    # Show per-device slices from executor
    executor = MultiDeviceExecutor(MESH)
    for ts in make_shape_tensors().values():
        executor.register_tensor(ts)
    executor.run_program(dist)
    slices = executor.final_slices()

    print(f"\n  Per-device tensor slices (TP={TP}):")
    sharded_tensors = ["W_q", "W_out", "W_gate", "W_down"]
    for name in sharded_tensors:
        parts = []
        for did in sorted(slices):
            s = slices[did].get(name)
            if s:
                parts.append(f"dev{did}={s.range_str()}")
        if parts:
            print(f"    {name:12s}: {', '.join(parts)}")

    # Z3 slice alignment results
    print()
    results = z3s.check_slice_alignment(dist)
    for r in results:
        status = "VERIFIED" if r.passed else "FAIL"
        print(f"  [{status}] {r.condition}: {r.details}")

    return all(r.passed for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Topology-Aware View
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_7():
    print("\n" + "=" * 70)
    print("  S7: TOPOLOGY VIEW — Computation graph overlaid on GPU topology")
    print("=" * 70)

    # Create physical topology: 2 GPUs with NVLink
    topo = DeviceTopology.fully_connected(TP, link_type="NVLink",
                                          bandwidth_gbps=300.0)
    mesh = DeviceMesh(shape=(TP,), dim_names=("tp",), topology=topo)

    print(f"\n  Hardware: {topo}")
    for link in topo.links:
        print(f"    {link}")

    # Validate topology connectivity
    errors = mesh.validate_topology()
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print(f"  [OK] All communication groups fully connected")

    # Build and run program (non-fused for shape correctness)
    dist = shape_correct_program()

    tensors = {
        "x": TensorState("x", (BS, H), (BS, H),
            ShardingSpec((Replicate(),), mesh), "x", requires_grad=True),
        "W_q": TensorState("W_q", (H, H), (H, H // TP),
            ShardingSpec((Shard(dim=1),), mesh), "W_q", requires_grad=True),
        "W_out": TensorState("W_out", (H, H), (H // TP, H),
            ShardingSpec((Shard(dim=0),), mesh), "W_out", requires_grad=True),
        "W_gate": TensorState("W_gate", (H, F), (H, F // TP),
            ShardingSpec((Shard(dim=1),), mesh), "W_gate", requires_grad=True),
        "W_down": TensorState("W_down", (F, H), (F // TP, H),
            ShardingSpec((Shard(dim=0),), mesh), "W_down", requires_grad=True),
    }

    executor = MultiDeviceExecutor(mesh)
    for ts in tensors.values():
        executor.register_tensor(ts)
    executor.run_program(dist)
    slices = executor.final_slices()

    # Per-device execution trace
    print(f"\n  Per-device execution trace:")
    for did in range(TP):
        print(f"\n    --- GPU {did} ---")
        dev_slices = slices[did]
        for op in dist.ops:
            if op.is_collective():
                link = topo.get_link(0, 1)
                link_info = f" ({link.link_type}, {link.bandwidth_gbps} GB/s)" if link else ""
                out_s = dev_slices.get(op.output_name)
                out_info = f" → full tensor" if out_s else ""
                print(f"      {op}{link_info}{out_info}")
            else:
                # Show input/output slice info
                parts = []
                for inp_name in op.input_names:
                    s = dev_slices.get(inp_name)
                    if s:
                        parts.append(f"{inp_name}{s.range_str()}")
                    else:
                        parts.append(inp_name)
                out_s = dev_slices.get(op.output_name)
                out_str = out_s.range_str() if out_s else ""
                print(f"      {type(op).__name__}({', '.join(parts)}) "
                      f"→ {op.output_name}{out_str}")

    # Summary: all verification levels
    print(f"\n  Verification summary:")
    z3s = Z3PlacementSolver()
    for name, pl in SHAPE_PLACEMENTS.items():
        z3s.add_input(name, pl)
    for name, ts in tensors.items():
        z3s.add_input_shape(name, ts.global_shape)
    z3s.encode_program(dist)
    z3s.encode_shape_constraints(dist, tp_size=TP)
    z3s.encode_slice_constraints(dist, tp_size=TP)

    eq = z3s.check_output_equivalence(["output"])
    shapes = z3s.check_shape_consistency()
    align = z3s.check_slice_alignment(dist)

    all_ok = True
    for r in eq:
        status = "VERIFIED" if r.passed else "FAIL"
        print(f"    [L0 {status}] {r.condition}: {r.details}")
        all_ok = all_ok and r.passed
    print(f"    [L1 {'VERIFIED' if shapes[0].passed else 'FAIL'}] "
          f"shape consistency: {shapes[0].details}")
    all_ok = all_ok and shapes[0].passed
    for r in align:
        status = "VERIFIED" if r.passed else "FAIL"
        print(f"    [L2 {status}] {r.condition}")
        all_ok = all_ok and r.passed

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CONSISTENCY DEMO: Single-GPU vs Distributed GPT-2 Layer")
    print("  Z3 SMT Solver — Placement + Shape + Slice Verification")
    print("=" * 70)

    results = {}
    results["S1"] = scenario_1()
    results["S2"] = scenario_2()
    results["S3"] = scenario_3()
    results["S4"] = scenario_4()
    results["S5"] = scenario_5()
    results["S6"] = scenario_6()
    results["S7"] = scenario_7()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    total = sum(1 for v in results.values() if v)
    print(f"\n  {total}/{len(results)} scenarios behaved as expected")
    print()
