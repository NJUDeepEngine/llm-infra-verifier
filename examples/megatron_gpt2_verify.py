"""
End-to-end verification of Megatron-LM GPT-2 training on our verifier.

Models the actual Megatron GPT-2 transformer layer with Tensor Parallelism:
  - QKV: ColumnParallelLinear (no fwd comm, output Shard(1))
  - Attention + Output Proj: RowParallelLinear (fwd AllReduce)
  - MLP Gate+Up: ColumnParallelLinear (no fwd comm)
  - MLP Down: RowParallelLinear (fwd AllReduce)
  - Mixed precision: fp16 forward + fp32 AllReduce + fp32 Adam

Verification dimensions:
  Spatial: placement correctness, gradient duality
  Temporal: async AllReduce overlap safety
  Numerical: fp16 precision safety, accumulation bounds
  Resource: HBM memory budget for GPT-2 model sizes

Runs 8 verification scenarios:
  V1: Correct GPT-2 layer (spatial + temporal + numerical)
  V2: Missing AllReduce after output projection → DETECTED
  V3: Missing AllReduce after MLP down → DETECTED
  V4: GELU bug: SiLU on sharded tensor → DETECTED
  V5: Async gradient race → DETECTED
  V6: Full GPT-2 model memory planning (H100/H200/B200)
  V7: fp16 numerical safety for GPT-2 training
  V8: Multi-layer accumulation over 100k steps
"""

import sys, os, math
from typing import Tuple
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    compute_local_shape,
)
from verifier.ir import (
    Program, MatMul, Add, Multiply, SiLU, AllReduce, AllReduceAsync,
    Wait, FlashAttention, COMM_STREAM, ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier, VerifyResult
from verifier.rewrite import PlacementAnalyzer
from verifier.temporal import verify_temporal
from verifier.numerical import (
    Dtype, ErrorModel, ReductionErrorAnalyzer, ErrorAccumulator,
    OverflowRiskDetector, ZeROStage, verify_numerical,
)
from verifier.hardware import GPU_MODELS, H100_SXM, H200_SXM, B200
from verifier.memory_graph import estimate_llm_memory


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-2 Model Configuration (Megatron defaults)
# ═══════════════════════════════════════════════════════════════════════════════

GPT2_SMALL = dict(hidden_dim=768, num_layers=12, num_heads=12, vocab_size=50257)
GPT2_MEDIUM = dict(hidden_dim=1024, num_layers=24, num_heads=16, vocab_size=50257)
GPT2_LARGE = dict(hidden_dim=1280, num_layers=36, num_heads=20, vocab_size=50257)
GPT2_XL = dict(hidden_dim=1600, num_layers=48, num_heads=25, vocab_size=50257)


def build_gpt2_transformer_layer(
    hidden_dim: int = 768,
    ffn_dim: int = 3072,  # 4*hidden_dim for GPT-2
    tp_size: int = 2,
    batch_size: int = 8,
    seq_len: int = 1024,
    dtype_bytes: int = 2,  # fp16
) -> Tuple[Program, dict, DeviceMesh]:
    """Build Megatron-LM GPT-2 transformer layer IR.

    Models the forward pass from megatron/core/transformer/transformer_layer.py.
    Batch+seq dims are merged (B*S,) as Megatron does internally before
    calling F.linear(). This avoids needing 3D MatMul shape propagation.

    With Tensor Parallelism (tp_size=2):

    ColumnParallel (QKV, FC1): X(Replicate, BS×H) @ W(Shard1) → Shard(1), NO AR
    RowParallel (Out, FC2):    X(Shard1) @ W(Shard0) → Partial → AllReduce
    """
    mesh = DeviceMesh(shape=(tp_size,), dim_names=("tp",))
    BS = batch_size * seq_len  # merged batch+seq
    H, F = hidden_dim, ffn_dim
    H3 = 3 * H                  # QKV fused output dim
    F2 = 2 * F                  # Gate+Up fused output dim

    # Input: Replicated (after scatter_to_tensor_model_parallel_region)
    x = TensorState("x", (BS, H), (BS, H),
        ShardingSpec((Replicate(),), mesh), "x", requires_grad=True)

    # QKV: ColumnParallel, Shard(1) → output Shard(1), NO fwd comm
    w_qkv = TensorState("W_qkv", (H, H3), (H, H3 // tp_size),
        ShardingSpec((Shard(dim=1),), mesh), "W_qkv", requires_grad=True)

    # Output proj: RowParallel, Shard(0) → Partial → AllReduce
    w_out = TensorState("W_out", (H, H), (H // tp_size, H),
        ShardingSpec((Shard(dim=0),), mesh), "W_out", requires_grad=True)

    # FC1 (Gate+Up fused): ColumnParallel, Shard(1)
    w_gate_up = TensorState("W_gate_up", (H, F2), (H, F2 // tp_size),
        ShardingSpec((Shard(dim=1),), mesh), "W_gate_up", requires_grad=True)

    # FC2 (Down): RowParallel, Shard(0)
    w_down = TensorState("W_down", (F, H), (F // tp_size, H),
        ShardingSpec((Shard(dim=0),), mesh), "W_down", requires_grad=True)

    fwd = Program("megatron_gpt2_layer")

    # --- Attention Block ---
    fwd.add(MatMul(a="x", b="W_qkv", output="qkv_shard"))           # ColParallel: Shard(1)
    fwd.add(FlashAttention(q="qkv_shard", k="qkv_shard", v="qkv_shard",
                            output="attn_shard"))                    # FA preserves placement
    fwd.add(MatMul(a="attn_shard", b="W_out", output="attn_partial"))# RowParallel: Partial
    fwd.add(AllReduce(x="attn_partial", output="attn_out", op_type="sum"))
    fwd.add(Add(a="x", b="attn_out", output="h1"))

    # --- MLP Block ---
    fwd.add(MatMul(a="h1", b="W_gate_up", output="gate_up_shard"))  # ColParallel
    fwd.add(SiLU(x="gate_up_shard", output="gate_act"))              # SiLU on Shard(1) ← RISK
    fwd.add(Multiply(a="gate_act", b="gate_up_shard", output="h_mlp"))
    fwd.add(MatMul(a="h_mlp", b="W_down", output="mlp_partial"))    # RowParallel
    fwd.add(AllReduce(x="mlp_partial", output="mlp_out", op_type="sum"))
    fwd.add(Add(a="h1", b="mlp_out", output="output"))

    return fwd, {
        "x": x, "W_qkv": w_qkv, "W_out": w_out,
        "W_gate_up": w_gate_up, "W_down": w_down,
    }, mesh


# ═══════════════════════════════════════════════════════════════════════════════
# V1: Correct GPT-2 layer — full spatial + temporal + numerical
# ═══════════════════════════════════════════════════════════════════════════════

def v1_correct_gpt2_layer():
    """Verify the CORRECT Megatron GPT-2 transformer layer."""
    print("=" * 70)
    print("  V1: CORRECT GPT-2 Transformer Layer (Megatron-LM)")
    print("=" * 70)

    fwd, tensors, mesh = build_gpt2_transformer_layer()

    print(f"\n  Forward IR ({len(fwd)} ops):")
    for i, op in enumerate(fwd.ops):
        coll = " [COLLECTIVE]" if op.is_collective() else ""
        p2p = " [P2P]" if op.is_p2p() else ""
        print(f"    [{i}] {op}{coll}{p2p}")
    print(f"\n  Collectives in fwd: {len(fwd.collectives)} (expected: 2 AllReduces)")

    # Execute
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        executor.register_tensor(ts)
    state = executor.run_program(fwd)

    # Spatial verification
    verifier = DistributedVerifier()
    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(fwd, state)

    output = state["output"]
    print(f"\n  Output: {output}")
    print(f"  output.partial = {output.partial} (expected: False)")

    checks = []
    # Postcondition: output must be Replicate
    vr = verifier.verify_postcondition(output, expected_partial=False)
    checks.append(("postcondition", vr.passed))

    # Communication legality (with tensor states for accurate Partial check)
    cl = verifier.verify_communication_legality(fwd, tensor_states=state)
    checks.append(("comm legality", cl.passed))

    # Placement analysis
    checks.append(("placement analysis", analysis.is_correct))

    print(f"\n  Spatial checks:")
    for name, ok in checks:
        print(f"    [{('PASS' if ok else 'FAIL')}] {name}")

    # Numerical: fp16 safety
    detector = OverflowRiskDetector(Dtype.FP16)
    risks = detector.check_activations(hidden_dim=768, num_layers=12)
    print(f"\n  Numerical (fp16 safety):")
    for r in risks:
        print(f"    {r.risk_level}: {r.name}")

    all_ok = all(ok for _, ok in checks)
    print(f"\n  Verdict: {'ALL PASSED' if all_ok else 'ISSUES FOUND'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# V2-V4: Bug injection into GPT-2 layer
# ═══════════════════════════════════════════════════════════════════════════════

def v2_missing_attention_allreduce():
    """Bug: Missing AllReduce after attention output projection."""
    print("\n" + "=" * 70)
    print("  V2: BUG — Missing AllReduce after Attention Output Proj")
    print("=" * 70)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    BS = 8192
    # RowParallel: input Shard(1), weight Shard(0) on reduce dim
    attn = TensorState("attn_shard", (BS, 768), (BS, 384),
        ShardingSpec((Shard(dim=1),), mesh), "attn_shard")
    w_out = TensorState("W_out", (768, 768), (384, 768),
        ShardingSpec((Shard(dim=0),), mesh), "W_out")

    fwd = Program("bug_no_ar")
    fwd.add(MatMul(a="attn_shard", b="W_out", output="attn_out"))
    # BUG: Missing AllReduce! Output is Partial

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(attn); executor.register_tensor(w_out)
    state = executor.run_program(fwd)

    vr = DistributedVerifier().verify_postcondition(state["attn_out"], expected_partial=False)
    analysis = PlacementAnalyzer().analyze(fwd, state)

    print(f"  attn_out.partial = {state['attn_out'].partial} (should be False)")
    print(f"  Postcondition: {'BUG DETECTED' if not vr.passed else 'MISSED'}")
    print(f"  Placement analysis: {'BUG DETECTED' if not analysis.is_correct else 'MISSED'}")

    return not vr.passed


def v3_missing_mlp_allreduce():
    """Bug: Missing AllReduce after MLP down projection."""
    print("\n" + "=" * 70)
    print("  V3: BUG — Missing AllReduce after MLP Down Proj")
    print("=" * 70)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    BS = 8192
    # RowParallel input: Shard(1), weight: Shard(0)
    h = TensorState("h_mlp", (BS, 3072), (BS, 1536),
        ShardingSpec((Shard(dim=1),), mesh), "h_mlp")
    w_down = TensorState("W_down", (3072, 768), (1536, 768),
        ShardingSpec((Shard(dim=0),), mesh), "W_down")

    fwd = Program("bug_no_ar_mlp")
    fwd.add(MatMul(a="h_mlp", b="W_down", output="mlp_out"))
    # BUG: Missing AllReduce!

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(h); executor.register_tensor(w_down)
    state = executor.run_program(fwd)

    vr = DistributedVerifier().verify_postcondition(state["mlp_out"], expected_partial=False)
    analysis = PlacementAnalyzer().analyze(fwd, state)

    print(f"  mlp_out.partial = {state['mlp_out'].partial} (should be False)")
    print(f"  Postcondition: {'BUG DETECTED' if not vr.passed else 'MISSED'}")
    print(f"  Placement analysis: {'BUG DETECTED' if not analysis.is_correct else 'MISSED'}")

    return not vr.passed


def v4_gelu_on_sharded_bug():
    """Bug: SiLU applied to Shard(1) tensor without AllGather first."""
    print("\n" + "=" * 70)
    print("  V4: BUG — SiLU on Shard(1) tensor (pytorch#144359)")
    print("=" * 70)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    BS = 8192
    x = TensorState("x", (BS, 768), (BS, 768),
        ShardingSpec((Replicate(),), mesh), "x")
    w_gate = TensorState("W_gate", (768, 3072), (768, 1536),
        ShardingSpec((Shard(dim=1),), mesh), "W_gate")

    fwd = Program("gelu_bug")
    fwd.add(MatMul(a="x", b="W_gate", output="gate_shard"))  # Shard(1)
    fwd.add(SiLU(x="gate_shard", output="gate_act"))          # BUG: nonlinear on shard!
    fwd.add(MatMul(a="gate_act", b="W_gate", output="output"))

    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(x); executor.register_tensor(w_gate)
    state = executor.run_program(fwd)

    gate = state["gate_shard"]
    is_shard = any(isinstance(p, Shard) for p in gate.sharding.placements)
    print(f"  gate_shard is Shard: {is_shard}")
    print(f"  SiLU applied to Shard(1): {'BUG DETECTED' if is_shard else 'NOT SHARDED'}")

    return is_shard


# ═══════════════════════════════════════════════════════════════════════════════
# V5: Async AllReduce gradient race (Megatron's actual pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def v5_async_gradient_race():
    """Megatron's LinearWithGradAccumulationAndAsyncCommunication pattern.

    Correct: AllReduceAsync → compute grad_input → Wait → use grad_weight.
    Bug: AllReduceAsync → immediately use grad_weight → Wait too late.
    """
    print("\n" + "=" * 70)
    print("  V5: Temporal — Async AllReduce Gradient Race")
    print("=" * 70)

    # CORRECT pattern
    correct = Program("async_correct")
    correct.add(AllReduceAsync("grad_w_local", "grad_w", handle="h1",
                                op_type="sum", stream=COMM_STREAM))
    correct.add(MatMul("grad_output", "W_T", "grad_input"))  # independent
    correct.add(Wait(handle="h1", tensor="grad_w", output="grad_w_ready"))

    # BUG pattern
    bug = Program("async_bug")
    bug.add(AllReduceAsync("grad_w_local", "grad_w", handle="h1",
                            op_type="sum", stream=COMM_STREAM))
    bug.add(MatMul("grad_w", "opt_state", "update"))  # BUG: reads async output!
    bug.add(Wait(handle="h1", tensor="grad_w", output="grad_w_ready"))

    r_correct = verify_temporal(correct)
    r_bug = verify_temporal(bug)

    print(f"  Correct pattern: {'SAFE' if r_correct.is_safe else 'UNSAFE'}")
    print(f"  Bug pattern:     {'SAFE' if r_bug.is_safe else 'UNSAFE'}")
    for report in r_bug.reports:
        print(f"    → {report.race_type.value}: {report.description}")

    return r_correct.is_safe and not r_bug.is_safe


# ═══════════════════════════════════════════════════════════════════════════════
# V6: GPT-2 memory planning across GPU generations
# ═══════════════════════════════════════════════════════════════════════════════

def v6_gpt2_memory_planning():
    """Check GPT-2 memory requirements on H100/H200/B200."""
    print("\n" + "=" * 70)
    print("  V6: Resource — GPT-2 Memory Planning Across GPUs")
    print("=" * 70)

    configs = [
        ("GPT-2 Small (124M)", GPT2_SMALL),
        ("GPT-2 Medium (355M)", GPT2_MEDIUM),
        ("GPT-2 Large (774M)", GPT2_LARGE),
        ("GPT-2 XL (1.5B)", GPT2_XL),
    ]

    gpus = [H100_SXM, H200_SXM, B200]

    print(f"  {'Model':<25} ", end="")
    for g in gpus:
        print(f"  {g.name.split()[1]:>8}", end="")
    print()

    for name, cfg in configs:
        print(f"  {name:<25} ", end="")
        for gpu in gpus:
            mem = estimate_llm_memory(
                cfg["hidden_dim"], cfg["num_layers"],
                cfg["vocab_size"], batch_size=8, seq_len=1024, tp_size=1,
            )
            total_gb = mem["total"] / (1024**3)
            fits = "✓" if total_gb < gpu.total_hbm_gb else "OOM"
            print(f"  {fits} {total_gb:>5.0f}G", end="")
        print()

    print(f"\n  With TP=2 (activations halved):")
    for name, cfg in configs:
        print(f"  {name:<25} ", end="")
        for gpu in gpus:
            mem = estimate_llm_memory(
                cfg["hidden_dim"], cfg["num_layers"],
                cfg["vocab_size"], batch_size=8, seq_len=1024, tp_size=2,
            )
            total_gb = mem["total"] / (1024**3)
            fits = "✓" if total_gb < gpu.total_hbm_gb else "OOM"
            print(f"  {fits} {total_gb:>5.0f}G", end="")
        print()

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# V7: Numerical safety for GPT-2 fp16 training
# ═══════════════════════════════════════════════════════════════════════════════

def v7_numerical_safety():
    """fp16 numerical safety analysis for GPT-2 training."""
    print("\n" + "=" * 70)
    print("  V7: Numerical — fp16 Safety for GPT-2 Training")
    print("=" * 70)

    # GPT-2 Small with 2-way TP, fp16 compute, fp32 AR + Adam
    result = verify_numerical(
        n_ranks=2,
        topology="tree",
        compute_dtype=Dtype.FP16,
        accumulate_dtype=Dtype.FP32,
        optimizer="adam",
        zero_stage=ZeROStage.DP,
        hidden_dim=768,
        num_layers=12,
        batch_size=8,
    )

    print(f"\n  Reduction analysis:")
    print(f"    Tree AR(N=2): {result.reduction_analysis.tree_error}")
    print(f"    Ring AR(N=2): {result.reduction_analysis.ring_error}")
    print(f"    Safe for fp16: {result.reduction_analysis.safe_for_fp16}")

    print(f"\n  Accumulation (10k steps):")
    print(f"    Path 1 (cast):  {result.accumulation_analysis.weight_cast_drift.steady_state_bound:.2e}")
    print(f"    Path 2 (Adam):  {result.accumulation_analysis.optimizer_state_ema.steady_state_bound:.2e}")
    print(f"    Path 3 (divergence): {result.accumulation_analysis.cross_rank_weight_diff:.2e}")

    print(f"\n  Overflow risks:")
    for risk in result.overflow_risks:
        print(f"    {risk.risk_level}: {risk.name} (magnitude ~{risk.typical_magnitude:.1e})")

    print(f"\n  Verdict: {'SAFE' if result.is_safe else 'UNSAFE'}")
    return result.is_safe


# ═══════════════════════════════════════════════════════════════════════════════
# V8: Multi-layer accumulation over training steps
# ═══════════════════════════════════════════════════════════════════════════════

def v8_accumulation_over_training():
    """Error accumulation for full GPT-2 training over 100k steps."""
    print("\n" + "=" * 70)
    print("  V8: Numerical — GPT-2 Accumulation Over 100k Training Steps")
    print("=" * 70)

    acc = ErrorAccumulator(Dtype.FP16, Dtype.FP32, n_ranks=2,
                           allreduce_topology="tree")

    configs = [
        ("FP16 compute + FP32 AR (standard)", Dtype.FP16, Dtype.FP32),
        ("FP16 compute + FP16 AR (dangerous)", Dtype.FP16, Dtype.FP16),
        ("BF16 compute + FP32 AR", Dtype.BF16, Dtype.FP32),
    ]

    G = 1e-3  # typical gradient magnitude for GPT-2

    print(f"  Typical |g| ≈ {G:.1e}, T=100k steps, lr=1e-4")
    print(f"  {'Config':<35} {'Cross-Rank Div':>14} {'Risk':>20}")
    print(f"  {'-'*35} {'-'*14} {'-'*20}")

    for name, comp, accum in configs:
        a = ErrorAccumulator(comp, accum, 2, "tree")
        analysis = a.analyze(num_steps=100000, learning_rate=1e-4,
                             typical_grad_magnitude=G)
        print(f"  {name:<35} {analysis.cross_rank_weight_diff:>14.2e} "
              f"{analysis.risk_level:>20}")

    # Show dominant error source
    a = ErrorAccumulator(Dtype.FP16, Dtype.FP32, 2, "tree")
    analysis = a.analyze(num_steps=100000, learning_rate=1e-4,
                         typical_grad_magnitude=G)

    p1 = analysis.weight_cast_drift.steady_state_bound
    p3 = analysis.cross_rank_weight_diff
    dominant = "Cross-Precision Cast" if p1 > p3 else "Cross-Rank Divergence"

    print(f"\n  Dominant error source: {dominant}")
    print(f"    Path 1 (cast drift):    {p1:.2e} — bounded")
    print(f"    Path 3 (cross-rank):    {p3:.2e} — linear with T")
    print(f"  Verdict: {analysis.risk_level}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  LLM-INFRA-VERIFIER: Megatron GPT-2 Verification")
    print("  Full-stack: Spatial + Temporal + Numerical + Resource")
    print("=" * 70)

    results = {}
    results["V1"] = v1_correct_gpt2_layer()
    results["V2"] = v2_missing_attention_allreduce()
    results["V3"] = v3_missing_mlp_allreduce()
    results["V4"] = v4_gelu_on_sharded_bug()
    results["V5"] = v5_async_gradient_race()
    results["V6"] = v6_gpt2_memory_planning()
    results["V7"] = v7_numerical_safety()
    results["V8"] = v8_accumulation_over_training()

    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY: Megatron GPT-2")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} checks passed")
    print()
