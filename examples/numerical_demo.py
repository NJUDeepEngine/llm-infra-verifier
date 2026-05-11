"""
Demo: Numerical Verification for Distributed Training.

Demonstrates:
  1. Dtype properties — IEEE 754 characteristics of each precision
  2. Reduction error analysis — Ring vs Tree AllReduce error bounds
  3. Optimizer invariants — Adam state consistency under DP/ZeRO
  4. Overflow/underflow risk — fp16 safety for given model config

Key findings (backed by IEEE 754 analysis):
  - Same-precision addition IS NOT associative (intermediate rounding)
  - Ring AllReduce error ∝ N, Tree AllReduce error ∝ log(N)
  - fp32→fp16 cast loses 13 bits of mantissa (DOMINANT error source)
  - Adam ε=1e-8 has NO effect in fp16 (below min_normal=6e-5)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.numerical import (
    Dtype, DtypeProperties, DTYPE_PROPS,
    ErrorModel, ErrorBound,
    ReductionErrorAnalyzer, ReductionAnalysis,
    OptimizerChecker, OptimizerCheckResult, ZeROStage,
    OverflowRiskDetector, OverflowRisk,
    verify_numerical, NumericalVerifyResult,
)


def demo_dtype_properties():
    """Show IEEE 754 properties for all dtypes."""
    print("=" * 70)
    print("  1. DTYPE PROPERTIES (IEEE 754)")
    print("=" * 70)
    print(f"  {'Dtype':<12} {'Mantissa':>8} {'ε (machine)':>14} {'Min Normal':>14} {'Max Normal':>14}")
    print(f"  {'-'*12} {'-'*8} {'-'*14} {'-'*14} {'-'*14}")
    for dtype, props in DTYPE_PROPS.items():
        print(f"  {dtype.value:<12} {props.mantissa_bits:>8} {props.machine_epsilon:>14.2e} "
              f"{props.min_normal:>14.2e} {props.max_normal:>14.2e}")


def demo_cast_errors():
    """Show error bounds for dtype conversions."""
    print("\n" + "=" * 70)
    print("  2. CAST ERROR BOUNDS (Cross-Precision)")
    print("=" * 70)
    model = ErrorModel(Dtype.FP16, Dtype.FP32)
    casts = [
        (Dtype.FP32, Dtype.FP16),
        (Dtype.FP32, Dtype.BF16),
        (Dtype.BF16, Dtype.FP16),
        (Dtype.FP16, Dtype.FP32),
        (Dtype.FP32, Dtype.FP8_E4M3),
    ]
    for src, dst in casts:
        err = model.cast_error(src, dst)
        lost = DTYPE_PROPS[src].mantissa_bits - DTYPE_PROPS[dst].mantissa_bits
        print(f"  {src.value} → {dst.value}: lost {max(0, lost)} bits, "
              f"rel_err ≈ {err.relative:.2e}")


def demo_reduction_analysis():
    """Compare Ring vs Tree AllReduce error."""
    print("\n" + "=" * 70)
    print("  3. REDUCTION ERROR ANALYSIS (Ring vs Tree)")
    print("=" * 70)

    for n_ranks in [4, 8, 16, 32, 64, 128, 256]:
        analyzer = ReductionErrorAnalyzer(Dtype.FP16, Dtype.FP32)
        result = analyzer.analyze(n_ranks, "tree")
        ring_rel = result.ring_error.relative
        tree_rel = result.tree_error.relative
        ratio = ring_rel / tree_rel if tree_rel > 0 else float('inf')
        safe = "SAFE" if result.safe_for_fp16 else "RISKY"
        print(f"  N={n_ranks:>4}: Ring={ring_rel:.2e}  Tree={tree_rel:.2e}  "
              f"Ratio={ratio:.1f}x  MaxRanks={result.max_recommended_ranks}  [{safe}]")


def demo_optimizer_invariants():
    """Check Adam invariants with fp16 compute."""
    print("\n" + "=" * 70)
    print("  4. OPTIMIZER INVARIANTS (Adam + fp16 + DP)")
    print("=" * 70)

    checker = OptimizerChecker(
        optimizer="adam",
        zero_stage=ZeROStage.DP,
        compute_dtype=Dtype.FP16,
        master_dtype=Dtype.FP32,
    )
    results = checker.verify_training_loop(num_steps=3, n_ranks=8)

    for r in results:
        print(f"  {r}")
        if not r.passed:
            print(f"    → INVARIANT BROKEN")


def demo_overflow_risks():
    """Check fp16 overflow/underflow for a typical LLM config."""
    print("\n" + "=" * 70)
    print("  5. OVERFLOW / UNDERFLOW RISK (fp16, Llama-like config)")
    print("=" * 70)

    detector = OverflowRiskDetector(Dtype.FP16)

    # Llama-7B-like: hidden=4096, layers=32
    act_risks = detector.check_activations(hidden_dim=4096, num_layers=32)
    grad_risks = detector.check_gradients(batch_size=128, hidden_dim=4096, num_layers=32)

    print("  Activation risks:")
    for r in act_risks:
        flag = " ← RISK!" if r.risk_level != "SAFE" else ""
        print(f"    {r.name}: mag≈{r.typical_magnitude:.1e}, safe=[{r.safe_range[0]:.1e}, {r.safe_range[1]:.1e}]{flag}")

    print("  Gradient risks:")
    for r in grad_risks:
        flag = " ← RISK!" if r.risk_level != "SAFE" else ""
        print(f"    {r.name}: mag≈{r.typical_magnitude:.1e}, safe=[{r.safe_range[0]:.1e}, {r.safe_range[1]:.1e}]{flag}")

    # Loss scale recommendation
    scale = detector.check_loss_scale(gradient_magnitude=1e-6)
    print(f"\n  Recommended loss scale for grad≈1e-6: {scale:.1f}")

    adam_warnings = detector.check_adam_state_precision()
    print("\n  Adam precision warnings:")
    for w in adam_warnings:
        print(f"    {w}")


def demo_full_verification():
    """Run full verification for a realistic training config."""
    print("\n" + "=" * 70)
    print("  6. FULL NUMERICAL VERIFICATION")
    print("=" * 70)

    # Llama-7B-style config on 8 GPUs
    result = verify_numerical(
        n_ranks=8,
        topology="tree",
        compute_dtype=Dtype.FP16,
        accumulate_dtype=Dtype.FP32,
        optimizer="adam",
        zero_stage=ZeROStage.DP,
        hidden_dim=4096,
        num_layers=32,
        batch_size=128,
    )
    print(result.summary())


def demo_zeor1_verification():
    """Compare DP vs ZeRO-1 numerical properties."""
    print("\n" + "=" * 70)
    print("  7. DP vs ZeRO-1 COMPARISON")
    print("=" * 70)

    for stage in [ZeROStage.DP, ZeROStage.ZERO1, ZeROStage.ZERO2]:
        checker = OptimizerChecker("adam", stage, Dtype.FP16, Dtype.FP32)
        result = checker.verify_step(step=1, n_ranks=8)

        n_v = len(result.violations)
        n_w = len(result.warnings)
        print(f"  {stage.name:8s}: {result.passed} ({n_v} violations, {n_w} warnings)")
        for w in result.warnings:
            print(f"    WARN: {w}")
        for v in result.violations:
            print(f"    VIOLATION: {v}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DTENSOR-VERIFIER: Numerical Verification Demo")
    print("  Floating-Point Error Analysis for Distributed Training")
    print("=" * 70)

    demo_dtype_properties()
    demo_cast_errors()
    demo_reduction_analysis()
    demo_optimizer_invariants()
    demo_overflow_risks()
    demo_full_verification()
    demo_zeor1_verification()

    print("\n" + "=" * 70)
    print("  KEY INSIGHTS")
    print("=" * 70)
    print("""
  1. Same-precision fp32 addition IS NOT associative
     → (a+b)+c ≠ a+(b+c) due to intermediate rounding
     → Error bounded by O(ε·logN) for tree, O(ε·N) for ring

  2. Cross-precision (fp32→fp16) is the DOMINANT error source
     → Loses 13 bits of mantissa (vs 1-2 bits for same-precision)
     → Every fp32→fp16 cast is an error injection point

  3. fp16 is dangerous for Adam optimizer state
     → ε=1e-8 < fp16 min_normal=6e-5 → ε has ZERO effect
     → v_t can underflow over many steps
     → ALWAYS use fp32 for Adam m/v

  4. Ring vs Tree AllReduce matters at scale
     → N=256: Ring error ~250x machine epsilon, Tree ~8x
     → Tree is ~30x more accurate at large N
     → But still negligible vs cross-precision error
""")
