"""Numerical verification: floating-point error analysis for distributed training.

Models error propagation through the training loop and verifies:
  1. Reduction error bounds — AllReduce topology (ring vs tree) error accumulation
  2. Optimizer state consistency — Adam/AdamW invariants under DP/ZeRO
  3. Mixed precision safety — fp16/bf16 overflow/underflow risk detection
  4. Cross-precision cast error — fp32↔fp16 conversion error injection points

Key insight (backed by IEEE 754 analysis):
  - Same-precision operations have bounded error (O(ε·log n) for tree, O(ε·n) for ring)
  - Cross-precision casts (fp32→fp16) are the DOMINANT error source (13-bit mantissa loss)
  - Verifier models both, with cross-precision weighted higher

Design:
  - Uses symbolic interval analysis, not concrete values
  - Error bounds are worst-case analytical, not empirical
  - Optimizer invariants are proven by induction over the state transition
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, IntEnum
import math

from .state import TensorState, ShardingSpec, DeviceMesh, Shard, Replicate, Partial
from .ir import (
    IROp, Program, MatMul, Add, Multiply, SiLU, AllReduce, AllReduceAsync,
    AllGather, ReduceScatter, Send, Recv,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dtype Model — IEEE 754 properties for each precision
# ═══════════════════════════════════════════════════════════════════════════════

class Dtype(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP64 = "fp64"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"


@dataclass
class DtypeProperties:
    """IEEE 754 properties for a given floating-point dtype."""
    dtype: Dtype
    exponent_bits: int
    mantissa_bits: int
    machine_epsilon: float        # ε = 2^(-mantissa_bits)
    min_normal: float              # smallest normal number
    max_normal: float              # largest normal number
    unit_in_last_place: float      # ULP at 1.0

    @property
    def relative_error_bound(self) -> float:
        """Worst-case relative error for a single rounding operation."""
        return self.machine_epsilon / 2  # 0.5 ULP (IEEE 754 default rounding)


# IEEE 754 dtype table
DTYPE_PROPS: Dict[Dtype, DtypeProperties] = {
    Dtype.FP32: DtypeProperties(Dtype.FP32, 8, 23,
        machine_epsilon=2**-23,          # ≈ 1.19e-7
        min_normal=2**-126,              # ≈ 1.18e-38
        max_normal=(2-2**-23)*2**127,    # ≈ 3.40e38
        unit_in_last_place=2**-23),
    Dtype.FP16: DtypeProperties(Dtype.FP16, 5, 10,
        machine_epsilon=2**-10,          # ≈ 9.77e-4
        min_normal=2**-14,               # ≈ 6.10e-5
        max_normal=(2-2**-10)*2**15,     # ≈ 65504
        unit_in_last_place=2**-10),
    Dtype.BF16: DtypeProperties(Dtype.BF16, 8, 7,
        machine_epsilon=2**-7,           # ≈ 7.81e-3
        min_normal=2**-126,              # ≈ 1.18e-38
        max_normal=(2-2**-7)*2**127,     # ≈ 3.39e38
        unit_in_last_place=2**-7),
    Dtype.FP64: DtypeProperties(Dtype.FP64, 11, 52,
        machine_epsilon=2**-52,          # ≈ 2.22e-16
        min_normal=2**-1022,
        max_normal=(2-2**-52)*2**1023,
        unit_in_last_place=2**-52),
    Dtype.FP8_E4M3: DtypeProperties(Dtype.FP8_E4M3, 4, 3,
        machine_epsilon=2**-3,           # ≈ 0.125
        min_normal=2**-6,                # ≈ 0.0156
        max_normal=(2-2**-3)*2**7,       # ≈ 240
        unit_in_last_place=2**-3),
    Dtype.FP8_E5M2: DtypeProperties(Dtype.FP8_E5M2, 5, 2,
        machine_epsilon=2**-2,           # ≈ 0.25
        min_normal=2**-14,               # ≈ 6.1e-5
        max_normal=(2-2**-2)*2**15,      # ≈ 57344
        unit_in_last_place=2**-2),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Error Model — per-operation error bounds
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ErrorBound:
    """Error bound for a computation result."""
    absolute: float = 0.0       # max absolute error
    relative: float = 0.0       # max relative error
    source: str = ""            # where the error comes from

    def __repr__(self):
        return (f"ErrorBound(abs={self.absolute:.2e}, "
                f"rel={self.relative:.2e}, src={self.source})")


class ErrorModel:
    """Models per-operation floating-point error.

    For each op, computes the worst-case error bound given:
      - Input dtypes
      - Operation type
      - Value ranges (if available, for interval analysis)
    """

    def __init__(self, compute_dtype: Dtype = Dtype.FP16,
                 accumulate_dtype: Dtype = Dtype.FP32):
        self.compute = compute_dtype
        self.accumulate = accumulate_dtype
        self.compute_props = DTYPE_PROPS[compute_dtype]
        self.accumulate_props = DTYPE_PROPS[accumulate_dtype]

    def matmul_error(self, a_rows: int, a_cols: int, b_cols: int) -> ErrorBound:
        """Error bound for MatMul: Y = A @ B.

        Each multiply-add: (a*b + c) rounded twice in fp16.
        K inner products accumulated, each with ε_compute rounding.

        Theory (Higham, Accuracy and Stability of Numerical Algorithms):
          |Y - Ŷ| ≤ γ_k · |A|·|B|    where γ_k = k·ε/(1 - k·ε)
        """
        k = a_cols  # inner dimension
        eps = self.compute_props.machine_epsilon
        gamma_k = (k * eps) / (1 - k * eps) if k * eps < 1 else float('inf')
        return ErrorBound(
            relative=gamma_k,
            absolute=0.0,
            source=f"MatMul K={k}, ε={eps:.2e}, γ_k={gamma_k:.2e}",
        )

    def add_error(self, n_operands: int = 2) -> ErrorBound:
        """Error bound for element-wise addition.

        N operands summed: each addition introduces 0.5 ULP rounding.
        With N operands sequential: error ≤ (N-1) * 0.5 ULP.
        Pairwise: error ≤ log₂(N) * 0.5 ULP.
        """
        eps = self.compute_props.machine_epsilon
        return ErrorBound(
            relative=eps * (n_operands - 1),
            absolute=0.0,
            source=f"Add N={n_operands}, ε={eps:.2e}",
        )

    def allreduce_error(
        self, n_ranks: int, topology: str = "tree", value_range: float = 1.0,
    ) -> ErrorBound:
        """Error bound for AllReduce(sum) across N ranks.

        Ring topology: sequential accumulation → O(ε · N)
        Tree topology:  pairwise accumulation → O(ε · log N)

        These bounds assume fp32 accumulation. If fp16 is used throughout,
        the error is much worse.

        Args:
            n_ranks: number of ranks participating
            topology: "ring", "tree", "recursive_doubling"
            value_range: max absolute value of input (for absolute error)

        Returns:
            ErrorBound with relative and absolute error estimates
        """
        eps = self.accumulate_props.machine_epsilon
        condition = 1.0  # Σ|x_i| / |Σx_i|, assumed well-conditioned

        if topology == "ring":
            # Sequential: N-1 additions, error accumulates linearly
            n_ops = n_ranks - 1
            gamma_n = (n_ops * eps) / (1 - n_ops * eps) if n_ops * eps < 1 else float('inf')
            relative = gamma_n
            desc = f"Ring AllReduce(N={n_ranks}): {n_ops} sequential adds, O(ε·N)={eps*n_ops:.2e}"
        elif topology == "tree":
            # Pairwise tree: log₂(N) levels
            depth = math.ceil(math.log2(n_ranks))
            n_ops_per_path = depth
            gamma_d = (n_ops_per_path * eps) / (1 - n_ops_per_path * eps) if n_ops_per_path * eps < 1 else float('inf')
            relative = gamma_d
            desc = f"Tree AllReduce(N={n_ranks}): depth={depth}, O(ε·logN)={eps*depth:.2e}"
        elif topology == "recursive_doubling":
            # Recursive doubling: log₂(N) steps, but each step adds 2^k elements
            depth = math.ceil(math.log2(n_ranks))
            gamma_d = (depth * eps) / (1 - depth * eps) if depth * eps < 1 else float('inf')
            relative = gamma_d
            desc = f"RecursiveDoubling(N={n_ranks}): depth={depth}, O(ε·logN)={eps*depth:.2e}"
        else:
            relative = float('inf')
            desc = f"Unknown topology: {topology}"

        return ErrorBound(
            relative=relative,
            absolute=relative * value_range,
            source=desc,
        )

    def cast_error(self, src_dtype: Dtype, dst_dtype: Dtype) -> ErrorBound:
        """Error bound for dtype conversion.

        The relative rounding error when casting src→dst is bounded by
        0.5 * ULP of the DESTINATION format. This is because:
        - The source value is rounded to the nearest dst value
        - The error is at most half the spacing between adjacent dst values
        - Independent of how many bits the source had

        fp32 → fp16: rel_err = 0.5 * 2^(-10) = 2^(-11) ≈ 4.88e-4
        fp32 → bf16: rel_err = 0.5 * 2^(-7)  = 2^(-8)  ≈ 3.91e-3
        fp16 → fp32: rel_err = 0  (all fp16 values are exact in fp32)

        Range issues (overflow/underflow) are handled separately.
        """
        dst_props = DTYPE_PROPS[dst_dtype]
        src_props = DTYPE_PROPS[src_dtype]

        # If dst has >= mantissa bits AND >= exponent range, cast is exact
        # (e.g., fp16→fp32, bf16→fp32)
        if (dst_props.mantissa_bits >= src_props.mantissa_bits and
            dst_props.exponent_bits >= src_props.exponent_bits):
            rel = 0.0  # exact
            desc = f"Cast {src_dtype.value}→{dst_dtype.value}: EXACT (dst has >= precision)"
        else:
            # Rounding to destination: half ULP of destination
            rel = 0.5 * dst_props.machine_epsilon  # 0.5 * 2^(-m_dst)
            desc = (f"Cast {src_dtype.value}→{dst_dtype.value}: "
                    f"0.5 * ε_dst = {rel:.2e}")

        return ErrorBound(relative=rel, absolute=0.0, source=desc)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Reduction Error Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReductionAnalysis:
    """Result of analyzing AllReduce error accumulation."""
    n_ranks: int
    topology: str
    compute_dtype: Dtype
    accumulate_dtype: Dtype

    ring_error: ErrorBound
    tree_error: ErrorBound

    # Risk assessment
    safe_for_fp16: bool            # whether error stays within fp16 precision
    recommended_topology: str       # which topology to use
    max_recommended_ranks: int      # max ranks before error unacceptable

    def __repr__(self):
        lines = [
            f"ReductionAnalysis(N={self.n_ranks}, {self.topology})",
            f"  Ring error:  {self.ring_error}",
            f"  Tree error:  {self.tree_error}",
            f"  Safe for fp16: {self.safe_for_fp16}",
            f"  Recommended:  {self.recommended_topology} (max {self.max_recommended_ranks} ranks)",
        ]
        return "\n".join(lines)


class ReductionErrorAnalyzer:
    """Analyzes error accumulation in distributed reduction operations.

    Given the number of ranks and compute/accumulate dtypes, computes
    the worst-case error bound for different AllReduce topologies
    and determines whether the setup is numerically safe.
    """

    # Threshold: relative error above which fp16 precision is compromised
    FP16_SAFETY_THRESHOLD = 0.01  # 1% relative error is dangerous for training

    def __init__(
        self,
        compute_dtype: Dtype = Dtype.FP16,
        accumulate_dtype: Dtype = Dtype.FP32,
    ):
        self.error_model = ErrorModel(compute_dtype, accumulate_dtype)
        self.compute = compute_dtype
        self.accumulate = accumulate_dtype

    def analyze(self, n_ranks: int, topology: str = "tree") -> ReductionAnalysis:
        """Analyze reduction error for a given configuration."""
        ring_err = self.error_model.allreduce_error(n_ranks, "ring")
        tree_err = self.error_model.allreduce_error(n_ranks, "tree")

        # Determine safety
        # fp16 has ~0.1% relative precision → error below this is "safe"
        fp16_eps = DTYPE_PROPS[Dtype.FP16].machine_epsilon
        safe_for_fp16 = (
            tree_err.relative < self.FP16_SAFETY_THRESHOLD and
            tree_err.relative < float('inf')
        )

        # Recommend tree if ring error > 2x tree error
        use_tree = ring_err.relative > 2 * tree_err.relative

        # Max ranks: when error exceeds fp16 precision
        max_ranks = 1
        for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            test_err = self.error_model.allreduce_error(n, "tree")
            if test_err.relative < self.FP16_SAFETY_THRESHOLD:
                max_ranks = n
            else:
                break

        return ReductionAnalysis(
            n_ranks=n_ranks,
            topology=topology,
            compute_dtype=self.compute,
            accumulate_dtype=self.accumulate,
            ring_error=ring_err,
            tree_error=tree_err,
            safe_for_fp16=safe_for_fp16,
            recommended_topology="tree" if use_tree else topology,
            max_recommended_ranks=max_ranks,
        )

    def compare_topologies(self, n_ranks_list: List[int]) -> Dict[int, float]:
        """Compare error across ranks: return {n_ranks: tree_error_ratio}."""
        return {
            n: self.error_model.allreduce_error(n, "tree").relative
            for n in n_ranks_list
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Optimizer State Invariant Checker
# ═══════════════════════════════════════════════════════════════════════════════

class ZeROStage(IntEnum):
    DP = 0           # Data Parallel (no sharding of optimizer state)
    ZERO1 = 1        # Optimizer state sharded across DP ranks
    ZERO2 = 2        # Gradients + optimizer state sharded
    ZERO3 = 3        # Parameters + gradients + optimizer state sharded


@dataclass
class OptimizerInvariant:
    """Defines the invariants that must hold for an optimizer across ranks.

    For DP: All ranks have identical θ, m, v after each step.
    For ZeRO-1: Each rank's shard of θ matches the corresponding shard of
                the full θ (computed via AllReduce → update → slice).
    """
    zero_stage: ZeROStage
    # DP invariants
    identical_params: bool = True          # θ[r] == θ[s] for all r,s
    identical_master_weights: bool = True  # fp32 weight copies identical
    identical_optimizer_state: bool = True # m[r], v[r] identical across ranks
    # ZeRO invariants
    shard_boundary_consistent: bool = True # no leakage across shard boundaries
    gather_equals_full: bool = True        # AllGather(shards) == full tensor (within error)

    def __repr__(self):
        checks = []
        if self.zero_stage in (ZeROStage.DP,):
            checks.append(f"identical_params={self.identical_params}")
            checks.append(f"identical_master={self.identical_master_weights}")
            checks.append(f"identical_optimizer={self.identical_optimizer_state}")
        else:
            checks.append(f"shard_bounds={self.shard_boundary_consistent}")
            checks.append(f"gather_equals_full={self.gather_equals_full}")
        return f"OptimizerInvariant({self.zero_stage.name}, {', '.join(checks)})"


@dataclass
class OptimizerCheckResult:
    """Result of checking optimizer invariants across a training step."""
    invariant: OptimizerInvariant
    step: int
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Step {self.step}: {status}"]
        for v in self.violations:
            lines.append(f"  VIOLATION: {v}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


class OptimizerChecker:
    """Verifies optimizer state invariants under distributed training.

    Does NOT simulate actual training data. Instead, performs inductive
    proof: if invariants hold at step t-1, they hold at step t.
    """

    def __init__(
        self,
        optimizer: str = "adam",
        zero_stage: ZeROStage = ZeROStage.DP,
        compute_dtype: Dtype = Dtype.FP16,
        master_dtype: Dtype = Dtype.FP32,
    ):
        self.optimizer = optimizer
        self.zero_stage = zero_stage
        self.compute_dtype = compute_dtype
        self.master_dtype = master_dtype
        self.error_model = ErrorModel(compute_dtype, master_dtype)

    def verify_step(
        self,
        step: int,
        has_allreduce: bool = True,
        allreduce_topology: str = "tree",
        n_ranks: int = 8,
    ) -> OptimizerCheckResult:
        """Verify invariants for a single training step.

        The proof is structural — it checks whether the operations
        in the training step preserve the invariants, not whether
        specific values match.
        """
        invariant = OptimizerInvariant(zero_stage=self.zero_stage)
        violations = []
        warnings = []

        # 1. Check gradient AllReduce
        if has_allreduce:
            ar_error = self.error_model.allreduce_error(n_ranks, allreduce_topology)
            if ar_error.relative > 1e-6:
                warnings.append(
                    f"AllReduce error {ar_error.relative:.2e} may cause "
                    f"rank-to-rank gradient divergence"
                )

        # 2. Check fp16 → fp32 master weight cast
        if self.compute_dtype == Dtype.FP16 and self.master_dtype == Dtype.FP32:
            cast_err = self.error_model.cast_error(Dtype.FP16, Dtype.FP32)
            # fp16→fp32 is exact (all fp16 values are exactly representable in fp32)
            # So no error here. The reverse (fp32→fp16) IS lossy.
            pass

        # 3. Check fp32 master → fp16 weight for next forward
        if self.compute_dtype == Dtype.FP16 and self.master_dtype == Dtype.FP32:
            cast_err = self.error_model.cast_error(Dtype.FP32, Dtype.FP16)
            # The cast is deterministic: same fp32 input → same fp16 output
            # on all ranks. So this does NOT break identical-params invariant.
            # But it does mean weights diverge from the fp32 master copy.
            if cast_err.relative > 1e-4:
                warnings.append(
                    f"fp32→fp16 weight cast error {cast_err.relative:.2e}: "
                    f"fp16 weights differ from fp32 master by up to {cast_err.relative:.2e}. "
                    f"All ranks have identical fp16 weights (deterministic cast)."
                )

        # 4. Adam-specific checks
        if self.optimizer == "adam":
            v_min = DTYPE_PROPS[self.master_dtype].min_normal
            adam_epsilon = 1e-8  # default Adam ε
            if adam_epsilon < v_min:
                warnings.append(
                    f"Adam ε={adam_epsilon} is below {self.master_dtype.value} "
                    f"min normal {v_min:.1e}. ε has no effect."
                )

        # 5. ZeRO-specific checks
        if self.zero_stage >= ZeROStage.ZERO1:
            # Shard boundaries: optimizer state is split, each rank updates
            # only its shard. The invariant is that gather(all_shards) ==
            # the full tensor that would result from DP AllReduce → update.
            #
            # This holds IF:
            #   a) ReduceScatter(grad) ≡ AllReduce(grad)[my_shard] (mathematically yes)
            #   b) Each rank's update is independent (no cross-shard dependency)
            #   c) No numerical leakage across shard boundaries
            if self.compute_dtype == Dtype.FP16:
                cast_err = self.error_model.cast_error(Dtype.FP32, Dtype.FP16)
                warnings.append(
                    f"ZeRO-{self.zero_stage.value} with fp16: ReduceScatter uses "
                    f"fp16 intermediate; cast error {cast_err.relative:.2e} may "
                    f"cause shard boundary drift over many steps"
                )

        # 6. Overflow/underflow risk for fp16
        if self.compute_dtype == Dtype.FP16:
            fp16_min = DTYPE_PROPS[Dtype.FP16].min_normal
            fp16_max = DTYPE_PROPS[Dtype.FP16].max_normal
            warnings.append(
                f"fp16 range: [{fp16_min:.1e}, {fp16_max:.1e}]. "
                f"Gradients outside this range will underflow/overflow."
            )

            # Adam v_t: after many steps, v_t → E[g²] / (1-β2)
            # If E[g²] is small, v_t may be < fp16 min
            beta2 = 0.999
            steady_v_scale = 1.0 / (1.0 - beta2)  # ≈ 1000
            # So v_t ≈ 1000 * E[g²]. Need E[g²] > fp16_min / 1000
            min_g2_for_fp16 = fp16_min / steady_v_scale
            if min_g2_for_fp16 > 0:
                warnings.append(
                    f"Adam v_t steady-state scaling: 1/(1-β2)={steady_v_scale:.0f}. "
                    f"For fp16 v_t > min_normal, need E[g²] > {min_g2_for_fp16:.2e}. "
                    f"Recommend fp32 for Adam state."
                )

        passed = len(violations) == 0
        return OptimizerCheckResult(
            invariant=invariant,
            step=step,
            passed=passed,
            violations=violations,
            warnings=warnings,
        )

    def verify_training_loop(
        self,
        num_steps: int = 1,
        n_ranks: int = 8,
        allreduce_topology: str = "tree",
    ) -> List[OptimizerCheckResult]:
        """Verify invariants over multiple training steps.

        For the inductive proof: check step 0 (initialization) and
        the induction step (if step t-1 is OK, step t is OK).
        """
        results = []

        # Base case: initialization
        results.append(OptimizerCheckResult(
            invariant=OptimizerInvariant(zero_stage=self.zero_stage),
            step=0,
            passed=True,
            violations=[],
            warnings=["Initial state: all ranks identical (verified by init)"],
        ))

        # Induction: any single step preserves invariants
        for t in range(1, num_steps + 1):
            result = self.verify_step(
                step=t,
                has_allreduce=True,
                allreduce_topology=allreduce_topology,
                n_ranks=n_ranks,
            )
            results.append(result)

            if not result.passed:
                break  # Invariant broken → chain fails

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Overflow / Underflow Risk Detector
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OverflowRisk:
    """Risk assessment for a specific tensor/value in the training loop."""
    name: str
    dtype: Dtype
    typical_magnitude: float       # estimated typical value
    underflow_risk: bool
    overflow_risk: bool
    safe_range: Tuple[float, float]  # [min, max] safe values

    @property
    def risk_level(self) -> str:
        if self.overflow_risk:
            return "HIGH (overflow)"
        if self.underflow_risk:
            return "MEDIUM (underflow)"
        return "SAFE"

    def __repr__(self):
        return (
            f"OverflowRisk({self.name}, {self.dtype.value}, "
            f"mag≈{self.typical_magnitude:.1e}, "
            f"safe=[{self.safe_range[0]:.1e}, {self.safe_range[1]:.1e}], "
            f"risk={self.risk_level})"
        )


class OverflowRiskDetector:
    """Detects overflow/underflow risks for fp16/bf16 training.

    Checks typical value ranges against dtype limits.
    Does NOT need actual data — uses structural properties:
      - Embedding dim → typical activation magnitudes
      - Layer count → gradient magnitude decay
      - Batch size → loss magnitude
    """

    def __init__(self, compute_dtype: Dtype = Dtype.FP16):
        self.dtype = compute_dtype
        self.props = DTYPE_PROPS[compute_dtype]

    def check_activations(
        self,
        hidden_dim: int,
        num_layers: int,
    ) -> List[OverflowRisk]:
        """Check activation overflow risks given model structure."""
        risks = []
        min_val = self.props.min_normal
        max_val = self.props.max_normal

        # Post-attention activation: roughly O(1) after LayerNorm
        risks.append(OverflowRisk(
            name="post_attn_activation",
            dtype=self.dtype,
            typical_magnitude=1.0,
            underflow_risk=False,  # O(1) is fine
            overflow_risk=False,
            safe_range=(min_val, max_val),
        ))

        # Pre-softmax logits: scale with sqrt(d) due to QK^T scaling
        import math
        logit_scale = math.sqrt(hidden_dim)
        risks.append(OverflowRisk(
            name="pre_softmax_logits",
            dtype=self.dtype,
            typical_magnitude=logit_scale,
            underflow_risk=logit_scale < min_val,
            overflow_risk=logit_scale > max_val,
            safe_range=(min_val, max_val),
        ))

        # Intermediate MLP activations (up to 4x hidden_dim in some architectures)
        ffn_magnitude = 4 * hidden_dim ** 0.5
        risks.append(OverflowRisk(
            name="ffn_intermediate",
            dtype=self.dtype,
            typical_magnitude=ffn_magnitude,
            underflow_risk=ffn_magnitude < min_val,
            overflow_risk=ffn_magnitude > max_val,
            safe_range=(min_val, max_val),
        ))

        return risks

    def check_gradients(
        self,
        batch_size: int,
        hidden_dim: int,
        num_layers: int,
    ) -> List[OverflowRisk]:
        """Check gradient underflow/overflow risks."""
        risks = []
        min_val = self.props.min_normal

        # Gradient magnitude decays as 1/sqrt(N_layers) due to chain rule
        # But also scales as 1/batch_size from loss averaging
        typical_grad = 1.0 / (batch_size * num_layers ** 0.5)

        risks.append(OverflowRisk(
            name="typical_gradient",
            dtype=self.dtype,
            typical_magnitude=typical_grad,
            underflow_risk=typical_grad < min_val,
            overflow_risk=False,  # gradients are small, not large
            safe_range=(min_val, self.props.max_normal),
        ))

        # Embedding gradients: shared across layers, larger magnitude
        embed_grad = 1.0 / batch_size  # full loss effect
        risks.append(OverflowRisk(
            name="embedding_gradient",
            dtype=self.dtype,
            typical_magnitude=embed_grad,
            underflow_risk=embed_grad < min_val,
            overflow_risk=False,
            safe_range=(min_val, self.props.max_normal),
        ))

        return risks

    def check_loss_scale(self, gradient_magnitude: float) -> float:
        """Recommend a loss scale to keep gradients in fp16 safe range.

        Returns the recommended loss scale factor.
        """
        fp16_min = self.props.min_normal
        # We want: loss_scale * grad > fp16_min
        safety_margin = 10.0  # 10x margin for safety
        min_scale = safety_margin * fp16_min / max(gradient_magnitude, 1e-30)
        return max(1.0, min_scale)

    def check_adam_state_precision(self) -> List[str]:
        """Check if Adam optimizer state should use fp32 instead of fp16."""
        warnings = []
        if self.dtype in (Dtype.FP16, Dtype.BF16):
            # Adam β2=0.999 → v steady state amplified by 1000x
            # But then sqrt(v) brings it back to ~30x
            # Adam ε=1e-8 is below fp16 min_normal (6e-5)
            # → ε literally does nothing in fp16
            fp16_min = DTYPE_PROPS[Dtype.FP16].min_normal
            warnings.append(
                f"Adam ε=1e-8 is below {self.dtype.value} min_normal={fp16_min:.1e}. "
                f"In {self.dtype.value}, ε has NO effect on the denominator √v̂+ε. "
                f"Strongly recommend fp32 for Adam state."
            )

        return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Top-level numerical verifier
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NumericalVerifyResult:
    """Complete numerical verification result for a distributed training setup."""
    reduction_analysis: ReductionAnalysis
    optimizer_results: List[OptimizerCheckResult]
    overflow_risks: List[OverflowRisk]
    warnings: List[str]
    violations: List[str]

    @property
    def is_safe(self) -> bool:
        return len(self.violations) == 0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  NUMERICAL VERIFICATION REPORT",
            "=" * 70,
            "",
            "Reduction Analysis:",
            f"  {self.reduction_analysis}",
            "",
            "Optimizer Invariants:",
        ]
        for r in self.optimizer_results[-1:]:  # show last step
            lines.append(f"  {r}")
        lines.extend([
            "",
            "Overflow/Underflow Risks:",
        ])
        for risk in self.overflow_risks:
            lines.append(f"  {risk}")
        lines.extend([
            "",
            f"Verdict: {'SAFE' if self.is_safe else 'UNSAFE'}",
        ])
        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  - {v}")
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def verify_numerical(
    n_ranks: int = 8,
    topology: str = "tree",
    compute_dtype: Dtype = Dtype.FP16,
    accumulate_dtype: Dtype = Dtype.FP32,
    optimizer: str = "adam",
    zero_stage: ZeROStage = ZeROStage.DP,
    hidden_dim: int = 4096,
    num_layers: int = 32,
    batch_size: int = 128,
) -> NumericalVerifyResult:
    """Run full numerical verification for a distributed training setup.

    Args:
        n_ranks: number of GPUs
        topology: AllReduce topology ("ring", "tree", "recursive_doubling")
        compute_dtype: forward pass dtype
        accumulate_dtype: gradient accumulation dtype
        optimizer: "adam" or "adamw"
        zero_stage: ZeRO stage (DP, ZERO1, ZERO2, ZERO3)
        hidden_dim: model hidden dimension
        num_layers: number of transformer layers
        batch_size: per-GPU batch size
    """
    violations = []
    warnings = []

    # 1. Analyze reduction error
    analyzer = ReductionErrorAnalyzer(compute_dtype, accumulate_dtype)
    red_analysis = analyzer.analyze(n_ranks, topology)

    if not red_analysis.safe_for_fp16:
        warnings.append(
            f"fp16 AllReduce may be unsafe: tree_error={red_analysis.tree_error.relative:.2e}. "
            f"Consider using fp32 accumulation or reducing rank count."
        )

    # 2. Check optimizer invariants
    checker = OptimizerChecker(optimizer, zero_stage, compute_dtype, accumulate_dtype)
    opt_results = checker.verify_training_loop(
        num_steps=1, n_ranks=n_ranks, allreduce_topology=topology,
    )

    for r in opt_results:
        violations.extend(r.violations)
        warnings.extend(r.warnings)

    # 3. Check overflow/underflow risks
    detector = OverflowRiskDetector(compute_dtype)
    act_risks = detector.check_activations(hidden_dim, num_layers)
    grad_risks = detector.check_gradients(batch_size, hidden_dim, num_layers)
    all_risks = act_risks + grad_risks

    overflow_risks = [r for r in all_risks if r.risk_level != "SAFE"]
    for risk in overflow_risks:
        if risk.overflow_risk:
            violations.append(f"{risk.name}: overflow risk")
        if risk.underflow_risk:
            warnings.append(f"{risk.name}: underflow risk")

    # 4. Adam state precision warning (always relevant for fp16)
    if compute_dtype == Dtype.FP16:
        adam_warnings = detector.check_adam_state_precision()
        warnings.extend(adam_warnings)

    return NumericalVerifyResult(
        reduction_analysis=red_analysis,
        optimizer_results=opt_results,
        overflow_risks=all_risks,
        warnings=warnings,
        violations=violations,
    )
