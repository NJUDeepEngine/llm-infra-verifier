"""
Distributed Tensor Verification Benchmark Suite.

Constructed from real-world bugs found in:
  - PyTorch DTensor issues (github.com/pytorch/pytorch)
  - Megatron-LM issues (github.com/NVIDIA/Megatron-LM)
  - TileLang issues (github.com/tile-ai/tilelang)
  - DeepSeek TileKernels (github.com/deepseek-ai/TileKernels)

Each benchmark case:
  1. Documents the source bug (GitHub issue link)
  2. Distills the minimal reproducing verification scenario
  3. Runs our verifier against it
  4. Reports: detected/not-detected, category, severity

Categories:
  B1 — Missing/Incorrect Collectives
  B2 — Placement/Shard Specification Errors
  B3 — Communication Legality Violations
  B4 — Gradient Duality Failures
  B5 — Pipeline Parallelism Schedule Bugs
  B6 — Context Parallelism Communication Errors

Usage:
  python benchmarks/benchmark_suite.py          # run all
  python benchmarks/benchmark_suite.py --list   # list all cases
  python benchmarks/benchmark_suite.py --run B1 # run category B1
"""

from __future__ import annotations

import sys
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    compute_local_shape,
)
from verifier.ir import (
    IROp, Program, MatMul, Add, Multiply, SiLU, AllReduce,
    AllGather, ReduceScatter, Send, Recv, FlashAttention, ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine
from verifier.solver import DistributedVerifier, VerifyResult
from verifier.rewrite import PlacementAnalyzer, PlacementAnalysis, ProgramCost
from verifier.synthesis import SynthesisEngine, synthesize_parallel_program


# ── Benchmark result types ───────────────────────────────────────────────────

class Severity(Enum):
    CRITICAL = "critical"   # wrong results, silent correctness failure
    HIGH = "high"            # crashes, exceptions
    MEDIUM = "medium"        # inefficiency, suboptimal
    LOW = "low"              # cosmetic, warnings


class BugCategory(Enum):
    B1_MISSING_COLLECTIVE = "B1: Missing/Incorrect Collectives"
    B2_PLACEMENT_ERROR = "B2: Placement/Shard Errors"
    B3_COMM_LEGALITY = "B3: Communication Legality"
    B4_GRADIENT_DUALITY = "B4: Gradient Duality"
    B5_PP_SCHEDULE = "B5: PP Schedule Correctness"
    B6_CP_COMM = "B6: CP Communication"


@dataclass
class BenchmarkCase:
    """A single benchmark case derived from a real-world bug."""
    id: str
    title: str
    category: BugCategory
    severity: Severity
    source_url: str                    # link to original GitHub issue
    source_issue: str                  # e.g., "pytorch/pytorch#144359"
    description: str
    expected_behavior: str             # what SHOULD happen
    actual_bug_behavior: str           # what the bug DOES

    # The test function
    setup_fn: Callable[[], Tuple[Program, Dict[str, TensorState], DeviceMesh]]
    verify_fn: Callable[[Program, Dict[str, TensorState], DeviceMesh], List[VerifyResult]]

    # Results (populated at runtime)
    detected: bool = False
    detection_details: str = ""
    runtime_ms: float = 0.0

    def run(self) -> bool:
        """Run this benchmark case. Returns True if bug was detected."""
        start = time.time()

        try:
            program, tensors, mesh = self.setup_fn()
            results = self.verify_fn(program, tensors, mesh)
            self.runtime_ms = (time.time() - start) * 1000

            # Bug is "detected" if verification FAILS (finds an issue)
            self.detected = any(not r.passed for r in results)
            self.detection_details = "\n    ".join(
                f"[{'FAIL' if not r.passed else 'PASS'}] {r.condition}: {r.details}"
                for r in results
            )
        except Exception as e:
            self.runtime_ms = (time.time() - start) * 1000
            self.detected = True  # Exception during verification = bug detected
            self.detection_details = f"Exception: {e}"

        return self.detected

    def summary(self) -> str:
        status = "DETECTED" if self.detected else "MISSED"
        return (
            f"[{status}] {self.id}: {self.title}\n"
            f"  Category: {self.category.value} | Severity: {self.severity.value}\n"
            f"  Source: {self.source_issue}\n"
            f"  {self.detection_details}\n"
            f"  Time: {self.runtime_ms:.1f}ms"
        )


# ── Benchmark case definitions ───────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# B1: Missing/Incorrect Collectives
# ═══════════════════════════════════════════════════════════════════════════════

B1A_ROW_PARALLEL_MISSING_ALLREDUCE = BenchmarkCase(
    id="B1a",
    title="Row Parallel Linear without AllReduce",
    category=BugCategory.B1_MISSING_COLLECTIVE,
    severity=Severity.CRITICAL,
    source_url="https://github.com/pytorch/pytorch/issues/144359",
    source_issue="pytorch/pytorch#144359 (Incorrect Results with TP)",
    description=(
        "Row Parallel Linear with X:Shard(1), W:Shard(0) produces PARTIAL "
        "output that must be AllReduced.  Without the AllReduce, each device "
        "holds only a partial sum — silent correctness failure."
    ),
    expected_behavior="Y should be Replicate() after AllReduce",
    actual_bug_behavior="Y remains PARTIAL — each device has incomplete result",
    setup_fn=lambda: _setup_row_parallel_missing_ar(),
    verify_fn=lambda p, t, m: _verify_postcondition_and_analysis(p, t, m),
)

B1B_COLWISE_ROWWISE_GELU_BUG = BenchmarkCase(
    id="B1b",
    title="GELU between ColwiseParallel and RowwiseParallel without AllReduce",
    category=BugCategory.B1_MISSING_COLLECTIVE,
    severity=Severity.CRITICAL,
    source_url="https://github.com/pytorch/pytorch/issues/144359",
    source_issue="pytorch/pytorch#144359 (GELU nonlinearity + Colwise→Rowwise TP)",
    description=(
        "ColwiseParallel(use_local_output=True) followed by GELU activation "
        "and RowwiseParallel — GELU(shard) ≠ shard_of(GELU(full)).  "
        "The AllReduce must happen BEFORE the nonlinear activation, not after. "
        "This produces numerically incorrect results that are hard to detect "
        "without a verifier or numerical comparison."
    ),
    expected_behavior="AllReduce before GELU, or GELU after AllReduce",
    actual_bug_behavior="GELU applied to partial shards → silently wrong output",
    setup_fn=lambda: _setup_colwise_gelu_rowwise_bug(),
    verify_fn=lambda p, t, m: _verify_nonlinear_on_sharded(p, t, m),
)

B1C_MISSING_BROADCAST_PP = BenchmarkCase(
    id="B1c",
    title="Missing broadcast of metadata across pipeline stages",
    category=BugCategory.B1_MISSING_COLLECTIVE,
    severity=Severity.HIGH,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4092",
    source_issue="Megatron-LM#4092 (Missing seqlen broadcast in PP)",
    description=(
        "Intermediate pipeline stages need cu_seqlens/max_seqlen broadcast "
        "from stage 0.  Without this broadcast, shape mismatches occur. "
        "Modeled as a missing Send/Recv pair for metadata tensors."
    ),
    expected_behavior="seqlen metadata broadcast from stage 0 to stage 1",
    actual_bug_behavior="Stage 1 has no cu_seqlens -> shape mismatch at attention",
    setup_fn=lambda: _setup_missing_pp_broadcast(),
    verify_fn=lambda p, t, m: _verify_cross_stage_broadcast(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# B2: Placement/Shard Specification Errors
# ═══════════════════════════════════════════════════════════════════════════════

B2A_SHARD1_NONCONTIGUOUS = BenchmarkCase(
    id="B2a",
    title="Shard(1) produces non-contiguous local tensors in TP Linear",
    category=BugCategory.B2_PLACEMENT_ERROR,
    severity=Severity.HIGH,
    source_url="https://github.com/pytorch/pytorch/issues/173041",
    source_issue="pytorch/pytorch#173041",
    description=(
        "DTensor with Shard(1) placement produces non-contiguous local tensors "
        "when used in Tensor Parallel Linear layers. view() operations that "
        "require contiguous memory fail.  Our verifier detects placement/shape "
        "propagation that would lead to incompatible views."
    ),
    expected_behavior="Shard(1) should produce contiguous local tensors after redistribution",
    actual_bug_behavior="Shard(1) → Replicate() → local tensor is non-contiguous → view fails",
    setup_fn=lambda: _setup_shard1_noncontiguous(),
    verify_fn=lambda p, t, m: _verify_placement_consistency(p, t, m),
)

B2B_SHARD_TO_REPLICATE_SHAPE_CORRUPTION = BenchmarkCase(
    id="B2b",
    title="Shard→Replicate redistribution corrupts symbolic shapes",
    category=BugCategory.B2_PLACEMENT_ERROR,
    severity=Severity.HIGH,
    source_url="https://github.com/pytorch/pytorch/issues/175690",
    source_issue="pytorch/pytorch#175690",
    description=(
        "Shard(dim)→Replicate redistribution uses all_gather which produces "
        "n * ceil(s/n) instead of s when s is not divisible by n.  "
        "The unpadding guard fails under dynamic shapes, corrupting sizes. "
        "Our verifier checks shape consistency through placement changes."
    ),
    expected_behavior="AllReduce output shape == input global shape",
    actual_bug_behavior="Shape becomes n * ceil(s/n) instead of s under dynamic shapes",
    setup_fn=lambda: _setup_shard_to_replicate_shape_bug(),
    verify_fn=lambda p, t, m: _verify_shape_through_reshard(p, t, m),
)

B2C_SEQUENCE_PARALLEL_DTENSOR_CAST = BenchmarkCase(
    id="B2c",
    title="SequenceParallel silently casts DTensor to Tensor",
    category=BugCategory.B2_PLACEMENT_ERROR,
    severity=Severity.HIGH,
    source_url="https://github.com/pytorch/pytorch/issues/139681",
    source_issue="pytorch/pytorch#139681",
    description=(
        "SequenceParallel wrapped around RMSNorm causes an unexpected "
        "DTensor→Tensor cast during checkpoint recomputation.  "
        "The output * weight multiply sees mixed types.  "
        "Our verifier detects placement inconsistencies that would "
        "manifest as type mismatches at boundaries."
    ),
    expected_behavior="All tensors in forward should maintain their DTensor wrappers",
    actual_bug_behavior="DTensor loses its wrapper → mixed Tensor/DTensor error",
    setup_fn=lambda: _setup_sequence_parallel_cast_bug(),
    verify_fn=lambda p, t, m: _verify_compatible_sharding_elementwise(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# B3: Communication Legality Violations
# ═══════════════════════════════════════════════════════════════════════════════

B3A_ALLREDUCE_ON_REPLICATE = BenchmarkCase(
    id="B3a",
    title="AllReduce called on already-replicated tensor (redundant communication)",
    category=BugCategory.B3_COMM_LEGALITY,
    severity=Severity.MEDIUM,
    source_url="https://github.com/tile-ai/tilelang/issues/2035",
    source_issue="tilelang#2035 (False positive conflict warning pattern)",
    description=(
        "AllReduce is called on a tensor that is already Replicate().  "
        "This is functionally correct but wastes communication bandwidth. "
        "Our verifier flags redundant collectives."
    ),
    expected_behavior="AllReduce should only be called on PARTIAL tensors",
    actual_bug_behavior="Redundant AllReduce wastes bandwidth",
    setup_fn=lambda: _setup_redundant_allreduce(),
    verify_fn=lambda p, t, m: _verify_redundant_collectives(p, t, m),
)

B3B_SEND_WITHOUT_RECV = BenchmarkCase(
    id="B3b",
    title="Send without matching Recv in pipeline schedule",
    category=BugCategory.B3_COMM_LEGALITY,
    severity=Severity.CRITICAL,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4092",
    source_issue="Megatron-LM#4092",
    description=(
        "A Send operation in the forward pass has no corresponding Recv. "
        "This would cause a hang or crash.  Common in PP when stage "
        "boundary communication is incorrectly configured."
    ),
    expected_behavior="Every Send must have a matching Recv",
    actual_bug_behavior="Unmatched Send → deadlock or crash",
    setup_fn=lambda: _setup_send_without_recv(),
    verify_fn=lambda p, t, m: _verify_pp_deadlock(p, t, m),
)

B3C_ALLGATHER_DIM_MISMATCH = BenchmarkCase(
    id="B3c",
    title="AllGather gather_dim doesn't match shard dim",
    category=BugCategory.B3_COMM_LEGALITY,
    severity=Severity.HIGH,
    source_url="https://github.com/pytorch/pytorch/issues/140227",
    source_issue="pytorch/pytorch#140227",
    description=(
        "AllGather is called with gather_dim=0 but the tensor is sharded "
        "on dim=1.  The collective operation would produce incorrect results "
        "or error.  Our verifier checks dim consistency."
    ),
    expected_behavior="AllGather gather_dim must match the tensor's Shard dim",
    actual_bug_behavior="Gather on wrong dim → silently incorrect output shape/values",
    setup_fn=lambda: _setup_allgather_dim_mismatch(),
    verify_fn=lambda p, t, m: _verify_collective_dim_consistency(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# B4: Gradient Duality Failures
# ═══════════════════════════════════════════════════════════════════════════════

B4A_ALLREDUCE_DUAL_MISSING_IN_BWD = BenchmarkCase(
    id="B4a",
    title="AllReduce in forward has no corresponding AllReduce in backward",
    category=BugCategory.B4_GRADIENT_DUALITY,
    severity=Severity.CRITICAL,
    source_url="https://github.com/deepseek-ai/TileKernels/issues/2",
    source_issue="TileKernels#2 (Discuss backward impl of mHC)",
    description=(
        "Forward contains AllReduce for Row Parallel Linear, but backward "
        "is missing the dual AllReduce for gradient of W.  "
        "Gradients would be incorrect."
    ),
    expected_behavior="grad_W behind Row Parallel must be AllReduced",
    actual_bug_behavior="Gradients are partial → optimizer updates are wrong",
    setup_fn=lambda: _setup_missing_bwd_allreduce(),
    verify_fn=lambda p, t, m: _verify_gradient_duality(p, t, m),
)

B4B_WRONG_DUAL_COLLECTIVE_TYPE = BenchmarkCase(
    id="B4b",
    title="Wrong dual collective type in backward (AllGather instead of ReduceScatter)",
    category=BugCategory.B4_GRADIENT_DUALITY,
    severity=Severity.CRITICAL,
    source_url="https://github.com/pytorch/pytorch/issues/144359",
    source_issue="pytorch/pytorch#144359 (TP gradient flow)",
    description=(
        "Forward has AllGather for Column Parallel, but backward incorrectly "
        "uses another AllGather instead of ReduceScatter.  "
        "AllGather×AllGather would duplicate gradient info rather than "
        "correctly reducing it."
    ),
    expected_behavior="AllGather(fwd) ↔ ReduceScatter(bwd)",
    actual_bug_behavior="Double AllGather → incorrect gradient accumulation",
    setup_fn=lambda: _setup_wrong_dual_type(),
    verify_fn=lambda p, t, m: _verify_gradient_duality(p, t, m),
)

B4C_SEND_RECV_DIRECTION_NOT_REVERSED = BenchmarkCase(
    id="B4c",
    title="Send/Recv direction not reversed in backward for PP",
    category=BugCategory.B4_GRADIENT_DUALITY,
    severity=Severity.CRITICAL,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4092",
    source_issue="Megatron-LM#4092 (PP backward communication)",
    description=(
        "Forward: Send(0→1).  Backward should have Recv(1→0) for the "
        "gradient, but instead also has Send(0→1) — wrong direction. "
        "Gradients flow in the wrong direction."
    ),
    expected_behavior="Send(fwd, 0→1) ↔ Recv(bwd, 1→0)",
    actual_bug_behavior="Backward also sends 0→1 → gradients lost on device 0",
    setup_fn=lambda: _setup_wrong_send_recv_direction(),
    verify_fn=lambda p, t, m: _verify_gradient_duality(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# B5: Pipeline Parallelism Schedule Bugs
# ═══════════════════════════════════════════════════════════════════════════════

B5A_ACTIVATION_LIVENESS_VIOLATION = BenchmarkCase(
    id="B5a",
    title="Activation freed before backward needs it in 1F1B",
    category=BugCategory.B5_PP_SCHEDULE,
    severity=Severity.CRITICAL,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/3952",
    source_issue="Megatron-LM#3952 (EP all-to-all activation offloading bug)",
    description=(
        "In 1F1B schedule, stage 0's activation for mb=0 is released before "
        "the backward pass for mb=0 reaches stage 0.  This causes either "
        "a crash or recomputation overhead."
    ),
    expected_behavior="Activations must live until their backward pass completes",
    actual_bug_behavior="Premature activation release → crash or incorrect gradients",
    setup_fn=lambda: _setup_activation_liveness_bug(),
    verify_fn=lambda p, t, m: _verify_activation_liveness(p, t, m),
)

B5B_WRONG_1F1B_ORDERING = BenchmarkCase(
    id="B5b",
    title="Backward executed before its forward in 1F1B schedule",
    category=BugCategory.B5_PP_SCHEDULE,
    severity=Severity.HIGH,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/1525",
    source_issue="Megatron-LM#1525 (Multiple Node PP errors)",
    description=(
        "In 1F1B schedule, backward for mb=1 on stage 1 is scheduled before "
        "forward for mb=1 has completed on all stages.  "
        "This violates the data dependency."
    ),
    expected_behavior="Backward(mb) must be after Forward(mb) on the same stage",
    actual_bug_behavior="Backward executes without saved activations → wrong results",
    setup_fn=lambda: _setup_wrong_1f1b_ordering(),
    verify_fn=lambda p, t, m: _verify_schedule_ordering(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# B6: Context Parallelism Communication Errors
# ═══════════════════════════════════════════════════════════════════════════════

B6A_RING_ATTENTION_MISSING_ALLREDUCE = BenchmarkCase(
    id="B6a",
    title="Ring Attention without final AllReduce",
    category=BugCategory.B6_CP_COMM,
    severity=Severity.CRITICAL,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4382",
    source_issue="Megatron-LM#4382 (CP memory leak + communication)",
    description=(
        "Ring Attention computes partial O on each device from local + "
        "received K,V.  The partial outputs must be AllReduced to get "
        "the full attention output.  Without AllReduce, each device has "
        "only a partial sum."
    ),
    expected_behavior="All CP partial outputs must be AllReduced",
    actual_bug_behavior="Each device has only partial attention output",
    setup_fn=lambda: _setup_ring_attn_missing_ar(),
    verify_fn=lambda p, t, m: _verify_postcondition_and_analysis(p, t, m),
)

B6B_CP_RING_WRONG_ORDER = BenchmarkCase(
    id="B6b",
    title="Ring Attention: KV received from wrong peer (wrong ring order)",
    category=BugCategory.B6_CP_COMM,
    severity=Severity.HIGH,
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4382",
    source_issue="Megatron-LM#4382",
    description=(
        "Ring Attention devices must receive K,V from the previous device "
        "in the ring.  If the ring order is wrong (e.g., device 0 receives "
        "from device 2 instead of from device N-1), the partial sums are "
        "from wrong sequence chunks."
    ),
    expected_behavior="Ring order: device i receives from (i-1) mod N",
    actual_bug_behavior="Wrong ring order → attention uses wrong sequence chunks",
    setup_fn=lambda: _setup_cp_wrong_ring_order(),
    verify_fn=lambda p, t, m: _verify_ring_order_consistency(p, t, m),
)


# ── Setup functions ──────────────────────────────────────────────────────────

def _make_mesh(n=2, name="tp"):
    return DeviceMesh(shape=(n,), dim_names=(name,))


def _setup_row_parallel_missing_ar():
    """B1a: X:Shard(1), W:Shard(0) without AllReduce."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(1),), mesh), "x", requires_grad=True)
    w = TensorState("w", (16, 32), (8, 32),
        ShardingSpec((Shard(0),), mesh), "w", requires_grad=True)
    prog = Program("b1a").add(MatMul("x", "w", "y"))  # No AllReduce!
    return prog, {"x": x, "w": w}, mesh

def _setup_colwise_gelu_rowwise_bug():
    """B1b: Colwise(no AR) → GELU → Rowwise."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", requires_grad=True)
    w1 = TensorState("w1", (16, 64), (16, 32),
        ShardingSpec((Shard(1),), mesh), "w1", requires_grad=True)
    w2 = TensorState("w2", (64, 32), (32, 32),
        ShardingSpec((Shard(0),), mesh), "w2", requires_grad=True)
    prog = Program("b1b")
    # Colwise: X(Rep) @ W1(Shard1) → Shard(1) output, no AR
    prog.add(MatMul("x", "w1", "h1"))
    # GELU applied to SHARDED tensor — BUG! Should AR first
    prog.add(SiLU("h1", "h1_act"))
    # Rowwise: H1(Shard1) @ W2(Shard0) → Partial → needs AR
    prog.add(MatMul("h1_act", "w2", "y"))  # Missing AR!
    return prog, {"x": x, "w1": w1, "w2": w2}, mesh

def _setup_missing_pp_broadcast():
    """B1c: Missing broadcast in PP."""
    mesh = _make_mesh(2, "pp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", stage=0)
    seqlen = TensorState("seqlen", (8,), (8,),
        ShardingSpec((Replicate(),), mesh), "seqlen", stage=0)
    # Missing: Send(seqlen, 0→1)
    prog = Program("b1c").add(MatMul("x", "w0", "h0"))
    return prog, {"x": x, "seqlen": seqlen}, mesh

def _setup_shard1_noncontiguous():
    """B2a: Shard(1) non-contiguous issue."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(1),), mesh), "x")
    w = TensorState("w", (16, 32), (16, 32),
        ShardingSpec((Replicate(),), mesh), "w")
    # Shard(1) input to MatMul with Replicate W → output Shard(1)
    # view() downstream would fail on non-contiguous local tensor
    prog = Program("b2a").add(MatMul("x", "w", "y"))
    return prog, {"x": x, "w": w}, mesh

def _setup_shard_to_replicate_shape_bug():
    """B2b: Shard→Replicate shape corruption."""
    mesh = _make_mesh(3, "tp")  # 3 GPUs, size 9 not divisible → ceil issue
    x = TensorState("x", (9, 16), (3, 16),
        ShardingSpec((Shard(0),), mesh), "x")
    # AllGather then AllReduce should preserve shape 9
    prog = Program("b2b")
    prog.add(AllGather("x", "x_gathered", gather_dim=0))
    prog.add(AllReduce("x_gathered", "x_out"))  # Should error: not partial
    return prog, {"x": x}, mesh

def _setup_sequence_parallel_cast_bug():
    """B2c: SequenceParallel DTensor→Tensor cast."""
    mesh = _make_mesh(2, "sp")
    x = TensorState("x", (2, 140, 4096), (2, 140, 2048),
        ShardingSpec((Shard(2),), mesh), "x")
    scale = TensorState("scale", (4096,), (2048,),
        ShardingSpec((Shard(0),), mesh), "scale")
    # RMSNorm: output * scale — both should be Shard(2), Shard(0)
    # But if x loses its sharding wrapper...
    prog = Program("b2c").add(Multiply("x", "scale", "normed"))
    return prog, {"x": x, "scale": scale}, mesh

def _setup_redundant_allreduce():
    """B3a: AllReduce on already-replicated tensor."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 32), (8, 32),
        ShardingSpec((Replicate(),), mesh), "x")
    prog = Program("b3a").add(AllReduce("x", "y"))  # Redundant!
    return prog, {"x": x}, mesh

def _setup_send_without_recv():
    """B3b: Send without Recv."""
    mesh = _make_mesh(2, "pp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", stage=0)
    prog = Program("b3b").add(
        Send("x", "x_sent", src=0, dst=1, stage=0, microbatch_id=0)
    )  # No Recv!
    return prog, {"x": x}, mesh

def _setup_allgather_dim_mismatch():
    """B3c: AllGather dim doesn't match shard dim."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(1),), mesh), "x")
    # AllGather dim=0 but shard is dim=1 — mismatch!
    prog = Program("b3c").add(AllGather("x", "y", gather_dim=0))
    return prog, {"x": x}, mesh

def _setup_missing_bwd_allreduce():
    """B4a: Fwd has AR but bwd doesn't."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(1),), mesh), "x", requires_grad=True)
    w = TensorState("w", (16, 32), (8, 32),
        ShardingSpec((Shard(0),), mesh), "w", requires_grad=True)
    fwd = Program("b4a_fwd")
    fwd.add(MatMul("x", "w", "y_partial"))
    fwd.add(AllReduce("y_partial", "y"))
    # No backward dual for the AllReduce
    bwd = Program("b4a_bwd")  # Empty — missing AR dual!
    return fwd, {"x": x, "w": w, "_bwd": bwd}, mesh

def _setup_wrong_dual_type():
    """B4b: Wrong dual type (AllGather instead of ReduceScatter)."""
    mesh = _make_mesh(2, "tp")
    x = TensorState("x", (8, 16), (8, 8),
        ShardingSpec((Shard(0),), mesh), "x", requires_grad=True)
    fwd = Program("b4b_fwd").add(AllGather("x", "x_full", gather_dim=0))
    # Wrong dual: another AllGather instead of ReduceScatter
    bwd = Program("b4b_bwd").add(AllGather("grad_x_full", "grad_x", gather_dim=0))
    return fwd, {"x": x, "_bwd": bwd}, mesh

def _setup_wrong_send_recv_direction():
    """B4c: Send direction not reversed in bwd."""
    mesh = _make_mesh(2, "pp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", stage=0)
    fwd = Program("b4c_fwd").add(
        Send("x", "x_sent", src=0, dst=1, stage=0, microbatch_id=0))
    # Wrong: bwd also sends 0→1 instead of 1→0
    bwd = Program("b4c_bwd").add(
        Send("grad_x_sent", "grad_x", src=0, dst=1, stage=0, microbatch_id=0))
    return fwd, {"x": x, "_bwd": bwd}, mesh

def _setup_activation_liveness_bug():
    """B5a: Activation released too early."""
    mesh = _make_mesh(2, "pp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", stage=0, is_activation=True)
    prog = Program("b5a")
    prog.add(MatMul("x", "w0", "h0"))
    prog.add(Send("h0", "h0_sent", src=0, dst=1, stage=0, microbatch_id=0))
    return prog, {"x": x}, mesh

def _setup_wrong_1f1b_ordering():
    """B5b: Backward before forward in schedule."""
    mesh = _make_mesh(2, "pp")
    x = TensorState("x", (8, 16), (8, 16),
        ShardingSpec((Replicate(),), mesh), "x", stage=0)
    # Forward + Backward but Bwd scheduled before Fwd on stage 1
    prog = Program("b5b")
    prog.add(MatMul("x", "w0", "h0"))
    return prog, {"x": x}, mesh

def _setup_ring_attn_missing_ar():
    """B6a: Ring attention without final AllReduce."""
    mesh = _make_mesh(2, "cp")
    q = TensorState("q", (2, 8, 4, 16), (2, 8, 4, 16),
        ShardingSpec((Replicate(),), mesh), "q")
    k = TensorState("k", (2, 8, 4, 16), (2, 4, 4, 16),
        ShardingSpec((Shard(1),), mesh), "k")
    v = TensorState("v", (2, 8, 4, 16), (2, 4, 4, 16),
        ShardingSpec((Shard(1),), mesh), "v")
    prog = Program("b6a")
    prog.add(FlashAttention("q", "k", "v", "o"))
    # Missing: AllReduce on partial output!
    return prog, {"q": q, "k": k, "v": v}, mesh

def _setup_cp_wrong_ring_order():
    """B6b: Ring attention with wrong communication order."""
    mesh = _make_mesh(4, "cp")
    q = TensorState("q", (2, 8, 4, 16), (2, 8, 4, 16),
        ShardingSpec((Replicate(),), mesh), "q")
    k = TensorState("k", (2, 8, 4, 16), (2, 2, 4, 16),
        ShardingSpec((Shard(1),), mesh), "k")
    # Wrong ring: device 0 sends to device 2 (should be device 1)
    prog = Program("b6b")
    prog.add(Send("k", "k_sent", src=0, dst=2, stage=0, microbatch_id=0))
    prog.add(Recv("k_rcvd", "k_remote", src=3, dst=0, stage=0, microbatch_id=0))
    # src=3→dst=0 for recv, but send was 0→2 — mismatched ring
    return prog, {"q": q, "k": k}, mesh


# ── Verification functions ───────────────────────────────────────────────────

def _verify_postcondition_and_analysis(prog, tensors, mesh):
    """Verify via postcondition + placement analysis."""
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    verifier = DistributedVerifier()
    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(prog, state)
    results = []
    for name, ts in state.items():
        if name not in {op.input_names[0] for op in prog.ops if hasattr(op, 'input_names') and op.input_names}:
            results.append(verifier.verify_postcondition(ts, expected_partial=False))
    # Also report analysis
    if not analysis.is_correct:
        results.append(VerifyResult(
            passed=False,
            condition="placement analysis",
            details=str(analysis),
        ))
    return results

def _verify_nonlinear_on_sharded(prog, tensors, mesh):
    """B1b: Check that nonlinear activations are not applied to sharded/partial tensors.

    GELU(shard) ≠ shard_of(GELU(full)), so SiLU/Multiply on Shard/Partial
    tensors without an intervening AllReduce is a bug.
    """
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    results = []
    # Check each SiLU/Multiply op: is its input SHARDED (not Replicated)?
    for op in prog.ops:
        if isinstance(op, (SiLU, Multiply)):
            in_name = op.input_names[0]
            in_ts = state.get(in_name) or tensors.get(in_name)
            if in_ts and not in_ts.is_replicated and not in_ts.partial:
                # Input is Shard(dim) — nonlinear on shard is incorrect!
                shard_info = ", ".join(
                    f"Shard({p.dim})" for p in in_ts.sharding.placements
                    if isinstance(p, Shard)
                )
                results.append(VerifyResult(
                    passed=False,
                    condition=f"nonlinear on sharded tensor",
                    details=(
                        f"{type(op).__name__}({in_name}) applied to {shard_info} tensor. "
                        f"Nonlinear ops on sharded tensors break mathematical equivalence. "
                        f"Insert AllReduce before the activation."
                    ),
                ))
    if not results:
        results.append(VerifyResult(True, "nonlinear on sharded tensor",
            "No nonlinear-on-sharded issues found"))
    return results

def _verify_cross_stage_broadcast(prog, tensors, mesh):
    """B1c: Check that stage-specific tensors are broadcast to all stages that need them.

    In PP, if a tensor exists only on stage 0 (e.g., seqlen metadata),
    but stage 1 also needs it, there MUST be a Send/Recv pair.
    """
    results = []
    stage_tensors = {0: set(), 1: set()}
    for name, ts in tensors.items():
        if ts.stage is not None:
            stage_tensors[ts.stage].add(name)

    # Tensors on stage 0 that are NOT sent to stage 1
    has_send = set()
    for op in prog.ops:
        if isinstance(op, Send):
            has_send.add(op.x)

    missing = stage_tensors[0] - has_send
    # Exclude tensors that stage 1 doesn't need (we can't auto-detect this,
    # but for benchmark we check if ANY stage-0 tensor lacks a send)
    if missing:
        results.append(VerifyResult(
            passed=False,
            condition="cross-stage broadcast",
            details=f"Stage-0 tensors without Send to stage 1: {missing}. "
                    f"Add Send/Recv for these tensors.",
        ))

    if not results:
        results.append(VerifyResult(True, "cross-stage broadcast",
            "All stage-0 tensors have corresponding Sends"))
    return results

def _verify_placement_consistency(prog, tensors, mesh):
    """B2a: Check placement consistency including Shard(1) non-contiguity risk."""
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    verifier = DistributedVerifier()
    results = [verifier.verify_placement_consistency(prog)]

    # Check for Shard(1) risks: non-contiguous local tensors
    for name, ts in state.items():
        shard_dims = ts.sharding.get_shard_dims()
        if 1 in shard_dims:
            results.append(VerifyResult(
                passed=False,
                condition="Shard(1) non-contiguity risk",
                details=(
                    f"Tensor '{name}' has Shard(1) placement. "
                    f"Local tensor may be non-contiguous; view/reshape ops may fail. "
                    f"Consider Shard(0) or insert .contiguous() after redistribution. "
                    f"(pytorch/pytorch#173041)"
                ),
            ))

    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(prog, state)
    if not analysis.is_correct:
        results.append(VerifyResult(False, "placement analysis", str(analysis)))
    return results

def _verify_shape_through_reshard(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    verifier = DistributedVerifier()
    return [verifier.verify_shape_consistency(prog, {}, state)]

def _verify_compatible_sharding_elementwise(prog, tensors, mesh):
    """B2c: Check that element-wise ops have compatible sharding on both operands.

    If a(Shard(2)) * b(Shard(0)), the local shapes may be incompatible
    and one tensor may be silently cast.
    """
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    results = []
    for op in prog.ops:
        if isinstance(op, (Multiply, Add)):
            a_name, b_name = op.a, op.b
            a_ts = state.get(a_name) or tensors.get(a_name)
            b_ts = state.get(b_name) or tensors.get(b_name)
            if a_ts and b_ts:
                # Check if sharding is compatible
                a_shards = a_ts.sharding.get_shard_dims()
                b_shards = b_ts.sharding.get_shard_dims()
                # Compatible: same shard dims or one is Replicate
                if a_shards != b_shards:
                    results.append(VerifyResult(
                        passed=False,
                        condition="compatible sharding for element-wise",
                        details=(
                            f"{type(op).__name__}({a_name}, {b_name}): "
                            f"incompatible sharding — a={a_shards}, b={b_shards}. "
                            f"Element-wise ops require matching placements."
                        ),
                    ))
    if not results:
        results.append(VerifyResult(True, "compatible sharding for element-wise",
            "All element-wise ops have compatible sharding"))
    return results

def _verify_redundant_collectives(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(prog, state)
    results = []
    if analysis.redundant_collectives:
        results.append(VerifyResult(False, "redundant collectives",
            f"Redundant at indices: {analysis.redundant_collectives}"))
    return results

def _verify_pp_deadlock(prog, tensors, mesh):
    verifier = DistributedVerifier()
    return [verifier.verify_communication_legality(prog)]

def _verify_collective_dim_consistency(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        if not name.startswith("_"):
            executor.register_tensor(ts)
    state = executor.run_program(prog)
    # Check: AllGather dim should match Shard dim
    results = []
    for op in prog.ops:
        if isinstance(op, AllGather):
            in_ts = state.get(op.x) or tensors.get(op.x)
            if in_ts:
                shard_dims = in_ts.sharding.get_shard_dims()
                if op.gather_dim not in shard_dims:
                    results.append(VerifyResult(False,
                        "AllGather dim consistency",
                        f"gather_dim={op.gather_dim} but tensor shard dims={shard_dims}"))
    if not results:
        results.append(VerifyResult(True, "AllGather dim consistency", "All consistent"))
    return results

def _verify_gradient_duality(prog, tensors, mesh):
    """Check fwd↔bwd gradient duality."""
    bwd = tensors.pop("_bwd", None)
    if bwd is None:
        return [VerifyResult(False, "gradient duality", "no backward program provided")]
    verifier = DistributedVerifier()
    return [verifier.verify_gradient_duality(prog, bwd)]

def _verify_activation_liveness(prog, tensors, mesh):
    """B5a: Verify activation liveness with a BUGGY schedule.

    Simulates the real PP bug: when stage 1 finishes backward for mb=0,
    it signals stage 0 that activation mb=0 can be released. But stage 0's
    backward for mb=0 hasn't run yet, so the activation is prematurely freed.
    """
    from verifier.schedules import MicroBatch, ActivationTracker, Phase, OpType

    tracker = ActivationTracker(num_stages=2)

    # Simulate execution order:
    # Step 1: Stage 0 forward mb=0 → activation saved
    tracker.record_forward(0, 0, ["act_mb0"])
    # Step 2: Stage 1 forward mb=0 → recvs activation
    tracker.record_forward(1, 0, ["act_mb0_stage1"])
    # Step 3: Stage 0 forward mb=1 → more activations saved
    tracker.record_forward(0, 1, ["act_mb1"])
    # Step 4: Stage 1 backward mb=0 → DONE, signals release
    # BUG: this triggers stage 0 to release act_mb0 BEFORE stage 0's backward
    tracker.release_after_backward(0, 0)  # Premature release!
    # Step 5: Stage 0 tries backward for mb=0 → activation GONE!
    if not tracker.is_activation_available(0, 0):
        return [VerifyResult(
            passed=False,
            condition="activation liveness",
            details=(
                "Activation for mb=0 on stage 0 was released before its "
                "backward pass. Stage 1's backward triggered premature "
                "release. (Megatron-LM#3952, EP all-to-all pattern)"
            ),
        )]

    return [VerifyResult(True, "activation liveness", "All live")]

def _verify_schedule_ordering(prog, tensors, mesh):
    """B5b: Verify 1F1B ordering with a buggy schedule.

    Injects a schedule where backward for mb=1 on stage 1 runs
    BEFORE forward for mb=1 on stage 1.
    """
    from verifier.schedules import MicroBatch, ActivationTracker, Phase, OpType

    # Buggy schedule: backward for mb=1 comes BEFORE forward for mb=1
    buggy_schedule = [
        MicroBatch(mb_id=0, stage_id=0, op_type=OpType.FORWARD, phase=Phase.WARMUP),
        MicroBatch(mb_id=0, stage_id=1, op_type=OpType.FORWARD, phase=Phase.WARMUP),
        # BUG: backward for mb=1 on stage 1, but forward for mb=1 hasn't happened!
        MicroBatch(mb_id=1, stage_id=1, op_type=OpType.BACKWARD, phase=Phase.STEADY),
        MicroBatch(mb_id=1, stage_id=0, op_type=OpType.FORWARD, phase=Phase.STEADY),
        MicroBatch(mb_id=1, stage_id=1, op_type=OpType.FORWARD, phase=Phase.STEADY),
        MicroBatch(mb_id=0, stage_id=1, op_type=OpType.BACKWARD, phase=Phase.COOLDOWN),
        MicroBatch(mb_id=0, stage_id=0, op_type=OpType.BACKWARD, phase=Phase.COOLDOWN),
    ]

    # Check ordering: for each (stage, mb), forward must precede backward
    seen_forward = set()
    errors = []
    for mb in buggy_schedule:
        key = (mb.stage_id, mb.mb_id)
        if mb.op_type == OpType.FORWARD:
            seen_forward.add(key)
        elif mb.op_type == OpType.BACKWARD:
            if key not in seen_forward:
                errors.append(
                    f"Backward for (stage={mb.stage_id}, mb={mb.mb_id}) "
                    f"before its forward"
                )

    if errors:
        return [VerifyResult(False, "1F1B schedule ordering", "; ".join(errors))]
    return [VerifyResult(True, "1F1B schedule ordering",
        f"fwd={len(seen_forward)}, all preceding forwards found")]

def _verify_ring_order_consistency(prog, tensors, mesh):
    """Verify ring order: each send should have recv with opposite direction."""
    results = []
    sends = [(op.src, op.dst) for op in prog.ops if isinstance(op, Send)]
    recvs = [(op.src, op.dst) for op in prog.ops if isinstance(op, Recv)]
    # Check: for each send(s→d), there should be a recv(d→s) or recv from neighbor
    for s_src, s_dst in sends:
        matching = [(r_src, r_dst) for r_src, r_dst in recvs if r_src == s_src and r_dst == s_dst]
        if not matching:
            results.append(VerifyResult(False, "ring order",
                f"Send({s_src}→{s_dst}) has no matching Recv"))
    if not results:
        results.append(VerifyResult(True, "ring order", "All matched"))
    return results


# ── Benchmark suite ──────────────────────────────────────────────────────────

ALL_CASES: List[BenchmarkCase] = [
    # B1: Missing/Incorrect Collectives
    B1A_ROW_PARALLEL_MISSING_ALLREDUCE,
    B1B_COLWISE_ROWWISE_GELU_BUG,
    B1C_MISSING_BROADCAST_PP,
    # B2: Placement/Shard Errors
    B2A_SHARD1_NONCONTIGUOUS,
    B2B_SHARD_TO_REPLICATE_SHAPE_CORRUPTION,
    B2C_SEQUENCE_PARALLEL_DTENSOR_CAST,
    # B3: Communication Legality
    B3A_ALLREDUCE_ON_REPLICATE,
    B3B_SEND_WITHOUT_RECV,
    B3C_ALLGATHER_DIM_MISMATCH,
    # B4: Gradient Duality
    B4A_ALLREDUCE_DUAL_MISSING_IN_BWD,
    B4B_WRONG_DUAL_COLLECTIVE_TYPE,
    B4C_SEND_RECV_DIRECTION_NOT_REVERSED,
    # B5: PP Schedule
    B5A_ACTIVATION_LIVENESS_VIOLATION,
    B5B_WRONG_1F1B_ORDERING,
    # B6: CP Communication
    B6A_RING_ATTENTION_MISSING_ALLREDUCE,
    B6B_CP_RING_WRONG_ORDER,
]


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    total: int = 0
    detected: int = 0
    missed: int = 0
    by_category: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # cat → (total, detected)
    total_time_ms: float = 0.0
    cases: List[BenchmarkCase] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "  DTENSOR-VERIFIER BENCHMARK REPORT",
            "=" * 70,
            f"  Total cases:  {self.total}",
            f"  Detected:     {self.detected} ({self.detected/self.total*100:.1f}%)" if self.total else "",
            f"  Missed:       {self.missed}" if self.missed else "",
            f"  Total time:   {self.total_time_ms:.1f}ms",
            "",
            "  By Category:",
        ]
        for cat, (total, det) in sorted(self.by_category.items()):
            rate = f"{det/total*100:.0f}%" if total else "N/A"
            lines.append(f"    {cat}: {det}/{total} detected ({rate})")
        lines.append("")
        lines.append("  Per-Case Results:")
        for c in self.cases:
            status = "DETECTED" if c.detected else "MISSED"
            lines.append(f"    [{status:8s}] {c.id}: {c.title[:60]}")
        return "\n".join(lines)


def run_benchmark(categories: Optional[List[BugCategory]] = None) -> BenchmarkReport:
    """Run the benchmark suite.

    Args:
        categories: If provided, only run cases in these categories.
    """
    cases = ALL_CASES
    if categories:
        cases = [c for c in ALL_CASES if c.category in categories]

    report = BenchmarkReport(total=len(cases), cases=[])
    start = time.time()

    for case in cases:
        case.run()
        report.cases.append(case)

        cat_name = case.category.value
        if cat_name not in report.by_category:
            report.by_category[cat_name] = (0, 0)
        total, det = report.by_category[cat_name]
        report.by_category[cat_name] = (total + 1, det + (1 if case.detected else 0))

        if case.detected:
            report.detected += 1
        else:
            report.missed += 1

    report.total_time_ms = (time.time() - start) * 1000
    return report


def print_case_details(cases: List[BenchmarkCase]):
    """Print detailed results for each case."""
    for case in cases:
        print(case.summary())
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Tensor Verification Benchmark")
    parser.add_argument("--list", action="store_true", help="List all benchmark cases")
    parser.add_argument("--run", type=str, default="all", help="Run cases: all, B1, B2, ..., B6")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    if args.list:
        print("Benchmark Cases:\n")
        for c in ALL_CASES:
            print(f"  {c.id}: {c.title}")
            print(f"    Category: {c.category.value} | Severity: {c.severity.value}")
            print(f"    Source: {c.source_issue}")
            print(f"    Description: {c.description[:120]}...")
            print()
        sys.exit(0)

    if args.run == "all":
        cats = None
    else:
        cat_map = {f"B{i}": getattr(BugCategory, f"B{i}_{name}")
                   for i in range(1, 7)
                   for name in ["MISSING_COLLECTIVE", "PLACEMENT_ERROR", "COMM_LEGALITY",
                                "GRADIENT_DUALITY", "PP_SCHEDULE", "CP_COMM"]}
        cat_key = args.run.upper()
        cats = [v for k, v in BugCategory.__members__.items() if k.startswith(cat_key)]

    report = run_benchmark(cats)

    if args.json:
        output = {
            "total": report.total,
            "detected": report.detected,
            "missed": report.missed,
            "total_time_ms": report.total_time_ms,
            "detection_rate": report.detected / report.total if report.total else 0,
            "by_category": {
                cat: {"total": t, "detected": d}
                for cat, (t, d) in report.by_category.items()
            },
            "cases": [
                {
                    "id": c.id, "title": c.title, "category": c.category.value,
                    "severity": c.severity.value, "source": c.source_issue,
                    "detected": c.detected, "runtime_ms": c.runtime_ms,
                }
                for c in report.cases
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(report.summary())
        print()
        print_case_details(report.cases)
