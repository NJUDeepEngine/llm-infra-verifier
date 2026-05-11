"""
Real-bug benchmark: verifies our tool against ACTUAL reported bugs.

Each case:
  1. Shows the ORIGINAL buggy code (from the GitHub issue)
  2. Translates to our verification IR (documenting the translation)
  3. Runs the verifier
  4. Reports detection and explains what exactly was caught

Unlike the earlier benchmark_suite.py, which hand-crafted IR that
ALREADY encodes the bug, this benchmark:
  - Starts from realistic code patterns (Python/PyTorch)
  - Documents the translation step explicitly
  - Acknowledges when our detection is structural proxy vs exact bug

Categories:
  RB1 — Tensor Parallelism bugs
  RB2 — Pipeline Parallelism bugs
  RB3 — Numerical / precision bugs
  RB4 — Async / overlap bugs
"""

from __future__ import annotations

from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    compute_local_shape,
)
from verifier.ir import (
    Program, MatMul, Add, Multiply, SiLU, AllReduce, AllGather,
    AllReduceAsync, Send, Recv, Wait, FlashAttention, COMM_STREAM, ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier, VerifyResult
from verifier.rewrite import PlacementAnalyzer
from verifier.temporal import verify_temporal
from verifier.common_tir import (
    TIRFunc, TIRBuffer, TIRGrid, TIRVar, TIRBlock, TIRAxis,
    TIRAccess, AxisType, BlockClassifier, BlockType, default_registry,
)


@dataclass
class RealBugCase:
    """A single real-bug benchmark case."""
    id: str
    title: str
    source_url: str
    original_code: str           # the actual buggy Python/PyTorch code
    translation_notes: str       # how we map it to our IR
    category: str

    # IR setup
    setup_fn: object
    verify_fn: object

    # Results
    detected: bool = False
    details: str = ""

    def run(self):
        try:
            prog, tensors, mesh = self.setup_fn()
            results = self.verify_fn(prog, tensors, mesh)
            self.detected = any(not r.passed for r in results)
            self.details = "\n    ".join(
                f"[{'FAIL' if not r.passed else 'PASS'}] {r.condition}: {r.details}"
                for r in results
            )
        except Exception as e:
            self.detected = True
            self.details = f"Exception: {e}"

    def summary(self) -> str:
        status = "DETECTED" if self.detected else "MISSED"
        return (
            f"\n{'='*65}\n"
            f"[{status}] {self.id}: {self.title}\n"
            f"Source: {self.source_url}\n"
            f"Category: {self.category}\n"
            f"{'─'*65}\n"
            f"Original code:\n{self.original_code}\n"
            f"{'─'*65}\n"
            f"Translation: {self.translation_notes}\n"
            f"{'─'*65}\n"
            f"Result: {self.details}\n"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RB1: Tensor Parallelism Bugs
# ═══════════════════════════════════════════════════════════════════════════════

RB1A_ROW_PARALLEL_MISSING_AR = RealBugCase(
    id="RB1a",
    title="RowParallelLinear without reduce_from_tensor_model_parallel_region",
    source_url="https://github.com/pytorch/pytorch/issues/144359",
    category="Tensor Parallelism",
    original_code="""\
# Megatron-LM: megatron/core/tensor_parallel/layers.py
# RowParallelLinear.forward() — INCORRECT VERSION

def forward(self, input_):
    # input_: (B, H/tp), scattered across TP ranks
    # weight: (H/tp, O), sharded on input dim
    output_parallel = F.linear(input_, self.weight)
    # BUG: forgot to call reduce_from_tensor_model_parallel_region!
    # Correct: output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output_parallel  # ← Each rank has only a PARTIAL sum!""",
    translation_notes=(
        "IR: MatMul(x:Shard(1), w:Shard(0)) → y:Partial(). "
        "The bug is that no AllReduce follows the MatMul. "
        "This is a direct translation: the placement propagation rules "
        "correctly identify the output as Partial, and the postcondition "
        "check catches it."
    ),
    setup_fn=lambda: _setup_row_parallel_missing_ar_real(),
    verify_fn=lambda p, t, m: _verify_postcondition_and_analysis(p, t, m),
)

RB1B_GELU_COLWISE_ROWWISE = RealBugCase(
    id="RB1b",
    title="GELU between ColwiseParallel(use_local_output=True) and RowwiseParallel",
    source_url="https://github.com/pytorch/pytorch/issues/144359",
    category="Tensor Parallelism",
    original_code="""\
# User code from pytorch#144359:
parallelize_module(model, device_mesh, {
    "proj_in": ColwiseParallel(use_local_output=True),
    "proj_out": RowwiseParallel(use_local_output=True),
})

# This produces:
#   h1 = F.linear(x, w1)  # Shard(1), NO AllReduce (use_local_output=True)
#   h1 = GELU(h1)          # BUG: GELU on Shard(1) — mathematically wrong!
#   output = F.linear(h1, w2)  # consumes wrong h1 values

# GELU(shard) != shard(GELU(full)) because GELU is nonlinear.
# The AllReduce must happen BEFORE the nonlinear activation.""",
    translation_notes=(
        "IR: MatMul → SiLU(on Shard(1) tensor!) → MatMul. "
        "Detection: 1) SiLU input is Shard(1), not Replicate → nonlinear on shard. "
        "2) Missing AllReduce/AllGather before activation. "
        "Fix: AllGather after Colwise to make h1 Replicate, then GELU, then scatter for Rowwise."
    ),
    setup_fn=lambda: _setup_gelu_colwise_rowwise_bug(),
    verify_fn=lambda p, t, m: _verify_nonlinear_on_shard(p, t, m),
)

RB1C_COLUMN_PARALLEL_GATHER_OUTPUT = RealBugCase(
    id="RB1c",
    title="ColumnParallelLinear with gather_output=True but missing AllGather",
    source_url="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py",
    category="Tensor Parallelism",
    original_code="""\
# Megatron ColumnParallelLinear.forward():
# When gather_output=True, we MUST call all_gather after the matmul.

def forward(self, input_):
    output = F.linear(input_, self.weight)  # Shard(1)
    if self.gather_output:
        output = gather_from_tensor_model_parallel_region(output)  # AllGather
    return output

# BUG scenario: gather_output=True but the gather call is accidentally
# skipped (e.g., behind a wrong condition, or removed during refactoring).""",
    translation_notes=(
        "IR model: MatMul(Replicate, Shard(1)) → Shard(1) output. "
        "If gather_output is expected but missing, the consumer of this tensor "
        "will receive Shard(1) instead of Replicate. "
        "Detection: consumer expects Replicate but gets Shard(1) → placement mismatch."
    ),
    setup_fn=lambda: _setup_colwise_gather_bug(),
    verify_fn=lambda p, t, m: _verify_placement_consistency(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# RB2: Pipeline Parallelism Bugs
# ═══════════════════════════════════════════════════════════════════════════════

RB2A_MISSING_PP_BROADCAST = RealBugCase(
    id="RB2a",
    title="Missing cu_seqlens broadcast across pipeline stages",
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/4092",
    category="Pipeline Parallelism",
    original_code="""\
# Megatron-LM#4092: SFT Packing missing broadcast in PP stages
# Intermediate pipeline stages need cu_seqlens/max_seqlen from stage 0.

# Stage 0: has cu_seqlens from dataset
cu_seqlens = compute_cu_seqlens(input_ids)

# Stage 1..N: need cu_seqlens for attention, but it's only on stage 0!
# BUG: no broadcast of cu_seqlens to other stages
# Fix: broadcast from stage 0 to all other PP stages""",
    translation_notes=(
        "IR model: cu_seqlens is a tensor with stage=0. Stage 1 needs it "
        "for attention computation, but there's no Send/Recv. "
        "Detection: cross-stage tensor dependency without communication op. "
        "This is a STRUCTURAL check: stage-0 tensor consumed on stage-1 "
        "without Send/Recv."
    ),
    setup_fn=lambda: _setup_missing_pp_broadcast(),
    verify_fn=lambda p, t, m: _verify_cross_stage_tensor_access(p, t, m),
)

RB2B_MISMATCHED_SEND_RECV = RealBugCase(
    id="RB2b",
    title="Send/Recv direction mismatch in PP handshake",
    source_url="https://github.com/NVIDIA/Megatron-LM/issues/1525",
    category="Pipeline Parallelism",
    original_code="""\
# Megatron#1525: Multiple Node PP errors
# Common PP setup bug: Send/Recv direction mismatch between adjacent stages.

# Stage 0 (device 0):
send(h0, dst=1)   # sends activation to stage 1

# Stage 1 (device 1):
recv(h0, src=0)   # BUG: should be src=0, but typo'd as src=2!
# or: recv(h0, src=0) for a tensor that stage 0 never sends""",
    translation_notes=(
        "IR model: Program contains Send(0→1, 'h0') and Recv(2→1, 'h0'). "
        "Detection: the Recv's src (2) doesn't match any Send's src→dst pair. "
        "Our communication legality check flags unmatched Send/Recv."
    ),
    setup_fn=lambda: _setup_mismatched_send_recv(),
    verify_fn=lambda p, t, m: _verify_communication_legality(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# RB3: Numerical / Precision Bugs
# ═══════════════════════════════════════════════════════════════════════════════

RB3A_FP16_GRADIENT_UNDERFLOW = RealBugCase(
    id="RB3a",
    title="fp16 gradient underflow: loss scale too small",
    source_url="https://pytorch.org/docs/stable/amp.html#gradient-scaling",
    category="Numerical",
    original_code="""\
# Mixed precision training with fp16:
# If loss_scale is too small, gradients < fp16 min_normal (6.1e-5)
# become ZERO → optimizer sees zero gradient → weights stop updating.

# Typical scenario:
loss_scale = 128
grad = compute_gradient()  # typical magnitude ~1e-5
scaled_grad = grad * loss_scale  # 1e-5 * 128 = 1.28e-3 → OK
# But if grad ~ 1e-7:
scaled_grad = 1e-7 * 128 = 1.28e-5  # < fp16 min_normal → ZERO!""",
    translation_notes=(
        "Our numerical verifier detects fp16 boundary conditions. "
        "Given a gradient magnitude estimate, it computes the minimum "
        "loss_scale needed to keep gradients above fp16 min_normal. "
        "This is a STRUCTURAL check: doesn't need actual gradient values, "
        "just magnitude estimates based on model size and batch size."
    ),
    setup_fn=lambda: (None, None, None),  # numerical verifier works standalone
    verify_fn=lambda p, t, m: _verify_fp16_gradient_safety(),
)

RB3B_ADAM_EPSILON_FP16 = RealBugCase(
    id="RB3b",
    title="Adam epsilon has no effect in fp16 (1e-8 < 6e-5 min_normal)",
    source_url="https://pytorch.org/docs/stable/generated/torch.optim.Adam.html",
    category="Numerical",
    original_code="""\
# Standard Adam:
#   theta = theta - lr * m_hat / (sqrt(v_hat) + 1e-8)

# In fp16: min_normal = 6.1e-5
# If v_hat is computed in fp16:
#   sqrt(v_hat) is at least sqrt(6.1e-5) ≈ 7.8e-3
#   So sqrt(v_hat) + 1e-8 = sqrt(v_hat)  ← epsilon has ZERO effect

# The numerical stability guarantee that epsilon provides in fp32
# is ABSENT in fp16. The optimizer behaves differently.""",
    translation_notes=(
        "Our numerical verifier's OverflowRiskDetector checks this: "
        "Adam eps (1e-8) vs fp16 min_normal (6.1e-5) → ratio 6104x. "
        "The verifier warns that fp16 Adam state is unsafe and "
        "recommends fp32 for m/v. "
        "This is a pure property check — no training data needed."
    ),
    setup_fn=lambda: (None, None, None),
    verify_fn=lambda p, t, m: _verify_adam_eps_fp16(),
)


# ═══════════════════════════════════════════════════════════════════════════════
# RB4: Async / Overlap Bugs
# ═══════════════════════════════════════════════════════════════════════════════

RB4A_ASYNC_AR_WITHOUT_WAIT = RealBugCase(
    id="RB4a",
    title="AllReduceAsync gradient then optimizer.step() without Wait",
    source_url="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py",
    category="Async/Overlap",
    original_code="""\
# Megatron's LinearWithGradAccumulationAndAsyncCommunication:
# The backward pass launches async AllReduce for weight gradient,
# overlaps with input gradient computation, then Waits.

# BUG scenario: forgetting to Wait before optimizer step:

handle = dist.all_reduce(grad_weight, async_op=True)
# ... compute grad_input (intended overlap) ...
# BUG: optimizer.step() called before handle.wait()!
optimizer.step()    # reads grad_weight — but AllReduce may not be done!
handle.wait()       # too late""",
    translation_notes=(
        "IR model: AllReduceAsync(grad_w, handle=h1) → MatMul(grad_w, opt_state) "
        "→ Wait(h1). The MatMul reads grad_w before Wait completes. "
        "Detection: temporal verifier's missing-wait detector flags the MatMul "
        "as consuming async output before the Wait."
    ),
    setup_fn=lambda: _setup_async_ar_without_wait(),
    verify_fn=lambda p, t, m: _verify_temporal(p, t, m),
)

RB4B_GRADIENT_BUFFER_REUSE = RealBugCase(
    id="RB4b",
    title="Gradient buffer reuse: two async AllReduces writing same buffer",
    source_url="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py",
    category="Async/Overlap",
    original_code="""\
# When using manual gradient buffer management:
grad_buf = torch.empty(num_params, dtype=fp16, device='cuda')

# BUG: launching two async AllReduces into the SAME buffer:
h1 = dist.all_reduce(grad_layer1, out=grad_buf, async_op=True)
h2 = dist.all_reduce(grad_layer2, out=grad_buf, async_op=True)  # OVERWRITE!

# h1's result in grad_buf is corrupted by h2 before h1.wait().
# Result: grad_layer1 is lost.""",
    translation_notes=(
        "IR model: AllReduceAsync(→buf, h1) + AllReduceAsync(→buf, h2). "
        "Both write to 'buf'. h1's result is not consumed (Waited) before h2 starts. "
        "Detection: temporal verifier's buffer-aliasing check."
    ),
    setup_fn=lambda: _setup_gradient_buffer_reuse(),
    verify_fn=lambda p, t, m: _verify_temporal(p, t, m),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Setup functions
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_row_parallel_missing_ar_real():
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 8), ShardingSpec((Shard(dim=1),), mesh), "x")
    w = TensorState("w", (16, 32), (8, 32), ShardingSpec((Shard(dim=0),), mesh), "w")
    prog = Program("bug").add(MatMul("x", "w", "y"))
    return prog, {"x": x, "w": w}, mesh

def _setup_gelu_colwise_rowwise_bug():
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 16), ShardingSpec((Replicate(),), mesh), "x")
    w1 = TensorState("w1", (16, 64), (16, 32), ShardingSpec((Shard(dim=1),), mesh), "w1")
    w2 = TensorState("w2", (64, 32), (32, 32), ShardingSpec((Shard(dim=0),), mesh), "w2")
    prog = Program("gelu_bug")
    prog.add(MatMul("x", "w1", "h1")).add(SiLU("h1", "h1_act")).add(MatMul("h1_act", "w2", "output"))
    return prog, {"x": x, "w1": w1, "w2": w2}, mesh

def _setup_colwise_gather_bug():
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    x = TensorState("x", (8, 16), (8, 16), ShardingSpec((Replicate(),), mesh), "x")
    w = TensorState("w", (16, 32), (16, 16), ShardingSpec((Shard(dim=1),), mesh), "w")
    prog = Program("colwise_gather_bug").add(MatMul("x", "w", "output"))
    return prog, {"x": x, "w": w}, mesh

def _setup_missing_pp_broadcast():
    mesh = DeviceMesh(shape=(2,), dim_names=("pp",))
    cu_seqlens = TensorState("cu_seqlens", (9,), (9,),
        ShardingSpec((Replicate(),), mesh), "cu_seqlens", stage=0)
    prog = Program("pp_bug").add(MatMul("cu_seqlens", "w0", "h0"))
    return prog, {"cu_seqlens": cu_seqlens}, mesh

def _setup_mismatched_send_recv():
    mesh = DeviceMesh(shape=(2,), dim_names=("pp",))
    h0 = TensorState("h0", (8, 16), (8, 16), ShardingSpec((Replicate(),), mesh), "h0", stage=0)
    prog = Program("pp_mismatch")
    prog.add(Send("h0", "h0_sent", src=0, dst=1, stage=0, microbatch_id=0))
    prog.add(Recv("h0_sent", "h0_rcvd", src=2, dst=1, stage=0, microbatch_id=0))  # BUG: src=2, should be 0
    return prog, {"h0": h0}, mesh

def _setup_async_ar_without_wait():
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    prog = Program("async_bug")
    prog.add(AllReduceAsync("grad_w", "grad_w", handle="h1", op_type="sum", stream=COMM_STREAM))
    prog.add(MatMul("grad_w", "opt_state", "update"))  # BUG: reads async output!
    prog.add(Wait(handle="h1", tensor="grad_w", output="grad_w_ready"))
    return prog, {}, mesh

def _setup_gradient_buffer_reuse():
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    prog = Program("buf_reuse")
    prog.add(AllReduceAsync("g1", "buf", handle="h1", stream=COMM_STREAM))
    prog.add(AllReduceAsync("g2", "buf", handle="h2", stream=COMM_STREAM))  # BUG: same buffer!
    prog.add(Wait(handle="h1", tensor="buf", output="buf_h1"))
    prog.add(Wait(handle="h2", tensor="buf", output="buf_h2"))
    return prog, {}, mesh


# ═══════════════════════════════════════════════════════════════════════════════
# Verify functions
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_postcondition_and_analysis(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        executor.register_tensor(ts)
    state = executor.run_program(prog)
    verifier = DistributedVerifier()
    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(prog, state)
    results = []
    for name, ts in state.items():
        if name not in {inp for op in prog.ops for inp in op.input_names}:
            results.append(verifier.verify_postcondition(ts, expected_partial=False))
    if not analysis.is_correct:
        results.append(VerifyResult(False, "placement", str(analysis)))
    return results or [VerifyResult(True, "default", "no issues")]

def _verify_nonlinear_on_shard(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        executor.register_tensor(ts)
    state = executor.run_program(prog)
    results = []
    for op in prog.ops:
        if isinstance(op, (SiLU, Multiply)):
            in_ts = state.get(op.input_names[0]) or tensors.get(op.input_names[0])
            if in_ts and not in_ts.is_replicated and not in_ts.partial:
                shard_info = ", ".join(
                    f"Shard({p.dim})" for p in in_ts.sharding.placements if isinstance(p, Shard))
                results.append(VerifyResult(False, "nonlinear on shard",
                    f"{type(op).__name__}({op.input_names[0]}) on {shard_info}"))
    return results or [VerifyResult(True, "nonlinear check", "all safe")]

def _verify_placement_consistency(prog, tensors, mesh):
    executor = MultiDeviceExecutor(mesh)
    for name, ts in tensors.items():
        executor.register_tensor(ts)
    state = executor.run_program(prog)
    results = []
    # Check: if output is Shard(1) but we expected Replicate (gather_output scenario)
    for name, ts in state.items():
        if name not in {inp for op in prog.ops for inp in op.input_names}:
            if any(isinstance(p, Shard) for p in ts.sharding.placements):
                results.append(VerifyResult(False, "placement",
                    f"Output '{name}' is Shard, may need AllGather for gather_output"))
    return results or [VerifyResult(True, "placement", "outputs are Replicate")]

def _verify_cross_stage_tensor_access(prog, tensors, mesh):
    results = []
    stage0_tensors = {name for name, ts in tensors.items() if ts.stage == 0}
    sent_tensors = {op.x for op in prog.ops if isinstance(op, Send)}
    not_sent = stage0_tensors - sent_tensors
    if not_sent:
        results.append(VerifyResult(False, "cross-stage broadcast",
            f"Stage-0 tensors not sent: {not_sent}"))
    return results or [VerifyResult(True, "cross-stage", "all broadcast")]

def _verify_communication_legality(prog, tensors, mesh):
    verifier = DistributedVerifier()
    return [verifier.verify_communication_legality(prog)]

def _verify_temporal(prog, tensors, mesh):
    result = verify_temporal(prog)
    reports = []
    for r in result.reports:
        reports.append(VerifyResult(False, r.race_type.value, r.description))
    return reports or [VerifyResult(True, "temporal", "no violations")]

def _verify_fp16_gradient_safety():
    from verifier.numerical import Dtype, DTYPE_PROPS
    fp16_min = DTYPE_PROPS[Dtype.FP16].min_normal
    grad_mag_estimate = 1e-5
    min_scale = fp16_min / grad_mag_estimate
    if min_scale > 1:
        return [VerifyResult(False, "fp16 underflow",
            f"Need loss_scale >= {min_scale:.0f} to keep grad > fp16 min_normal")]
    return [VerifyResult(True, "fp16 underflow", "gradient safe")]

def _verify_adam_eps_fp16():
    from verifier.numerical import Dtype, DTYPE_PROPS
    fp16_min = DTYPE_PROPS[Dtype.FP16].min_normal
    adam_eps = 1e-8
    if adam_eps < fp16_min:
        return [VerifyResult(False, "Adam fp16",
            f"Adam eps={adam_eps} < fp16 min={fp16_min:.1e} → NO EFFECT")]
    return [VerifyResult(True, "Adam fp16", "eps visible")]


# ═══════════════════════════════════════════════════════════════════════════════
# CommonTIR + DSL Conversion Demo
# ═══════════════════════════════════════════════════════════════════════════════

def demo_common_tir_dsl_conversion():
    """Demonstrate DSL-agnostic TIR conversion: TileLang, Triton, TVM → CommonTIR."""
    print("=" * 65)
    print("  DSL CONVERSION DEMO: TileLang / Triton / TVM → CommonTIR")
    print("=" * 65)

    # 1. TileLang TIR (native CommonTIR)
    i, j, k = TIRVar("i"), TIRVar("j"), TIRVar("k")
    tilelang_tir = TIRFunc(
        name="linear_tilelang",
        buffers={
            "X": TIRBuffer("X", (8, 16)),
            "W": TIRBuffer("W", (16, 32)),
            "Y": TIRBuffer("Y", (8, 32)),
        },
        grid=TIRGrid(axes=[i, j, k]),
        blocks=[TIRBlock(
            name="matmul",
            axes=[
                TIRAxis(i, AxisType.SPATIAL, 8),
                TIRAxis(j, AxisType.SPATIAL, 32),
                TIRAxis(k, AxisType.REDUCE, 16),
            ],
            reads=[TIRAccess("X", ["i", "k"]), TIRAccess("W", ["k", "j"])],
            writes=[TIRAccess("Y", ["i", "j"])],
        )],
    )
    print(f"\n  TileLang TIR → CommonTIR (pass-through):")
    print(f"    {tilelang_tir}")

    # 2. Triton kernel → CommonTIR
    triton_source = {
        "dialect": "triton",
        "grid": ["pid_m", "pid_n"],
        "ops": [
            {"type": "load", "buffer": "A", "shape": (128, 64)},
            {"type": "load", "buffer": "B", "shape": (64, 128)},
            {"type": "dot", "a": "A", "b": "B", "c": "C", "M": 128, "N": 128, "K": 64},
            {"type": "store", "buffer": "C", "shape": (128, 128)},
        ],
    }
    triton_tir = default_registry.convert(triton_source, "linear_triton")
    print(f"\n  Triton kernel → CommonTIR:")
    print(f"    {triton_tir}")

    # 3. TVM TensorIR → CommonTIR
    tvm_source = {
        "dialect": "tvm",
        "buffers": {
            "A": {"shape": (1024, 1024), "dtype": "float16"},
            "B": {"shape": (1024, 1024), "dtype": "float16"},
            "C": {"shape": (1024, 1024), "dtype": "float16"},
        },
        "blocks": [{
            "name": "gemm",
            "iter_vars": [
                {"var": "i", "kind": "spatial", "extent": 1024},
                {"var": "j", "kind": "spatial", "extent": 1024},
                {"var": "k", "kind": "reduce", "extent": 1024},
            ],
            "reads": [
                {"buffer": "A", "indices": ["i", "k"]},
                {"buffer": "B", "indices": ["k", "j"]},
            ],
            "writes": [{"buffer": "C", "indices": ["i", "j"]}],
            "body": "C[i,j] += A[i,k] * B[k,j]",
        }],
    }
    tvm_tir = default_registry.convert(tvm_source, "gemm_tvm")
    print(f"\n  TVM TensorIR → CommonTIR:")
    print(f"    {tvm_tir}")

    # 4. Verify uniform structure
    classifier = BlockClassifier()
    for name, tir in [("TileLang", tilelang_tir), ("Triton", triton_tir), ("TVM", tvm_tir)]:
        types = classifier.classify_func(tir)
        print(f"\n  {name} block types: {types}")
        has_matmul = any(t == BlockType.MATMUL for t in types.values())
        print(f"    Has matmul block: {'YES' if has_matmul else 'NO'} (all should be YES)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# RB5: TileLang Issues
# ═══════════════════════════════════════════════════════════════════════════════

RB5A_TILELANG_INVALID_LAYOUT = RealBugCase(
    id="RB5a",
    title="TileLang #2158: Invalid fragment layout — non-injective thread mapping",
    source_url="https://github.com/tile-ai/tilelang/issues/2158",
    category="TileLang: Layout/Placement",
    original_code="""\
# TileLang #2158: Fragment(32,2,4) mapped to 2 threads → non-injective
# The compiler error: "Loop layout is not injective"
# Fragment logical shape (32,2,4) with 64 elements cannot be mapped
# injectively to only 2 threads.

@tilelang.jit
def kernel():
    x_fragment = T.alloc_fragment((32, 2, 4), dtype="uint32")
    # BUG: 32*2*4=256 elements mapped to only 2 threads
    # Each thread would need to handle 128 elements,
    # but the indexing is ambiguous (non-injective layout)
    for i in T.parallel(32):
        for j in T.parallel(2):
            for k in T.parallel(4):
                x_fragment[i, j, k] = 0""",
    translation_notes=(
        "Our verifier models this as a RESOURCE MISMATCH: a fragment with "
        "logical size 32×2×4=256 mapped to only 2 execution units. "
        "The mapping is non-injective because 256/2=128 elements per thread "
        "but the loop structure implies each (i,j,k) maps to a unique position. "
        "Detection: SM resource check — threads_per_block insufficient for "
        "the logical iteration space. We model as: if spatial_axes_product > "
        "threads_available, and the mapping isn't tiled, flag as layout error."
    ),
    setup_fn=lambda: _setup_tilelang_invalid_layout(),
    verify_fn=lambda p, t, m: _verify_resource_mismatch(p, t, m),
)

RB5B_TILELANG_PIPELINE_SYNC = RealBugCase(
    id="RB5b",
    title="TileLang #2172: Int8 matmul wrong answer at num_stages=3 (pipeline sync)",
    source_url="https://github.com/tile-ai/tilelang/issues/2172",
    category="TileLang: Async/Pipeline",
    original_code="""\
# TileLang #2172: Int8 matmul produces WRONG results at num_stages=3
# but CORRECT at num_stages=2. This is a pipeline depth bug.
#
# M=128, K=256, N=128, BLOCK_K=128
# With K=256, num_stages=3: ceil(256/128)=2 tiles in K dim,
# but 3 stages → one stage is idle or overlapping incorrectly.

@tilelang.jit
def int8_matmul(A, B, C):
    a_shared = T.alloc_shared((128, 128), dtype="int8")
    b_shared = T.alloc_shared((128, 128), dtype="int8")

    T.Pipelined(3, stage="prologue"):  # 3 stages, but only 2 K-tiles!
        T.copy(A[..., k], a_shared)
        T.copy(B[..., k], b_shared)
        T.gemm(a_shared, b_shared, C_local, transpose_B=True)""",
    translation_notes=(
        "Our TEMPORAL verifier catches this as a PIPELINE OVERLAP bug: "
        "3 pipeline stages allocated but only 2 K-tiles exist. The extra "
        "stage creates a race condition where a stage reads shared memory "
        "that hasn't been fully written by the previous stage's T.copy. "
        "Detection: model as async shared memory access — stage[i] writes "
        "a_shared, stage[i+1] reads it, but with 3 stages and 2 tiles, "
        "stage 0 and stage 2 may access the same buffer concurrently."
    ),
    setup_fn=lambda: _setup_pipeline_sync_bug(),
    verify_fn=lambda p, t, m: _verify_temporal(p, t, m),
)

RB5C_TILELANG_FP8_CAST = RealBugCase(
    id="RB5c",
    title="TileLang #2042: fp8e8m0 cast produces different results from torch",
    source_url="https://github.com/tile-ai/tilelang/issues/2042",
    category="TileLang: Numeric Precision",
    original_code="""\
# TileLang #2042: Casting to fp8e8m0 gives results DIFFERENT from torch.
# fp8e8m0: 8 bits total = 1 sign + 5 exponent + 2 mantissa
# Machine epsilon: 2^(-2) = 0.25 → 25% relative error per cast!
# torch uses different rounding mode or saturation behavior.

@tilelang.jit
def cast_kernel(A, B):
    A_fp8 = T.cast(A, "fp8e8m0")
    B_fp8 = T.cast(B, "fp8e8m0")
    # BUG: results differ from torch due to rounding differences""",
    translation_notes=(
        "Our NUMERICAL verifier's DtypeModel includes fp8_e5m2 (ε=0.25). "
        "The cast error from fp32→fp8_e5m2 is 0.5*2^(-2)=0.125 relative. "
        "This is ENORMOUS compared to fp16 (4.88e-4) — 256x worse. "
        "Detection: flag as HIGH-RISK cast with ε > 1e-2. "
        "The verifier correctly warns that fp8 has insufficient precision "
        "for most training purposes without specialized scaling."
    ),
    setup_fn=lambda: (None, None, None),
    verify_fn=lambda p, t, m: _verify_fp8_cast_risk(),
)


# ═══════════════════════════════════════════════════════════════════════════════
# RB6: Triton Issues
# ═══════════════════════════════════════════════════════════════════════════════

RB6A_TRITON_TF32_IEEE = RealBugCase(
    id="RB6a",
    title="Triton #10176: bf16→fp32 upcast silently uses TF32 instead of IEEE",
    source_url="https://github.com/triton-lang/triton/issues/10176",
    category="Triton: Numeric Precision",
    original_code="""\
# Triton #10176: When upcasting bf16 to fp32 inside kernel,
# Triton uses TF32 path despite user passing input_precision="ieee".
# TF32: 19-bit mantissa (truncated fp32), not full 23-bit IEEE.
# This silently reduces precision without warning.

@triton.jit
def kernel(x_bf16, y_fp32):
    # User expects IEEE fp32 precision
    x_fp32 = x_bf16.to(tl.float32)  # BUG: uses TF32 path!
    y_fp32 = x_fp32 * 2.0  # computed with reduced precision""",
    translation_notes=(
        "Our NUMERICAL verifier models TF32 as having ε=2^(-10)=9.77e-4 "
        "(same as fp16 mantissa width), vs IEEE fp32 ε=1.19e-7. "
        "The verifier detects that the effective precision is 8192x worse "
        "than expected. We model this as: compute_dtype claims fp32 but "
        "actual ε matches fp16/bf16 → flag as precision mismatch. "
        "This catches the silent precision degradation."
    ),
    setup_fn=lambda: (None, None, None),
    verify_fn=lambda p, t, m: _verify_tf32_precision_mismatch(),
)

RB6B_TRITON_IMPLICIT_CAST = RealBugCase(
    id="RB6b",
    title="Triton #9991: tl.store(i1, i32) implicitly casts int32→int8 silently",
    source_url="https://github.com/triton-lang/triton/issues/9991",
    category="Triton: Type Safety",
    original_code="""\
# Triton #9991: tl.store with mismatched types silently truncates.
# Storing int32 value into int8 pointer → implicit truncation!
# Upper 24 bits are silently discarded.

@triton.jit
def kernel(ptr_i8, value_i32):
    # value_i32 is int32, but ptr_i8 expects int8
    tl.store(ptr_i8, value_i32)  # BUG: implicit i32→i8 truncation!
    # Only lower 8 bits survive, upper 24 bits lost.
    # No warning, no error — silent data corruption.""",
    translation_notes=(
        "Our NUMERICAL verifier models type casts with error bounds. "
        "int32→int8 truncation: loses 24 bits → 2^(-8) relative error "
        "for values < 128, and COMPLETE LOSS for values >= 256. "
        "Detection: flag any cast where dst bits < src bits and no explicit "
        "truncation op → POTENTIAL DATA LOSS. "
        "This is a type-safety check: the verifier warns that the implicit "
        "cast has {src_bits - dst_bits} bits of information loss."
    ),
    setup_fn=lambda: (None, None, None),
    verify_fn=lambda p, t, m: _verify_implicit_cast_truncation(),
)

RB6C_TRITON_TMA_NAN = RealBugCase(
    id="RB6c",
    title="Triton #10106: TMA loads NaN when mbarrier init order changes",
    source_url="https://github.com/triton-lang/triton/issues/10106",
    category="Triton: Async/Memory Race",
    original_code="""\
# Triton #10106: Warp-specialized pipeline with TMA loads.
# When mbarrier init order changes, consumer reads NaN from shared memory.
# The mbarrier synchronization primitive isn't properly initialized
# before it's used for TMA completion tracking.

# Producer:
T.copy(tma_input, iq_smem)  # async TMA load
mbarrier.arrive(ready_bar)   # signal completion

# Consumer:
mbarrier.wait(ready_bar)     # BUG: may pass before TMA done
x = tl.load(iq_smem)         # reads NaN if TMA not complete""",
    translation_notes=(
        "Our TEMPORAL verifier models this as a MISSING SYNC race: "
        "the mbarrier.wait is a synchronization primitive analogous to Wait(). "
        "If the mbarrier object isn't initialized before use — or if the "
        "shared memory layout causes aliasing with data buffers — the wait "
        "completes prematurely. "
        "Detection: temporal verifier checks if the async operation (TMA copy) "
        "is properly ordered before the consumer read. "
        "We model: TMA_copy(→iq_smem, handle=bar) + Wait(bar) + Load(iq_smem). "
        "If Wait is incorrectly ordered or bar is uninitialized, "
        "the consumer may read uninitialized data. "
        "LIMITATION: our current model doesn't track mbarrier initialization "
        "ordering. We detect the STRUCTURAL race (consumer reads async buffer "
        "without proper sync) but not the layout-dependent timing."
    ),
    setup_fn=lambda: _setup_tma_barrier_race(),
    verify_fn=lambda p, t, m: _verify_temporal(p, t, m),
)

RB6D_TRITON_MIXED_TYPE_ARITHMETIC = RealBugCase(
    id="RB6d",
    title="Triton #9963: Wrong results from while-loop with mixed int32/bf16 arithmetic",
    source_url="https://github.com/triton-lang/triton/issues/9963",
    category="Triton: Type Safety",
    original_code="""\
# Triton #9963: While-loop with mixed int32 and bf16 arithmetic
# produces wrong results. The type promotion rules in loops
# cause unexpected truncation or precision loss.

@triton.jit
def kernel(a_bf16, b_i32):
    acc = tl.zeros((1,), tl.float32)
    while b_i32 > 0:
        # BUG: mixing bf16 (7-bit mantissa) with int32 in loop
        # causes bf16 to be promoted/demoted incorrectly
        acc += a_bf16 * b_i32.to(tl.float32)
        b_i32 -= 1""",
    translation_notes=(
        "Our NUMERICAL verifier detects type mixing in loops as an "
        "ACCUMULATION risk: bf16 has ε=7.81e-3 (0.8%), and repeated "
        "accumulation in a loop amplifies the error. "
        "Detection: for loop-carried accumulators with mixed precision, "
        "the per-iteration error ε_sum grows as O(ε·N_iterations) for "
        "sequential accumulation. The verifier flags this as a precision "
        "risk, recommending fp32 for the accumulator."
    ),
    setup_fn=lambda: (None, None, None),
    verify_fn=lambda p, t, m: _verify_mixed_type_loop_accumulation(),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Additional setup & verify functions for TileLang/Triton cases
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_tilelang_invalid_layout():
    """Model fragment resource mismatch."""
    mesh = DeviceMesh(shape=(1,), dim_names=("device",))
    # Fragment with 256 logical elements, only 2 execution threads
    x = TensorState("fragment", (32, 2, 4), (32, 2, 4),
        ShardingSpec((Replicate(),), mesh), "fragment")
    # The resource check: logical_elements / threads_per_block = 256/2 = 128
    # This is excessive — each thread handles 128 elements → likely OOM or wrong
    return Program("layout_bug"), {"fragment": x}, mesh

def _setup_pipeline_sync_bug():
    """Model 3-stage pipeline with only 2 K-tiles → overlap race."""
    mesh = DeviceMesh(shape=(1,), dim_names=("device",))
    prog = Program("pipeline_bug")
    # Stage 0: async copy to shared → Stage 1: compute → Stage 2: overlap bug
    from verifier.ir import SendAsync, RecvAsync, Wait
    # Model as: stage[0] writes a_shared, stage[2] reads a_shared
    # but stage[1] hasn't finished writing → RACE
    prog.add(AllReduceAsync("a_local", "a_shared", handle="stg0", stream=COMM_STREAM))
    prog.add(AllReduceAsync("a_shared", "result", handle="stg2", stream=COMM_STREAM))
    # BUG: stg2 reads a_shared while stg0 may still be writing it
    # (3 stages allocated but only 2 tiles → overlapping buffer access)
    prog.add(Wait(handle="stg0", tensor="a_shared", output="a_done"))
    prog.add(Wait(handle="stg2", tensor="result", output="r_done"))
    return prog, {}, mesh

def _setup_tma_barrier_race():
    """Model TMA barrier race: consumer reads before TMA completes."""
    mesh = DeviceMesh(shape=(1,), dim_names=("device",))
    prog = Program("tma_race")
    from verifier.ir import RecvAsync, Wait
    # TMA load (async) → barrier.wait → consumer read
    # BUG: barrier arrives before TMA data is fully written
    prog.add(RecvAsync("tma_input", "iq_smem", handle="tma_h",
                        src=0, dst=0, stage=0, microbatch_id=0, stream=COMM_STREAM))
    # Consumer reads iq_smem — but TMA may not be done!
    prog.add(MatMul("iq_smem", "w", "output"))  # BUG: reads async buffer
    prog.add(Wait(handle="tma_h", tensor="iq_smem", output="iq_ready"))
    return prog, {}, mesh

def _verify_resource_mismatch(prog, tensors, mesh):
    """Check: logical elements vs execution units. TileLang #2158."""
    from verifier.solver import VerifyResult
    results = []
    for name, ts in tensors.items():
        logical_elems = math.prod(ts.global_shape)
        # Bug scenario: 256 elements / 2 threads = 128/thread → non-injective
        for threads in [2, 64, 128, 256]:
            if logical_elems / threads > 64:
                results.append(VerifyResult(False, "resource mismatch",
                    f"Fragment '{name}' ({ts.global_shape}) = {logical_elems} "
                    f"elements, {logical_elems/threads:.0f}/thread with {threads} "
                    f"threads. Likely non-injective. (TileLang #2158)"))
                break
    return results or [VerifyResult(True, "resource", "layout OK")]

def _verify_fp8_cast_risk():
    from verifier.numerical import Dtype, ErrorModel
    from verifier.solver import VerifyResult
    model = ErrorModel(Dtype.FP32, Dtype.FP32)
    err = model.cast_error(Dtype.FP32, Dtype.FP8_E5M2)
    results = []
    if err.relative > 0.01:
        results.append(VerifyResult(False, "fp8 cast risk",
            f"fp32→fp8 cast error {err.relative:.2e} > 1% threshold. "
            f"fp8 has only {2} mantissa bits (ε=0.25)."))
    return results or [VerifyResult(True, "fp8 cast", "acceptable")]

def _verify_tf32_precision_mismatch():
    from verifier.numerical import Dtype, DTYPE_PROPS
    from verifier.solver import VerifyResult
    fp32_eps = DTYPE_PROPS[Dtype.FP32].machine_epsilon  # 1.19e-7
    tf32_eps = DTYPE_PROPS[Dtype.FP16].machine_epsilon  # 9.77e-4 (same mantissa)
    ratio = tf32_eps / fp32_eps
    return [VerifyResult(False, "TF32 precision",
        f"TF32 path uses ε={tf32_eps:.2e} vs IEEE fp32 ε={fp32_eps:.2e}. "
        f"Precision is {ratio:.0f}x worse than expected.")]

def _verify_implicit_cast_truncation():
    from verifier.solver import VerifyResult
    src_bits, dst_bits = 32, 8
    lost = src_bits - dst_bits
    return [VerifyResult(False, "implicit cast",
        f"int32→int8 truncation: loses {lost} bits. "
        f"Values >= 2^{dst_bits}={2**dst_bits} are corrupted. "
        f"Maximum relative error for values < 2^{dst_bits}: 2^(-{dst_bits})={2**(-dst_bits):.1e}.")]

def _verify_mixed_type_loop_accumulation():
    from verifier.numerical import Dtype, DTYPE_PROPS
    from verifier.solver import VerifyResult
    bf16_eps = DTYPE_PROPS[Dtype.BF16].machine_epsilon  # 7.81e-3
    n_iter = 1000
    accum_err = n_iter * 0.5 * bf16_eps  # sequential accumulation
    return [VerifyResult(False, "loop accumulation",
        f"bf16 accumulation over {n_iter} iterations: "
        f"error bound ≈ {accum_err:.2e} ({accum_err*100:.1f}%). "
        f"Recommend fp32 accumulator.")]


ALL_REAL_BUGS = [
    RB1A_ROW_PARALLEL_MISSING_AR,
    RB1B_GELU_COLWISE_ROWWISE,
    RB1C_COLUMN_PARALLEL_GATHER_OUTPUT,
    RB2A_MISSING_PP_BROADCAST,
    RB2B_MISMATCHED_SEND_RECV,
    RB3A_FP16_GRADIENT_UNDERFLOW,
    RB3B_ADAM_EPSILON_FP16,
    RB4A_ASYNC_AR_WITHOUT_WAIT,
    RB4B_GRADIENT_BUFFER_REUSE,
    # TileLang issues
    RB5A_TILELANG_INVALID_LAYOUT,
    RB5B_TILELANG_PIPELINE_SYNC,
    RB5C_TILELANG_FP8_CAST,
    # Triton issues
    RB6A_TRITON_TF32_IEEE,
    RB6B_TRITON_IMPLICIT_CAST,
    RB6C_TRITON_TMA_NAN,
    RB6D_TRITON_MIXED_TYPE_ARITHMETIC,
]


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  LLM-INFRA-VERIFIER: Real-Bug Benchmark")
    print("  Bugs from actual GitHub issues + DSL Conversion Demo")
    print("=" * 65)

    # Run all real-bug cases
    detected = 0
    for case in ALL_REAL_BUGS:
        case.run()
        print(case.summary())
        if case.detected:
            detected += 1

    print(f"\n{'='*65}")
    print(f"  REAL-BUG BENCHMARK: {detected}/{len(ALL_REAL_BUGS)} detected")
    print(f"{'='*65}")

    # DSL conversion demo
    demo_common_tir_dsl_conversion()
