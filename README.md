# LLM-Infra-Verifier

> **Static verification framework for distributed LLM training infrastructure.**
> Catch placement bugs, communication races, numerical drift, and OOM —
> before you launch a single GPU job.

[![Tests](https://img.shields.io/badge/tests-42%20passed-green)]()
[![Benchmarks](https://img.shields.io/badge/benchmarks-33%20cases%20%7C%20100%25%20detection-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)]()

---

## Why this exists

Distributed training bugs are **silent, intermittent, and catastrophic at scale**:

| Bug | Symptom | Can you detect it by testing? |
|-----|---------|-------------------------------|
| Missing AllReduce | Wrong gradients, silently diverging weights | Only if you compare against single-GPU baseline |
| GELU on sharded tensor | Numerically wrong but structurally valid output | Only if you check mathematical equivalence |
| Async AllReduce without Wait | Race condition — works on some GPUs, fails on others | Flaky, CI-unfriendly |
| fp16 Adam state underflow | Optimizer silently stops updating | After thousands of steps of no progress |
| Activation memory > HBM | OOM halfway through training | After wasting GPU hours |

**This verifier catches them all at compile time — in milliseconds, with zero GPUs.**

---

## Four-Dimensional Verification

The core insight: verify **what**, **when**, **how precise**, and **does it fit** — as independent dimensions.

```
                         Input Source
                 TileLang TIR / PyTorch / Megatron
                             │
                             ▼
                   ┌─────────────────┐
                   │  Verification IR │  Unified intermediate representation
                   │  (ir.py, 35 ops) │  Symbolic, not numeric
                   └────────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   SPATIAL     │   │   TEMPORAL    │   │   NUMERICAL   │
│ Where things  │   │ When things   │   │ How precise   │
│   go          │   │   happen      │   │               │
│               │   │               │   │               │
│ Z3 SMT solver │   │ HB graph + Z3 │   │ IEEE 754      │
│ 6 checks      │   │ 4 checks      │   │ analytical     │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │    RESOURCE     │
                   │  Does it fit?   │
                   │                 │
                   │ Memory graph    │
                   │ HBM→Shared→Reg  │
                   │ Occupancy calc  │
                   └─────────────────┘
```

### Spatial: "Where does each tensor go?"

| Check | What it verifies | Method |
|-------|-----------------|--------|
| Postcondition | Output tensors not PARTIAL at boundaries | Z3: `Bool("partial")==True → unsat` |
| Communication legality | AllReduce only on Partial; Send↔Recv matched | Structural + Z3 |
| Gradient duality | fwd collective has matching bwd dual | Type-based duality table |
| Placement consistency | Output placement follows from inputs | Symbolic propagation rules |
| Shape consistency | Shapes survive collectives unchanged | Structural validation |
| PP deadlock freedom | No unmatched Send/Recv; no circular wait | DFS cycle detection |

### Temporal: "When do things happen?"

| Check | What it verifies | Method |
|-------|-----------------|--------|
| Data race | Different streams, same buffer, ≥1 write, unordered | HB interval overlap |
| Missing Wait | Async output consumed before `Wait(handle)` | Handle-waited_by analysis |
| Buffer aliasing | Two async ops writing same buffer concurrently | Write-after-write check |
| Dependency violation | Recv before Send for same (src,dst,mb) | HB ordering constraint |

### Numerical: "How much error can accumulate?"

| Pathway | Accumulation type | Bound |
|---------|------------------|-------|
| Weight cast drift | **None** (per-step independent) | ≤ ε_cast, regardless of T |
| Optimizer EMA state | **Bounded** (converges to input error) | ≤ g_error |
| Cross-rank divergence | **Linear** (unbounded, grows with T) | ≤ T × lr × ε_ar × |g| |

### Resource: "Does it fit in GPU memory?"

| Level | Capacity (H100) | What we check |
|-------|-----------------|---------------|
| HBM | 80 GB | Peak live tensors + params + optimizer + activations |
| Shared memory | 228 KB/SM | Per-block shared mem × concurrent blocks |
| Registers | 65536/SM | Per-thread regs × threads/block |
| Occupancy | 2048 threads/SM | Bottleneck analysis (threads vs regs vs shared) |

---

## How It Works: Static Symbolic Simulation

The key design decision: **we never run the program with concrete data**. Instead:

### 1. Symbolic execution, not numeric execution

```python
# Traditional execution: compute actual values
y = x @ w  # y = [[1.2, 3.4], ...]

# Our symbolic execution: propagate metadata
y = MatMul(x, w).apply(ctx)
# y.placement = Partial()
# y.shape = (8, 32)
# y.expr = "(x @ w)"
# y.local_shape = (8, 32)
```

The executor (`executor.py`) tracks **what each device holds** — placement, shape, symbolic expression — but never touches actual numbers.

### 2. Z3 as the correctness oracle

Placement correctness is encoded as SMT constraints:

```
Given: tensor y has sharding spec S
Check:  is it possible that y.partial == True at the output?

Z3 encoding:
  Bool("partial") == y.partial      // actual state
  (partial == True)                 // negated postcondition
  → sat: bug found (y IS partial)
  → unsat: safe (y cannot be partial)
```

The elegance: we don't need to enumerate all possible sharding configurations. Z3 searches the space for us.

### 3. Happens-Before for temporal reasoning

Each op gets a time interval `[issue, complete)`. Constraints:

```
Same stream:     complete_i < issue_{i+1}
Wait sync:       complete_async < issue_wait
Data dependency: complete_writer < issue_reader
```

Z3 checks: "Does there exist a schedule where two unordered ops on different streams overlap on the same buffer?" If yes → race.

### 4. IEEE 754 bounds for numerical analysis

We don't simulate training. We compute worst-case error bounds from first principles:

```
fp32 AllReduce (tree, N=256):  ε ≤ 0.5·2^(-23) · log₂(256) = 9.54e-7
fp16 AllReduce (tree, N=256):  ε ≤ 0.5·2^(-10) · log₂(256) = 3.91e-3
fp32→fp16 cast:                ε ≤ 0.5·2^(-10) = 4.88e-4  (DOMINANT)

Cross-rank divergence after T steps:  |Δθ| ≤ T × lr × ε_ar × |g|
```

These bounds are **valid for ALL inputs** — conservative but never miss a violation.

### 5. Memory graph for resource analysis

Each tensor is a node with a **lifetime** `[first_use, last_use]`. At each program point we sum live tensors and compare against hardware limits. Same idea as register allocation, but for GPU HBM.

---

## Project Structure

```
verifier/
├── state.py           # TensorState, ShardingSpec, DeviceMesh (248 lines)
├── ir.py              # 35 op types with fwd+VJP (~1400 lines)
├── executor.py        # Multi-device symbolic executor (316 lines)
├── autograd.py        # VJP autograd + gradient duality (368 lines)
│
├── solver.py          # Z3 spatial verifier, 6 checks (572 lines)
├── temporal.py        # HB graph + Z3 race detection, 4 checks (~580 lines)
├── numerical.py       # IEEE 754 error model, 3-pathway accumulation (~800 lines)
│
├── hardware.py        # H100/H200/B200/A100 GPU specs (~300 lines)
├── memory_graph.py    # Memory graph builder + OOM detector (~400 lines)
│
├── rewrite.py         # Pattern matching, rewrite rules, cost model (574 lines)
├── synthesis.py       # Verified parallelization synthesis (539 lines)
├── llm_frontend.py    # PyTorch → IR via LLM + feedback loop (658 lines)
├── tir_lifter.py      # TileLang TIR → distributed IR (656 lines)
└── schedules.py       # 1F1B schedule + deadlock checker (397 lines)

examples/              # 8 runnable demos
benchmarks/            # 33 cases across 4 suites
tests/                 # 42 tests
docs/                  # GitHub Pages documentation
```

---

## Quick Start

```bash
pip install -r requirements.txt  # z3-solver, pytest
```

### Verify a Row Parallel Linear layer (spatial)

```python
from verifier import *

mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
x = TensorState("x", (8, 16), (8, 8), ShardingSpec((Shard(dim=1),), mesh))
w = TensorState("w", (16, 32), (8, 32), ShardingSpec((Shard(dim=0),), mesh))

fwd = Program("tp").add(MatMul("x", "w", "y_partial")).add(AllReduce("y_partial", "y"))
executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x); executor.register_tensor(w)
state = executor.run_program(fwd)

verifier = DistributedVerifier()
result = verifier.verify_postcondition(state["y"], expected_partial=False)
print(f"Postcondition: {'PASSED' if result.passed else 'FAILED'}")
# Remove the AllReduce line → verifier catches the bug
```

### Detect async communication race (temporal)

```python
from verifier.temporal import verify_temporal

prog = Program("race")
prog.add(MatMul("x", "w", "y_p"))
prog.add(AllReduceAsync("y_p", "y", handle="h1", stream=COMM_STREAM))
prog.add(MatMul("y", "w2", "z"))  # BUG: reads before Wait!

result = verify_temporal(prog)
# Detected: MISSING_WAIT on tensor 'y'
```

### Check numerical safety for your training config (numerical)

```python
from verifier.numerical import verify_numerical, Dtype, ZeROStage

result = verify_numerical(
    n_ranks=256, topology="tree",
    compute_dtype=Dtype.FP16, accumulate_dtype=Dtype.FP32,
    optimizer="adam", zero_stage=ZeROStage.ZERO1,
    hidden_dim=4096, num_layers=32, batch_size=128,
)
print(result.summary())
# Shows: reduction error, 3-pathway accumulation, overflow risks, optimizer safety
```

### Check if your model fits in GPU memory (resource)

```python
from verifier.hardware import H100_SXM
from verifier.memory_graph import estimate_llm_memory

mem = estimate_llm_memory(hidden_dim=8192, num_layers=80, tp_size=8)
fits = mem["total"] / (1024**3) < H100_SXM.total_hbm_gb
print(f"Llama-70B on H100: {'FITS' if fits else 'OOM'} ({mem['total']/(1024**3):.1f}GB)")
```

---

## Demos & Benchmarks

```bash
# Spatial verification
python examples/tp_linear.py       # Row Parallel: correct vs bug
python examples/tp_mlp.py          # Megatron MLP
python examples/pp_2stage.py       # 2-Stage 1F1B Pipeline
python examples/cp_ring_attn.py    # Ring Attention

# Temporal verification
python examples/overlap_demo.py    # 5 cases: data race, missing wait, buffer alias

# Numerical verification
python examples/numerical_demo.py  # 7 sections: dtype, cast, reduction, accumulation

# Resource / OOM detection
python examples/oom_demo.py        # 6 sections: GPU specs, LLM memory, occupancy

# Synthesis + LLM
python examples/synthesis_demo.py  # Auto-synthesis + LLM extraction

# Benchmarks
python benchmarks/benchmark_suite.py          # 16 synthetic cases (100% detection)
python benchmarks/real_code_validation.py     # 8 real-code cases (Megatron + TileLang)
python benchmarks/numerical_benchmark.py      # 9 numerical cases

# Tests
python -m pytest tests/test_verifier.py -v    # 42 tests
```

---

## Design Philosophy

### 1. LLM proposes, Verifier checks — never the reverse

The LLM frontend can **suggest** IR translations, collective insertions, or parallelization tactics. But the formal verifier (Z3 + HB graph + IEEE 754 bounds) is the **final authority**. A program only passes if the verifier says so — no matter how confident the LLM is.

### 2. Symbolic over numeric — verify all inputs at once

We don't sample specific tensor values. We verify properties (placement correctness, race freedom, error bounds) that hold for **every possible input**. This is what makes static verification different from testing.

### 3. Dimensions are orthogonal — verify independently, compose results

Spatial correctness doesn't imply temporal safety. Temporal safety doesn't imply numerical stability. Each dimension has its own verification technique, and a program is only fully verified when all four pass.

### 4. Conservative bounds over empirical estimates

Our numerical error bounds are **worst-case analytical** (IEEE 754 + Higham). They may overestimate error in practice, but they will **never miss a real violation**. For safety-critical infrastructure, this is the right trade-off.

### 5. Pattern library over ad-hoc rules

Known distributed patterns (RowParallel, ColumnParallel, 1F1B, Ring Attention) are first-class citizens. This enables:
- **Detection**: recognize when a pattern is used incorrectly (e.g., GELU between Colwise and Rowwise without AllGather)
- **Synthesis**: generate the correct collective insertion for a given compute pattern
- **Optimization**: fuse or eliminate redundant collectives

---

## Verification Coverage Matrix

| | Spatial | Temporal | Numerical | Resource |
|---|---|---|---|---|
| **Tensor Parallelism** | ✓ placement, postcond | ✓ async AR gradient | ✓ AR error, cast | ✓ HBM, shared mem |
| **Pipeline Parallelism** | ✓ Send/Recv match | ✓ 1F1B schedule order | ✓ | ✓ activation liveness |
| **Context Parallelism** | ✓ ring order | ✓ async ring comm | ✓ | ✓ |
| **Data Parallelism** | ✓ gradient duality | ✓ async AR | ✓ cross-rank div | ✓ |
| **ZeRO-1/2/3** | ✓ shard consistency | ✓ ReduceScatter | ✓ shard boundary drift | ✓ |
| **Mixed Precision** | — | — | ✓ fp16/bf16 bounds | — |
| **Adam/AdamW** | — | — | ✓ state invariants | ✓ optimizer memory |
| **FlashAttention** | ✓ CP placement | ✓ async overlap | — | ✓ shared mem occupancy |
| **H100/B200 GPUs** | — | — | — | ✓ per-SM limits |
