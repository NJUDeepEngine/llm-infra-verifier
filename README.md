# LLM-Infra-Verifier

> **Static verification framework for distributed LLM training infrastructure.**
> Catch placement bugs and communication races — before you launch a single GPU job.

[![Tests](https://img.shields.io/badge/tests-42%20passed-green)]()
[![Benchmarks](https://img.shields.io/badge/benchmarks-35%20cases%20%7C%20100%25%20detection-blue)]()
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

**This verifier catches them all at compile time — in milliseconds, with zero GPUs.**

---

## Two-Dimensional Verification

The core insight: verify **what** and **when** — as independent dimensions.

```
                         Input Source
                 TileLang TIR / PyTorch / Megatron
                             │
                             ▼
                   ┌─────────────────┐
                   │  Verification IR │  Unified intermediate representation
                   │  (ir.py, 20 ops) │  Symbolic, not numeric
                   └────────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │   SPATIAL     │           │   TEMPORAL    │
      │ Where things  │           │ When things   │
      │   go          │           │   happen      │
      │               │           │               │
      │ Z3 SMT solver │           │ HB graph + Z3 │
      │ 6 checks      │           │ 4 checks      │
      └───────────────┘           └───────────────┘
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

---

## Project Structure

```
verifier/
├── state.py           # TensorState, ShardingSpec, DeviceMesh
├── ir.py              # 20 op types with fwd+VJP
├── executor.py        # Multi-device symbolic executor
├── autograd.py        # VJP autograd + gradient duality
│
├── solver.py          # Z3 spatial verifier, 6 checks
├── temporal.py        # HB graph + Z3 race detection, 4 checks
│
├── rewrite.py         # Pattern matching, rewrite rules, cost model
├── synthesis.py       # Verified parallelization synthesis
├── llm_frontend.py    # PyTorch → IR via LLM + feedback loop
├── tir_lifter.py      # TileLang TIR → distributed IR
└── schedules.py       # 1F1B schedule + deadlock checker

examples/              # 6 runnable demos
benchmarks/            # 35 cases across 3 suites
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

---

## Demos & Benchmarks

### Examples (6 demo scripts)

```bash
# Spatial
python examples/tp_linear.py       # Row Parallel: correct vs bug
python examples/tp_mlp.py          # Megatron MLP: Column+Row Parallel
python examples/pp_2stage.py       # 2-Stage 1F1B Pipeline
python examples/cp_ring_attn.py    # Ring Attention with FlashAttention

# Temporal
python examples/overlap_demo.py    # 5 cases: race, missing wait, buffer alias

# Synthesis + LLM
python examples/synthesis_demo.py  # Auto-synthesis + LLM extraction flow
```

### Benchmarks (35 cases, 3 suites, 100% detection)

```bash
# Suite 1: Synthetic bug patterns (16 cases from PyTorch/Megatron/TileLang GitHub issues)
python benchmarks/benchmark_suite.py

# Suite 2: Real-code validation (8 cases lifted from Megatron-LM + TileLang source)
python benchmarks/real_code_validation.py

# Suite 3: Real-bug benchmark (11 cases with original buggy code + translation notes)
python benchmarks/real_bug_benchmark.py

python benchmarks/benchmark_suite.py --list   # List all 16 cases by category
python benchmarks/benchmark_suite.py --run B1 # Run specific category
python benchmarks/benchmark_suite.py --json   # JSON output
```

### Tests

```bash
python -m pytest tests/test_verifier.py -v    # 42 tests
```

### Issue Coverage

| Repository | Issues Used |
|---|---|
| `pytorch/pytorch` | #144359, #173041, #175690, #139681, #140227 |
| `NVIDIA/Megatron-LM` | #4092, #3952, #1525, #4382 |
| `tile-ai/tilelang` | #2035, #2042, #2054, #2158, #2172 |
| `triton-lang/triton` | #9991, #9963, #10106, #10176 |
| `deepseek-ai/TileKernels` | #2 |

---

## Design Philosophy

### 1. LLM proposes, Verifier checks — never the reverse

The LLM frontend can **suggest** IR translations, collective insertions, or parallelization tactics. But the formal verifier (Z3 + HB graph + IEEE 754 bounds) is the **final authority**. A program only passes if the verifier says so — no matter how confident the LLM is.

### 2. Symbolic over numeric — verify all inputs at once

We don't sample specific tensor values. We verify properties (placement correctness, race freedom, error bounds) that hold for **every possible input**. This is what makes static verification different from testing.

### 3. Dimensions are orthogonal — verify independently, compose results

Spatial correctness doesn't imply temporal safety. Each dimension has its own verification technique, and a program is only fully verified when both pass.

### 4. Pattern library over ad-hoc rules

Known distributed patterns (RowParallel, ColumnParallel, 1F1B, Ring Attention) are first-class citizens. This enables:
- **Detection**: recognize when a pattern is used incorrectly (e.g., GELU between Colwise and Rowwise without AllGather)
- **Synthesis**: generate the correct collective insertion for a given compute pattern
- **Optimization**: fuse or eliminate redundant collectives

---

## Verification Coverage Matrix

| | Spatial | Temporal |
|---|---|---|
| **Tensor Parallelism** | ✓ placement, postcond | ✓ async AR gradient |
| **Pipeline Parallelism** | ✓ Send/Recv match | ✓ 1F1B schedule order |
| **Context Parallelism** | ✓ ring order | ✓ async ring comm |
| **Data Parallelism** | ✓ gradient duality | ✓ async AR |
| **ZeRO-1/2/3** | ✓ shard consistency | ✓ ReduceScatter |
| **FlashAttention** | ✓ CP placement | ✓ async overlap |
