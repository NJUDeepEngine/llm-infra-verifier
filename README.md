# LLM-Infra-Verifier

> **Static verification framework for distributed LLM training.**
> Catches placement bugs and communication races at compile time — no GPUs needed.

[![Tests](https://img.shields.io/badge/tests-660%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)]()

---

## Why This Exists

Distributed training bugs are **silent, intermittent, and catastrophic at scale**:

| Bug Type | Symptom | Traditional Detection |
|----------|---------|----------------------|
| Missing AllReduce after row-parallel MatMul | Silently diverging weights | Compare against single-GPU baseline |
| LayerNorm on sharded hidden dim | Numerically wrong normalization | Check mathematical equivalence |
| Async AllReduce without Wait | Race condition — works on some runs | Flaky, CI-unfriendly |
| Mismatched Send/Recv in pipeline | Deadlock under specific schedules | Only surfaces at scale |

**This verifier catches them all statically — in milliseconds, with zero GPUs.**

---

## Architecture

```
Input Source (PyTorch / Megatron / TileLang TIR)
                    │
                    ▼
         ┌──────────────────┐
         │  Verification IR  │   48 ops, symbolic metadata
         │  verifier/ir/     │   not numeric values
         └────────┬─────────┘
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
┌──────────────┐        ┌──────────────┐
│   SPATIAL    │        │   TEMPORAL   │
│  "Where"     │        │  "When"      │
│              │        │              │
│ Z3 SMT       │        │ HB graph     │
│ 6 checks     │        │ 5 checks     │
└──────────────┘        └──────────────┘
```

### Spatial Verification (Z3-based)

| Check | Property | Method |
|-------|----------|--------|
| Postcondition | Output not Partial at boundaries | Z3 UNSAT proof |
| Communication legality | Collectives on valid inputs only | Structural + Z3 |
| Gradient duality | fwd collective ↔ bwd dual | Type duality table |
| Placement consistency | Output placement follows inputs | Symbolic propagation |
| Shape consistency | Shapes valid through collectives | Divisibility constraints |
| PP deadlock freedom | No circular Send/Recv waits | DFS cycle detection |

**Multi-dim mesh support:** N-dimensional meshes (e.g. TP×DP) with per-dim Z3 variables. Collectives with `mesh_dim` target only the specified dimension, preserving placements on other dims.

### Temporal Verification (Happens-Before + Z3)

| Check | Property | Method |
|-------|----------|--------|
| Data race | Unordered concurrent R/W on same buffer | HB interval overlap |
| Missing Wait | Async output read before sync | Handle-waited analysis |
| Buffer aliasing | Two async ops write same buffer | WAW detection |
| Dependency violation | Recv before matching Send | HB ordering |
| Orphaned handle | Async handle never waited on | Handle lifecycle tracking |

---

## IR Operations (48 ops)

### Compute (13 ops)

| Category | Ops | Sharding Semantics |
|----------|-----|-------------------|
| Linear algebra | MatMul | S(1)×S(0)→P, R×S(1)→S(1), S(0)×R→S(0) |
| Element-wise binary | Add, Multiply | Merge placements; P×P forbidden |
| Activations | SiLU, GELU, ReLU | Passthrough |
| Regularization | Dropout | Passthrough |
| Normalization | LayerNorm, RMSNorm | Error if Shard(norm_dim) |
| Reduction | Softmax | Error if Shard(reduction_dim) |
| Attention | FlashAttention | Follows Q placement |
| Vocab-parallel | Embedding | weight Shard(vocab)→Partial |
| Loss | CrossEntropyLoss | logits Shard(vocab)→Partial |

### Communication (8 NCCL collectives + 4 P2P)

All collectives support optional `mesh_dim` for multi-dim meshes — targets only the specified dimension.

| Op | Forward | Backward Dual |
|----|---------|---------------|
| AllReduce | Partial→Replicate | AllReduce (self) |
| AllGather | Shard(d)→Replicate | ReduceScatter |
| ReduceScatter | R/P→Shard(d) | AllGather |
| Broadcast | any→Replicate | Reduce |
| Reduce | Partial→Replicate(root) | Broadcast |
| AllToAll | Shard(split)→Shard(concat) | AllToAll (dim-swap) |
| Scatter | Replicate→Shard(d) | Gather |
| Gather | Shard(d)→Replicate | Scatter |
| Send/Recv | P2P data movement | Recv/Send (reversed) |
| SendAsync/RecvAsync | Non-blocking P2P | — |

### Async & Overlap (4 ops)
AllReduceAsync, Wait, WaitAll, OverlapRegion

### Precision (5 ops)
Cast, LossScale, FP8Quantize, FP8Dequantize, AmaxUpdate

### Parallelism-Specific (10 ops)

| Domain | Ops |
|--------|-----|
| ZeRO | ZeROGatherParam, ZeROScatterGrad, ZeROPartitionOptState |
| Context Parallelism | RingRotate, RingAttentionStep, RingAttention |
| Mixture of Experts | TopKGate, MoEDispatch, MoECombine, ExpertCompute |

### Shape & SPMD (4 ops)
Reshape, Transpose, Reinterpret, Convert

---

## Project Structure

```
verifier/
├── state/              # TensorState, ShardingSpec, DeviceMesh, placements
├── ir/                 # 48 IR ops across 10 modules
│   ├── compute.py      #   MatMul, activations, norm, attention, embedding, loss
│   ├── collective.py   #   8 NCCL collectives
│   ├── p2p.py          #   Send/Recv (sync + async)
│   ├── async_ops.py    #   AllReduceAsync, Wait, OverlapRegion
│   ├── precision.py    #   Cast, FP8, LossScale
│   ├── shape.py        #   Reshape, Transpose
│   ├── spmd.py         #   Reinterpret, Convert, SPMDGuard
│   ├── zero.py         #   ZeRO-1/2/3 ops
│   ├── cp.py           #   Ring attention ops
│   └── moe.py          #   MoE dispatch/combine
├── executor.py         # Multi-device symbolic executor
├── solver.py           # Z3 spatial verifier (6 checks, multi-dim mesh)
├── temporal.py         # HB graph + race detection (5 checks)
├── autograd.py         # VJP engine + gradient duality
├── rewrite.py          # Pattern matching + cost model
├── synthesis.py        # Beam-search tactic synthesis
├── schedules.py        # PP 1F1B schedule + deadlock checker
├── llm_frontend.py     # PyTorch → IR via LLM + feedback loop
└── tir_lifter.py       # TileLang TIR → IR lifter

tests/                  # 660 tests across 13 files
examples/               # 9 runnable demos
benchmarks/             # 3 suites
docs/                   # Architecture & API docs
```

---

## Quick Start

```bash
pip install -r requirements.txt  # z3-solver, pytest
python -m pytest tests/ -v       # run all 660 tests
```

### Example: Verify Tensor-Parallel Linear

```python
from verifier import *

mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
x_spec = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
w_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

x = TensorState("x", (8, 16), compute_local_shape((8, 16), x_spec), x_spec, expr="x")
w = TensorState("w", (16, 32), compute_local_shape((16, 32), w_spec), w_spec, expr="w")

program = Program("tp_linear", ops=[
    MatMul(a="x", b="w", output="y_partial"),
    AllReduce(x="y_partial", output="y"),
])

executor = MultiDeviceExecutor(mesh=mesh)
executor.register_tensor(x)
executor.register_tensor(w)
final = executor.run_program(program)

verifier = DistributedVerifier()
results = verifier.verify_all(program, final)
print(verifier.summary())
# Verification Summary: 4 passed, 0 failed
```

### Example: Detect Async Race

```python
from verifier import *
from verifier.temporal import verify_temporal

prog = Program("race", ops=[
    MatMul(a="x", b="w", output="y_p"),
    AllReduceAsync(x="y_p", output="y", handle="h1"),
    MatMul(a="y", b="w2", output="z"),  # BUG: reads y before Wait!
])

result = verify_temporal(prog)
# Detected: MISSING_WAIT on tensor 'y'
```

### Example: Vocab-Parallel Embedding

```python
from verifier import *

mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
w_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
ids_spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)

ids = TensorState("ids", (32,), (32,), ids_spec, expr="ids")
W = TensorState("W", (50000, 128), compute_local_shape((50000, 128), w_spec), w_spec, expr="W")

program = Program("vocab_parallel", ops=[
    Embedding(indices="ids", weight="W", output="emb_partial"),
    AllReduce(x="emb_partial", output="emb"),
])

executor = MultiDeviceExecutor(mesh=mesh)
executor.register_tensor(ids)
executor.register_tensor(W)
final = executor.run_program(program)

assert not final["emb"].partial  # Output is Replicate after AllReduce
```

---

## Demos

```bash
# Tensor Parallelism
python examples/tp_linear.py          # Row-parallel: correct vs bug detection
python examples/tp_mlp.py             # Megatron MLP: Column+Row parallel

# Pipeline Parallelism
python examples/pp_2stage.py          # 2-stage 1F1B schedule verification

# Context Parallelism
python examples/cp_ring_attn.py       # Ring attention with FlashAttention

# Temporal / Async
python examples/overlap_demo.py       # Race, missing wait, buffer aliasing

# SPMD Type System
python examples/spmd_demo.py          # R/I/V/P types + gradient duality

# Synthesis
python examples/synthesis_demo.py     # Auto-synthesis from unannotated compute

# End-to-End
python examples/megatron_gpt2_verify.py   # Full Megatron GPT-2 verification
python examples/consistency_demo.py       # Single-GPU equivalence proof
```

---

## SPMD Type System

Four local types with gradient duality:

| Type | Meaning | Gradient Dual |
|------|---------|---------------|
| **R** (Replicate) | Same data on all ranks | P (Partial) |
| **I** (Invariant) | Same data, no gradient comm needed | I (Invariant) |
| **V** (Varying) | Different data per rank | V (Varying) |
| **P** (Partial) | Pending sum across ranks | R (Replicate) |

Key rules enforced by `SPMDGuard`:
- `Partial × Partial` is **forbidden** (doesn't distribute over pending sum)
- `AllReduce(Replicate)` is an **error** (no pending sum exists)
- `AllReduce(Invariant)` is an **error** (gradient already identical)

---

## Verification Coverage

| Parallelism Strategy | Spatial Checks | Temporal Checks |
|---------------------|----------------|-----------------|
| Tensor Parallelism | Placement, postcondition, shape | Async AllReduce races |
| Pipeline Parallelism | Send/Recv matching, deadlock | 1F1B schedule ordering |
| Context Parallelism | Ring placement propagation | Async ring communication |
| Data Parallelism | Gradient duality | Async gradient AllReduce |
| TP + DP (multi-dim mesh) | Per-dim placement, mesh_dim targeting | Per-dim async races |
| ZeRO-1/2/3 | Shard/gather consistency | ReduceScatter ordering |
| MoE (Expert Parallel) | AllToAll dispatch/combine | Token routing races |
| Mixed Precision (FP8) | Scale freshness, format usage | Delayed scaling ordering |

---

## Design Principles

1. **Symbolic over numeric** — verify properties for ALL possible inputs, not samples
2. **LLM proposes, Verifier decides** — Z3 is the final authority, not model confidence
3. **Spatial × Temporal** — independent dimensions, composed results
4. **Every op has VJP** — forward and backward are equally verifiable
5. **Zero dependencies beyond Z3** — no PyTorch, no CUDA, no cluster needed

---

## References

Bugs sourced from real distributed training issues:

| Repository | Issues |
|------------|--------|
| pytorch/pytorch | #144359, #173041, #175690, #139681, #140227 |
| NVIDIA/Megatron-LM | #4092, #3952, #1525, #4382 |
| tile-ai/tilelang | #2035, #2042, #2054, #2158, #2172 |
| triton-lang/triton | #9991, #9963, #10106, #10176 |
| deepseek-ai/TileKernels | #2 |
