---
title: Getting Started
nav_order: 2
---

# Getting Started

## Installation

```bash
git clone git@github.com:NJUDeepEngine/llm-infra-verifier.git
cd llm-infra-verifier
pip install -r requirements.txt
```

**Dependencies:** `z3-solver>=4.12.0`, `pytest>=7.0.0`

## 5-minute verification

### 1. Verify a Row Parallel Linear layer

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

### 2. Detect a placement bug

Remove the `AllReduce` — the verifier catches it:

```python
program = Program("tp_linear_bug", ops=[
    MatMul(a="x", b="w", output="y"),  # Output is Partial, no AllReduce!
])

# ...run executor and verify...
# Postcondition check: FAILED
#   Found counterexample: tensor.partial=True, expected=False
```

### 3. Check temporal correctness

```python
from verifier.temporal import verify_temporal

prog = Program("overlap", ops=[
    MatMul(a="x", b="w", output="y_p"),
    AllReduceAsync(x="y_p", output="y", handle="h1", stream=COMM_STREAM),
    MatMul(a="y", b="w2", output="z"),   # BUG: reads y before Wait!
    Wait(handle="h1", tensor="y", output="y_safe"),
])

result = verify_temporal(prog)
print(f"Temporal: {'SAFE' if result.is_safe else 'UNSAFE'}")
for r in result.reports:
    print(f"  {r.race_type.value}: {r.description}")
# Detected: MISSING_WAIT on tensor 'y'
```

### 4. Verify vocab-parallel embedding

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

## Run the demos

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

## Run benchmarks

```bash
# Synthetic benchmark (16 cases from real GitHub issues)
python benchmarks/benchmark_suite.py

# Real-code validation (8 cases from Megatron-LM + TileLang source)
python benchmarks/real_code_validation.py

# Scaling benchmark
python benchmarks/scaling_benchmark.py
```

## Run tests

```bash
# All 660 tests
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_verifier.py -v
python -m pytest tests/test_op_verification.py -v
python -m pytest tests/test_executor.py -v
python -m pytest tests/test_temporal.py -v
python -m pytest tests/test_synthesis.py -v
python -m pytest tests/test_autograd.py -v
python -m pytest tests/test_tir_lifter.py -v
python -m pytest tests/test_schedules.py -v
python -m pytest tests/test_real_world_bugs.py -v
python -m pytest tests/test_solver_deep.py -v
python -m pytest tests/test_temporal_ops.py -v
```
