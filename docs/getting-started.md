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
x = TensorState("x", (8, 16), (8, 8),
    ShardingSpec((Shard(dim=1),), mesh), expr="x")
w = TensorState("w", (16, 32), (8, 32),
    ShardingSpec((Shard(dim=0),), mesh), expr="w")

fwd = Program("tp")
fwd.add(MatMul(a="x", b="w", output="y_partial"))
fwd.add(AllReduce(x="y_partial", output="y"))

executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x); executor.register_tensor(w)
state = executor.run_program(fwd)

verifier = DistributedVerifier()
result = verifier.verify_postcondition(state["y"], expected_partial=False)
print(f"Verification: {'PASSED' if result.passed else 'FAILED'}")
```

### 2. Detect a bug

Remove the `AllReduce` line — the verifier will catch it:

```
Postcondition check: FAILED
  Found counterexample: tensor.partial=True, expected=False
```

### 3. Check temporal correctness

```python
from verifier.temporal import verify_temporal
from verifier.ir import AllReduceAsync, Wait, COMM_STREAM

prog = Program("overlap")
prog.add(MatMul("x", "w", "y_p"))
prog.add(AllReduceAsync("y_p", "y", handle="h1", stream=COMM_STREAM))
prog.add(MatMul("y", "w2", "z"))    # BUG: reads before Wait
prog.add(Wait(handle="h1", tensor="y", output="y_safe"))

result = verify_temporal(prog)
print(f"Temporal: {'SAFE' if result.is_safe else 'UNSAFE'}")
for r in result.reports:
    print(f"  {r.race_type.value}: {r.description}")
```

## Run the demos

```bash
# Spatial verification
python examples/tp_linear.py       # Row Parallel: correct vs bug
python examples/tp_mlp.py          # Megatron MLP: Column + Row Parallel
python examples/pp_2stage.py       # 2-Stage 1F1B Pipeline
python examples/cp_ring_attn.py    # Ring Attention with FlashAttention

# Temporal verification
python examples/overlap_demo.py    # Data race, missing wait, buffer aliasing

# Synthesis + LLM frontend
python examples/synthesis_demo.py  # Auto-synthesis + LLM extraction loop
```

## Run benchmarks

```bash
# Synthetic benchmark (16 cases from real GitHub issues)
python benchmarks/benchmark_suite.py

# Real-code validation (8 cases from Megatron-LM + TileLang source)
python benchmarks/real_code_validation.py
```

## Run tests

```bash
python -m pytest tests/test_verifier.py -v    # 42 tests
```
