---
title: Home
nav_order: 1
permalink: /
---

# LLM-Infra-Verifier

{: .fs-8 }
Formal verification for distributed tensor programs — ensuring correctness of Tensor Parallelism, Pipeline Parallelism, and Context Parallelism.

{: .fs-5 }
Built on **Z3 SMT solver** + **symbolic execution** + **Happens-Before temporal analysis**, targeting TileLang TIR and Megatron-LM as verification sources.

[Get Started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/NJUDeepEngine/llm-infra-verifier){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Why this exists

Distributed training programs are notoriously hard to get right. Bugs like **missing AllReduce**, **GELU applied to sharded tensors**, or **async communication races** produce numerically incorrect results that pass unit tests but fail in production at scale.

`llm-infra-verifier` catches these bugs **at compile time**, before you ever launch a distributed job.

### Two dimensions of verification

| | Spatial | Temporal |
|---|---|---|
| **What** | Placement propagation, postconditions, gradient duality | Data races, missing waits, buffer aliasing |
| **How** | Z3 SMT + symbolic execution | Happens-Before graph + Z3 constraints |
| **Bugs caught** | Missing AllReduce, wrong placement, gradient inconsistencies | Async read before Wait, concurrent buffer writes |
| **Example** | RowParallel without AllReduce → PARTIAL output | `AllReduceAsync` + `MatMul` on same buffer without `Wait` |

### Real-code validation

Every verification capability is validated against **actual implementations**:

| Source | Module | Reference |
|---|---|---|
| Megatron-LM | `ColumnParallelLinear`, `RowParallelLinear` | `megatron/core/tensor_parallel/layers.py` |
| Megatron-LM | Async gradient AllReduce | `LinearWithGradAccumulationAndAsyncCommunication` |
| Megatron-LM | TP MLP, Sequence Parallel + TP | `megatron_mlp.py`, `layers.py` |
| PyTorch | GELU-between-CP-and-RP bug | `pytorch/pytorch#144359` |
| TileLang | TIR block → distributed IR lifting | `examples/gemm/example_gemm.py` |

---

## Project status

| Metric | Value |
|---|---|
| Tests | 42 (all passing) |
| Benchmark cases | 35 (100% detection) |
| Verification dimensions | Spatial (6 checks) + Temporal (4 checks) |
| Parallelism types | TP, PP, CP (forward + backward) |

---

## Quick overview

```python
from verifier import *

# Row Parallel Linear: both operands sharded on reduce dim
mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
x = TensorState("x", (8, 16), (8, 8),
    ShardingSpec((Shard(dim=1),), mesh))
w = TensorState("w", (16, 32), (8, 32),
    ShardingSpec((Shard(dim=0),), mesh))

# Verify: MatMul → AllReduce produces Replicate output
fwd = Program("tp_linear")
fwd.add(MatMul(a="x", b="w", output="y_partial"))
fwd.add(AllReduce(x="y_partial", output="y"))

executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x); executor.register_tensor(w)
state = executor.run_program(fwd)

verifier = DistributedVerifier()
result = verifier.verify_postcondition(state["y"], expected_partial=False)
# PASSED — y is Replicate, not Partial
```

### Temporal verification

```python
from verifier.temporal import verify_temporal

prog = Program("overlap_bug")
prog.add(MatMul("x", "w", "y_partial"))
prog.add(AllReduceAsync("y_partial", "y", handle="h1", stream=COMM_STREAM))
prog.add(MatMul("y", "w2", "z"))  # BUG: reads async output before Wait!

result = verify_temporal(prog)
# Detected: MISSING_WAIT — MatMul reads 'y' before Wait(h1)
```

### Verified parallelization synthesis

```python
from verifier.synthesis import synthesize_parallel_program

result = synthesize_parallel_program(
    compute_ops=[MatMul(a="x", b="w", output="y")],
    input_shapes={"x": (8, 16), "w": (16, 32)},
    sharding_specs={
        "x": ShardingSpec((Shard(dim=1),), mesh),
        "w": ShardingSpec((Shard(dim=0),), mesh),
    },
)
# Auto-synthesizes: MatMul → AllReduce → y_reduced
```
