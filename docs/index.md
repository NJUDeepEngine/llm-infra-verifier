---
title: Home
nav_order: 1
permalink: /
---

# LLM-Infra-Verifier

{: .fs-8 }
Static verification framework for distributed LLM training — catches placement bugs and communication races at compile time.

{: .fs-5 }
Built on **Z3 SMT solver** + **symbolic execution** + **Happens-Before temporal analysis**. 48 IR ops, 11 verification checks, 660 tests. Multi-dim mesh aware. Zero GPU required.

[Get Started](getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/NJUDeepEngine/llm-infra-verifier){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Why This Exists

Distributed training bugs are silent, intermittent, and catastrophic at scale:

| Bug Type | Symptom | Traditional Detection |
|----------|---------|----------------------|
| Missing AllReduce after row-parallel MatMul | Silently diverging weights | Compare against single-GPU baseline |
| LayerNorm on sharded hidden dim | Incorrect normalization statistics | Check mathematical equivalence |
| Async AllReduce without Wait | Race condition | Flaky, CI-unfriendly |
| Mismatched Send/Recv in pipeline | Deadlock at specific scale | Only surfaces in production |

**This verifier catches them all statically — in milliseconds, with zero GPUs.**

---

## Two-Dimensional Verification

| | Spatial (6 checks) | Temporal (5 checks) |
|---|---|---|
| **Question** | Where does each tensor go? | When do things happen? |
| **Method** | Z3 SMT + symbolic execution | Happens-Before graph + Z3 |
| **Bugs caught** | Missing collectives, wrong placement, gradient inconsistencies | Races, missing waits, buffer aliasing, orphaned handles |

---

## Quick Overview

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

---

## Project Status

| Metric | Value |
|--------|-------|
| IR Operations | 48 (compute, collective, P2P, async, shape, precision, ZeRO, CP, MoE) |
| Verification Checks | 11 (6 spatial + 5 temporal) |
| Tests | 660 (all passing) |
| Examples | 9 runnable demos |
| Dependencies | 2 (z3-solver, pytest) |
| Parallelism Coverage | TP, PP, CP, DP, TP+DP (multi-dim mesh), ZeRO, MoE, FP8 |
