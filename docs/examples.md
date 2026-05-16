---
title: Examples
nav_order: 6
---

# Examples & Demos

All 9 examples are runnable from the `examples/` directory.

## Spatial Verification

### TP Linear (`tp_linear.py`)

Two cases:
1. **Correct:** Row Parallel with `MatMul → AllReduce` — passes verification
2. **Bug:** Missing AllReduce — output remains Partial, **detected**

```bash
python examples/tp_linear.py
```

### TP MLP (`tp_mlp.py`)

Megatron-style MLP with Column Parallel (gate, up) + Row Parallel (down):

- Column Parallel produces Shard(1) — **no forward communication**
- Row Parallel needs AllReduce — **1 collective in forward**
- Element-wise ops (SiLU, GELU) preserve sharding

```bash
python examples/tp_mlp.py
```

### PP 2-Stage 1F1B (`pp_2stage.py`)

2-stage pipeline with 2 micro-batches:
- Stage 0: `MatMul → Send →`  Stage 1: `Recv → MatMul`
- Send/Recv matching, deadlock freedom, activation liveness
- 1F1B schedule generation (warmup → steady → cooldown)

```bash
python examples/pp_2stage.py
```

### CP Ring Attention (`cp_ring_attn.py`)

Ring Attention with FlashAttention on 2 devices:
- Q replicated, K/V sharded on seq_len
- Ring Send/Recv for K,V rotation
- Local FA + Remote FA → Add → AllReduce

```bash
python examples/cp_ring_attn.py
```

## Temporal Verification

### Overlap Demo (`overlap_demo.py`)

Five cases demonstrating temporal bug detection:

| Case | Bug Type | Detection |
|------|----------|-----------|
| 1 | Data race (read async output before Wait) | MISSING_WAIT |
| 2 | No Wait at all | MISSING_WAIT |
| 3 | Two async ops writing same buffer | BUFFER_ALIASING |
| 4 | Cross-stream AllReduceAsync vs MatMul | MISSING_WAIT |
| 5 | Correct overlap (independent compute + Wait) | SAFE |

```bash
python examples/overlap_demo.py
```

## SPMD Type System

### SPMD Demo (`spmd_demo.py`)

Demonstrates the R/I/V/P SPMD type system:
- Type propagation through compute ops
- Gradient duality verification (R↔P, I↔I, V↔V)
- SPMDGuard assertion checking
- Forbidden operations (Partial × Partial)

```bash
python examples/spmd_demo.py
```

## Synthesis & LLM Frontend

### Synthesis Demo (`synthesis_demo.py`)

Full "Verified Parallelization Synthesis" pipeline:

1. **Input:** compute-only program + sharding spec
2. **Analysis:** find Partial tensors, propose tactics
3. **Synthesis:** beam search over tactic combinations
4. **Selection:** minimal-cost correct program

Also demonstrates the LLM frontend:
- Mock LLM extracts IR from PyTorch code
- Feedback loop: verifier errors → LLM refines

```bash
python examples/synthesis_demo.py
```

## End-to-End Verification

### Megatron GPT-2 (`megatron_gpt2_verify.py`)

Full Megatron-style GPT-2 layer verification:
- Vocab-parallel Embedding with AllReduce
- Tensor-parallel self-attention (Column + Row parallel)
- Tensor-parallel MLP (SiLU gate, up/down projections)
- LayerNorm placement checking
- Complete forward + backward gradient duality

```bash
python examples/megatron_gpt2_verify.py
```

### Consistency Demo (`consistency_demo.py`)

Single-GPU equivalence proof:
- Shows that distributed program produces same results as single-GPU
- Symbolic expression comparison after collective resolution
- Validates that AllReduce/AllGather properly reconstruct full tensors

```bash
python examples/consistency_demo.py
```

## Usage in Your Own Code

```python
from verifier import *
from verifier.temporal import verify_temporal

# 1. Define device mesh and tensor states
mesh = DeviceMesh(shape=(4,), dim_names=("tp",))
spec = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
x = TensorState("x", (B, H), compute_local_shape((B, H), spec),
    spec, requires_grad=True, expr="x")

# 2. Build IR program
prog = Program("my_layer", ops=[
    MatMul(a="x", b="w", output="y_partial"),
    AllReduce(x="y_partial", output="y"),
    LayerNorm(x="y", output="y_norm", norm_dim=-1),
    SiLU(x="y_norm", output="y_act"),
])

# 3. Execute + verify spatial
executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x)
state = executor.run_program(prog)

verifier = DistributedVerifier()
results = verifier.verify_all(prog, state)
print(verifier.summary())

# 4. Verify temporal (if async ops present)
temporal_result = verify_temporal(prog)
if not temporal_result.is_safe:
    for report in temporal_result.reports:
        print(f"VIOLATION: {report.race_type.value} — {report.description}")

# 5. Generate backward + check gradient duality
autograd = AutogradEngine()
bwd = autograd.generate_backward("y_act")
check = autograd.verify_gradient_correctness(prog, bwd)
```
