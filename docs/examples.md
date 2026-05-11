---
title: Examples
nav_order: 6
---

# Examples & Demos

All examples are runnable from the `examples/` directory.

## Spatial Verification

### TP Linear (`tp_linear.py`)

Two cases:
1. **Correct:** Row Parallel with `MatMul → AllReduce` — passes verification
2. **Bug:** Missing AllReduce — output remains PARTIAL, **detected**

```bash
python examples/tp_linear.py
```

### TP MLP (`tp_mlp.py`)

Megatron-style MLP with Column Parallel (gate, up) + Row Parallel (down):

- Column Parallel produces Shard(1) — **no forward communication**
- Row Parallel needs AllReduce — **1 collective in forward**
- Element-wise ops preserve sharding

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

### Expected output (case 5 — correct):

```
Temporal Verification: SAFE
  Ops: 5 total, 1 async
  HB edges: 7
  Violations: 0
```

## Synthesis & LLM Frontend

### Synthesis Demo (`synthesis_demo.py`)

Demonstrates the full "Verified Parallelization Synthesis" pipeline:

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

### Synthesis result example:

```
Row Parallel synthesis:  SUCCESS
  Best program: MatMul(x, w) → y  +  AllReduce(y) → y_reduced
  Cost: AR=1 (total=2)
  Final verification: PASSED
```

## Usage in your own code

```python
from verifier import *
from verifier.temporal import verify_temporal

# 1. Define device mesh and tensor states
mesh = DeviceMesh(shape=(4,), dim_names=("tp",))
x = TensorState("x", (B, H), (B, H//4),
    ShardingSpec((Shard(dim=1),), mesh), requires_grad=True)

# 2. Build IR program
prog = Program("my_layer")
prog.add(MatMul("x", "w", "y_partial"))
prog.add(AllReduce("y_partial", "y"))

# 3. Execute + verify spatial
executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x)
state = executor.run_program(prog)

verifier = DistributedVerifier()
verifier.verify_all(prog, state)

# 4. Verify temporal (if async ops)
temporal_result = verify_temporal(prog)
if not temporal_result.is_safe:
    for report in temporal_result.reports:
        print(f"VIOLATION: {report.race_type.value} — {report.description}")
```
