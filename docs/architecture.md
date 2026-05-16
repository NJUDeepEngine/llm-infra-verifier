---
title: Architecture
nav_order: 3
---

# Architecture

## Design Philosophy

{: .important }
> **LLM proposes, Verifier checks.** LLM generates candidate IR and parallelization tactics; formal verification (Z3 SMT) is the final correctness arbiter.

The system is built on five core principles:

1. **Symbolic over numeric** — verify properties for ALL possible inputs, not samples
2. **LLM proposes, Verifier decides** — Z3 is the final authority, not model confidence
3. **Spatial × Temporal** — independent dimensions, composed results
4. **Every op has VJP** — forward and backward are equally verifiable
5. **Zero dependencies beyond Z3** — no PyTorch, no CUDA, no cluster needed

## System Pipeline

```
                    TileLang TIR / PyTorch Code
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
    ┌─────────────────────┐          ┌─────────────────────┐
    │   LLM Frontend       │          │   TIR Lifter         │
    │   (llm_frontend.py)  │          │   (tir_lifter.py)    │
    │   PyTorch → IR       │          │   TileLang TIR → IR  │
    │   + feedback loop    │          │   Block-level access  │
    └─────────┬───────────┘          └─────────┬───────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
              ┌───────────────────────────────────┐
              │         Verification IR            │
              │         (verifier/ir/)             │
              │                                   │
              │  48 ops across 10 sub-modules:     │
              │  compute, collective, p2p,         │
              │  async_ops, shape, spmd,           │
              │  precision, zero, cp, moe          │
              └───────────────┬───────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Spatial Verifier │           │ Temporal Verifier │
    │ (solver.py)      │           │ (temporal.py)     │
    │                  │           │                   │
    │ • Postcondition  │           │ • Data Race       │
    │ • Comm Legality  │           │ • Missing Wait    │
    │ • Gradient Dual  │           │ • Buffer Aliasing │
    │ • Placement Cons │           │ • Dep. Violation  │
    │ • Shape Cons     │           │                   │
    │ • PP Deadlock    │           │                   │
    └────────┬─────────┘           └────────┬─────────┘
             │                               │
             └──────────┬────────────────────┘
                        ▼
              ┌──────────────────┐
              │  Synthesis Engine │
              │  (synthesis.py)   │
              │                   │
              │  Tactic Proposer  │
              │  Beam Search      │
              │  Cost Model       │
              └──────────────────┘
```

## Module Map

| Module | Role |
|--------|------|
| [`verifier/state/`](modules#state) | `TensorState`, `ShardingSpec`, `DeviceMesh`, placements, `LocalSPMDType` |
| [`verifier/ir/`](modules#ir) | 48 IR ops across 10 sub-modules (compute, collective, p2p, async, shape, spmd, precision, zero, cp, moe) |
| [`verifier/executor.py`](modules#executor) | Multi-device symbolic executor with registry-based dispatch |
| [`verifier/autograd.py`](modules#autograd) | Formal VJP autograd engine + gradient duality verification |
| [`verifier/solver.py`](modules#solver) | Z3 spatial verification (6 checks, 3 levels: L0 placement, L1 shape, L2 slice) |
| [`verifier/temporal.py`](modules#temporal) | HB graph + Z3 temporal race detection (4 checks) |
| [`verifier/rewrite.py`](modules#rewrite) | Pattern matching, placement analysis, cost model, rewrite rules |
| [`verifier/synthesis.py`](modules#synthesis) | Verified parallelization synthesis via beam search |
| [`verifier/llm_frontend.py`](modules#llm-frontend) | PyTorch → IR with LLM + feedback refinement loop |
| [`verifier/tir_lifter.py`](modules#tir-lifter) | TileLang TIR → distributed IR lifter |
| [`verifier/schedules.py`](modules#schedules) | 1F1B pipeline schedule + deadlock checker (DFS cycle detection) |

## Data Flow

### Spatial verification flow

```
TensorState(input) ──> IROp.apply() ──> TensorState(output)
     │                      │                    │
     │ ShardingSpec         │ Placement          │ New placement
     │ DeviceMesh           │ propagation        │ Updated shape
     │ Local shape          │ VJP rule           │ Symbolic expr
     │ dtype                │                    │ dtype propagated
     │                      │                    │
     └──────────────────────┴────────────────────┘
                     Executor tracks per-device state
                            │
                            ▼
                     Z3 Solver checks:
                     - Not Partial at boundaries
                     - Collectives on valid inputs
                     - Gradient duality holds
                     - Placement consistency
                     - Shape divisibility
                     - PP deadlock freedom
```

### Temporal verification flow

```
IR Program
    │
    ▼
TemporalGraph ──> Events[i] = {issue_time, complete_time, reads, writes, stream}
    │
    ├── Program order: same stream → sequential
    ├── Wait sync: async complete < Wait issue
    └── Data deps: writer complete < reader issue
    │
    ▼
Happens-Before Graph (Z3 constraints)
    │
    ▼
RaceDetector ──> For each pair of unordered ops on different streams:
                 - Same buffer + ≥1 write → DATA RACE
                 - Async output read before Wait → MISSING WAIT
                 - Two async writes to same buffer → BUFFER ALIASING
                 - Recv before matching Send → DEPENDENCY VIOLATION
```

## Key Abstractions

### Placement System

```
Placement = Shard(dim) | Replicate() | Partial()

Shard(0):   tensor split along dim 0 across devices
Shard(1):   tensor split along dim 1 across devices
Replicate:  full copy on each device
Partial:    locally computed sum, needs AllReduce
```

### SPMD Type System

Four local types with gradient duality:

| Type | Meaning | Gradient Dual |
|------|---------|---------------|
| **R** (Replicate) | Same data on all ranks | P (Partial) |
| **I** (Invariant) | Same data, no gradient comm needed | I (Invariant) |
| **V** (Varying) | Different data per rank | V (Varying) |
| **P** (Partial) | Pending sum across ranks | R (Replicate) |

### Parallelism Patterns

| Pattern | Input | Weight | Output | Fwd Comm | Bwd Comm |
|---------|-------|--------|--------|----------|----------|
| Row Parallel | Shard(1) | Shard(0) | Partial → AR → R | AllReduce | AllReduce |
| Column Parallel | Replicate | Shard(1) | Shard(1) | None | AllReduce |
| Data Parallel | Shard(0) | Replicate | Shard(0) | None | AllReduce |
| Sequence Parallel | Shard(1) | Replicate | Shard(1) | AllGather | ReduceScatter |
| Vocab Parallel | Replicate | Shard(0) | Partial → AR → R | AllReduce | — |

### Gradient Duality

| Forward Collective | Backward Dual |
|---|---|
| `AllReduce` | `AllReduce` (self-dual) |
| `AllGather` | `ReduceScatter` |
| `ReduceScatter` | `AllGather` |
| `Broadcast` | `Reduce` |
| `Reduce` | `Broadcast` |
| `AllToAll` | `AllToAll` (dim-swap) |
| `Scatter` | `Gather` |
| `Gather` | `Scatter` |
| `Send(src→dst)` | `Recv(dst→src)` |
| `Recv(src→dst)` | `Send(dst→src)` |
