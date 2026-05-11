---
title: Architecture
nav_order: 3
---

# Architecture

## Design Philosophy

{: .important }
> **LLM proposes, Verifier checks.** LLM generates candidate IR and parallelization tactics; formal verification (Z3 SMT) is the final correctness arbiter.

The system is built on four core principles:

1. **Explicit over implicit** — all sharding, placement, and communication must be represented in IR
2. **Symbolic over numeric** — verify properties over all possible inputs, not just specific values
3. **Z3 as oracle, not runtime** — Z3 checks correctness; a separate symbolic executor handles propagation
4. **Two-dimensional verification** — spatial (placement) and temporal (overlap) are orthogonal concerns

## System Pipeline

```
                        TileLang TIR / PyTorch Code
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LLM Frontend       │  PyTorch → IR extraction
                    │   (llm_frontend.py)  │  + feedback refinement loop
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   TIR Lifter         │  TileLang TIR → Distributed IR
                    │   (tir_lifter.py)    │  Block-level access analysis
                    └─────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       Verification IR          │
              │         (ir.py)                │
              │                               │
              │  Compute: MatMul, FA, SiLU...  │
              │  Collective: AllReduce, AG, RS │
              │  P2P: Send, Recv               │
              │  Async: ARAsync, Wait, SendAsync│
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │ Spatial Verifier │           │ Temporal Verifier│
    │ (solver.py)      │           │ (temporal.py)    │
    │                  │           │                  │
    │ • Postcondition  │           │ • Data Race      │
    │ • Comm Legality  │           │ • Missing Wait   │
    │ • Gradient Duality│          │ • Buffer Aliasing│
    │ • Placement Cons. │           │ • Dep. Violation │
    │ • Shape Cons.     │           │                  │
    │ • PP Deadlock     │           │                  │
    └────────┬─────────┘           └────────┬─────────┘
             │                              │
             └──────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │  Synthesis Engine │
              │  (synthesis.py)   │
              │                   │
              │  Tactic Proposer  │
              │  Search/Refine    │
              │  Cost Model       │
              │  Beam Search      │
              └──────────────────┘
```

## Module Map

| Module | Lines | Role |
|--------|-------|------|
| [`state.py`](modules/state) | 248 | `TensorState`, `ShardingSpec`, `DeviceMesh`, placement types |
| [`ir.py`](modules/ir) | ~1400 | All IR ops with forward+VJP, sync + async variants |
| [`executor.py`](modules/executor) | 316 | Multi-device symbolic executor |
| [`autograd.py`](modules/autograd) | 368 | Formal VJP autograd + gradient duality |
| [`solver.py`](modules/solver) | 572 | Z3 spatial verification engine |
| [`temporal.py`](modules/temporal) | ~580 | HB graph + Z3 temporal (race) detection |
| [`rewrite.py`](modules/rewrite) | 574 | Pattern matching, placement analysis, rewrite rules |
| [`synthesis.py`](modules/synthesis) | 539 | Verified parallelization synthesis |
| [`llm_frontend.py`](modules/llm-frontend) | 658 | PyTorch → IR with LLM + feedback loop |
| [`tir_lifter.py`](modules/tir-lifter) | 656 | TileLang TIR → distributed IR |
| [`schedules.py`](modules/schedules) | 397 | 1F1B schedule + deadlock checker |

## Data Flow

### Spatial verification flow

```
TensorState(input) ──> IROp.apply() ──> TensorState(output)
     │                      │                    │
     │ ShardingSpec         │ Placement          │ New placement
     │ DeviceMesh           │ propagation        │ Updated shape
     │ Local shape          │ VJP rule           │ Symbolic expr
     │                      │                    │
     └──────────────────────┴────────────────────┘
                     Executor tracks per-device state
                            │
                            ▼
                     Z3 Solver checks:
                     - Not Partial at boundaries
                     - Collectives on valid inputs
                     - Gradient duality
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
```

## Key Abstractions

### Placement System

```
Placement = Shard(dim) | Replicate() | Partial()

Shard(0):  tensor split along dim 0 across devices
Shard(1):  tensor split along dim 1 across devices
Replicate: full copy on each device
Partial:   locally computed sum, needs AllReduce
```

### Parallelism Patterns

| Pattern | Input Placement | Weight Placement | Output | Fwd Comm | Bwd Comm |
|---------|----------------|-----------------|--------|----------|----------|
| Row Parallel | Shard(1) | Shard(0) | Partial → AR → Replicate | AllReduce | AllReduce |
| Column Parallel | Replicate | Shard(1) | Shard(1) | None | AllReduce |
| Data Parallel | Shard(0) | Replicate | Shard(0) | None | AllReduce |
| Sequence Parallel | Shard(1) | Replicate | Shard(1) | AllGather | ReduceScatter |

### Gradient Duality

| Forward Collective | Backward Dual |
|---|---|
| `AllReduce(x, "sum")` | `AllReduce(grad_x, "sum")` |
| `AllGather(x, dim)` | `ReduceScatter(grad_x, dim)` |
| `ReduceScatter(x, dim)` | `AllGather(grad_x, dim)` |
| `Send(x, src→dst)` | `Recv(grad_x, dst→src)` |
| `Recv(x, src→dst)` | `Send(grad_x, dst→src)` |
