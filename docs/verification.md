---
title: Verification
nav_order: 4
---

# Verification Capabilities

## Spatial Verification (6 checks)

### 1. Postcondition

Ensures output tensors meet expected properties.

{: .note }
The most fundamental check: output tensors must not be `PARTIAL` at program boundaries.

**Z3 encoding:**
```
Bool("partial") == tensor.partial
Assert(partial == True)   // negated postcondition
Result: unsat → passes, sat → counterexample found
```

**What it catches:**
- RowParallel without AllReduce → output remains PARTIAL
- Ring Attention without final AllReduce → partial sums

### 2. Communication Legality

Ensures collectives are only called on valid input states.

| Collective | Requires |
|---|---|
| `AllReduce(x)` | `x` must be `PARTIAL` |
| `AllGather(x, dim)` | `x` must be `Shard(dim)` |
| `ReduceScatter(x, dim)` | `x` must be `Replicate()` on that dim |
| `Send(x)` | Must have matching `Recv` |
| `Recv(x)` | Must have matching `Send` |

### 3. Gradient Duality

Ensures every forward collective has a matching backward dual.

**Key insight:** `AllReduce(sum)` is **self-dual** — its VJP is itself. `AllGather` ↔ `ReduceScatter`. `Send` ↔ `Recv` (reversed).

### 4. Placement Consistency

Ensures output placement follows from input placements through propagation rules.

For example: `MatMul(Shard(1), Shard(0))` must produce `Partial`, not `Replicate` or `Shard(1)`.

### 5. Shape Consistency

Ensures shapes propagate correctly through all ops, including collectives that change local shapes (AllGather, ReduceScatter).

### 6. PP Deadlock Freedom

Ensures:
- Every `Send` has a matching `Recv`
- Communication graph has no circular waits
- DFS cycle detection on the wait-for graph

## Temporal Verification (5 checks)

### 1. Data Race

{: .warning }
Two ops on **different streams** access the **same tensor**, at least one writes, and they are **not ordered** by Happens-Before.

**Detection:** Build HB graph → for each unordered pair on different streams → check if intervals overlap → flag race.

### 2. Missing Wait

{: .warning }
An async op's output is read by compute **before** `Wait(handle)` completes.

**Common pattern in Megatron:**
```python
# BUG
handle = dist.all_reduce(grad, async_op=True)
optimizer.step()  # reads grad — but AllReduce may not be done!
handle.wait()     # too late
```

### 3. Buffer Aliasing

{: .warning }
Two async ops write to the **same output buffer**, and the first result is not consumed before the second write.

**Real-world pattern:**
```python
buf = torch.empty(...)
h1 = dist.all_reduce(g1, out=buf, async_op=True)  # starts writing buf
h2 = dist.all_reduce(g2, out=buf, async_op=True)  # also writes buf! BUG!
```

### 4. Dependency Violation

Ensures required ordering (e.g., `SendAsync` must complete before matching `RecvAsync` starts) is not violated by the HB graph.

### 5. Orphaned Handle

{: .warning }
An async op (e.g., `AllReduceAsync`) produces a handle that is **never consumed by any `Wait` or `WaitAll`**, meaning the async result may never be synchronized.

**Detection:** Collect all handles created by async ops, subtract handles referenced by Wait/WaitAll, flag remaining as orphans.

**Common pattern:**
```python
# BUG
handle = dist.all_reduce(grad, async_op=True)
# ...other computation...
# handle.wait() never called! Result may be incomplete
next_layer(grad)
```

## Happens-Before Model

Each IR op is modeled as an interval `[issue_time, complete_time]`:

| Op Type | Constraint |
|---|---|
| Sync compute (`MatMul`, etc.) | `issue == complete` (atomic) |
| Async (`AllReduceAsync`, etc.) | `issue < complete` (has duration) |
| `Wait` | `async_complete < Wait_issue` |
| Same stream | `complete_i < issue_{i+1}` |
| Data dependency | writer's `complete` < reader's `issue` |

Z3 solves for a satisfying assignment of all `issue`/`complete` times. If the solver can find a schedule where two unordered ops overlap on the same buffer, it's a violation.
