# Verifier Development Guide

This skill encodes the core invariants, design patterns, and verification checklist for the LLM Infra Verifier project. Invoke it before implementing new IR ops, modifying TensorState, or writing/reviewing tests.

---

## 1. Architecture Overview

The verifier performs **static symbolic analysis** of distributed LLM programs. No real tensors flow — only `TensorState` metadata (shape, placement, dtype, SPMD type) propagates through an IR op graph.

```
TensorState  ─→  IR Ops  ─→  Executor (per-device simulation)
    │                │               │
    │                │               └─ MultiDeviceExecutor: runs ops on virtual DeviceMesh
    │                └─ IROp.apply(ctx) → TensorState  (forward)
    │                   IROp.vjp(ctx, grad) → {name: TensorState}  (backward)
    └─ ShardingSpec(placements, mesh) → compute_local_shape()
```

**Key files:**
- `verifier/state/tensor.py` — TensorState dataclass
- `verifier/ir/base.py` — IROp abstract base
- `verifier/ir/collective.py` — CollectiveOp base + all collective ops
- `verifier/executor.py` — MultiDeviceExecutor
- `verifier/autograd.py` — AutogradEngine (tape + VJP + duality)

---

## 2. TensorState Invariants

### 2.1 local_shape Must Be Derived, Never Hardcoded

`TensorState.__post_init__` enforces:
```python
expected = compute_local_shape(self.global_shape, self.sharding)
if self.local_shape != expected:
    raise ValueError(...)
```
**Never manually compute local_shape**. Always use `compute_local_shape(global_shape, spec)`.

### 2.2 New Fields Must Be Optional with None Default

All metadata fields beyond the core 4 (name, global_shape, local_shape, sharding) must be `Optional[X] = None`. This guarantees backward compatibility — existing TensorState constructors continue to work.

### 2.3 Hash Contract

`__hash__` uses: `(name, global_shape, local_shape, dtype, placements, mesh.shape, mesh.dim_names)`. If you add a field that affects tensor identity, add it to the hash tuple.

### 2.4 SPMD Type Auto-Derivation

`local_type` is auto-derived from placements in `__post_init__`:
- Has `Partial` placement → `PARTIAL`
- Has `Shard` placement → `VARYING`
- All `Replicate` → `REPLICATE`

`INVARIANT` cannot be auto-derived; it requires explicit `Reinterpret(R→I)`.

---

## 3. IR Op Patterns

### 3.1 Two Base Classes

| Base | When to Use | Key Methods |
|------|-------------|-------------|
| `IROp` | Non-collective ops (compute, bookkeeping, dtype) | `apply()`, `vjp()`, `input_names`, `output_name`, `clone_with_names()` |
| `CollectiveOp(IROp)` | Ops with cross-device communication | Inherits IROp + `_transform_placements()`, `_validate()`, `_validate_spmd()`, `_make_grad()` |

### 3.2 CollectiveOp Template

Subclasses of `CollectiveOp` get a standard `apply()` for free:
1. `_validate(x)` — precondition checks (e.g., AllReduce requires Partial)
2. `_validate_spmd(x)` — SPMD type precondition
3. `_transform_placements(placements, x)` — the core placement transformation
4. Auto-constructs output TensorState with new spec + `compute_local_shape()`

You only need to implement `_transform_placements()`, `vjp()`, `clone_with_names()`.
Optionally override `_validate()` for precondition checks.

### 3.3 Custom apply() Pattern

For ops that need to annotate extra metadata on the output (e.g., MoEDispatch sets `num_experts`):
```python
def apply(self, ctx):
    result = super().apply(ctx)       # CollectiveOp standard pipeline
    result.num_experts = self.num_experts  # Annotate after
    return result
```

For non-collective ops with custom output shapes (e.g., TopKGate dual output):
```python
def apply(self, ctx):
    x = ctx[self.x]
    result = replace(x, name=self.output, ...)
    ctx[self.output] = result           # MUST store in ctx
    # ... create second output ...
    ctx[self.indices_output] = indices   # dual output
    return result
```

**Critical**: Always store result in `ctx[self.output]`. The executor reads from ctx.

### 3.4 Placement Transformation Rules

| Collective | Transform | Pattern |
|-----------|-----------|---------|
| AllReduce | Partial → Replicate | `Replicate() if isinstance(p, Partial) else p` |
| AllGather | Shard(dim) → Replicate | `Replicate() if isinstance(p, Shard) and p.dim == gather_dim else p` |
| ReduceScatter | Replicate/Partial → Shard(dim) | First R/P becomes Shard; rest unchanged |
| AllToAll / MoE | Shard(split) → Shard(concat) | `Shard(concat) if isinstance(p, Shard) and p.dim == split else p` |

**2D mesh rule**: Transformation ONLY affects the matching mesh dimension. A `(Shard(1), Shard(0))` on mesh `(tp, dp)` where you gather `dim=0` (DP axis): only the second placement changes → `(Shard(1), Replicate())`. The TP placement is untouched.

### 3.5 Composite Op Pattern (expand)

For complex ops decomposable into primitives (e.g., RingAttention):
```python
def apply(self, ctx):       # Quick verification: directly produces final result
def expand(self) -> List[IROp]:  # Temporal analysis: primitive op sequence
```
Executor dispatches `expand()`:
```python
elif isinstance(op, RingAttention):
    for sub in op.expand():
        self._execute_op(sub)
```

---

## 4. VJP (Backward) Rules

### 4.1 Structural Invariant

**For every op, the gradient of each input MUST have the same `global_shape`, `local_shape`, and `sharding.placements` as that forward input.** This is the fundamental correctness property.

```python
# In vjp():
grad = TensorState(
    name=f"grad_{self.x}",
    global_shape=x.global_shape,    # MUST match forward input
    local_shape=x.local_shape,      # MUST match forward input
    sharding=x.sharding,            # MUST match forward input
    expr=...,
)
```

CollectiveOp provides `_make_grad(x, expr)` helper that enforces this automatically.

### 4.2 Autograd Duality

Every forward collective has a backward dual:

| Forward | Backward Dual | Dim Relationship |
|---------|---------------|------------------|
| AllReduce | AllReduce | Self-dual |
| AllGather(dim) | ReduceScatter(dim) | Same dim |
| ReduceScatter(dim) | AllGather(dim) | Same dim |
| Send(src→dst) | Recv(dst→src) | Reversed |
| ZeROGatherParam(dim) | ZeROScatterGrad(dim) | Same dim |
| RingRotate(size) | RingRotate(size) | Self-dual, same ring_size |
| MoEDispatch(s,c) | MoECombine(c,s) | Dims swapped |
| FP8Quantize(src→dst) | FP8Dequantize(dst→src) | Reversed dtype + scale |
| FP8Dequantize(src→dst) | FP8Quantize(dst→src) | Reversed dtype + scale |

`_is_dual(fwd, bwd)` checks structural match. `_generate_dual_collective(entry, grads)` produces the backward op.

### 4.3 dtype in VJP

Cast VJP reverses dtype: `Cast(fp32→fp16)` backward is `Cast(fp16→fp32)`.
LossScale VJP reverses direction: `scale` → `unscale`.
FP8Quantize VJP dequantizes: `FP8Quantize(fp32→fp8e4m3)` backward is dequantize (fp8→fp32), clears `fp8_scale_expr`.
FP8Dequantize VJP quantizes: `FP8Dequantize(fp8e4m3→fp32)` backward is quantize (fp32→fp8), attaches `fp8_scale_expr`.
AmaxUpdate VJP is empty (observation-only, not differentiable).

### 4.4 FP8 Conventions

- Forward activations/weights: `fp8e4m3` (higher precision, 4 exponent + 3 mantissa bits)
- Backward gradients: `fp8e5m2` (wider dynamic range, 5 exponent + 2 mantissa bits)
- `DtypeGuard.check_fp8_format_usage(tensor, phase)` enforces this convention
- Per-tensor scale is symbolic (`scale_expr: str`), tracked on `TensorState.fp8_scale_expr`
- Delayed scaling: `AmaxUpdate` at iter N must happen-before `FP8Quantize` at iter N+1
- `DtypeGuard.check_fp8_scale_freshness(quantize_idx, amax_idx)` verifies ordering

---

## 5. Executor Dispatch

### 5.1 Adding a New Op

In `executor.py._execute_op()`, add an isinstance branch. Prefer reusing existing handlers:

| New Op | Reuse Handler | Reason |
|--------|--------------|--------|
| ZeROGatherParam | `_exec_allgather` | Same gather semantics |
| ZeROScatterGrad | `_exec_reducescatter` | Same scatter semantics |
| Cast, LossScale, ExpertCompute, ZeROPartitionOptState | `_exec_unary` | Shape/device-independent |
| FP8Quantize, FP8Dequantize, AmaxUpdate | `_exec_unary` | Shape/device-independent |
| MoEDispatch, MoECombine | `_exec_collective_unary` | Standard collective |
| RingAttentionStep | `_exec_flash_attn` | Same QKV→output pattern |

Only write a new handler when the semantics diverge:
- `_exec_ring_rotate`: needs cross-device data rotation (snapshot all → permute)
- `_exec_topk_gate`: needs dual-output storage

### 5.2 Ring Rotation

```python
def _exec_ring_rotate(self, op):
    current = {did: deepcopy(dev.get(op.x)) for ...}  # Snapshot FIRST
    for did in self.devices:
        src = (did - 1) % op.ring_size   # ring_size, NOT num_devices
        rotated = deepcopy(current[src])
        ...
```

**Wrap uses `ring_size`, not `num_devices`.** The ring may be smaller than the mesh.

---

## 6. Test Checklist

Every new op or extension MUST have tests covering all of these:

### 6.1 Unit Tests (per op)

- [ ] `apply()` produces correct output shape, local_shape, placements
- [ ] `apply()` stores result in ctx
- [ ] Metadata annotation (num_experts, ring_step, dtype, zero_stage, etc.)
- [ ] `is_collective()` returns correct value
- [ ] `clone_with_names()` preserves all op-specific fields
- [ ] `propagate_spmd_type()` returns expected type

### 6.2 VJP Tests (per op)

**Never just check key existence.** Always verify structural correctness:
```python
def _assert_grad_matches_fwd(grad, fwd):
    assert grad.global_shape == fwd.global_shape
    assert grad.local_shape == fwd.local_shape
    assert grad.sharding.placements == fwd.sharding.placements
```
Plus op-specific checks:
- Cast VJP: `grad.dtype == src_dtype`
- LossScale VJP: reversed direction in expr
- Ring VJP: `ring_step` preserved
- Collective VJP: dual op name in expr (e.g., `"ZeROScatterGrad" in expr`)

### 6.3 Executor Integration Tests

- [ ] Run op on a real `MultiDeviceExecutor` + `DeviceMesh`
- [ ] Verify all devices produce correct results (`for did in range(mesh.num_devices)`)
- [ ] End-to-end pipeline test combining multiple ops (e.g., Gate→Dispatch→Compute→Combine)

### 6.4 Multi-Dim Mesh Tests

- [ ] Test on 2D mesh (e.g., `(2, 4)` for TP+DP or TP+EP)
- [ ] Verify only the targeted mesh dim changes placement
- [ ] Verify untouched dims remain exactly as before

### 6.5 Autograd Duality Tests

- [ ] `_is_dual(fwd_op, bwd_op)` returns True for correct pair
- [ ] `_is_dual(fwd_op, wrong_bwd)` returns False (dim/size mismatch)
- [ ] Full flow: `record() → generate_backward() → verify bwd op type + fields`

### 6.6 Anti-Patterns to Avoid

- **Empty assertions**: `assert result is not None` proves nothing. Verify shape/placement/metadata.
- **Hardcoded device counts**: Use `ring_size`/`num_experts` from the op, not magic numbers.
- **Shallow VJP tests**: `assert "key" in grads` without structural verification.
- **Missing dim verification**: `any(isinstance(p, Shard))` without checking `p.dim`.
- **No local_shape check**: Shape is the most concrete invariant — always verify it changes correctly.

### 6.7 Data Flow Verification Pattern

For ops that move data between devices (ring rotate, send/recv), use expr tagging:
```python
for did in range(mesh.num_devices):
    exe.devices[did].tensors["k"].expr = f"k_dev{did}"  # Tag each device

# After op execution:
for did in range(mesh.num_devices):
    expected_src = (did - 1) % ring_size  # Compute from ring_size
    assert result.expr == f"k_dev{expected_src}"
```

---

## 7. Step-by-Step: Adding a New Extension

1. **TensorState fields**: Add `Optional[X] = None` fields to `tensor.py`. Update `__hash__` if identity-affecting.

2. **IR ops**: Create `verifier/ir/<extension>.py`. Use `CollectiveOp` for communication ops, `IROp` for bookkeeping/compute.

3. **Exports**: Add to `verifier/ir/__init__.py` (import + `__all__`) and `verifier/__init__.py`.

4. **Executor**: Add isinstance dispatch in `_execute_op()`. Reuse handlers when possible.

5. **Autograd**: Add `_generate_dual_collective()` elif + `_is_dual()` isinstance check in `autograd.py`.

6. **Tests**: Write `tests/test_<extension>.py` covering ALL items in Section 6.

7. **Verify**:
   ```bash
   python -m pytest tests/ -q                        # Full regression
   python -m pytest tests/test_<extension>.py -v      # New tests
   ```
