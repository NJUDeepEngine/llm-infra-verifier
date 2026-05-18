---
title: Module Reference
nav_order: 7
---

# Module Reference

## Core State

### `verifier/state/` — Tensor State & Placements {#state}

Central data structures for the entire framework.

```python
# Placement types
Shard(dim=0)       # Tensor split along dim 0
Shard(dim=1)       # Tensor split along dim 1
Replicate()        # Full copy on each device
Partial()          # Locally computed, needs AllReduce

# SPMD local types (with gradient duality)
LocalSPMDType.R    # Replicate (dual: P)
LocalSPMDType.I    # Invariant (dual: I)
LocalSPMDType.V    # Varying   (dual: V)
LocalSPMDType.P    # Partial   (dual: R)

# Device topology
DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"))  # 2D mesh: 8 devices

# Sharding specification
ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

# Core state
TensorState(
    name="x",
    global_shape=(8, 16),      # before sharding
    local_shape=(8, 8),        # on this device
    sharding=spec,
    expr="x",                  # symbolic expression
    requires_grad=True,
    dtype="bf16",              # optional: None, fp16, bf16, fp8e4m3, fp8e5m2
    stage=0,                   # PP stage (optional)
    _async_handle="h1",       # in-flight async handle (optional)
)

# Utility
compute_local_shape(global_shape, spec)  # Derive local shape from sharding
compute_tensor_slices(tensor)            # Per-device slice information
```

---

## IR Operations

### `verifier/ir/` — 48 Operations across 10 Sub-modules {#ir}

Every `IROp` implements:
- `apply(ctx) → TensorState` — forward placement propagation
- `vjp(ctx, grad_output) → {input: grad}` — vector-Jacobian product
- `propagate_spmd_type(input_types) → type` — SPMD type propagation
- `input_names` / `output_name` — tensor name bindings
- `clone_with_names(input_map, output_name)` — structural clone
- `is_collective()` / `is_p2p()` / `is_async()` — classification

### `ir/compute.py` — Compute Ops (13)

| Op | Sharding Semantics | VJP |
|---|---|---|
| `MatMul(a, b, output)` | S(1)×S(0)→P, R×S(1)→S(1), S(0)×R→S(0) | MatMul with transposed weights |
| `Add(a, b, output)` | Merge placements; P×P forbidden | Identity to both |
| `Multiply(a, b, output)` | Merge placements; P×P forbidden | Scale by other input |
| `SiLU(x, output)` | Pass-through | SiLU derivative × grad |
| `GELU(x, output)` | Pass-through | GELU derivative × grad |
| `ReLU(x, output)` | Pass-through | ReLU derivative × grad |
| `Dropout(x, output, p)` | Pass-through | Same mask |
| `LayerNorm(x, output, norm_dim)` | Error if Shard(norm_dim) | Same placement as input |
| `RMSNorm(x, output, norm_dim)` | Error if Shard(norm_dim) | Same placement as input |
| `Softmax(x, output, dim)` | Error if Shard(reduction_dim) | Same placement as input |
| `FlashAttention(q, k, v, output)` | Follows Q placement | Through Q/K/V |
| `Embedding(indices, weight, output)` | weight Shard(0) → Partial | grad_weight same as weight |
| `CrossEntropyLoss(logits, labels, output)` | logits Shard → scalar Partial | grad same as logits |

### `ir/collective.py` — NCCL Collectives (8)

All collectives support `mesh_dim: Optional[int] = None` for multi-dim meshes. When set, only the targeted mesh dimension is transformed; other dimensions pass through.

| Op | Forward | Backward Dual |
|---|---|---|
| `AllReduce(x, output, mesh_dim)` | Partial → Replicate | AllReduce (self) |
| `AllGather(x, output, dim, mesh_dim)` | Shard(d) → Replicate | ReduceScatter |
| `ReduceScatter(x, output, dim, mesh_dim)` | R/P → Shard(d) | AllGather |
| `Broadcast(x, output)` | any → Replicate | Reduce |
| `Reduce(x, output)` | Partial → Replicate(root) | Broadcast |
| `AllToAll(x, output, split_dim, concat_dim, mesh_dim)` | Shard(split) → Shard(concat) | AllToAll (dim-swap) |
| `Scatter(x, output, dim, mesh_dim)` | Replicate → Shard(d) | Gather |
| `Gather(x, output, dim, mesh_dim)` | Shard(d) → Replicate | Scatter |

### `ir/p2p.py` — Point-to-Point (4)

| Op | Description |
|---|---|
| `Send(x, output, src, dst)` | Synchronous P2P send |
| `Recv(x, output, src, dst)` | Synchronous P2P receive |
| `SendAsync(x, output, src, dst, handle)` | Non-blocking send |
| `RecvAsync(x, output, src, dst, handle)` | Non-blocking receive |

### `ir/async_ops.py` — Async & Overlap (4)

| Op | Description |
|---|---|
| `AllReduceAsync(x, output, handle, stream)` | Non-blocking AllReduce |
| `Wait(handle, tensor, output)` | Wait for async completion |
| `WaitAll(handles, tensors, outputs)` | Wait for multiple handles |
| `OverlapRegion(ops, stream)` | Compute/comm overlap region |

### `ir/precision.py` — Mixed Precision (5)

| Op | Description |
|---|---|
| `Cast(x, output, dtype)` | Dtype conversion |
| `LossScale(x, output, scale)` | Loss scaling for mixed precision |
| `FP8Quantize(x, output, scale_expr, src_dtype, dst_dtype)` | Quantize to FP8 |
| `FP8Dequantize(x, output, scale_expr, src_dtype, dst_dtype)` | Dequantize from FP8 |
| `AmaxUpdate(x, output, tensor_name, iteration_expr)` | Delayed scaling amax |

### `ir/shape.py` — Shape Operations (2)

| Op | Description |
|---|---|
| `Reshape(x, output, new_shape)` | Reshape with placement propagation |
| `Transpose(x, output, dim0, dim1)` | Transpose with shard dim update |

### `ir/spmd.py` — SPMD Type System (3)

| Op | Description |
|---|---|
| `Reinterpret(x, output, new_type)` | Change SPMD type annotation |
| `Convert(x, output, target_type)` | Convert between SPMD types with comm |
| `SPMDGuard(x, output, expected_type)` | Assert SPMD type (error on mismatch) |

### `ir/zero.py` — ZeRO Parallelism (3)

| Op | Description |
|---|---|
| `ZeROGatherParam(x, output)` | AllGather partitioned parameter |
| `ZeROScatterGrad(x, output)` | ReduceScatter gradient |
| `ZeROPartitionOptState(x, output)` | Partition optimizer state |

### `ir/cp.py` — Context Parallelism (3)

| Op | Description |
|---|---|
| `RingRotate(x, output, direction)` | Rotate KV along ring |
| `RingAttentionStep(q, k, v, output, step)` | Single ring attention step |
| `RingAttention(q, k, v, output, num_steps)` | Full ring attention |

### `ir/moe.py` — Mixture of Experts (4)

| Op | Description |
|---|---|
| `TopKGate(x, output, num_experts, top_k)` | Expert routing |
| `MoEDispatch(x, output, gate)` | AllToAll token dispatch |
| `MoECombine(x, output, gate)` | AllToAll token combine |
| `ExpertCompute(x, output, expert_id)` | Per-expert computation |

---

## Execution & Verification

### `verifier/executor.py` — Multi-Device Executor {#executor}

Symbolic executor with registry-based dispatch tracking per-device `TensorState`:

```python
executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x_tensor)
executor.register_tensor(w_tensor)
state = executor.run_program(program)  # Dict[str, TensorState]

# Access per-device state
dev0_tensor = executor.get_tensor("y", device_id=0)
all_devices = executor.final_state()
```

### `verifier/autograd.py` — Formal Autograd Engine {#autograd}

VJP-based gradient computation with duality verification:

```python
autograd = AutogradEngine()
for op in fwd.ops:
    autograd.record(op, tensor_states)

bwd_program = autograd.generate_backward("loss")
check = autograd.verify_gradient_correctness(fwd, bwd_program)
# check.passed, check.collective_pairs, check.errors
```

### `verifier/solver.py` — Z3 Spatial Verifier (6 checks) {#solver}

Encoding verification conditions as SMT formulas. Supports multi-dimensional meshes (e.g. TP×DP) with per-dim Z3 variables and `mesh_dim`-aware collective encoding.

```python
verifier = DistributedVerifier()

# Individual checks
verifier.verify_postcondition(tensor, expected_partial=False)
verifier.verify_communication_legality(program, tensor_states=state)
verifier.verify_gradient_duality(fwd, bwd)
verifier.verify_pp_deadlock_free(schedule, sends, recvs)

# Run all checks at once
results = verifier.verify_all(program, final_tensors, bwd_program=bwd)
print(verifier.summary())

# Z3 placement solver (low-level)
solver = Z3PlacementSolver(mesh_ndim=2)  # multi-dim mesh
solver.add_input("x", (Partial(), Shard(dim=0)))
solver.encode_program(program)
result = solver.check()  # SAT/UNSAT
```

**Key features:**
- Multi-dim mesh: N Z3 Int variables per tensor (one per mesh dimension)
- `mesh_dim`-aware collectives: `AllReduce(mesh_dim=0)` transforms only dim 0, preserves others
- Wait/WaitAll passthrough: async sync ops correctly propagate placements through Z3
- Shape constraints respect `mesh_dim` targeting

### `verifier/temporal.py` — Temporal Verifier (5 checks) {#temporal}

Happens-Before analysis + Z3 race detection:

```python
from verifier.temporal import verify_temporal, TemporalGraph, RaceDetector, RaceType

result = verify_temporal(program)
# result.is_safe, result.reports (list of RaceReport)
# result.num_missing_waits, result.num_orphaned_handles

# Low-level access
graph = TemporalGraph(program)   # Build HB graph
detector = RaceDetector(graph)   # Run detection
reports = detector.detect_all()  # List of RaceReport
```

**Race types:** `DATA_RACE`, `MISSING_WAIT`, `BUFFER_ALIASING`, `DEPENDENCY_VIOLATION`, `ORPHANED_HANDLE`

---

## Synthesis & Optimization

### `verifier/rewrite.py` — Rewrite System {#rewrite}

```python
analyzer = PlacementAnalyzer()
analysis = analyzer.analyze(program, state)
# analysis.partial_tensors, analysis.missing_collectives,
# analysis.redundant_collectives, analysis.is_correct

cost = ProgramCost.from_program(program)
# cost.num_allreduce, cost.total_communication

optimizer = ProgramOptimizer()
optimized, history = optimizer.optimize(program, state)
```

### `verifier/synthesis.py` — Synthesis Engine {#synthesis}

```python
engine = SynthesisEngine(max_tactics=3, max_search_depth=2)
result = engine.synthesize(compute_program, tensors, mesh)
# result.best_candidate, result.all_candidates

# One-shot convenience
result = synthesize_parallel_program(
    compute_ops=[MatMul("x", "w", "y")],
    input_shapes={"x": (8, 16), "w": (16, 32)},
    sharding_specs={...},
)
```

### `verifier/llm_frontend.py` — LLM Frontend {#llm-frontend}

```python
# Prompt building
builder = PromptBuilder()
prompt = builder.build_extraction_prompt(pytorch_code)
feedback = builder.build_feedback_prompt(code, previous_ir, errors)

# LLM verification loop
llm = MockLLM()  # or Anthropic/OpenAI backend
loop = LLMVerificationLoop(llm=llm, max_iterations=5)
result = loop.verify_code(pytorch_code, mesh=mesh)

# Parse LLM output
ir_response = LLMIRResponse.from_json(llm_json_response)
program = ir_response.to_program("extracted")
```

---

## Parallelism Support

### `verifier/tir_lifter.py` — TileLang TIR Lifter {#tir-lifter}

```python
# Model TileLang TIR
tir_func = TIRFunc(
    name="gemm",
    buffers={"A": (1024, 1024), "B": (1024, 1024), "C": (1024, 1024)},
    grid=TIRGrid(axes=[i, j, k]),
    blocks=[TIRBlock(
        axes=[
            TIRBlockAxis(i, "S", 1024),  # spatial
            TIRBlockAxis(j, "S", 1024),  # spatial
            TIRBlockAxis(k, "R", 1024),  # reduce
        ],
        reads=[TIRBufferRegion("A", ["i", "k"]),
               TIRBufferRegion("B", ["k", "j"])],
        writes=[TIRBufferRegion("C", ["i", "j"])],
    )],
)

# Lift to distributed IR
lifter = TIRLifter(sharding_specs)
result = lifter.lift(tir_func)
# result.fwd_program, result.bwd_program, result.collectives_inserted
```

### `verifier/schedules.py` — PP Schedules {#schedules}

```python
# 1F1B schedule generation
sched = PP1F1BSchedule(num_stages=2, num_microbatches=4)
schedule = sched.generate_simple()

# Activation tracking
tracker = ActivationTracker(num_stages=2)
passed, errors = tracker.verify_activation_liveness(schedule)

# Deadlock checking
checker = DeadlockChecker()
checker.add_send(0, 1, "h0")
checker.add_recv(0, 1, "h0")
is_free, errors = checker.check()
```
