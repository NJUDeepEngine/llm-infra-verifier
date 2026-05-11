---
title: Module Reference
nav_order: 7
---

# Module Reference

## Core Verification

### `verifier/state.py` — Tensor State & Placements

Central data structures for the entire framework.

```python
# Placement types
Shard(dim=0)       # Tensor split along dim 0
Shard(dim=1)       # Tensor split along dim 1
Replicate()        # Full copy on each device
Partial()          # Locally computed, needs AllReduce

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
    stage=0,                   # PP stage (optional)
    _async_handle="h1",        # in-flight async handle (optional)
)
```

### `verifier/ir.py` — IR Operations

All operations with **forward placement propagation** and **VJP rules**.

**Compute ops:** `MatMul`, `Add`, `Multiply`, `SiLU`, `FlashAttention`, `Reshape`, `Transpose`

**Sync collectives:** `AllReduce`, `AllGather`, `ReduceScatter`

**P2P:** `Send`, `Recv`

**Async ops:** `AllReduceAsync`, `SendAsync`, `RecvAsync`

**Synchronization:** `Wait`, `WaitAll`, `OverlapRegion`

**Infrastructure:** `Handle`, `Stream`, `Program`

Each op implements:
- `apply(ctx) → TensorState` — forward pass with placement propagation
- `vjp(ctx, grad_output) → {input: grad}` — vector-Jacobian product
- `is_collective()`, `is_p2p()`, `is_async()`, `is_sync()` — classification

### `verifier/executor.py` — Multi-Device Executor

Symbolic executor tracking per-device `TensorState`:

```python
executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x_tensor)
executor.register_tensor(w_tensor)
state = executor.run_program(fwd)  # Dict[str, TensorState]

# Access per-device state
dev0_tensor = executor.get_tensor("y", device_id=0)
all_devices = executor.final_state()
```

### `verifier/autograd.py` — Formal Autograd Engine

VJP-based gradient computation with duality verification:

```python
autograd = AutogradEngine()
for op in fwd.ops:
    autograd.record(op, tensor_states)

bwd_program = autograd.generate_backward("loss")
check = autograd.verify_gradient_correctness(fwd, bwd_program)
# check.passed, check.collective_pairs, check.errors
```

### `verifier/solver.py` — Z3 Spatial Verifier

Encoding verification conditions as SMT formulas:

```python
verifier = DistributedVerifier()

# Postcondition check
verifier.verify_postcondition(tensor, expected_partial=False)

# Communication legality
verifier.verify_communication_legality(program, tensor_states=state)

# Gradient duality
verifier.verify_gradient_duality(fwd, bwd)

# PP deadlock freedom
verifier.verify_pp_deadlock_free(schedule, sends, recvs)

# Run all checks
verifier.verify_all(program, final_tensors, bwd_program=bwd)
```

### `verifier/temporal.py` — Temporal Verifier

Happens-Before analysis + Z3 race detection:

```python
from verifier.temporal import verify_temporal, TemporalGraph, RaceDetector

result = verify_temporal(program)
# result.is_safe, result.reports (RaceReport list)

# Low-level access
graph = TemporalGraph(program)   # Build HB graph
detector = RaceDetector(graph)   # Run detection
reports = detector.detect_all()  # List of RaceReport
```

## Synthesis & Optimization

### `verifier/rewrite.py` — Rewrite System

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

### `verifier/synthesis.py` — Synthesis Engine

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

### `verifier/llm_frontend.py` — LLM Frontend

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

## Parallelism Support

### `verifier/tir_lifter.py` — TileLang TIR Lifter

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

### `verifier/schedules.py` — PP Schedules

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
