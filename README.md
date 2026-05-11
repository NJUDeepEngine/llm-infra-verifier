# LLM-Infra-Verifier

> A formal verification framework for distributed tensor programs — covering Tensor Parallelism (TP), Pipeline Parallelism (PP), and Context Parallelism (CP) with both spatial (placement) and temporal (overlap) correctness guarantees.

Built on **Z3 SMT solver** + **symbolic execution** + **Happens-Before temporal analysis**, targeting TileLang TIR as the verification IR source.

## Overview

Distributed training programs are notoriously hard to get right. Silent bugs — missing AllReduce, GELU applied to sharded tensors, async communication races — produce **numerically incorrect results** that pass standard tests but fail in production.

`llm-infra-verifier` provides **formal, compile-time verification** of distributed tensor programs across two dimensions:

| Dimension | What it checks | Key technique |
|-----------|---------------|---------------|
| **Spatial** | Placement propagation, postconditions, gradient duality | Z3 SMT + symbolic execution |
| **Temporal** | Data races, missing waits, buffer aliasing, dependency violations | Happens-Before graph + Z3 constraints |

## Architecture

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
                    │   (tir_lifter.py)    │  (Scheme A: direct block analysis)
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

## Key Modules

### Core Verification

| Module | Lines | Purpose |
|--------|-------|---------|
| `state.py` | 248 | `TensorState`, `ShardingSpec`, `DeviceMesh`, placement types (`Shard`/`Replicate`/`Partial`) |
| `ir.py` | ~1400 | All IR ops with forward placement propagation + VJP rules. Sync and async variants: `MatMul`, `AllReduce`, `AllReduceAsync`, `Wait`, `FlashAttention`, `Send`/`Recv`/`SendAsync`/`RecvAsync`, `OverlapRegion` |
| `executor.py` | 316 | Multi-device symbolic executor tracking per-device `TensorState` |
| `autograd.py` | 368 | Formal VJP-based autograd engine with gradient duality checking |
| `solver.py` | 572 | Z3 encoding for postcondition, communication legality, gradient duality, placement consistency, shape consistency, PP deadlock freedom |
| `temporal.py` | ~580 | Happens-Before graph builder + Z3 temporal constraints. Detects data races, missing waits, buffer aliasing, dependency violations |

### Synthesis & Optimization

| Module | Purpose |
|--------|---------|
| `rewrite.py` | Pattern matching, placement analysis (`PlacementAnalyzer`), rewrite rules (`InsertAllReduceRule`, `RemoveRedundantAllReduceRule`), cost model (`ProgramCost`) |
| `synthesis.py` | Verified Parallelization Synthesis: `TacticProposer` → beam search over tactic space → verify each candidate → rank by communication cost → return minimal correct program |
| `llm_frontend.py` | PyTorch code → IR extraction via LLM with few-shot prompts. Feedback loop: verifier errors → LLM refines IR. Includes `MockLLM` for testing |

### Parallelism Support

| Module | Purpose |
|--------|---------|
| `tir_lifter.py` | TileLang TIR subset model + lift to distributed IR (Scheme A: direct block-level access pattern analysis) |
| `schedules.py` | 1F1B pipeline schedule generation, activation memory tracking, deadlock freedom checking |

## Installation

```bash
git clone git@github.com:NJUDeepEngine/llm-infra-verifier.git
cd llm-infra-verifier
pip install -r requirements.txt
```

Dependencies: `z3-solver>=4.12.0`, `pytest>=7.0.0`

## Quick Start

### 1. TP Linear Verification

```python
from verifier import *

# Row Parallel Linear: X(Shard H) @ W(Shard H) → Partial → AllReduce → Replicate
mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
x = TensorState("x", (8, 16), (8, 8),
    ShardingSpec((Shard(dim=1),), mesh), expr="x", requires_grad=True)
w = TensorState("w", (16, 32), (8, 32),
    ShardingSpec((Shard(dim=0),), mesh), expr="w", requires_grad=True)

# Correct program
fwd = Program("tp_linear")
fwd.add(MatMul(a="x", b="w", output="y_partial"))
fwd.add(AllReduce(x="y_partial", output="y"))

executor = MultiDeviceExecutor(mesh)
executor.register_tensor(x); executor.register_tensor(w)
state = executor.run_program(fwd)

# Verify postcondition
verifier = DistributedVerifier()
result = verifier.verify_postcondition(state["y"], expected_partial=False)
print(f"Postcondition: {'PASSED' if result.passed else 'FAILED'}")
```

### 2. Temporal Overlap Verification

```python
from verifier.temporal import verify_temporal

prog = Program("overlap_bug")
prog.add(MatMul("x", "w", "y_partial"))
prog.add(AllReduceAsync("y_partial", "y", handle="h1", stream=COMM_STREAM))
prog.add(MatMul("y", "w2", "z"))  # BUG: reads 'y' before Wait!
prog.add(Wait(handle="h1", tensor="y", output="y_safe"))

result = verify_temporal(prog)
# Detects: MISSING_WAIT — MatMul reads async output before Wait(h1)
```

### 3. Verified Parallelization Synthesis

```python
from verifier.synthesis import synthesize_parallel_program

mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
result = synthesize_parallel_program(
    compute_ops=[MatMul(a="x", b="w", output="y")],
    input_shapes={"x": (8, 16), "w": (16, 32)},
    sharding_specs={
        "x": ShardingSpec((Shard(dim=1),), mesh),
        "w": ShardingSpec((Shard(dim=0),), mesh),
    },
    verbose=True,
)
# Auto-synthesizes: MatMul → AllReduce → y_reduced
```

## Verification Capabilities

### Spatial Verification (6 checks)

| Check | Description | Z3 Encoding |
|-------|-------------|-------------|
| Postcondition | Output tensors must not be PARTIAL | `Bool("partial") == True → unsat` |
| Communication Legality | AllReduce only on Partial; Send/Recv matched | Structural check + Z3 fallback |
| Gradient Duality | fwd collective → bwd dual (AR↔AR, AG↔RS, Send↔Recv) | Type-based duality matching |
| Placement Consistency | Output placement follows from input placements | Trust executor (structural) |
| Shape Consistency | Shapes propagate correctly through ops | Structural validation |
| PP Deadlock Freedom | Send/Recv matching + no circular waits | DFS cycle detection + pair matching |

### Temporal Verification (4 checks)

| Check | Description | Detection Mechanism |
|-------|-------------|---------------------|
| Data Race | Two ops on different streams access same tensor, ≥1 write, unordered | HB graph + interval overlap |
| Missing Wait | Async output consumed before `Wait(handle)` | Handle-waited_by analysis |
| Buffer Aliasing | Two async ops write same buffer, first unconsumed before second | Write-after-write check |
| Dependency Violation | Recv before Send for same (src,dst,mb) | HB ordering constraint |

## Examples

Run all examples:

```bash
# Spatial verification
python examples/tp_linear.py       # Row Parallel: correct vs missing AllReduce
python examples/tp_mlp.py          # Megatron MLP: Column + Row Parallel
python examples/pp_2stage.py       # 2-Stage 1F1B Pipeline Parallelism
python examples/cp_ring_attn.py    # Ring Attention with FlashAttention

# Temporal verification
python examples/overlap_demo.py    # Data race, missing wait, buffer aliasing

# Synthesis + LLM frontend
python examples/synthesis_demo.py  # Auto-synthesis + LLM extraction loop
```

## Benchmark

Built from **real-world GitHub issues** across PyTorch DTensor, Megatron-LM, TileLang, and DeepSeek TileKernels.

```bash
python benchmarks/benchmark_suite.py          # Run all 16 cases
python benchmarks/benchmark_suite.py --list   # List all cases
python benchmarks/benchmark_suite.py --json   # JSON output
python benchmarks/benchmark_suite.py --run B1 # Run specific category
```

### Benchmark Sources & Results

| Category | Cases | Source Issues |
|----------|-------|---------------|
| **B1: Missing/Incorrect Collectives** | 3 | pytorch#144359, Megatron#4092 |
| **B2: Placement/Shard Errors** | 3 | pytorch#173041, pytorch#175690, pytorch#139681 |
| **B3: Communication Legality** | 3 | tilelang#2035, Megatron#4092, pytorch#140227 |
| **B4: Gradient Duality** | 3 | TileKernels#2, pytorch#144359, Megatron#4092 |
| **B5: PP Schedule** | 2 | Megatron#3952, Megatron#1525 |
| **B6: CP Communication** | 2 | Megatron#4382 |
| **Total** | **16/16 detected (100%)** | |

### Detection Methods per Category

| Bug Pattern | Detection |
|-------------|-----------|
| Missing AllReduce after MatMul | `verify_postcondition` (Z3 partial check) |
| GELU on sharded tensor | `verify_nonlinear_on_sharded` |
| Missing PP cross-stage broadcast | `verify_cross_stage_broadcast` |
| Shard(1) non-contiguity | Shard(1) risk analysis |
| Shard→Replicate shape corruption | AllReduce legality (Partial check) |
| SeqParallel DTensor→Tensor cast | Element-wise sharding compatibility |
| Redundant AllReduce | Redundant collective detection |
| Unmatched Send/Recv | Communication legality |
| AllGather dim mismatch | Collective dim consistency |
| Missing bwd collective | Gradient duality |
| Wrong dual collective type | Gradient duality |
| Send direction not reversed in bwd | Gradient duality |
| Activation premature release | Activation liveness checker |
| Backward before forward in 1F1B | Schedule ordering check |
| Missing AllReduce in ring attn | Postcondition check |
| Wrong ring order | Send/Recv ring consistency |

## Tests

```bash
python -m pytest tests/test_verifier.py -v    # 42 tests
```

Test coverage:
- `TestTensorState` — placement types, local shape computation
- `TestIROps` — MatMul placement propagation, AllReduce conversion, replicated ops
- `TestExecutor` — multi-device Row Parallel execution, state isolation
- `TestAutograd` — gradient duality, gradient correctness check
- `TestSolver` — postcondition (pass/fail), communication legality
- `TestSchedules` — 1F1B generation, deadlock checker, bidirectional matching
- `TestRewrite` — placement analysis, program cost
- `TestSynthesis` — tactic proposer, synthesis finds valid program
- `TestLLMFrontend` — op parsing, mock LLM, verification loop
- `TestTemporal` — correct overlap is safe, missing wait, buffer aliasing, data race

## Design Principles

1. **No false negatives on safety**: if the verifier says SAFE, the program is genuinely safe under the modeled abstractions
2. **Explicit over implicit**: all sharding, placement, and communication must be explicitly represented in IR
3. **Z3 as oracle, not runtime**: Z3 is used for verification, not for execution — the symbolic executor handles propagation
4. **Pattern library > one-off rules**: recognized patterns (Row Parallel, Column Parallel, 1F1B) are first-class, enabling synthesis and optimization
5. **LLM as proposer, verifier as checker**: LLM generates candidate IR / tactics; formal verification is the final arbiter

## Future Directions

- **E-graph equivalence system**: Rewrite rules for DTensor program equivalence (AllReduce fusion, reshard elimination)
- **Real LLM API integration**: Anthropic/OpenAI backend for PyTorch → IR extraction
- **Real TileLang TIR parser**: Replace the TIR subset model with actual TileLang AST parsing
- **Attention + FSDP**: Extend verification to FlashAttention variants and FSDP sharding strategies
- **Autograd theorem proving**: Prove gradient correctness via symbolic differentiation rather than VJP rule checking
- **CUDA stream-aware analysis**: Extend temporal model with CUDA stream semantics and event-based synchronization

## License

Apache 2.0
