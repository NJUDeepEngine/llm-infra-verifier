# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Install dependencies (only z3-solver and pytest)
pip install -r requirements.txt

# Run all tests (416 tests)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_verifier.py -v

# Run a specific test class or test
python -m pytest tests/test_op_verification.py::TestPlacementRuleEvaluation -v
python -m pytest tests/test_verifier.py::TestDistributedVerifier::test_row_parallel_bug -v

# Import smoke test
python -c "from verifier import *"

# Run examples
python examples/tp_linear.py
python examples/tp_mlp.py
python examples/overlap_demo.py
```

## Core Architecture

This is a **static verification framework for distributed LLM training** that catches placement bugs and communication races at compile time — no GPUs needed. It verifies properties over all possible inputs using symbolic execution + Z3 SMT solver + Happens-Before graph analysis.

### Two Independent Verification Dimensions

```
Input Source (PyTorch / Megatron / TileLang TIR)
            │
            ▼
    Verification IR (verifier/ir/)  ← symbolic, not numeric
            │
   ┌────────┴────────┐
   ▼                 ▼
Spatial (solver.py)  Temporal (temporal.py)
"Where things go"    "When things happen"
Z3 SMT solver        HB graph + Z3
6 checks             4 checks
```

**Spatial** (Z3-based): postcondition (output not partial), communication legality, gradient duality, placement consistency, shape consistency, PP deadlock freedom.

**Temporal** (HB graph + Z3): data races, missing Wait, buffer aliasing, dependency violations.

### Execution Model: Symbolic, Not Numeric

The executor (`executor.py`) tracks **metadata** (placement, shape, symbolic expression) per device — never actual tensor values. For example, `MatMul(x, w)` produces a `TensorState` with derived `placement=Partial()` and `shape=(8,32)`, not numeric results.

Z3 encodes placement as Ints: `R=0, S0=1, S1=2, P=3`. Propagation rules become SMT constraints (`If(And(a==S1, b==S0), P, ...)`). Correctness is checked by asking Z3 to find counterexamples — UNSAT means proven correct.

### Package Structure

- **`verifier/state/`** — `TensorState`, `ShardingSpec`, `DeviceMesh`, placement types (`Shard`, `Replicate`, `Partial`), `LocalSPMDType` enum (R/I/V/P with R↔P gradient duality)
- **`verifier/ir/`** — All IR ops. Each op encodes `apply(ctx)` for forward placement propagation, `vjp(ctx, grad)` for backward, and SPMD type rules (`propagate_spmd_type`). Sub-modules: `base.py`, `compute.py`, `collective.py`, `p2p.py`, `async_ops.py`, `shape.py`, `spmd.py`, `precision.py`, `zero.py`, `cp.py`, `moe.py`, `program.py`
- **`verifier/executor.py`** — Multi-device symbolic executor. Maintains per-device `DeviceState` (tensors + slices). Dispatches ops to `_exec_*` methods that propagate metadata across devices.
- **`verifier/solver.py`** — Z3 spatial verifier. `Z3PlacementSolver` encodes op propagation rules as SMT constraints. `DistributedVerifier` provides `verify_all()` running 5 checks. Three levels: L0 placement, L1 shape constraints, L2 slice alignment.
- **`verifier/temporal.py`** — HB graph construction + race detection. `TemporalGraph` builds events with `[issue_time, complete_time)` intervals, encoding program order, Wait sync, and data dependencies as Z3 constraints. `RaceDetector` finds unordered conflicting access pairs.
- **`verifier/autograd.py`** — `AutogradEngine` records ops on a tape, replays in reverse applying VJP rules to generate backward program with correct collective duals (AllReduce↔AllReduce, AllGather↔ReduceScatter, Send↔Recv).
- **`verifier/synthesis.py`** — Beam search over tactic space (`INSERT_ALLREDUCE`, `INSERT_ALLGATHER`, `INSERT_REDUCESCATTER`) to produce verified parallel programs from unannotated compute graphs.
- **`verifier/llm_frontend.py`** — PyTorch → IR extraction via LLM with feedback loop (LLM proposes, verifier checks).
- **`verifier/tir_lifter.py`** — TileLang TIR → distributed IR lifter.
- **`verifier/rewrite.py`** — Placement analysis, pattern matching (`InsertAllReduceRule`, `RemoveRedundantAllReduceRule`), cost model.
- **`verifier/schedules.py`** — 1F1B pipeline schedule + `DeadlockChecker` (DFS cycle detection).

### Key Patterns

**Writing a new IR op:** Every `IROp` subclass must implement `apply(ctx)`, `vjp(ctx, grad_output)`, `input_names`, `output_name`, `clone_with_names(input_map, output_name)`, `__repr__`, and appropriately override `is_collective()`/`is_p2p()`/`is_async()`.

**Placement propagation:** Ops merge input placements element-wise per mesh dimension. `Replicate` acts as identity (absorbs the other). `Shard(dim)` only merges with same-dim `Shard`. `Partial` propagates through most ops. `MatMul` has special rules: `S(1)×S(0)→P`, `R×S(1)→S(1)`, `S(0)×R→S(0)`.

**Backward compatibility:** All symbols must be re-exported from both `verifier.ir.__init__` and `verifier.__init__`. Existing `from verifier.ir import X` imports must continue working after any refactoring.

**Refactoring checklist** (from user constraints):
1. IR package sub-module separation maintained (base/compute/collective/p2p/async_ops/shape/spmd/program)
2. All 8 NCCL collectives in `collective.py`
3. Every IROp implements the full interface contract
4. No breaking changes for consumer files (tests, examples, benchmarks)
5. After each step: run `pytest tests/test_verifier.py -q`, import smoke test, and `python examples/tp_linear.py`
6. Placement semantics correct: VJP duals, SPMD guard rules (Partial*Partial forbidden)
7. `@dataclass` for ops, type hints on public signatures, relative imports within verifier

### Mixed Precision & FP8

**dtype system:** `TensorState.dtype` is `Optional[str]`. Values: `None` (fp32), `"fp16"`, `"bf16"`, `"fp8e4m3"`, `"fp8e5m2"`. Properties: `is_fp8`, `is_fp8e4m3`, `is_fp8e5m2`, `is_fp16`, `is_bf16`, `is_fp32`. All IR ops must propagate `dtype` in `TensorState` construction (both forward and VJP).

**FP8 ops** (`precision.py`):
- `FP8Quantize(x, output, scale_expr, src_dtype, dst_dtype)` — quantize to fp8e4m3 (forward) or fp8e5m2 (backward), attaches symbolic `scale_expr` to output via `fp8_scale_expr` field
- `FP8Dequantize(x, output, scale_expr, src_dtype, dst_dtype)` — inverse, clears `fp8_scale_expr`
- `AmaxUpdate(x, output, tensor_name, iteration_expr)` — records amax observation for delayed scaling; input must be fp8, output is replicated fp32 scalar; VJP is empty

**DtypeGuard** checks: `check_fp8_format_usage(tensor, phase)` verifies e4m3 in forward / e5m2 in backward. `check_fp8_scale_freshness(quantize_idx, amax_idx)` verifies delayed scaling ordering.

**Delayed scaling pattern:** AmaxUpdate at iteration N must happen-before FP8Quantize at iteration N+1. Model by unrolling two iterations with distinct tensor names.
