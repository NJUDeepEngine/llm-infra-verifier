---
title: Benchmarks
nav_order: 5
---

# Benchmarks

**49 cases across 5 suites, 100% detection rate.**
Derived from 16 real GitHub issues across PyTorch, Megatron-LM, TileLang, Triton, and DeepSeek TileKernels.

## Quick Run

```bash
python benchmarks/benchmark_suite.py           # Suite 1: 16 synthetic cases
python benchmarks/real_code_validation.py      # Suite 2: 8 real-code cases
python benchmarks/real_bug_benchmark.py        # Suite 3: 16 real-bug cases
python benchmarks/numerical_benchmark.py       # Suite 4: 9 numerical cases
```

## Suite 1: Synthetic Bug Patterns (16 cases)

Derived from GitHub issues. Tests our verifier's ability to detect known bug patterns.

```bash
python benchmarks/benchmark_suite.py --list    # list all
python benchmarks/benchmark_suite.py --run B1  # specific category
python benchmarks/benchmark_suite.py --json    # JSON output
```

| Category | Cases | Source Issues |
|----------|-------|---------------|
| B1: Missing Collectives | 3 | pytorch#144359, Megatron#4092 |
| B2: Placement Errors | 3 | pytorch#173041, #175690, #139681 |
| B3: Comm Legality | 3 | tilelang#2035, Megatron#4092, pytorch#140227 |
| B4: Gradient Duality | 3 | TileKernels#2, pytorch#144359, Megatron#4092 |
| B5: PP Schedule | 2 | Megatron#3952, #1525 |
| B6: CP Communication | 2 | Megatron#4382 |

## Suite 2: Real-Code Validation (8 cases)

Lifted from actual Megatron-LM and TileLang source patterns. Each case cites exact file and line numbers.

```bash
python benchmarks/real_code_validation.py
```

| Case | Source |
|------|--------|
| Megatron ColumnParallelLinear | `megatron/core/tensor_parallel/layers.py ~L200` |
| Megatron RowParallelLinear | `megatron/core/tensor_parallel/layers.py ~L290` |
| RowParallel missing AllReduce (bug) | pytorch#144359 |
| Async AllReduce gradient pattern | `layers.py ~L100` |
| GELU between CP and RP (bug + fix) | pytorch#144359 |
| TileLang TIR → IR lifting | `tilelang/examples/gemm` |
| Megatron TP MLP | `megatron/.../megatron_mlp.py` |
| Sequence Parallel + TP | `layers.py ~L200` |

## Suite 3: Real-Bug Benchmark (16 cases)

Each case shows the **original buggy code** from the GitHub issue, explains how we **translate it to our IR**, and reports the detection result. This is the most rigorous benchmark: it starts from actual code, not pre-encoded IR.

```bash
python benchmarks/real_bug_benchmark.py
```

### PyTorch / Megatron-LM Issues (9 cases)

| ID | Bug | Source | Category |
|----|-----|--------|----------|
| RB1a | RowParallel without AllReduce | pytorch#144359 | Spatial |
| RB1b | GELU on sharded tensor | pytorch#144359 | Spatial |
| RB1c | Colwise missing AllGather | Megatron layers.py | Spatial |
| RB2a | PP missing broadcast | Megatron#4092 | PP |
| RB2b | Send/Recv direction mismatch | Megatron#1525 | PP |
| RB3a | fp16 gradient underflow | PyTorch AMP docs | Numerical |
| RB3b | Adam eps invisible in fp16 | PyTorch Adam docs | Numerical |
| RB4a | Async AR without Wait | Megatron layers.py | Temporal |
| RB4b | Gradient buffer reuse | Megatron layers.py | Temporal |

### TileLang Issues (3 cases)

| ID | Bug | Source | Category |
|----|-----|--------|----------|
| RB5a | Invalid fragment layout (non-injective) | tilelang#2158 | Resource |
| RB5b | Int8 matmul pipeline sync (num_stages) | tilelang#2172 | Temporal |
| RB5c | fp8 cast mismatch vs torch | tilelang#2042 | Numerical |

### Triton Issues (4 cases)

| ID | Bug | Source | Category |
|----|-----|--------|----------|
| RB6a | TF32 path instead of IEEE fp32 | triton#10176 | Numerical |
| RB6b | Implicit int32→int8 truncation | triton#9991 | Type Safety |
| RB6c | TMA NaN from mbarrier init race | triton#10106 | Temporal |
| RB6d | Mixed int32/bf16 loop error | triton#9963 | Numerical |

## Suite 4: Numerical Benchmarks (9 cases)

IEEE 754 analytical error bounds and accumulation analysis.

```bash
python benchmarks/numerical_benchmark.py
```

| Category | Cases |
|----------|-------|
| N1: Same-Precision | Tree vs Ring, dtype effects, non-associativity |
| N2: Cross-Precision | Cast magnitudes, fp16 boundaries, Adam eps visibility |
| N3: Accumulation | 3-pathway comparison, multi-config, ZeRO effects |

## Detection Methods

| Verifier Dimension | Cases Using It |
|---|---|
| Spatial (Z3 SMT) | RB1a-c, RB2a-b, B1-B6 (22 cases) |
| Temporal (HB Graph) | RB4a-b, RB5b, RB6c, RB6d (7 cases) |
| Numerical (IEEE 754) | RB3a-b, RB5c, RB6a-b (8 cases) |
| Resource (Memory Graph) | RB5a (1 case) |

## Issue Coverage

| Repository | Issues |
|---|---|
| `pytorch/pytorch` | #144359, #173041, #175690, #139681, #140227 |
| `NVIDIA/Megatron-LM` | #4092, #3952, #1525, #4382 |
| `tile-ai/tilelang` | #2035, #2042, #2158, #2172 |
| `triton-lang/triton` | #9991, #9963, #10106, #10176 |
| `deepseek-ai/TileKernels` | #2 |
