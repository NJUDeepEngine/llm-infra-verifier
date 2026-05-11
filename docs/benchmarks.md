---
title: Benchmarks
nav_order: 5
---

# Benchmarks

Two benchmark suites validate the verifier against real-world bugs and source-code patterns.

## Synthetic Benchmark Suite

**16 cases across 6 categories, 100% detection rate.** Each case is derived from a real GitHub issue.

Run:
```bash
python benchmarks/benchmark_suite.py          # all 16 cases
python benchmarks/benchmark_suite.py --run B1 # specific category
python benchmarks/benchmark_suite.py --json   # JSON output
```

### B1: Missing/Incorrect Collectives (3/3)

| ID | Bug | Source |
|----|-----|--------|
| B1a | Row Parallel without AllReduce | `pytorch/pytorch#144359` |
| B1b | GELU on sharded tensor (Colwise→Rowwise) | `pytorch/pytorch#144359` |
| B1c | Missing cross-stage broadcast in PP | `NVIDIA/Megatron-LM#4092` |

### B2: Placement/Shard Errors (3/3)

| ID | Bug | Source |
|----|-----|--------|
| B2a | Shard(1) non-contiguous local tensors | `pytorch/pytorch#173041` |
| B2b | Shard→Replicate symbolic shape corruption | `pytorch/pytorch#175690` |
| B2c | SequenceParallel DTensor→Tensor cast | `pytorch/pytorch#139681` |

### B3: Communication Legality (3/3)

| ID | Bug | Source |
|----|-----|--------|
| B3a | AllReduce on already-replicated tensor | `tile-ai/tilelang#2035` |
| B3b | Send without matching Recv | `NVIDIA/Megatron-LM#4092` |
| B3c | AllGather dim mismatch | `pytorch/pytorch#140227` |

### B4: Gradient Duality (3/3)

| ID | Bug | Source |
|----|-----|--------|
| B4a | Missing bwd AllReduce for fwd AllReduce | `deepseek-ai/TileKernels#2` |
| B4b | Wrong dual (AllGather instead of ReduceScatter) | `pytorch/pytorch#144359` |
| B4c | Send direction not reversed in bwd | `NVIDIA/Megatron-LM#4092` |

### B5: PP Schedule (2/2)

| ID | Bug | Source |
|----|-----|--------|
| B5a | Activation premature release in 1F1B | `NVIDIA/Megatron-LM#3952` |
| B5b | Backward before forward in 1F1B | `NVIDIA/Megatron-LM#1525` |

### B6: CP Communication (2/2)

| ID | Bug | Source |
|----|-----|--------|
| B6a | Ring Attention without final AllReduce | `NVIDIA/Megatron-LM#4382` |
| B6b | Wrong ring order in CP | `NVIDIA/Megatron-LM#4382` |

## Real-Code Validation Suite

**8 end-to-end cases** that lift from actual TileLang TIR and Megatron-LM source patterns, model them in IR, and verify.

Run:
```bash
python benchmarks/real_code_validation.py
```

| # | Case | Source File |
|---|------|-------------|
| 1 | Megatron ColumnParallelLinear (correct) | `megatron/core/tensor_parallel/layers.py ~L200` |
| 2 | Megatron RowParallelLinear (correct) | `megatron/core/tensor_parallel/layers.py ~L290` |
| 3 | RowParallel WITHOUT AllReduce (bug) | `pytorch/pytorch#144359` |
| 4 | Megatron Async AllReduce Gradient | `layers.py ~L100` (`LinearWithGradAccumulation...`) |
| 5 | GELU between Colwise+Rowwise (bug + fix) | `pytorch/pytorch#144359` |
| 6 | TileLang TIR → IR Lifting | `tilelang examples/gemm/example_gemm.py` |
| 7 | Megatron TP MLP | `megatron/core/transformer/moe/megatron_mlp.py` |
| 8 | Sequence Parallel + TP Interaction | `layers.py ~L200` |

### What makes this "real"

1. **Source code references** — each case cites the exact file and approximate line numbers
2. **Original code patterns** — shows the Megatron/TileLang source pattern in comments before the lifted IR
3. **End-to-end pipeline** — especially Case 6 demonstrates TIR block → `TIRLifter` → IR → `MultiDeviceExecutor` → `DistributedVerifier`
4. **Bug detection + fix verification** — Cases 3, 5 show bug detection, then propose and verify the fix

## Source Issue Coverage

| Repository | Issues Used |
|---|---|
| `pytorch/pytorch` | #144359, #173041, #175690, #139681, #140227 |
| `NVIDIA/Megatron-LM` | #4092, #3952, #1525, #4382 |
| `tile-ai/tilelang` | #2035 |
| `deepseek-ai/TileKernels` | #2 |
