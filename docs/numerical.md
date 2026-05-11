---
title: Numerical Verification
nav_order: 4.5
---

# Numerical Verification

{: .note }
Floating-point error analysis for distributed training. Models three error dimensions: same-precision, cross-precision, and accumulation over time.

## Overview

The numerical verifier provides **compile-time bounds** on floating-point error in distributed training, without requiring actual training data. It uses IEEE 754 analytical error bounds, not empirical sampling.

### Three-dimensional error model

```
                           Error Accumulation (time)
                                   ▲
                                  /|\
                                 / | \    Path 3: Cross-Rank Divergence
                                /  |  \   LINEAR growth: T × lr × ε_ar × |g|
                               /   |   \
                              /    |    \
                             / Path 2: \
                            / Adam EMA   \
                           /  bounded     \
                          /_______________\
                          Same-Precision   Cross-Precision
                         (AR non-assoc)    (fp32→fp16 cast)
```

## Module: `verifier/numerical.py`

### Quick Start

```bash
python examples/numerical_demo.py    # Full demo with all sections
python benchmarks/numerical_benchmark.py  # 9 benchmark cases
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `DtypeProperties` | IEEE 754 characteristics: ε, min/max normal, mantissa/exponent bits |
| `ErrorModel` | Per-operation error bounds (MatMul, element-wise, AllReduce, cast) |
| `ReductionErrorAnalyzer` | Ring vs Tree topology error comparison |
| `ErrorAccumulator` | Three-pathway accumulation over training steps |
| `OptimizerChecker` | Adam/AdamW invariants under DP, ZeRO-1/2/3 |
| `OverflowRiskDetector` | fp16/bf16 boundary safety for activations and gradients |

### Dtype Properties

| Dtype | Mantissa | ε (machine) | Min Normal | Max Normal |
|-------|----------|-------------|------------|------------|
| fp32 | 23 bits | 1.19e-07 | 1.18e-38 | 3.40e+38 |
| fp16 | 10 bits | 9.77e-04 | 6.10e-05 | 6.55e+04 |
| bf16 | 7 bits | 7.81e-03 | 1.18e-38 | 3.39e+38 |
| fp64 | 52 bits | 2.22e-16 | 2.23e-308 | 1.80e+308 |

## Error Pathways

### Path 1: Weight Cast Drift (cross-precision)

{: .note }
**No memory.** Each step: `fp32_master → fp16_weight`. Same fp32 input → same fp16 output every time. Error is bounded by `ε_cast` regardless of training duration.

```
|δ| ≤ 0.5 × 2^(-m_dst) = ε_cast

fp32 → fp16: ε_cast = 4.88e-4
fp32 → bf16: ε_cast = 3.91e-3
```

### Path 2: Optimizer State EMA (same + cross)

{: .note }
**Bounded steady-state.** EMA filter converges to input mean. Error in m_t = error in g_t. Neither amplifies nor attenuates.

```
m_t = β1·m_{t-1} + (1-β1)·g_t
v_t = β2·v_{t-1} + (1-β2)·g_t²

Steady state: m_error ≈ g_error (converges, doesn't grow)
```

### Path 3: Cross-Rank Divergence (same-precision)

{: .warning }
**LINEARLY UNBOUNDED.** If AllReduce doesn't produce bit-exact gradients across ranks, the weight difference grows without bound.

```
|θ_T[r] - θ_T[s]| ≈ T × lr × ε_ar × |g_typical|

Where ε_ar depends on:
  - AllReduce topology (ring: O(ε×N), tree: O(ε×logN))
  - Accumulation dtype (fp32: ε≈1e-7, fp16: ε≈1e-3)
```

## Reduction Error Analysis

### Topology Comparison (N=256, fp32 accumulation)

| Topology | Operations | Error Bound | Ratio vs Tree |
|----------|-----------|-------------|---------------|
| Ring | 255 sequential adds | 3.04e-05 | 31.9x |
| Tree | 8 levels (log₂256) | 9.54e-07 | 1x |

### Cross-Precision Comparison

| Error Source | Relative Error | Notes |
|---|---|---|
| Tree AR (N=256, fp32) | 9.54e-07 | Same-precision, negligible |
| Ring AR (N=256, fp32) | 3.04e-05 | 32x worse but still small |
| fp32 → fp16 cast | **4.88e-04** | **512x larger than Tree AR!** |
| fp32 → bf16 cast | **3.91e-03** | 4100x larger than Tree AR |

## Accumulation Over Training

### Standard config (fp16 compute + fp32 AR + fp32 Adam)

| T (steps) | Path 1 (Cast) | Path 2 (Adam) | Path 3 (Divergence) | Risk |
|-----------|--------------|---------------|---------------------|------|
| 10,000 | 4.88e-04 | 8.87e-05 | 3.58e-10 | SAFE |
| 100,000 | 4.88e-04 | 8.87e-05 | 3.58e-09 | SAFE |
| 1,000,000 | 4.88e-04 | 8.87e-05 | 3.58e-08 | SAFE |

{: .tip }
The dominant short-term error is Path 1 (cast). Path 3 (divergence) only dominates at extreme scale.

### Dangerous config (fp16 compute + fp16 AR)

| T (steps) | Cross-Rank Divergence | Risk |
|-----------|----------------------|------|
| 10,000 | 2.93e-06 | SAFE |
| 100,000 | 2.93e-05 | SAFE |
| 1,000,000 | 2.93e-04 | SAFE |

{: .warning }
fp16 AR is **8,192x worse** than fp32 AR for cross-rank divergence. At scale (N=256, ring, T=1M): divergence reaches **7.8%**.

## Optimizer Safety

### Adam in fp16

{: .important }
**Adam ε=1e-8 has NO effect in fp16.** fp16 min_normal=6.1e-05 means ε is 6000x smaller than the smallest representable value. In fp16, `√v̂ + ε = √v̂` always.

```
Recommendation: ALWAYS use fp32 for Adam m/v state
```

### ZeRO Stage Effects

| Stage | Additional Risk | Source |
|-------|----------------|--------|
| DP | Baseline | AllReduce gradients, identical state |
| ZeRO-1 | Low | Optimizer state sharded, grad still AllReduced |
| ZeRO-2 | Medium | ReduceScatter introduces fp16 intermediate |
| ZeRO-3 | Medium-High | Parameter gather/scatter adds cast cycles |

## Usage

```python
from verifier.numerical import *

# Full analysis for your training config
result = verify_numerical(
    n_ranks=64,
    topology="tree",
    compute_dtype=Dtype.FP16,
    accumulate_dtype=Dtype.FP32,
    optimizer="adam",
    zero_stage=ZeROStage.DP,
    hidden_dim=4096,
    num_layers=32,
    batch_size=128,
)

print(result.summary())
# Shows: reduction analysis, accumulation pathways,
# overflow risks, optimizer invariants

# Compare multiple configurations
accumulator = ErrorAccumulator(Dtype.FP16, Dtype.FP32, n_ranks=256)
configs = [
    ("fp16+f32 AR", Dtype.FP16, Dtype.FP32),
    ("fp16+fp16 AR", Dtype.FP16, Dtype.FP16),
]
results = accumulator.compare_configs(configs, num_steps=100000)
for name, a in results.items():
    print(f"{name}: cross-rank div = {a.cross_rank_weight_diff:.2e}")
```

## Benchmarks

```bash
python benchmarks/numerical_benchmark.py    # 9 cases across 3 categories
```

### Category N1: Same-Precision

| Case | What it verifies |
|------|-----------------|
| N1a | Tree vs Ring error growth rates |
| N1b | Dtype effect on AllReduce error |
| N1c | IEEE 754 non-associativity |

### Category N2: Cross-Precision

| Case | What it verifies |
|------|-----------------|
| N2a | Cast error vs AR error ratio (>500x) |
| N2b | fp16 range safety for model scales |
| N2c | Adam ε invisibility in fp16 |

### Category N3: Accumulation

| Case | What it verifies |
|------|-----------------|
| N3a | Three-pathway transition over time |
| N3b | Multi-config safety (4 to 1024 GPUs) |
| N3c | ZeRO stage accumulation effects |

## Realism & Limitations

### What the model captures

- IEEE 754 worst-case error bounds (valid for ALL inputs)
- Topology-dependent reduction error
- Cast error at precision boundaries
- Linear divergence accumulation

### What the model does NOT capture

- Actual gradient distributions (uses structural upper bounds)
- Loss landscape effects on gradient magnitude
- Stochastic rounding (uses round-to-nearest)
- Hardware-specific behaviors (tensor cores, fused multiply-add)

### To validate against YOUR cluster

The model is fully parameterized. Replace defaults with your actual:
- `n_ranks`, `topology`, `compute_dtype`, `accumulate_dtype`
- `hidden_dim`, `num_layers`, `batch_size`
- `learning_rate`, `num_steps`

The bounds are conservative (worst-case), so they will NEVER miss a real violation.
