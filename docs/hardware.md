---
title: Hardware & OOM
nav_order: 8
---

# GPU Hardware & OOM Detection

## GPU Models

| GPU | Generation | HBM | SMs | Shared/SM | TFLOPS |
|-----|-----------|-----|-----|-----------|--------|
| A100 SXM | Ampere | 80 GB | 108 | 164 KB | 312 |
| H100 SXM | Hopper | 80 GB | 132 | 228 KB | 989 |
| H200 SXM | Hopper | 141 GB | 132 | 228 KB | 989 |
| B100 | Blackwell | 128 GB | 128 | 256 KB | 1800 |
| B200 | Blackwell | 192 GB | 160 | 256 KB | 2250 |

```bash
python examples/oom_demo.py    # Full hardware + OOM demo
```

## Memory Hierarchy

```
GPU → HBM (80-192 GB)
   → L2 Cache (40-96 MB)
      → SM (108-160)
         → Shared Memory (164-256 KB)
         → Register File (65536 × 32-bit)
         → Warps (×64, 32 threads each)
```

## OOM Detection: Four Levels

### HBM (Global Memory)

Sum all live tensors at each program point vs capacity. Includes parameters, gradients, optimizer state, activations, communication buffers.

### Shared Memory (per SM)

Per-block shared memory vs SM limit. FlashAttention @ 96KB/block → max 2 concurrent blocks on H100 (228KB).

### Registers (per SM)

Per-thread registers × threads/block vs 65536. High register kernels (128+/thread) severely limit occupancy.

### Occupancy

`min(thread_limit, register_limit, shared_limit, hw_limit)` → concurrent warps → utilization.

## LLM Memory Budget

```python
from verifier.memory_graph import estimate_llm_memory

mem = estimate_llm_memory(hidden_dim=8192, num_layers=80, tp_size=8)
# params=15.1G, grads=15.1G, optimizer=60.4G, activations=5.3G → total=95.9G
# H100(80G): OOM, H200(141G): OK, B200(192G): OK
```

| Model | TP | Per-GPU | H100 | H200 | B200 |
|-------|----|---------|------|------|------|
| Llama-7B | 1 | 82.8 GB | OOM | ✓ | ✓ |
| Llama-70B | 8 | 95.9 GB | OOM | ✓ | ✓ |
| GPT-3 175B | 8 | 253.5 GB | OOM | OOM | OOM |
