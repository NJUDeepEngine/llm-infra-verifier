"""
Demo: GPU Memory Hierarchy OOM Detection.

Shows memory resource analysis for H100/H200/B200 GPUs:
  1. GPU specs comparison
  2. LLM memory estimation for different model sizes
  3. OOM detection across HBM → Shared → Register levels
  4. Occupancy analysis for GEMM/Attention kernels
  5. Multi-GPU cluster memory planning (TP/PP/DP)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.hardware import (
    GPUModel, MemoryLevel, MemoryTier, SMResources,
    GPU_MODELS, H100_SXM, H200_SXM, A100_SXM, B200, B100,
    ClusterConfig,
)
from verifier.memory_graph import (
    MemoryNode, MemoryGraph, KernelResources,
    OOMDetector, OOMReport, OOMCheck, OOMSeverity,
    compute_occupancy, estimate_llm_memory,
)


def demo_gpu_specs():
    """Compare GPU models."""
    print("=" * 70)
    print("  1. GPU SPECIFICATIONS COMPARISON")
    print("=" * 70)

    gpus = [A100_SXM, H100_SXM, H200_SXM, B100, B200]
    print(f"  {'Model':<25} {'HBM':>10} {'SMs':>5} {'Shared/SM':>10} {'Regs/SM':>8} {'TFLOPS':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*5} {'-'*10} {'-'*8} {'-'*8}")
    for g in gpus:
        print(f"  {g.name:<25} {g.total_hbm_gb:>7.0f}GB {g.sm.num_sms:>5} "
              f"{g.sm.shared_memory_bytes//1024:>7}KB {g.sm.registers_total:>8} "
              f"{g.tensor_core_fp16_tflops:>6.0f}")


def demo_llm_memory_estimation():
    """Estimate memory for standard LLM sizes."""
    print("\n" + "=" * 70)
    print("  2. LLM MEMORY ESTIMATION (per GPU)")
    print("=" * 70)

    configs = [
        ("Llama-7B", 4096, 32, 50000, 1, 2048),
        ("Llama-13B", 5120, 40, 50000, 1, 2048),
        ("Llama-70B (TP=8)", 8192, 80, 50000, 1, 2048, 8),
        ("DeepSeek-V2 (TP=8)", 7168, 60, 128000, 1, 4096, 8),
        ("GPT-3 175B (TP=8)", 12288, 96, 50000, 1, 2048, 8),
    ]

    print(f"  {'Model':<28} {'Params':>8} {'Grads':>8} {'Optim':>8} {'Activ':>8} {'Total':>9} {'H100?':>6}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*6}")
    for name, hd, nl, vs, bs, sl, *tp in configs:
        tp_size = tp[0] if tp else 1
        mem = estimate_llm_memory(hd, nl, vs, bs, sl, tp_size=tp_size)
        total_gb = mem["total"] / (1024**3)
        fits = "YES" if total_gb < 80 else "OOM!"
        print(f"  {name:<28} {mem['params']/(1024**3):>7.1f}G {mem['gradients']/(1024**3):>7.1f}G "
              f"{mem['optimizer']/(1024**3):>7.1f}G {mem['activations']/(1024**3):>7.1f}G "
              f"{total_gb:>8.1f}G {fits:>6}")


def demo_oom_detection():
    """Run OOM detection for a realistic model on H100."""
    print("\n" + "=" * 70)
    print("  3. OOM DETECTION — Llama-7B on H100")
    print("=" * 70)

    gpu = H100_SXM
    detector = OOMDetector(gpu)

    # Build memory graph for Llama-7B training
    graph = MemoryGraph(gpu=gpu, num_program_ops=10)
    hidden, layers, vocab = 4096, 32, 50000
    batch, seq = 1, 2048
    dtype = 2  # fp16

    # Parameters
    params_per_layer = 12 * hidden * hidden
    total_params = layers * params_per_layer + vocab * hidden
    param_bytes = total_params * dtype

    # Add memory nodes
    graph.add_tensor_node(MemoryNode(
        "params", dtype, total_params, hbm_bytes=param_bytes,
        location=MemoryLevel.HBM, first_use=0, last_use=9,
    ))
    graph.add_tensor_node(MemoryNode(
        "gradients", dtype, total_params, hbm_bytes=param_bytes,
        location=MemoryLevel.HBM, first_use=5, last_use=9,
        is_communication_buffer=True,
    ))
    graph.add_tensor_node(MemoryNode(
        "optimizer_m", 4, total_params, hbm_bytes=total_params * 4,
        location=MemoryLevel.HBM, first_use=0, last_use=9,
    ))
    graph.add_tensor_node(MemoryNode(
        "optimizer_v", 4, total_params, hbm_bytes=total_params * 4,
        location=MemoryLevel.HBM, first_use=0, last_use=9,
    ))

    # Activations (peak at middle layer)
    act_per_layer = batch * seq * hidden * dtype * 34  # 34x factor
    # Only some layers have activations live simultaneously (PP reduces this)
    peak_live_layers = layers // 4  # with activation ckpt, ~1/4 live at once
    act_bytes = act_per_layer * peak_live_layers
    for l in range(peak_live_layers):
        graph.add_tensor_node(MemoryNode(
            f"act_layer_{l}", dtype, batch * seq * hidden,
            hbm_bytes=act_per_layer,
            location=MemoryLevel.HBM,
            first_use=2, last_use=7,
            is_activation=True,
        ))

    # Communication buffer
    comm_buf = batch * seq * hidden * dtype * 2
    graph.add_tensor_node(MemoryNode(
        "allreduce_buffer", dtype, batch * seq * hidden,
        hbm_bytes=comm_buf,
        location=MemoryLevel.HBM,
        first_use=4, last_use=6,
        is_communication_buffer=True,
    ))

    # Kernel resources (GEMM tile)
    graph.add_kernel(KernelResources(
        name="qkv_projection",
        threads_per_block=256,
        registers_per_thread=64,
        shared_mem_per_block_bytes=48 * 1024,  # 48KB
        num_blocks=gpu.sm.num_sms * 2,         # 2 blocks per SM
    ))
    graph.add_kernel(KernelResources(
        name="attention_score",
        threads_per_block=128,
        registers_per_thread=128,
        shared_mem_per_block_bytes=96 * 1024,  # 96KB for QK^T tiles
        num_blocks=gpu.sm.num_sms,
    ))

    report = detector.analyze(graph)
    print(report.summary())


def demo_occupancy():
    """Show occupancy analysis for different kernel configs."""
    print("\n" + "=" * 70)
    print("  4. OCCUPANCY ANALYSIS — H100 SM")
    print("=" * 70)

    sm = H100_SXM.sm
    print(f"  SM resources: {sm.max_threads} threads, "
          f"{sm.registers_total} regs, {sm.shared_memory_bytes//1024}KB shared")

    kernels = [
        ("Low regs, low shared", 256, 32, 16*1024),
        ("Medium regs, medium shared", 256, 64, 48*1024),
        ("High regs, high shared (FlashAttn)", 128, 128, 96*1024),
        ("Extreme regs (unrolled)", 128, 255, 32*1024),     # may fail
        ("Max shared (GEMM tile)", 256, 64, 200*1024),       # near max
    ]

    print(f"\n  {'Config':<35} {'Threads':>8} {'Regs':>5} {'Shared':>8} {'Occ%':>6} {'Bottleneck':>12}")
    print(f"  {'-'*35} {'-'*8} {'-'*5} {'-'*8} {'-'*6} {'-'*12}")
    for name, tb, rpt, smem in kernels:
        k = KernelResources(name, tb, rpt, smem)
        occ = compute_occupancy(k, sm)
        print(f"  {name:<35} {occ.max_threads_achievable:>8} {rpt:>5} "
              f"{smem//1024:>5}KB {occ.occupancy_pct:>5.0f}% {occ.bottleneck:>12}")


def demo_cluster_planning():
    """Memory planning for multi-GPU clusters."""
    print("\n" + "=" * 70)
    print("  5. MULTI-GPU CLUSTER MEMORY PLANNING")
    print("=" * 70)

    # Simulate 70B model on different GPU configs
    hidden, layers, vocab = 8192, 80, 50000
    batch, seq = 1, 4096

    configs = [
        ("8× A100 (80GB)", A100_SXM, 8, 1, 1, 8),
        ("8× H100 (80GB)", H100_SXM, 8, 1, 1, 8),
        ("8× H200 (141GB)", H200_SXM, 8, 1, 1, 8),
        ("16× H100 (TP=8, DP=2)", H100_SXM, 16, 8, 1, 2),
        ("8× B200 (192GB)", B200, 8, 1, 1, 8),
    ]

    print(f"  Model: {hidden}D, {layers}L, vocab={vocab}, batch={batch}, seq={seq}")
    print(f"\n  {'Config':<30} {'Per-GPU':>10} {'Capacity':>10} {'Usage':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name, gpu, n_gpus, tp, pp, dp in configs:
        mem = estimate_llm_memory(hidden, layers, vocab, batch, seq,
                                   tp_size=tp, pp_size=pp, dp_size=dp)
        per_gpu_gb = mem["total"] / (1024**3)
        capacity_gb = gpu.total_hbm_gb
        pct = (per_gpu_gb / capacity_gb) * 100
        status = "OK" if per_gpu_gb < capacity_gb else "OOM!"
        print(f"  {name:<30} {per_gpu_gb:>7.1f}GB {capacity_gb:>7.0f}GB {pct:>6.0f}% {status:>8}")


def demo_cross_generation_comparison():
    """Compare what model sizes fit on different GPU generations."""
    print("\n" + "=" * 70)
    print("  6. CROSS-GENERATION CAPACITY COMPARISON")
    print("=" * 70)

    gpus = [A100_SXM, H100_SXM, H200_SXM, B200]
    model_sizes = [
        ("7B (4096/32L)", 4096, 32),
        ("13B (5120/40L)", 5120, 40),
        ("34B (7168/60L)", 7168, 60),
        ("70B (8192/80L)", 8192, 80),
        ("130B (12288/96L)", 12288, 96),
    ]

    print(f"  {'Model':<22}", end="")
    for g in gpus:
        print(f"  {g.name.split()[1]:>8}", end="")
    print()

    for name, hd, nl in model_sizes:
        print(f"  {name:<22}", end="")
        for g in gpus:
            mem = estimate_llm_memory(hd, nl, 50000, 1, 2048, tp_size=8)
            per_gpu = mem["total"] / (1024**3)
            fits = "✓" if per_gpu < g.total_hbm_gb else "✗"
            print(f"  {fits} {per_gpu:>5.0f}G", end="")
        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DTENSOR-VERIFIER: GPU Memory Hierarchy OOM Detection")
    print("  HBM → L2 → Shared Memory → Registers")
    print("=" * 70)

    demo_gpu_specs()
    demo_llm_memory_estimation()
    demo_oom_detection()
    demo_occupancy()
    demo_cluster_planning()
    demo_cross_generation_comparison()

    print("\n" + "=" * 70)
    print("  KEY INSIGHTS")
    print("=" * 70)
    print("""
  1. Memory hierarchy is a GRAPH:
     HBM (global) → L2 (chip-wide) → Shared (per-SM) → Registers (per-thread)
     Each level has a hard capacity. Exceeding any level = OOM.

  2. OOM bottlenecks by model size:
     < 7B:  HBM rarely the issue (fits in 80GB with TP=1)
     7-70B: HBM becomes critical → need TP/PP/activation ckpt
     > 70B: Multiple parallelism dimensions REQUIRED

  3. Per-SM limits matter:
     FlashAttention: shared memory heavy (96KB per block)
     Unrolled GEMM: register heavy (128+ regs per thread)
     Both limit occupancy → fewer concurrent warps → lower throughput

  4. H200 vs H100: 76% more HBM (141GB vs 80GB), same SMs
     → Same compute, more model fits. Memory-bound workloads benefit most.

  5. B200 vs H100: 140% more HBM (192GB vs 80GB), ~21% more SMs
     → Bigger models AND faster. But still finite resources.
""")
