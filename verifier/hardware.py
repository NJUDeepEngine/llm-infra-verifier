"""GPU hardware models: H100, H200, B100, B200, A100, and generational specs.

Provides exact SM-level, HBM, register, and shared memory specifications.
Used by the memory graph OOM detector to check resource feasibility.

Modeled as a hierarchical resource graph:
  GPU → SMs → Warps → Threads → Registers
  GPU → HBM → L2 Cache → L1/Shared Memory → Registers

Each memory level has:
  - capacity (bytes)
  - bandwidth (bytes/sec)
  - latency (cycles, optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Level
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryLevel(Enum):
    HBM = "hbm"                   # Global GPU memory
    L2_CACHE = "l2"               # L2 cache (shared across SMs)
    L1_CACHE = "l1"               # L1 cache (per SM)
    SHARED_MEMORY = "shared"      # Shared memory (per SM, software-managed)
    REGISTER = "register"         # Register file (per SM, per thread)
    CONSTANT = "constant"         # Constant memory
    TEXTURE = "texture"           # Texture memory (read-only)


@dataclass
class MemoryTier:
    """A tier in the GPU memory hierarchy."""
    level: MemoryLevel
    capacity_bytes: int
    bandwidth_bytes_per_sec: float   # peak bandwidth
    scope: str = "global"            # "global", "per_sm", "per_thread"
    latency_cycles: int = 0          # approximate access latency

    @property
    def capacity_gb(self) -> float:
        return self.capacity_bytes / (1024**3)

    @property
    def capacity_mb(self) -> float:
        return self.capacity_bytes / (1024**2)

    @property
    def capacity_kb(self) -> float:
        return self.capacity_bytes / 1024

    @property
    def bandwidth_gb_s(self) -> float:
        return self.bandwidth_bytes_per_sec / (1024**3)

    def __repr__(self):
        return (f"{self.level.value}({self.capacity_gb:.1f}GB"
                + (f", {self.capacity_kb:.0f}KB" if self.capacity_bytes < 1024**3 else "")
                + f", {self.bandwidth_gb_s:.0f}GB/s)")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SMResources:
    """Resources available per Streaming Multiprocessor."""
    num_sms: int
    max_threads: int          # max concurrent threads
    max_warps: int            # max concurrent warps (threads / 32)
    max_blocks: int           # max concurrent thread blocks
    registers_total: int      # total 32-bit registers
    shared_memory_bytes: int  # shared memory + L1 carveout (max shared config)


@dataclass
class GPUModel:
    """Complete model of a GPU for resource analysis.

    Attributes:
        name: Human-readable name
        generation: Architecture generation (Hopper, Blackwell, etc.)
        compute_capability: CUDA compute capability (e.g., "9.0" for Hopper)
        hbm: Global HBM specification
        l2_cache: L2 cache specification
        sm: SM-level resources
        max_grid_dim: Maximum grid dimensions (x, y, z)
        max_block_dim: Maximum block dimensions (x, y, z)
        tensor_core_fp16_tflops: Peak FP16 tensor core throughput
        nvlink_bandwidth: NVLink interconnect bandwidth (bytes/sec)
    """
    name: str
    generation: str
    compute_capability: str

    # Memory tiers
    hbm: MemoryTier
    l2_cache: MemoryTier

    # SM resources
    sm: SMResources

    # Launch limits
    max_grid_dim: Tuple[int, int, int] = (2**31 - 1, 65535, 65535)
    max_block_dim: Tuple[int, int, int] = (1024, 1024, 64)

    # Performance
    tensor_core_fp16_tflops: float = 0.0
    nvlink_bandwidth_bytes_per_sec: float = 0.0

    @property
    def total_hbm_bytes(self) -> int:
        return self.hbm.capacity_bytes

    @property
    def total_hbm_gb(self) -> float:
        return self.hbm.capacity_gb

    @property
    def total_registers(self) -> int:
        return self.sm.registers_total * self.sm.num_sms

    @property
    def total_shared_memory_bytes(self) -> int:
        return self.sm.shared_memory_bytes * self.sm.num_sms

    def __repr__(self):
        return (f"GPUModel({self.name}, {self.total_hbm_gb:.0f}GB HBM, "
                f"{self.sm.num_sms} SMs, {self.generation})")


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-configured GPU models
# ═══════════════════════════════════════════════════════════════════════════════

# --- NVIDIA H100 SXM5 (Hopper, 2022) ---
H100_SXM = GPUModel(
    name="NVIDIA H100 SXM5",
    generation="Hopper",
    compute_capability="9.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=80 * 1024**3,       # 80 GB HBM3
        bandwidth_bytes_per_sec=3.35e12,   # 3.35 TB/s
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=50 * 1024**2,       # 50 MB
        bandwidth_bytes_per_sec=12.0e12,   # ~12 TB/s (internal)
        scope="global",
    ),
    sm=SMResources(
        num_sms=132,
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,              # 64K × 32-bit registers
        shared_memory_bytes=228 * 1024,     # 228 KB (max carveout)
    ),
    tensor_core_fp16_tflops=989.0,         # FP16 dense
    nvlink_bandwidth_bytes_per_sec=900e9,  # 900 GB/s
)

# --- NVIDIA H100 PCIe (Hopper, 2022) ---
H100_PCIE = GPUModel(
    name="NVIDIA H100 PCIe",
    generation="Hopper",
    compute_capability="9.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=80 * 1024**3,
        bandwidth_bytes_per_sec=2.0e12,    # 2.0 TB/s (PCIe limited)
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=50 * 1024**2,
        bandwidth_bytes_per_sec=12.0e12,
        scope="global",
    ),
    sm=SMResources(
        num_sms=114,                        # fewer SMs than SXM
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,
        shared_memory_bytes=228 * 1024,
    ),
    tensor_core_fp16_tflops=756.0,
    nvlink_bandwidth_bytes_per_sec=600e9,
)

# --- NVIDIA H200 SXM (Hopper, 2024) ---
H200_SXM = GPUModel(
    name="NVIDIA H200 SXM",
    generation="Hopper",
    compute_capability="9.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=141 * 1024**3,      # 141 GB HBM3e
        bandwidth_bytes_per_sec=4.8e12,    # 4.8 TB/s
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=50 * 1024**2,
        bandwidth_bytes_per_sec=12.0e12,
        scope="global",
    ),
    sm=SMResources(
        num_sms=132,
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,
        shared_memory_bytes=228 * 1024,
    ),
    tensor_core_fp16_tflops=989.0,
    nvlink_bandwidth_bytes_per_sec=900e9,
)

# --- NVIDIA A100 SXM (Ampere, 2020) ---
A100_SXM = GPUModel(
    name="NVIDIA A100 SXM",
    generation="Ampere",
    compute_capability="8.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=80 * 1024**3,       # 80 GB HBM2e
        bandwidth_bytes_per_sec=2.0e12,    # 2.0 TB/s
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=40 * 1024**2,       # 40 MB
        bandwidth_bytes_per_sec=8.0e12,
        scope="global",
    ),
    sm=SMResources(
        num_sms=108,
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,
        shared_memory_bytes=164 * 1024,    # 164 KB (max carveout, configurable)
    ),
    tensor_core_fp16_tflops=312.0,
    nvlink_bandwidth_bytes_per_sec=600e9,
)

# --- NVIDIA B200 (Blackwell, 2024) — estimated specs ---
# Publicly known: dual-die, ~208B transistors, 192GB HBM3e
# SM count and bandwidth are estimates based on Blackwell architecture scaling
B200 = GPUModel(
    name="NVIDIA B200 (Blackwell)",
    generation="Blackwell",
    compute_capability="10.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=192 * 1024**3,      # 192 GB HBM3e
        bandwidth_bytes_per_sec=8.0e12,    # 8 TB/s
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=96 * 1024**2,       # ~96 MB (estimated)
        bandwidth_bytes_per_sec=20.0e12,   # ~20 TB/s (estimated)
        scope="global",
    ),
    sm=SMResources(
        num_sms=160,                        # estimated (dual-die, ~80 per die)
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,
        shared_memory_bytes=256 * 1024,    # 256 KB (estimated, Blackwell increase)
    ),
    tensor_core_fp16_tflops=2250.0,        # estimated (2.25 PFLOPS FP16)
    nvlink_bandwidth_bytes_per_sec=1.8e12, # 1.8 TB/s (NVLink 5)
)

# --- NVIDIA B100 (Blackwell, 2024) — estimated ---
B100 = GPUModel(
    name="NVIDIA B100 (Blackwell)",
    generation="Blackwell",
    compute_capability="10.0",
    hbm=MemoryTier(
        level=MemoryLevel.HBM,
        capacity_bytes=128 * 1024**3,      # 128 GB HBM3e
        bandwidth_bytes_per_sec=6.0e12,    # 6 TB/s (estimated)
        scope="global",
    ),
    l2_cache=MemoryTier(
        level=MemoryLevel.L2_CACHE,
        capacity_bytes=72 * 1024**2,       # ~72 MB (estimated)
        bandwidth_bytes_per_sec=16.0e12,
        scope="global",
    ),
    sm=SMResources(
        num_sms=128,                        # estimated
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers_total=65536,
        shared_memory_bytes=256 * 1024,
    ),
    tensor_core_fp16_tflops=1800.0,
    nvlink_bandwidth_bytes_per_sec=1.8e12,
)


# Registry of all models
GPU_MODELS: Dict[str, GPUModel] = {
    "H100": H100_SXM,
    "H100-SXM": H100_SXM,
    "H100-PCIE": H100_PCIE,
    "H200": H200_SXM,
    "H200-SXM": H200_SXM,
    "A100": A100_SXM,
    "A100-SXM": A100_SXM,
    "B200": B200,
    "B100": B100,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-GPU cluster model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterConfig:
    """Configuration for a multi-GPU training cluster."""
    gpu_model: GPUModel
    num_gpus: int
    # Interconnect
    intranode_bandwidth_bytes_per_sec: float   # NVLink / PCIe within node
    internode_bandwidth_bytes_per_sec: float   # InfiniBand / RoCE across nodes
    num_nodes: int = 1
    gpus_per_node: int = 8

    @property
    def total_hbm_bytes(self) -> int:
        return self.gpu_model.total_hbm_bytes * self.num_gpus

    @property
    def total_hbm_gb(self) -> float:
        return self.total_hbm_bytes / (1024**3)

    def __repr__(self):
        return (f"Cluster({self.num_gpus}x {self.gpu_model.name}, "
                f"{self.total_hbm_gb:.0f}GB total HBM, "
                f"{self.num_nodes} nodes × {self.gpus_per_node} GPUs)")
