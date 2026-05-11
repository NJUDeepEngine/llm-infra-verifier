"""Memory graph builder and OOM detector for GPU resource analysis.

Models the GPU memory hierarchy as a directed graph:
  HBM → L2 → L1/Shared Memory → Registers

Each tensor is a node with:
  - size (bytes) at each memory level
  - lifetime (which ops it's live between)
  - location (HBM, shared, register)
  - async handle (if in-flight for overlap)

The graph is traversed to compute peak memory usage at each level
and detect OOM conditions.

Key checks:
  1. HBM OOM: total active tensors > HBM capacity
  2. Shared memory OOM: per-block shared memory > SM shared memory
  3. Register pressure: per-thread registers × threads/block > SM registers
  4. Occupancy: register/shared usage limiting concurrent blocks
  5. Activation memory: PP schedule + saved activations
  6. Communication buffers: AllReduce/AllGather temporary buffers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math

from .hardware import (
    GPUModel, MemoryLevel, MemoryTier, SMResources,
    GPU_MODELS, H100_SXM, H200_SXM, B200,
)
from .state import TensorState


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Node — a tensor's footprint at each memory level
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryNode:
    """A tensor node in the memory graph with per-level footprint.

    Tracks where the tensor lives (HBM/shared/register), its size
    in bytes, and which ops it's live between (for lifetime analysis).
    """
    name: str
    dtype_bytes: int = 2          # bytes per element (fp16=2, fp32=4, bf16=2)
    num_elements: int = 0         # total elements
    hbm_bytes: int = 0            # global memory footprint
    shared_mem_bytes: int = 0     # shared memory footprint (if in shared)
    register_count: int = 0       # register usage (if in registers)
    location: MemoryLevel = MemoryLevel.HBM

    # Lifetime: which op indices this tensor is live between
    first_use: int = 0
    last_use: int = 0
    is_activation: bool = False   # saved for backward (PP)
    is_communication_buffer: bool = False  # temp buffer for collectives

    # Async
    async_handle: Optional[str] = None  # if in-flight from async op

    @property
    def total_bytes(self) -> int:
        return self.hbm_bytes + self.shared_mem_bytes

    @property
    def size_mb(self) -> float:
        return self.hbm_bytes / (1024**2)

    @property
    def size_gb(self) -> float:
        return self.hbm_bytes / (1024**3)

    def is_live_at(self, op_index: int) -> bool:
        """Check if this tensor is live at a given program point."""
        return self.first_use <= op_index <= self.last_use

    def __repr__(self):
        loc = self.location.value
        return (f"MemNode({self.name}, {self.size_mb:.1f}MB "
                f"@{loc}, live=[{self.first_use},{self.last_use}])")


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel Resource Usage — per-kernel SM resource consumption
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KernelResources:
    """Resource usage for a single kernel launch."""
    name: str
    threads_per_block: int = 256
    registers_per_thread: int = 64
    shared_mem_per_block_bytes: int = 0
    num_blocks: int = 1

    @property
    def registers_per_block(self) -> int:
        return self.registers_per_thread * self.threads_per_block

    @property
    def total_shared_mem_bytes(self) -> int:
        return self.shared_mem_per_block_bytes * self.num_blocks

    @property
    def total_registers(self) -> int:
        return self.registers_per_block * self.num_blocks


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Graph
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryGraph:
    """Complete memory graph for a program running on a specific GPU.

    Builds the graph from tensor states and kernel resource estimates,
    then analyzes peak memory usage at each level.
    """
    gpu: GPUModel
    nodes: List[MemoryNode] = field(default_factory=list)
    kernels: List[KernelResources] = field(default_factory=list)
    num_program_ops: int = 0

    def add_tensor_node(self, node: MemoryNode):
        self.nodes.append(node)

    def add_kernel(self, kernel: KernelResources):
        self.kernels.append(kernel)

    def build_from_tensor_states(
        self,
        tensors: Dict[str, TensorState],
        op_to_tensor_map: Dict[int, List[str]] = None,
        activations: Set[str] = None,
    ):
        """Build memory nodes from TensorState objects.

        Args:
            tensors: {name: TensorState} after symbolic execution
            op_to_tensor_map: {op_index: [tensor_names live after this op]}
            activations: names of tensors saved for backward
        """
        activations = activations or set()

        for name, ts in tensors.items():
            num_elements = math.prod(ts.global_shape) if ts.global_shape else 0
            dtype_bytes = 2  # default fp16/bf16

            # Estimate per-level footprint
            hbm_bytes = num_elements * dtype_bytes
            shared_bytes = 0
            regs = 0

            # If tensor is in shared memory (tiled), estimate tile size
            if ts.local_shape and ts.local_shape != ts.global_shape:
                # Locally sharded → smaller HBM footprint
                hbm_bytes = math.prod(ts.local_shape) * dtype_bytes

            node = MemoryNode(
                name=name,
                dtype_bytes=dtype_bytes,
                num_elements=num_elements,
                hbm_bytes=hbm_bytes,
                shared_mem_bytes=shared_bytes,
                register_count=regs,
                location=MemoryLevel.HBM,
                is_activation=name in activations,
                is_communication_buffer="_partial" in name or "_temp" in name,
                async_handle=ts._async_handle,
            )
            self.add_tensor_node(node)

    def peak_hbm_at_op(self, op_index: int) -> int:
        """Compute peak HBM usage at a specific program point."""
        total = 0
        for node in self.nodes:
            if node.is_live_at(op_index):
                total += node.hbm_bytes
        return total

    def peak_hbm_over_program(self) -> Tuple[int, int]:
        """Return (peak_bytes, op_index) of maximum HBM usage."""
        if self.num_program_ops == 0:
            return 0, 0

        peak = 0
        peak_idx = 0
        for i in range(self.num_program_ops):
            usage = self.peak_hbm_at_op(i)
            if usage > peak:
                peak = usage
                peak_idx = i
        return peak, peak_idx

    def peak_shared_memory_per_sm(self) -> int:
        """Return maximum shared memory per SM across all kernels."""
        if not self.kernels:
            return 0
        return max(k.shared_mem_per_block_bytes for k in self.kernels)

    def peak_registers_per_sm(self) -> Tuple[int, int]:
        """Return (max_registers_per_sm, max_registers_per_thread)."""
        if not self.kernels:
            return 0, 0
        max_per_block = max(k.registers_per_block for k in self.kernels)
        max_per_thread = max(k.registers_per_thread for k in self.kernels)
        return max_per_block, max_per_thread


# ═══════════════════════════════════════════════════════════════════════════════
# Occupancy Calculator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OccupancyResult:
    """Kernel occupancy analysis result."""
    kernel_name: str
    threads_per_block: int
    registers_per_thread: int
    shared_mem_per_block_bytes: int

    # Limits
    max_blocks_by_threads: int
    max_blocks_by_registers: int
    max_blocks_by_shared_mem: int
    max_blocks_total: int       # bottleneck
    max_threads_achievable: int
    occupancy_pct: float        # % of max theoretical occupancy

    @property
    def bottleneck(self) -> str:
        limits = {
            "threads": self.max_blocks_by_threads,
            "registers": self.max_blocks_by_registers,
            "shared_mem": self.max_blocks_by_shared_mem,
        }
        return min(limits, key=limits.get)

    def __repr__(self):
        return (
            f"Occupancy({self.kernel_name}: {self.occupancy_pct:.0f}%, "
            f"{self.max_threads_achievable} threads, "
            f"bottleneck={self.bottleneck}, "
            f"blocks={self.max_blocks_total})"
        )


def compute_occupancy(
    kernel: KernelResources,
    sm: SMResources,
) -> OccupancyResult:
    """Compute theoretical occupancy for a kernel on a given SM.

    Occupancy = min(
        blocks limited by threads,
        blocks limited by registers,
        blocks limited by shared memory
    )

    Args:
        kernel: Kernel resource usage
        sm: SM resource limits

    Returns:
        OccupancyResult with breakdown
    """
    # Blocks limited by max threads per SM
    max_blocks_by_threads = sm.max_threads // kernel.threads_per_block
    max_blocks_by_threads = min(max_blocks_by_threads, sm.max_blocks)

    # Blocks limited by registers per SM
    regs_per_block = kernel.registers_per_thread * kernel.threads_per_block
    if regs_per_block > 0:
        # Register allocation granularity: 256 registers per warp on H100
        # Round up to nearest 256 (warp allocation unit)
        regs_per_warp = kernel.registers_per_thread * 32
        regs_per_warp = ((regs_per_warp + 255) // 256) * 256
        regs_per_block_aligned = regs_per_warp * (kernel.threads_per_block // 32)
        max_blocks_by_registers = sm.registers_total // max(regs_per_block_aligned, 1)
    else:
        max_blocks_by_registers = sm.max_blocks
    max_blocks_by_registers = min(max_blocks_by_registers, sm.max_blocks)

    # Blocks limited by shared memory
    if kernel.shared_mem_per_block_bytes > 0:
        max_blocks_by_shared_mem = sm.shared_memory_bytes // max(
            kernel.shared_mem_per_block_bytes, 1
        )
    else:
        max_blocks_by_shared_mem = sm.max_blocks
    max_blocks_by_shared_mem = min(max_blocks_by_shared_mem, sm.max_blocks)

    # Bottleneck
    max_blocks = min(max_blocks_by_threads, max_blocks_by_registers,
                     max_blocks_by_shared_mem)
    max_threads = max_blocks * kernel.threads_per_block
    occupancy_pct = (max_threads / sm.max_threads) * 100.0

    return OccupancyResult(
        kernel_name=kernel.name,
        threads_per_block=kernel.threads_per_block,
        registers_per_thread=kernel.registers_per_thread,
        shared_mem_per_block_bytes=kernel.shared_mem_per_block_bytes,
        max_blocks_by_threads=max_blocks_by_threads,
        max_blocks_by_registers=max_blocks_by_registers,
        max_blocks_by_shared_mem=max_blocks_by_shared_mem,
        max_blocks_total=max_blocks,
        max_threads_achievable=max_threads,
        occupancy_pct=occupancy_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OOM Detector
# ═══════════════════════════════════════════════════════════════════════════════

class OOMSeverity(Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class OOMCheck:
    """Result of a single OOM check."""
    level: MemoryLevel
    severity: OOMSeverity
    peak_usage_bytes: int
    capacity_bytes: int
    usage_pct: float
    description: str
    recommendation: str = ""

    @property
    def passed(self) -> bool:
        return self.severity == OOMSeverity.SAFE

    def __repr__(self):
        status = "SAFE" if self.passed else f"{self.severity.value.upper()}"
        return (
            f"[{status}] {self.level.value}: "
            f"{self.peak_usage_bytes/(1024**3):.2f}GB / "
            f"{self.capacity_bytes/(1024**3):.2f}GB "
            f"({self.usage_pct:.0f}%)"
            + (f" — {self.recommendation}" if self.recommendation else "")
        )


@dataclass
class OOMReport:
    """Complete OOM analysis report."""
    gpu: GPUModel
    peak_hbm_bytes: int
    peak_hbm_op_idx: int
    peak_shared_mem_per_sm: int
    peak_registers_per_sm: int
    peak_registers_per_thread: int
    checks: List[OOMCheck] = field(default_factory=list)
    occupancy_results: List[OccupancyResult] = field(default_factory=list)
    activation_bytes: int = 0
    communication_buffer_bytes: int = 0

    @property
    def is_safe(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def peak_hbm_gb(self) -> float:
        return self.peak_hbm_bytes / (1024**3)

    @property
    def hbm_usage_pct(self) -> float:
        return (self.peak_hbm_bytes / self.gpu.total_hbm_bytes) * 100.0

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"  OOM ANALYSIS — {self.gpu.name}",
            "=" * 70,
            "",
            f"  GPU: {self.gpu.total_hbm_gb:.0f}GB HBM, {self.gpu.sm.num_sms} SMs",
            f"  Peak HBM: {self.peak_hbm_gb:.2f}GB ({self.hbm_usage_pct:.0f}%) at op [{self.peak_hbm_op_idx}]",
            f"  Activations: {self.activation_bytes/(1024**3):.2f}GB",
            f"  Comm buffers: {self.communication_buffer_bytes/(1024**3):.2f}GB",
            "",
            "  Memory Hierarchy Checks:",
        ]
        for c in self.checks:
            lines.append(f"    {c}")
        if self.occupancy_results:
            lines.append("")
            lines.append("  Kernel Occupancy:")
            for o in self.occupancy_results:
                lines.append(f"    {o}")
        lines.append("")
        lines.append(f"  Verdict: {'SAFE' if self.is_safe else 'OOM RISK DETECTED'}")
        return "\n".join(lines)


class OOMDetector:
    """Detects out-of-memory conditions across the GPU memory hierarchy.

    Checks at four levels:
      1. HBM (global memory) — total live tensors
      2. Shared memory (per SM) — per-block shared memory
      3. Registers (per SM) — per-block register usage
      4. Occupancy — concurrent block limits
    """

    # Thresholds for warnings
    HBM_WARNING_PCT = 85.0      # warn at 85% HBM usage
    HBM_CRITICAL_PCT = 95.0     # critical at 95% HBM usage
    SHARED_MEM_WARNING_PCT = 80.0
    REGISTER_WARNING_PCT = 80.0
    OCCUPANCY_WARNING_PCT = 50.0  # warn if occupancy < 50%

    def __init__(self, gpu: GPUModel):
        self.gpu = gpu

    def analyze(
        self,
        memory_graph: MemoryGraph,
        activation_nodes: Optional[List[MemoryNode]] = None,
        comm_buffer_nodes: Optional[List[MemoryNode]] = None,
    ) -> OOMReport:
        """Run full OOM analysis on a memory graph.

        Args:
            memory_graph: Built memory graph with nodes and kernels
            activation_nodes: Tensors saved for backward (PP)
            comm_buffer_nodes: Temporary buffers for collectives

        Returns:
            OOMReport with all checks
        """
        checks = []

        # ── 1. HBM Check ────────────────────────────────────────────────
        peak_hbm, peak_idx = memory_graph.peak_hbm_over_program()
        activation_bytes = sum(
            n.hbm_bytes for n in (activation_nodes or [])
            if n.is_activation
        )
        comm_bytes = sum(
            n.hbm_bytes for n in (comm_buffer_nodes or [])
            if n.is_communication_buffer
        )

        # Add communication buffer overhead (AllReduce needs 2x buffer)
        total_peak = peak_hbm + comm_bytes

        hbm_pct = (total_peak / self.gpu.total_hbm_bytes) * 100.0
        if hbm_pct >= self.HBM_CRITICAL_PCT:
            severity = OOMSeverity.CRITICAL
            rec = "Reduce batch size, enable activation checkpointing, or use ZeRO-3"
        elif hbm_pct >= self.HBM_WARNING_PCT:
            severity = OOMSeverity.WARNING
            rec = "Consider activation checkpointing or gradient accumulation"
        else:
            severity = OOMSeverity.SAFE
            rec = ""

        checks.append(OOMCheck(
            level=MemoryLevel.HBM,
            severity=severity,
            peak_usage_bytes=total_peak,
            capacity_bytes=self.gpu.total_hbm_bytes,
            usage_pct=hbm_pct,
            description=f"Peak HBM at op [{peak_idx}]: activations + params + comm buffers",
            recommendation=rec,
        ))

        # ── 2. Shared Memory Check (per SM) ─────────────────────────────
        peak_shared = memory_graph.peak_shared_memory_per_sm()
        shared_pct = (peak_shared / self.gpu.sm.shared_memory_bytes) * 100.0

        if peak_shared > self.gpu.sm.shared_memory_bytes:
            severity = OOMSeverity.CRITICAL
            rec = "Launch would FAIL: reduce shared memory per block"
        elif shared_pct >= self.SHARED_MEM_WARNING_PCT:
            severity = OOMSeverity.WARNING
            rec = "High shared memory usage may limit occupancy"
        else:
            severity = OOMSeverity.SAFE
            rec = ""

        checks.append(OOMCheck(
            level=MemoryLevel.SHARED_MEMORY,
            severity=severity,
            peak_usage_bytes=peak_shared,
            capacity_bytes=self.gpu.sm.shared_memory_bytes,
            usage_pct=shared_pct,
            description=f"Per-SM shared memory (max {self.gpu.sm.shared_memory_bytes//1024}KB)",
            recommendation=rec,
        ))

        # ── 3. Register Check (per SM) ──────────────────────────────────
        peak_regs_per_sm, peak_regs_per_thread = memory_graph.peak_registers_per_sm()
        reg_pct = (peak_regs_per_sm / self.gpu.sm.registers_total) * 100.0

        if peak_regs_per_sm > self.gpu.sm.registers_total:
            severity = OOMSeverity.CRITICAL
            rec = "Launch would FAIL: kernel uses too many registers"
        elif reg_pct >= self.REGISTER_WARNING_PCT:
            severity = OOMSeverity.WARNING
            rec = "High register pressure may cause spilling to L1"
        else:
            severity = OOMSeverity.SAFE
            rec = ""

        checks.append(OOMCheck(
            level=MemoryLevel.REGISTER,
            severity=severity,
            peak_usage_bytes=peak_regs_per_sm * 4,  # 4 bytes per register
            capacity_bytes=self.gpu.sm.registers_total * 4,
            usage_pct=reg_pct,
            description=(
                f"Per-SM registers: {peak_regs_per_sm}/{self.gpu.sm.registers_total} "
                f"({peak_regs_per_thread} per thread)"
            ),
            recommendation=rec,
        ))

        # ── 4. Occupancy ────────────────────────────────────────────────
        occupancy_results = []
        for kernel in memory_graph.kernels:
            occ = compute_occupancy(kernel, self.gpu.sm)
            occupancy_results.append(occ)
            if occ.occupancy_pct < self.OCCUPANCY_WARNING_PCT:
                if not any(
                    c.level == MemoryLevel.REGISTER
                    and c.severity != OOMSeverity.SAFE
                    for c in checks
                ):
                    checks.append(OOMCheck(
                        level=MemoryLevel.REGISTER,
                        severity=OOMSeverity.WARNING,
                        peak_usage_bytes=0,
                        capacity_bytes=0,
                        usage_pct=occ.occupancy_pct,
                        description=f"Low occupancy: {occ.kernel_name} at {occ.occupancy_pct:.0f}%",
                        recommendation=f"Bottleneck: {occ.bottleneck}. "
                                       f"Reduce register/shared memory usage.",
                    ))

        return OOMReport(
            gpu=self.gpu,
            peak_hbm_bytes=total_peak,
            peak_hbm_op_idx=peak_idx,
            peak_shared_mem_per_sm=peak_shared,
            peak_registers_per_sm=peak_regs_per_sm,
            peak_registers_per_thread=peak_regs_per_thread,
            checks=checks,
            occupancy_results=occupancy_results,
            activation_bytes=activation_bytes,
            communication_buffer_bytes=comm_bytes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: estimate memory for common model sizes
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_llm_memory(
    hidden_dim: int,
    num_layers: int,
    vocab_size: int = 50000,
    batch_size: int = 1,
    seq_len: int = 2048,
    dtype_bytes: int = 2,           # fp16/bf16
    optimizer_dtype_bytes: int = 4, # fp32 Adam
    tp_size: int = 1,
    pp_size: int = 1,
    dp_size: int = 1,
) -> Dict[str, int]:
    """Estimate memory usage for a transformer LLM.

    Returns breakdown in bytes:
      - param_memory: model parameters
      - grad_memory: gradients
      - optimizer_memory: Adam m/v state
      - activation_memory: saved activations (peak)
      - total per GPU

    Based on standard transformer memory analysis (Narayanan et al., 2021).
    """
    n_gpus = tp_size * pp_size * dp_size

    # Parameters per layer: 4 × hidden_dim² (QKV + output for attention)
    #                        + 2 × hidden_dim × ffn_dim (MLP, roughly 8× hidden_dim)
    # ~ 12 × hidden_dim² per layer
    params_per_layer = 12 * hidden_dim * hidden_dim
    total_params = num_layers * params_per_layer + vocab_size * hidden_dim
    param_bytes = total_params * dtype_bytes

    # TP sharding: parameters split across TP ranks
    param_bytes_per_gpu = param_bytes // tp_size

    # Gradients: same size as parameters (before AllReduce)
    grad_bytes_per_gpu = param_bytes_per_gpu

    # Optimizer state (Adam): 2× parameters (m + v) in fp32
    optimizer_bytes_per_gpu = 2 * param_bytes_per_gpu * (optimizer_dtype_bytes // dtype_bytes)

    # Activations: roughly batch_size × seq_len × hidden_dim × num_layers × factor
    # Factor ~34 bytes per element for standard transformer (Korthikanti et al., 2023)
    activation_factor = 34  # bytes per element per layer
    activation_bytes = (
        batch_size * seq_len * hidden_dim * num_layers * activation_factor
    )
    activation_bytes_per_gpu = activation_bytes // tp_size

    # Communication buffers (AllReduce): 2× largest tensor
    largest_tensor = batch_size * seq_len * hidden_dim * dtype_bytes
    comm_buffer_bytes = 2 * largest_tensor

    total_per_gpu = (
        param_bytes_per_gpu
        + grad_bytes_per_gpu
        + optimizer_bytes_per_gpu
        + activation_bytes_per_gpu
        + comm_buffer_bytes
    )

    return {
        "params": param_bytes_per_gpu,
        "gradients": grad_bytes_per_gpu,
        "optimizer": optimizer_bytes_per_gpu,
        "activations": activation_bytes_per_gpu,
        "comm_buffers": comm_buffer_bytes,
        "total": total_per_gpu,
    }
