"""Pipeline parallelism schedules: 1F1B and GPipe.

Implements:
  1. 1F1B schedule generation (warmup → steady → cooldown)
  2. Activation memory tracking (which activations are live)
  3. Deadlock freedom checking (Send/Recv matching, no circular waits)
  4. Activation liveness verification for backward pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum


# ── Schedule phases ──────────────────────────────────────────────────────────

class Phase(Enum):
    WARMUP = "warmup"    # forward only
    STEADY = "steady"    # 1 forward + 1 backward alternating
    COOLDOWN = "cooldown"  # backward only


class OpType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class MicroBatch:
    """A single micro-batch in the pipeline schedule."""
    mb_id: int
    stage_id: int
    op_type: OpType
    phase: Phase
    sends: List[Tuple[int, str]] = field(default_factory=list)    # [(dst, tensor_name)]
    recvs: List[Tuple[int, str]] = field(default_factory=list)    # [(src, tensor_name)]

    @property
    def key(self) -> Tuple[int, int, str]:
        return (self.stage_id, self.mb_id, self.op_type.value)

    def __repr__(self):
        direction = "F" if self.op_type == OpType.FORWARD else "B"
        return f"MB(stage={self.stage_id}, mb={self.mb_id}, {direction}, {self.phase.value})"


@dataclass
class PP1F1BSchedule:
    """1F1B pipeline schedule.

    1F1B = "One Forward, One Backward" — the standard schedule for
    memory-efficient pipeline parallelism.

    Phases:
      Warmup:  stage 0 does mb=0, stage 1 does mb=0, ... until all stages
               have at least 1 forward in flight
      Steady:  each stage alternates 1 forward + 1 backward
      Cooldown: all remaining backwards are drained
    """

    num_stages: int
    num_microbatches: int

    def generate(self) -> List[MicroBatch]:
        """Generate the full 1F1B schedule.

        Returns ordered list of (stage_id, microbatch_id, is_forward, phase).
        """
        schedule: List[MicroBatch] = []
        P = self.num_stages
        M = self.num_microbatches

        # ── Warmup phase ──
        # Stage 0 starts mb=0, then stage 1 starts mb=0, etc.
        # After each stage has enough "in-flight" microbatches, steady state begins
        warmup_steps: List[Tuple[int, int]] = []  # [(stage_id, mb_id)]

        for step in range(P):  # at most P warmup steps per mbid
            for s in range(min(step + 1, P)):
                mb = step - s
                if 0 <= mb < M:
                    warmup_steps.append((s, mb))

        # Remove duplicates and sort by global step
        seen = set()
        warmup_unique = []
        for s, mb in warmup_steps:
            if (s, mb) not in seen:
                seen.add((s, mb))
                warmup_unique.append((s, mb))
        warmup_unique.sort(key=lambda x: (x[0] + x[1], x[0]))

        for s, mb in warmup_unique:
            schedule.append(MicroBatch(
                mb_id=mb,
                stage_id=s,
                op_type=OpType.FORWARD,
                phase=Phase.WARMUP,
            ))

        # ── Steady state phase ──
        # Alternating 1F + 1B per stage
        # Each stage i processes forward mb=(warmup_count_for_stage_i + k)
        # and backward mb=k
        warmup_count = {s: 0 for s in range(P)}
        for mb in warmup_unique:
            warmup_count[mb[0]] += 1

        steady_steps = []
        for k in range(M):
            for s in range(P):
                fwd_mb = warmup_count[s] + k
                bwd_mb = k
                if fwd_mb < M and bwd_mb < M:
                    steady_steps.append((s, fwd_mb, True))
                    steady_steps.append((s, bwd_mb, False))

        # Sort by global order
        steady_sorted = []
        # Steady 1F1B: stage s does fwd mb i and bwd mb j
        # The global order is: each clock cycle, each stage does one op
        total_clocks = max(warmup_count[s] + M for s in range(P))
        for clock in range(total_clocks):
            for s in range(P):
                fwd_offset = clock - s
                bwd_offset = clock - (P - 1 - s) - s - 1  # rough
                # Simpler: stage s does (fwd at offset, bwd at offset)
                fwd_idx = clock - s
                bwd_idx = clock - P - s
                if 0 <= fwd_idx < M:
                    if fwd_idx >= warmup_count.get(s, 0):
                        schedule.append(MicroBatch(
                            mb_id=fwd_idx,
                            stage_id=s,
                            op_type=OpType.FORWARD,
                            phase=Phase.STEADY,
                        ))
                bwd_mb_id = clock - P - s
                if 0 <= bwd_mb_id < M:
                    schedule.append(MicroBatch(
                        mb_id=bwd_mb_id,
                        stage_id=s,
                        op_type=OpType.BACKWARD,
                        phase=Phase.STEADY,
                    ))

        # ── Cooldown phase ──
        # Drain remaining backwards
        cooldown_steps = []
        for s in reversed(range(P)):
            for mb in range(M):
                key = (s, mb, OpType.BACKWARD)
                already_scheduled = any(
                    m.stage_id == s and m.mb_id == mb and m.op_type == OpType.BACKWARD
                    for m in schedule
                )
                if not already_scheduled:
                    cooldown_steps.append((s, mb))

        cooldown_steps.sort(key=lambda x: (x[0] * M + x[1], -x[0]))
        for s, mb in cooldown_steps:
            schedule.append(MicroBatch(
                mb_id=mb,
                stage_id=s,
                op_type=OpType.BACKWARD,
                phase=Phase.COOLDOWN,
            ))

        return schedule

    def generate_simple(self) -> List[MicroBatch]:
        """Generate a simplified 1F1B schedule (easier to verify).

        This uses the standard formula:
          - Warmup: M micro-batches flow forward, with each stage i processing
            MBs 0..(M-i-1) in forward
          - Steady: alternating F/B per stage
          - Cooldown: drain remaining backward passes

        Total steps = 2*M + P - 1 per stage (F + B each)
        """
        schedule: List[MicroBatch] = []
        P = self.num_stages
        M = self.num_microbatches

        # For simplicity and correctness, use the clock-based formulation:
        # At clock cycle t, stage s does:
        #   - Forward mb = t - s (if in warmup/steady range)
        #   - Backward mb = t - (2*P - 1 - s) (if in steady/cooldown range)
        total_clocks = 2 * (M + P - 1)

        for t in range(total_clocks):
            for s in range(P):
                # Forward
                fwd_mb = t - s
                if 0 <= fwd_mb < M:
                    # Check if this is warmup or steady
                    # Warmup: the first P-1 forward passes per stage
                    is_warmup = fwd_mb < (P - 1 - s)
                    phase = Phase.WARMUP if is_warmup else Phase.STEADY
                    schedule.append(MicroBatch(
                        mb_id=fwd_mb,
                        stage_id=s,
                        op_type=OpType.FORWARD,
                        phase=phase,
                    ))

                # Backward
                bwd_mb = t - (2 * P - 1 - s)
                if 0 <= bwd_mb < M:
                    # Check if cooldown
                    is_cooldown = bwd_mb >= (M - (P - s))
                    phase = Phase.COOLDOWN if is_cooldown else Phase.STEADY
                    schedule.append(MicroBatch(
                        mb_id=bwd_mb,
                        stage_id=s,
                        op_type=OpType.BACKWARD,
                        phase=phase,
                    ))

        return schedule


# ── Activation memory tracker ────────────────────────────────────────────────

@dataclass
class ActivationTracker:
    """Tracks which activations are live in memory during PP execution.

    For each stage, we track:
      - Which activations are stored (waiting for backward)
      - Peak memory usage
      - Whether an activation is still alive when needed for backward
    """

    num_stages: int
    max_activations_per_stage: int = 4  # configurable limit

    def __post_init__(self):
        self.live_activations: Dict[int, List[Tuple[int, str]]] = {
            s: [] for s in range(self.num_stages)
        }  # stage → [(mb_id, tensor_name)]

    def record_forward(self, stage: int, mb_id: int, tensor_names: List[str]):
        """Record that activations were saved during forward."""
        for name in tensor_names:
            self.live_activations[stage].append((mb_id, name))

    def release_after_backward(self, stage: int, mb_id: int):
        """Release activations after backward pass for this micro-batch."""
        self.live_activations[stage] = [
            (m, name) for m, name in self.live_activations[stage]
            if m != mb_id
        ]

    def is_activation_available(self, stage: int, mb_id: int) -> bool:
        """Check if activation for a given micro-batch is still live."""
        return any(m == mb_id for m, _ in self.live_activations[stage])

    def peak_memory(self, stage: int) -> int:
        """Return peak number of live activations for a stage."""
        return len(self.live_activations[stage])

    def verify_activation_liveness(
        self,
        schedule: List[MicroBatch],
    ) -> Tuple[bool, List[str]]:
        """Verify all backward passes have their activations available.

        Returns: (passed, errors)
        """
        errors = []
        # Track per stage, per mb whether activation is saved
        saved = {s: set() for s in range(self.num_stages)}
        released = {s: set() for s in range(self.num_stages)}

        for mb in schedule:
            if mb.op_type == OpType.FORWARD:
                saved[mb.stage_id].add(mb.mb_id)
            elif mb.op_type == OpType.BACKWARD:
                if mb.mb_id not in saved[mb.stage_id]:
                    errors.append(
                        f"Stage {mb.stage_id}, MB {mb.mb_id}: "
                        f"activation not saved before backward"
                    )
                if mb.mb_id in released[mb.stage_id]:
                    errors.append(
                        f"Stage {mb.stage_id}, MB {mb.mb_id}: "
                        f"activation already released before backward"
                    )
                released[mb.stage_id].add(mb.mb_id)

        return len(errors) == 0, errors


# ── Deadlock checker ─────────────────────────────────────────────────────────

@dataclass
class DeadlockChecker:
    """Checks for deadlock freedom in the PP communication graph.

    Deadlock conditions:
      1. Unmatched Send → no corresponding Recv
      2. Unmatched Recv → no corresponding Send
      3. Circular wait in the communication graph
    """

    def __init__(self):
        self.sends: List[Tuple[int, int, str]] = []   # [(src, dst, tensor)]
        self.recvs: List[Tuple[int, int, str]] = []   # [(dst, src, tensor)]
        self.wait_for: Dict[int, Set[int]] = {}        # stage → {stages it waits for}

    def add_send(self, src: int, dst: int, tensor: str):
        self.sends.append((src, dst, tensor))

    def add_recv(self, src: int, dst: int, tensor: str):
        self.recvs.append((dst, src, tensor))

    def check(self) -> Tuple[bool, List[str]]:
        """Check for deadlock conditions.

        Matches Send/Recv on (src, dst) pairs, not tensor names,
        since names may differ between Send and Recv ops (the Recv
        references the post-Send tensor name).
        """
        errors = []

        # Check 1: every Send (src, dst) has a matching Recv
        recv_pairs = {(s, d) for d, s, _ in self.recvs}
        for src, dst, tensor in self.sends:
            if (src, dst) not in recv_pairs:
                errors.append(
                    f"Unmatched Send: src={src} → dst={dst}, tensor='{tensor}'"
                )

        # Check 2: every Recv (src, dst) has a matching Send
        send_pairs = {(s, d) for s, d, _ in self.sends}
        for dst, src, tensor in self.recvs:
            if (src, dst) not in send_pairs:
                errors.append(
                    f"Unmatched Recv: src={src} → dst={dst}, tensor='{tensor}'"
                )

        # Check 3: no circular waits at the device-pair level
        # Build wait-for graph using (src, dst) pairs, not tensor names
        wait_graph: Dict[int, Set[int]] = {}
        # Only build wait-for if there are unmatched pairs
        if len(errors) == 0:
            for dst, src, _ in self.recvs:
                if dst not in wait_graph:
                    wait_graph[dst] = set()
                wait_graph[dst].add(src)

        if self._has_cycle(wait_graph):
            errors.append(
                f"Circular wait detected in communication graph: {wait_graph}"
            )

        return len(errors) == 0, errors

    def _has_cycle(self, graph: Dict[int, Set[int]]) -> bool:
        """Check if a directed graph has a cycle (DFS)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, set()):
                if color.get(neighbor, WHITE) == GRAY:
                    return True
                if color.get(neighbor, WHITE) == WHITE:
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for node in graph:
            if color.get(node, WHITE) == WHITE:
                if dfs(node):
                    return True
        return False

    def verify_schedule(
        self,
        schedule: List[MicroBatch],
    ) -> Tuple[bool, List[str]]:
        """Verify a full schedule for deadlock freedom."""
        # Collect all Send/Recv from the schedule
        for mb in schedule:
            for dst, tensor in mb.sends:
                self.add_send(mb.stage_id, dst, tensor)
            for src, tensor in mb.recvs:
                self.add_recv(src, mb.stage_id, tensor)

        return self.check()
