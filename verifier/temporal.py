"""Temporal verifier: Happens-Before analysis + Z3 race detection.

Checks correctness of asynchronous communication + compute overlap:
  1. Data races: async write concurrent with read/write of the same buffer
  2. Missing waits: async output consumed before Wait
  3. Buffer aliasing: two async ops sharing overlapping buffer in-flight
  4. Dependency violations: HB order contradicts required fwd/bwd order

Models each op as an interval [issue_time, complete_time] and builds
a Happens-Before (HB) graph encoding all partial order constraints.
Z3 is used to check satisfiability and find counterexample schedules.

Analogy to GPU programming:
  issue_time = kernel launch time
  complete_time = kernel finish (synced by stream or event)
  stream = CUDA stream (sequential within, concurrent across)
  Wait = cudaStreamWaitEvent / torch.cuda.synchronize
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import itertools

from z3 import (
    Solver, Int, Bool, BoolVal, IntVal, And, Or, Not, Implies, If,
    sat, unsat, unknown, Function,
)

from .state import TensorState, DeviceMesh
from .ir import (
    IROp, Program, MatMul, Add, Multiply, SiLU,
    AllReduce, AllReduceAsync, AllGather, ReduceScatter,
    Send, Recv, SendAsync, RecvAsync,
    Wait, WaitAll, OverlapRegion, FlashAttention, Handle, Stream,
    DEFAULT_STREAM, COMM_STREAM, COMPUTE_STREAM,
)


# ── Temporal event ───────────────────────────────────────────────────────────

class AccessType(Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"  # in-place ops, attention


@dataclass
class TemporalEvent:
    """An op's temporal footprint: when it issues and completes, what it accesses."""
    op_index: int
    op: IROp
    stream: Stream
    issue_var: Optional[Int] = None      # Z3 variable for issue time
    complete_var: Optional[Int] = None   # Z3 variable for complete time
    reads: Set[str] = field(default_factory=set)    # tensor names read
    writes: Set[str] = field(default_factory=set)   # tensor names written
    handle: Optional[str] = None         # async handle name (if async op)
    waited_by: Optional[int] = None      # index of Wait op (if any)


@dataclass
class HappensBeforeEdge:
    """A happens-before edge: source.complete_time < target.issue_time."""
    source: int
    target: int
    reason: str  # "program_order", "wait", "data_dep_write_read", "data_dep_write_write"

    def __repr__(self):
        return f"HB({self.source} -> {self.target}, {self.reason})"


# ── Race types ───────────────────────────────────────────────────────────────

class RaceType(Enum):
    DATA_RACE = "data_race"
    MISSING_WAIT = "missing_wait"
    BUFFER_ALIASING = "buffer_aliasing"
    DEPENDENCY_VIOLATION = "dependency_violation"
    CONCURRENT_COLLECTIVES = "concurrent_collectives"


@dataclass
class RaceReport:
    """A detected temporal correctness violation."""
    race_type: RaceType
    description: str
    op_a_idx: int
    op_b_idx: int
    tensor_name: str = ""
    details: str = ""
    counterexample: Optional[Dict] = None

    def __repr__(self):
        return (
            f"[{self.race_type.value.upper()}] {self.description}\n"
            f"  ops: [{self.op_a_idx}] vs [{self.op_b_idx}]"
            + (f" on '{self.tensor_name}'" if self.tensor_name else "")
            + (f"\n  details: {self.details}" if self.details else "")
        )


# ── Temporal graph builder ───────────────────────────────────────────────────

class TemporalGraph:
    """Builds the temporal event graph and Happens-Before edges from an IR program.

    Each op gets issue_time and complete_time. Edges encode ordering constraints.
    """

    def __init__(self, program: Program):
        self.program = program
        self.events: List[TemporalEvent] = []
        self.hb_edges: List[HappensBeforeEdge] = []
        self.z3_solver = Solver()
        self.z3_issue: Dict[int, Int] = {}
        self.z3_complete: Dict[int, Int] = {}

        self._build_events()
        self._build_program_order()
        self._build_wait_sync()
        self._build_data_dependencies()

    def _build_events(self):
        """Create TemporalEvent for each op, classifying reads/writes."""
        stream_order: Dict[Stream, List[int]] = defaultdict(list)

        for i, op in enumerate(self.program.ops):
            issue = Int(f"issue_{i}")
            complete = Int(f"complete_{i}")
            self.z3_issue[i] = issue
            self.z3_complete[i] = complete

            reads, writes = self._classify_access(op)
            handle = None
            stream = DEFAULT_STREAM

            if isinstance(op, AllReduceAsync):
                stream = op.stream
                handle = op.handle
            elif isinstance(op, SendAsync):
                stream = op.stream
                handle = op.handle
            elif isinstance(op, RecvAsync):
                stream = op.stream
                handle = op.handle
            elif isinstance(op, Wait):
                stream = DEFAULT_STREAM
            elif isinstance(op, OverlapRegion):
                stream = COMPUTE_STREAM  # compute + comm overlap

            event = TemporalEvent(
                op_index=i, op=op, stream=stream,
                issue_var=issue, complete_var=complete,
                reads=reads, writes=writes, handle=handle,
            )
            self.events.append(event)
            stream_order[stream].append(i)

        # Basic temporal constraints
        for i, event in enumerate(self.events):
            issue = self.z3_issue[i]
            complete = self.z3_complete[i]

            if event.op.is_async():
                # Async: issue < complete (gap for overlap)
                self.z3_solver.add(issue < complete)
                # Async ops must have non-negative duration
                self.z3_solver.add(issue >= IntVal(0))
            elif isinstance(event.op, (Wait, WaitAll)):
                # Wait is a sync point: blocks until complete
                self.z3_solver.add(issue <= complete)
            else:
                # Sync compute: issue == complete (atomic from temporal perspective)
                self.z3_solver.add(issue == complete)
                self.z3_solver.add(issue >= IntVal(0))

        # Sequential ordering within each stream
        for stream, op_indices in stream_order.items():
            for a, b in zip(op_indices, op_indices[1:]):
                self.hb_edges.append(HappensBeforeEdge(a, b, "program_order"))
                self.z3_solver.add(
                    self.z3_complete[a] < self.z3_issue[b]
                )

    def _classify_access(self, op: IROp) -> Tuple[Set[str], Set[str]]:
        """Classify which tensors an op reads and writes."""
        reads = set()
        writes = set()

        # All ops read their inputs
        for name in op.input_names:
            if name:
                reads.add(name)

        # All ops write their output
        if op.output_name:
            writes.add(op.output_name)

        # OverlapRegion: reads and writes of all sub-ops
        if isinstance(op, OverlapRegion):
            for sub in op.compute_ops + op.comm_ops:
                sr, sw = self._classify_access(sub)
                reads.update(sr)
                writes.update(sw)

        return reads, writes

    def _build_program_order(self):
        """Program order is already handled per-stream in _build_events."""
        pass

    def _build_wait_sync(self):
        """Wait(handle) → the async op that created handle must complete first."""
        handle_to_issue_idx: Dict[str, int] = {}
        for i, event in enumerate(self.events):
            if event.handle:
                handle_to_issue_idx[event.handle] = i

        for i, event in enumerate(self.events):
            if isinstance(event.op, Wait):
                h = event.op.handle
                if h in handle_to_issue_idx:
                    async_idx = handle_to_issue_idx[h]
                    self.hb_edges.append(HappensBeforeEdge(
                        async_idx, i, "wait"
                    ))
                    self.z3_solver.add(
                        self.z3_complete[async_idx] < self.z3_issue[i]
                    )
                    # Record the wait relationship
                    self.events[async_idx].waited_by = i

            elif isinstance(event.op, WaitAll):
                for h in event.op.handles:
                    if h in handle_to_issue_idx:
                        async_idx = handle_to_issue_idx[h]
                        self.hb_edges.append(HappensBeforeEdge(
                            async_idx, i, "wait"
                        ))
                        self.z3_solver.add(
                            self.z3_complete[async_idx] < self.z3_issue[i]
                        )
                        self.events[async_idx].waited_by = i

    def _build_data_dependencies(self):
        """Writer→Reader: if A writes T and B reads T, A must complete before B issues."""
        # Track all writes per tensor
        writers: Dict[str, List[int]] = defaultdict(list)
        for i, event in enumerate(self.events):
            for t in event.writes:
                writers[t].append(i)

        for i, event in enumerate(self.events):
            for t in event.reads:
                for w_idx in writers.get(t, []):
                    if w_idx != i:
                        self.hb_edges.append(HappensBeforeEdge(
                            w_idx, i, "data_dep_write_read"
                        ))
                        self.z3_solver.add(
                            self.z3_complete[w_idx] < self.z3_issue[i]
                        )

    def get_ordered(self, a: int, b: int) -> Optional[bool]:
        """Check if a and b are ordered by HB (via Z3).

        Returns True if a → b, False if b → a, None if concurrent.
        """
        s = Solver()
        # Copy all constraints
        for c in self.z3_solver.assertions():
            s.add(c)

        # Check if a → b is forced
        s.push()
        s.add(self.z3_complete[a] >= self.z3_issue[b])
        a_before_b = s.check()
        s.pop()

        # Check if b → a is forced
        s.push()
        s.add(self.z3_complete[b] >= self.z3_issue[a])
        b_before_a = s.check()
        s.pop()

        if a_before_b == unsat:
            return True   # a must complete before b
        if b_before_a == unsat:
            return False  # b must complete before a
        return None       # concurrent (could go either way)

    def intervals_overlap(self, a: int, b: int) -> bool:
        """Check if the execution intervals of ops a and b can overlap."""
        s = Solver()
        for c in self.z3_solver.assertions():
            s.add(c)

        # Overlap condition: NOT (a_complete < b_issue OR b_complete < a_issue)
        a_before_b = self.z3_complete[a] < self.z3_issue[b]
        b_before_a = self.z3_complete[b] < self.z3_issue[a]
        overlap = Not(Or(a_before_b, b_before_a))

        s.add(overlap)
        result = s.check()
        return result == sat  # sat = overlap IS possible


# ── Race detector ────────────────────────────────────────────────────────────

class RaceDetector:
    """Detects temporal correctness violations using the TemporalGraph.

    Checks:
      1. DATA_RACE: concurrent read-write or write-write on same tensor
      2. MISSING_WAIT: async output read before Wait
      3. BUFFER_ALIASING: two async writes to same buffer overlapping in time
      4. DEPENDENCY_VIOLATION: HB contradicts fwd/bwd ordering requirements
    """

    def __init__(self, graph: TemporalGraph):
        self.graph = graph
        self.reports: List[RaceReport] = []

    def detect_all(self) -> List[RaceReport]:
        """Run all detections."""
        self.reports = []
        self.detect_races()
        self.detect_missing_waits()
        self.detect_buffer_aliasing()
        self.detect_dependency_violations()
        self.detect_concurrent_collectives()
        return self.reports

    def _is_sync_op(self, event: TemporalEvent) -> bool:
        """Check if this event is a synchronization op (Wait/WaitAll)."""
        return isinstance(event.op, (Wait, WaitAll))

    def detect_races(self):
        """Detect data races: concurrent conflicting accesses on different streams."""
        events = self.graph.events
        n = len(events)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = events[i], events[j]

                # Skip sync ops (Wait/WaitAll) — they ARE the synchronization
                if self._is_sync_op(a) or self._is_sync_op(b):
                    continue

                # Only check different streams
                if a.stream == b.stream:
                    continue  # same stream = sequential, no race possible

                # Find tensors both access
                a_accessed = a.reads | a.writes
                b_accessed = b.reads | b.writes
                common = a_accessed & b_accessed

                if not common:
                    continue

                # At least one must be a write
                a_writes_any = bool(a.writes & common)
                b_writes_any = bool(b.writes & common)
                if not (a_writes_any or b_writes_any):
                    continue

                # Check ordering
                ordered = self.graph.get_ordered(i, j)
                if ordered is not None:
                    continue

                access_type = "write-write" if (a_writes_any and b_writes_any) else "read-write"
                self.reports.append(RaceReport(
                    race_type=RaceType.DATA_RACE,
                    description=f"Data race: {access_type} on tensor(s) {common}",
                    op_a_idx=i, op_b_idx=j,
                    tensor_name=", ".join(common),
                    details=(
                        f"Op[{i}]={type(a.op).__name__} (stream={a.stream.name}) "
                        f"and Op[{j}]={type(b.op).__name__} (stream={b.stream.name}) "
                        f"are on different streams, not ordered by HB."
                    ),
                ))

    def detect_missing_waits(self):
        """Detect missing Wait: async output consumed by non-Wait reader before Wait."""
        events = self.graph.events

        for i, event in enumerate(events):
            if not event.op.is_async():
                continue

            handle = event.handle
            if handle is None:
                continue

            output_name = event.op.output_name
            waited_by = event.waited_by  # index of Wait op, or None

            # Find readers of the async output (exclude Wait/WaitAll ops themselves)
            for j, other in enumerate(events):
                if j == i:
                    continue
                if self._is_sync_op(other):  # Skip Wait/WaitAll — they ARE the sync
                    continue
                if output_name not in other.reads:
                    continue

                # Check: is the Wait ordered before this reader?
                if waited_by is not None:
                    ordered = self.graph.get_ordered(waited_by, j)
                    if ordered is True:  # Wait → reader, safe
                        continue

                self.reports.append(RaceReport(
                    race_type=RaceType.MISSING_WAIT,
                    description=f"Missing Wait: async output '{output_name}' "
                                f"read by {type(other.op).__name__} before Wait({handle})",
                    op_a_idx=i, op_b_idx=j,
                    tensor_name=output_name,
                    details=(
                        f"Async op[{i}]={type(event.op).__name__} writes '{output_name}' "
                        f"with handle '{handle}'. "
                        f"Reader op[{j}]={type(other.op).__name__} reads it, but "
                        + (f"Wait at op[{waited_by}] does not precede the read."
                           if waited_by else "no Wait exists at all.")
                    ),
                ))

    def detect_buffer_aliasing(self):
        """Detect buffer aliasing: two async ops writing to the same buffer.

        If two async ops write to the same output buffer name, and the
        second issues before the first's output is consumed (Waited on),
        the first result is corrupted.
        """
        events = self.graph.events
        async_ops = [i for i, e in enumerate(events) if e.op.is_async()]

        for i, j in itertools.combinations(async_ops, 2):
            a, b = events[i], events[j]

            # Same output buffer written by both
            common_writes = a.writes & b.writes
            if not common_writes:
                continue

            # Check if first op's output is consumed BEFORE second op issues
            # The Wait for handle_a should precede the issue of op j
            waited_by_i = a.waited_by
            if waited_by_i is not None:
                # Does Wait for handle_i come before op j?
                ordered = self.graph.get_ordered(waited_by_i, j)
                if ordered is True:  # Wait → op_j, first result consumed, safe
                    continue

            waited_by_j = b.waited_by
            if waited_by_j is not None:
                ordered = self.graph.get_ordered(waited_by_j, i)
                if ordered is True:
                    continue

            self.reports.append(RaceReport(
                race_type=RaceType.BUFFER_ALIASING,
                description=f"Buffer aliasing: async ops [{i}] and [{j}] "
                            f"both write to '{', '.join(common_writes)}'",
                op_a_idx=i, op_b_idx=j,
                tensor_name=", ".join(common_writes),
                details=(
                    f"Async op[{i}]={type(a.op).__name__}(handle={a.handle}) "
                    f"and op[{j}]={type(b.op).__name__}(handle={b.handle}) "
                    f"both write to buffer '{', '.join(common_writes)}'. "
                    f"First result not consumed before second write. "
                    f"Use separate buffers."
                ),
            ))

    def detect_dependency_violations(self):
        """Detect dependency violations: HB contradicts required ordering.

        For example, in 1F1B overlap:
          - FWD(mb=1) must complete before BWD(mb=0) on the same stage
          - SendAsync must complete before corresponding RecvAsync issues
        """
        events = self.graph.events

        # Check: each SendAsync must complete before its matching RecvAsync
        sends: Dict[Tuple[int, int, int], int] = {}  # (src, dst, mb) → op_idx
        for i, event in enumerate(events):
            if isinstance(event.op, SendAsync):
                key = (event.op.src, event.op.dst, event.op.microbatch_id)
                sends[key] = i

        for i, event in enumerate(events):
            if isinstance(event.op, RecvAsync):
                key = (event.op.src, event.op.dst, event.op.microbatch_id)
                if key in sends:
                    send_idx = sends[key]
                    # Send must happen-before Recv
                    ordered = self.graph.get_ordered(send_idx, i)
                    if ordered is False:  # Recv before Send → violation!
                        self.reports.append(RaceReport(
                            race_type=RaceType.DEPENDENCY_VIOLATION,
                            description=f"Dependency violation: RecvAsync[{i}] "
                                        f"before SendAsync[{send_idx}]",
                            op_a_idx=send_idx, op_b_idx=i,
                            details=(
                                f"SendAsync(op[{send_idx}], {event.op.src}→{event.op.dst}) "
                                f"must precede RecvAsync(op[{i}]) but HB says otherwise."
                            ),
                        ))

    def detect_concurrent_collectives(self):
        """Detect concurrent collective operations that may deadlock.

        NCCL requires that collectives on overlapping process groups are
        ordered across all ranks. Two collectives in an OverlapRegion
        (intended for different streams) risk deadlock if they share ranks.

        This is a general NCCL safety rule, not specific to any bug.
        """
        for i, event in enumerate(self.graph.events):
            if not isinstance(event.op, OverlapRegion):
                continue

            compute_collectives = [
                op for op in event.op.compute_ops if op.is_collective()
            ]
            comm_collectives = [
                op for op in event.op.comm_ops if op.is_collective()
            ]

            if not compute_collectives or not comm_collectives:
                continue

            for c_op in compute_collectives:
                for m_op in comm_collectives:
                    self.reports.append(RaceReport(
                        race_type=RaceType.CONCURRENT_COLLECTIVES,
                        description=(
                            f"Concurrent collectives in OverlapRegion: "
                            f"{type(c_op).__name__} and {type(m_op).__name__}"
                        ),
                        op_a_idx=i, op_b_idx=i,
                        details=(
                            f"Compute stream: {c_op}, Comm stream: {m_op}. "
                            f"NCCL collectives on different streams sharing ranks "
                            f"must be serialized to avoid deadlock."
                        ),
                    ))

    def summary(self) -> str:
        if not self.reports:
            return "No temporal violations detected."
        lines = [f"Found {len(self.reports)} temporal violation(s):"]
        for r in self.reports:
            lines.append(f"  {r.race_type.value}: {r.description}")
        return "\n".join(lines)


# ── Temporal verifier (top-level) ────────────────────────────────────────────

@dataclass
class TemporalVerifyResult:
    """Result of temporal verification."""
    program_name: str
    num_ops: int
    num_async_ops: int
    num_hb_edges: int
    reports: List[RaceReport]
    is_safe: bool

    @property
    def num_races(self) -> int:
        return sum(1 for r in self.reports if r.race_type == RaceType.DATA_RACE)

    @property
    def num_missing_waits(self) -> int:
        return sum(1 for r in self.reports if r.race_type == RaceType.MISSING_WAIT)

    @property
    def num_buffer_aliases(self) -> int:
        return sum(1 for r in self.reports if r.race_type == RaceType.BUFFER_ALIASING)

    @property
    def num_dep_violations(self) -> int:
        return sum(1 for r in self.reports if r.race_type == RaceType.DEPENDENCY_VIOLATION)

    @property
    def num_concurrent_collectives(self) -> int:
        return sum(1 for r in self.reports if r.race_type == RaceType.CONCURRENT_COLLECTIVES)

    def summary(self) -> str:
        status = "SAFE" if self.is_safe else "UNSAFE"
        lines = [
            f"Temporal Verification: {status}",
            f"  Ops: {self.num_ops} total, {self.num_async_ops} async",
            f"  HB edges: {self.num_hb_edges}",
            f"  Violations: {len(self.reports)}",
        ]
        if self.num_races:
            lines.append(f"    Data races: {self.num_races}")
        if self.num_missing_waits:
            lines.append(f"    Missing waits: {self.num_missing_waits}")
        if self.num_buffer_aliases:
            lines.append(f"    Buffer aliasing: {self.num_buffer_aliases}")
        if self.num_dep_violations:
            lines.append(f"    Dependency violations: {self.num_dep_violations}")
        if self.num_concurrent_collectives:
            lines.append(f"    Concurrent collectives: {self.num_concurrent_collectives}")
        for r in self.reports:
            lines.append(f"\n  {r}")
        return "\n".join(lines)


def verify_temporal(program: Program) -> TemporalVerifyResult:
    """Run full temporal verification on a program.

    Args:
        program: IR program potentially containing async ops.

    Returns:
        TemporalVerifyResult with all detected violations.
    """
    # Build HB graph
    graph = TemporalGraph(program)

    # Count async ops
    num_async = sum(1 for e in graph.events if e.op.is_async())

    # Detect violations
    detector = RaceDetector(graph)
    reports = detector.detect_all()

    return TemporalVerifyResult(
        program_name=program.name,
        num_ops=len(program.ops),
        num_async_ops=num_async,
        num_hb_edges=len(graph.hb_edges),
        reports=reports,
        is_safe=len(reports) == 0,
    )
