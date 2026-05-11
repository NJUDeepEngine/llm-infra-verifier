"""DTensor rewrite system for program equivalence and optimization.

Implements pattern-based rewrite rules that:
  1. Recognize known distributed patterns (Row Parallel, Column Parallel, etc.)
  2. Detect missing/redundant collectives
  3. Generate minimal correct programs via rule application
  4. Compute cost models for comparing alternative parallelization strategies

The rewrite system operates on IR programs and preserves global semantics
(i.e., the final output is equivalent to single-device execution).

Key rewrite rules (each preserves equivalence):
  R1: MatMul(X:Shard(r), W:Shard(r)) without AllReduce → insert AllReduce
      (Row Parallel — both sharded on reduce dim requires reduction)
  R2: MatMul(X:Rep, W:Shard(c)) → no fwd collective needed
      (Column Parallel — output naturally sharded)
  R3: AllReduce @ AllReduce → AllReduce
      (fusion — consecutive AllReduces on same tensor can be merged)
  R4: AllReduce after MatMul where neither input is sharded on reduce dim
      → remove AllReduce (redundant — output is already replicated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import copy

from .state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    Placement,
    compute_local_shape,
)
from .ir import (
    IROp,
    Program,
    MatMul,
    Add,
    Multiply,
    AllReduce,
    AllGather,
    ReduceScatter,
    Send,
    Recv,
)


# ── Rewrite rule result ──────────────────────────────────────────────────────

class RuleStatus(Enum):
    APPLIED = "applied"
    NOT_MATCHED = "not_matched"
    ERROR = "error"


@dataclass
class RewriteResult:
    """Result of applying a rewrite rule."""
    status: RuleStatus
    rule_name: str
    original_program: Program
    rewritten_program: Optional[Program] = None
    description: str = ""
    cost_delta: int = 0

    @property
    def changed(self) -> bool:
        return self.status == RuleStatus.APPLIED


# ── Pattern definition ───────────────────────────────────────────────────────

@dataclass
class IRPattern:
    """A structural pattern to match in IR programs.

    Each pattern describes a sub-sequence of ops and their placement
    constraints. Used both for recognition (does this program contain
    this pattern?) and generation (insert ops to complete this pattern).
    """
    name: str
    description: str

    # Required ops (checked in order, may have gaps)
    required_op_types: Tuple[type, ...]

    # Placement conditions on inputs
    input_conditions: Dict[str, Callable[[TensorState], bool]] = field(default_factory=dict)

    # What the pattern produces (placement after completion)
    output_placement: Optional[Placement] = None


# ── Known distributed patterns ────────────────────────────────────────────────

ROW_PARALLEL_LINEAR = IRPattern(
    name="row_parallel_linear",
    description="Row Parallel Linear: both operands sharded on reduce dim → AllReduce needed",
    required_op_types=(MatMul, AllReduce),
    input_conditions={
        "a": lambda t: any(isinstance(p, Shard) for p in t.sharding.placements),
        "b": lambda t: any(isinstance(p, Shard) for p in t.sharding.placements),
    },
    output_placement=Replicate(),
)

COLUMN_PARALLEL_LINEAR = IRPattern(
    name="column_parallel_linear",
    description="Column Parallel Linear: W sharded on output dim → no fwd communication",
    required_op_types=(MatMul,),
    input_conditions={
        "b": lambda t: any(
            isinstance(p, Shard) and p.dim == 1 for p in t.sharding.placements
        ),
    },
    output_placement=Shard(dim=1),
)

ROW_PARALLEL_MISSING_AR = IRPattern(
    name="row_parallel_missing_allreduce",
    description="Row Parallel with missing AllReduce — output is still Partial",
    required_op_types=(MatMul,),
    input_conditions={
        "a": lambda t: any(isinstance(p, Shard) for p in t.sharding.placements),
        "b": lambda t: any(isinstance(p, Shard) for p in t.sharding.placements),
    },
    output_placement=Partial(),
)


# ── Placement analyzer ───────────────────────────────────────────────────────

@dataclass
class PlacementAnalysis:
    """Analysis of a program's placement: which tensors need collectives."""
    partial_tensors: List[str]         # tensors that are PARTIAL
    missing_collectives: List[Tuple[int, str, type]]  # (op_idx, tensor_name, needed_collective_type)
    redundant_collectives: List[int]   # indices of unnecessary collective ops
    collectives_ok: List[int]          # indices of correct collectives

    @property
    def is_correct(self) -> bool:
        return len(self.missing_collectives) == 0 and len(self.redundant_collectives) == 0

    def __repr__(self):
        lines = ["PlacementAnalysis:"]
        if self.partial_tensors:
            lines.append(f"  Partial tensors: {self.partial_tensors}")
        if self.missing_collectives:
            lines.append(f"  Missing collectives: {len(self.missing_collectives)}")
            for idx, name, ctype in self.missing_collectives:
                lines.append(f"    [{idx}] {name} needs {ctype.__name__}")
        if self.redundant_collectives:
            lines.append(f"  Redundant collectives: {self.redundant_collectives}")
        if self.is_correct:
            lines.append("  Status: CORRECT")
        else:
            lines.append("  Status: NEEDS FIX")
        return "\n".join(lines)


class PlacementAnalyzer:
    """Analyze a program to find placement issues.

    Given a program and tensor states after execution, identifies:
      - Tensors that are Partial at the end (need AllReduce)
      - Missing collectives (where a Partial output should have been reduced)
      - Redundant collectives (AllReduce on already-Replicated tensor)
    """

    def analyze(
        self,
        program: Program,
        tensor_states: Dict[str, TensorState],
    ) -> PlacementAnalysis:
        partial_tensors = []
        missing_collectives = []
        redundant_collectives = []
        collectives_ok = []

        for i, op in enumerate(program.ops):
            if isinstance(op, MatMul):
                # Check if matmul output is partial
                out_name = op.output
                if out_name in tensor_states:
                    out_ts = tensor_states[out_name]
                    if out_ts.partial:
                        partial_tensors.append(out_name)
                        # Check: is there an AllReduce after this matmul?
                        has_ar = self._has_subsequent_allreduce(program, i, out_name)
                        if not has_ar:
                            missing_collectives.append((i, out_name, AllReduce))
                    else:
                        # Check: is there an unnecessary AllReduce after this?
                        pass  # No Partial → no AllReduce needed

            elif isinstance(op, AllReduce):
                in_name = op.x
                if in_name in tensor_states:
                    in_ts = tensor_states[in_name]
                    if in_ts.partial:
                        collectives_ok.append(i)
                    else:
                        redundant_collectives.append(i)

        return PlacementAnalysis(
            partial_tensors=partial_tensors,
            missing_collectives=missing_collectives,
            redundant_collectives=redundant_collectives,
            collectives_ok=collectives_ok,
        )

    def _has_subsequent_allreduce(
        self, program: Program, from_idx: int, tensor_name: str
    ) -> bool:
        """Check if there's an AllReduce on tensor_name after from_idx."""
        for j in range(from_idx + 1, len(program.ops)):
            op = program.ops[j]
            if isinstance(op, AllReduce) and op.x == tensor_name:
                return True
        return False


# ── Rewrite rules ────────────────────────────────────────────────────────────

class RewriteRule:
    """A single rewrite rule: match → transform."""

    def __init__(self, name: str, description: str, cost_delta: int = 0):
        self.name = name
        self.description = description
        self.cost_delta = cost_delta

    def matches(self, program: Program, idx: int) -> bool:
        """Check if this rule matches at the given program index."""
        raise NotImplementedError

    def apply(self, program: Program, idx: int) -> Program:
        """Apply the rule, returning a new program."""
        raise NotImplementedError


class InsertAllReduceRule(RewriteRule):
    """Insert AllReduce after a MatMul that produces Partial output."""

    def __init__(self):
        super().__init__(
            name="insert_allreduce",
            description="Insert AllReduce after MatMul with Partial output",
            cost_delta=+1,  # adds communication
        )

    def matches(
        self,
        program: Program,
        idx: int,
        tensor_states: Optional[Dict[str, TensorState]] = None,
    ) -> bool:
        if idx >= len(program.ops):
            return False
        op = program.ops[idx]
        if not isinstance(op, MatMul):
            return False
        # Check if the output needs AllReduce
        if tensor_states and op.output in tensor_states:
            return tensor_states[op.output].partial
        return True  # Optimistic: assume it might need it

    def apply(
        self,
        program: Program,
        idx: int,
        tensor_states: Optional[Dict[str, TensorState]] = None,
    ) -> Program:
        op = program.ops[idx]
        assert isinstance(op, MatMul)

        new_program = Program(name=f"{program.name}_fixed")
        # Copy ops up to and including the matmul
        for i in range(idx + 1):
            new_program.ops.append(program.ops[i])

        # Insert AllReduce
        partial_name = op.output
        final_name = partial_name.replace("_partial", "")
        if final_name == partial_name:
            final_name = f"{partial_name}_reduced"

        ar_op = AllReduce(x=partial_name, output=final_name, op_type="sum")
        new_program.ops.append(ar_op)

        # Remap subsequent ops that reference the partial output
        for i in range(idx + 1, len(program.ops)):
            remapped = program.ops[i].clone_with_names(
                input_map={partial_name: final_name},
                output_name=program.ops[i].output_name,
            )
            new_program.ops.append(remapped)

        return new_program


class RemoveRedundantAllReduceRule(RewriteRule):
    """Remove AllReduce on already-replicated tensors."""

    def __init__(self):
        super().__init__(
            name="remove_redundant_allreduce",
            description="Remove AllReduce on non-Partial tensor",
            cost_delta=-1,  # removes unnecessary communication
        )

    def matches(
        self,
        program: Program,
        idx: int,
        tensor_states: Optional[Dict[str, TensorState]] = None,
    ) -> bool:
        if idx >= len(program.ops):
            return False
        op = program.ops[idx]
        if not isinstance(op, AllReduce):
            return False
        if tensor_states and op.x in tensor_states:
            return not tensor_states[op.x].partial
        return False

    def apply(
        self,
        program: Program,
        idx: int,
        tensor_states: Optional[Dict[str, TensorState]] = None,
    ) -> Program:
        op = program.ops[idx]
        new_program = Program(name=f"{program.name}_optimized")
        for i, p_op in enumerate(program.ops):
            if i == idx:
                continue
            # Remap references from AllReduce output to its input
            new_program.ops.append(
                p_op.clone_with_names(
                    input_map={op.output: op.x},
                    output_name=p_op.output_name,
                )
            )
        return new_program


# ── Cost model ───────────────────────────────────────────────────────────────

@dataclass
class ProgramCost:
    """Estimated communication cost of a distributed program."""
    num_allreduce: int = 0
    num_allgather: int = 0
    num_reducescatter: int = 0
    num_send_recv: int = 0
    total_communication: int = 0

    @classmethod
    def from_program(cls, program: Program) -> ProgramCost:
        cost = cls()
        for op in program.ops:
            if isinstance(op, AllReduce):
                cost.num_allreduce += 1
            elif isinstance(op, AllGather):
                cost.num_allgather += 1
            elif isinstance(op, ReduceScatter):
                cost.num_reducescatter += 1
            elif isinstance(op, (Send, Recv)):
                cost.num_send_recv += 1
        # Simple cost model: AllReduce = 2x, AllGather/ReduceScatter = 1x, P2P = 1x
        cost.total_communication = (
            2 * cost.num_allreduce
            + cost.num_allgather
            + cost.num_reducescatter
            + cost.num_send_recv
        )
        return cost

    def __lt__(self, other: ProgramCost) -> bool:
        return self.total_communication < other.total_communication

    def __repr__(self):
        return (
            f"Cost(AR={self.num_allreduce}, AG={self.num_allgather}, "
            f"RS={self.num_reducescatter}, P2P={self.num_send_recv}, "
            f"total={self.total_communication})"
        )


# ── Program optimizer ────────────────────────────────────────────────────────

class ProgramOptimizer:
    """Apply rewrite rules to optimize a distributed program.

    Uses a simple greedy approach:
      1. Analyze placement
      2. Fix missing collectives
      3. Remove redundant collectives
      4. Verify correctness
    """

    def __init__(self):
        self.analyzer = PlacementAnalyzer()
        self.rules: List[RewriteRule] = [
            InsertAllReduceRule(),
            RemoveRedundantAllReduceRule(),
        ]

    def optimize(
        self,
        program: Program,
        tensor_states: Dict[str, TensorState],
        max_iterations: int = 10,
    ) -> Tuple[Program, List[RewriteResult]]:
        """Optimize a program to be correct and minimal.

        Returns (optimized_program, history).
        """
        current = copy.deepcopy(program)
        history: List[RewriteResult] = []

        for iteration in range(max_iterations):
            analysis = self.analyzer.analyze(current, tensor_states)

            if analysis.is_correct:
                break

            changed = False

            # First: fix missing collectives
            for op_idx, tensor_name, collective_type in analysis.missing_collectives:
                rule = self._find_rule_for(collective_type)
                if rule and rule.matches(current, op_idx, tensor_states):
                    current = rule.apply(current, op_idx, tensor_states)
                    history.append(RewriteResult(
                        status=RuleStatus.APPLIED,
                        rule_name=rule.name,
                        original_program=program,
                        rewritten_program=current,
                        description=f"Inserted {collective_type.__name__} after op {op_idx} for tensor '{tensor_name}'",
                        cost_delta=rule.cost_delta,
                    ))
                    changed = True
                    break  # Re-analyze after each fix

            if changed:
                continue

            # Second: remove redundant collectives
            for op_idx in analysis.redundant_collectives:
                rule = RemoveRedundantAllReduceRule()
                if rule.matches(current, op_idx, tensor_states):
                    current = rule.apply(current, op_idx, tensor_states)
                    history.append(RewriteResult(
                        status=RuleStatus.APPLIED,
                        rule_name=rule.name,
                        original_program=program,
                        rewritten_program=current,
                        description=f"Removed redundant AllReduce at op {op_idx}",
                        cost_delta=rule.cost_delta,
                    ))
                    changed = True
                    break

            if not changed:
                break

        return current, history

    def _find_rule_for(self, collective_type: type) -> Optional[RewriteRule]:
        if collective_type == AllReduce:
            return InsertAllReduceRule()
        return None


# ── Pattern-based program synthesis ──────────────────────────────────────────

class PatternSynthesizer:
    """Synthesize collectives based on known patterns.

    Given a single-device program (compute ops only) and a sharding spec,
    determines what collectives are needed and where to insert them.

    This is the core of "Verified Parallelization Synthesis":
      Single-device program + Sharding spec → Distributed program
    """

    def synthesize(
        self,
        compute_program: Program,
        sharding_specs: Dict[str, ShardingSpec],
        tensor_states: Dict[str, TensorState],
    ) -> Tuple[Program, List[str]]:
        """Synthesize collectives for a compute-only program.

        Args:
            compute_program: Program with only compute ops (no collectives)
            sharding_specs: {tensor_name: ShardingSpec} for all tensors
            tensor_states: Tensor states after executing compute ops

        Returns:
            (full_program_with_collectives, list_of_inserted_collective_descriptions)
        """
        analysis = PlacementAnalyzer().analyze(compute_program, tensor_states)
        optimizer = ProgramOptimizer()

        result, history = optimizer.optimize(compute_program, tensor_states)

        descriptions = [h.description for h in history if h.changed]
        return result, descriptions

    def synthesize_from_spec(
        self,
        compute_ops: List[IROp],
        input_shapes: Dict[str, Tuple[int, ...]],
        sharding_specs: Dict[str, ShardingSpec],
    ) -> Tuple[Program, Dict[str, TensorState], List[str]]:
        """Full synthesis: create tensor states, execute, synthesize collectives.

        Args:
            compute_ops: List of compute-only IR ops (e.g., just MatMul)
            input_shapes: {name: shape} for input tensors
            sharding_specs: {name: ShardingSpec} for all tensors

        Returns:
            (synthesized_program, final_tensor_states, descriptions)
        """
        from .executor import MultiDeviceExecutor

        # Create tensor states
        tensors = {}
        mesh = list(sharding_specs.values())[0].mesh

        for name, shape in input_shapes.items():
            spec = sharding_specs.get(name)
            if spec is None:
                spec = ShardingSpec(
                    placements=(Replicate(),) * mesh.ndim,
                    mesh=mesh,
                )
            local = compute_local_shape(shape, spec)
            tensors[name] = TensorState(
                name=name,
                global_shape=shape,
                local_shape=local,
                sharding=spec,
                expr=name.lower(),
                requires_grad=True,
            )

        # Execute compute ops to get initial tensor states
        compute_program = Program(name="compute_only")
        for op in compute_ops:
            compute_program.add(op)

        executor = MultiDeviceExecutor(mesh)
        for name, t in tensors.items():
            executor.register_tensor(t)
        exec_result = executor.run_program(compute_program)

        # Synthesize collectives
        synthesized, descriptions = self.synthesize(
            compute_program, sharding_specs, exec_result
        )

        return synthesized, exec_result, descriptions
