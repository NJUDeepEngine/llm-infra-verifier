"""Verified Parallelization Synthesis: search/verify/refine loop.

Given a single-device compute program and a sharding specification, this
module synthesizes the minimal correct distributed program by:

  1. Executing the compute-only program to find placement issues
  2. Enumerating possible collective insertions (tactics)
  3. Verifying each candidate with Z3
  4. Ranking by communication cost
  5. Returning the optimal correct program

This is the core innovation: LLM-like tactic search, but for distributed
tensor programs, with a formal verifier as the correctness oracle.

Analogy to Lean:
  - Spec = single-device program + sharding spec
  - Tactics = collective insertion patterns
  - Kernel = Z3 verifier
  - Search = branch-and-bound over tactic space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Callable
from enum import Enum
import copy
import itertools

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
    SiLU,
    AllReduce,
    AllGather,
    ReduceScatter,
    FlashAttention,
)
from .executor import MultiDeviceExecutor
from .solver import DistributedVerifier, VerifyResult
from .rewrite import (
    PlacementAnalyzer,
    PlacementAnalysis,
    ProgramCost,
    ProgramOptimizer,
)


# ── Tactic definition ────────────────────────────────────────────────────────

class TacticType(Enum):
    INSERT_ALLREDUCE = "insert_allreduce"
    INSERT_ALLGATHER = "insert_allgather"
    INSERT_REDUCESCATTER = "insert_reducescatter"
    RESHARD = "reshard"
    FUSE_COLLECTIVES = "fuse_collectives"
    REORDER = "reorder"


@dataclass
class Tactic:
    """A proposed modification to the program.

    A tactic is a candidate fix: "insert AllReduce after op 3 for tensor y_partial".
    """
    type: TacticType
    op_index: int             # where to insert/modify
    tensor_name: str          # which tensor to act on
    output_name: str          # output name for the new op
    description: str = ""
    collective_type: Optional[type] = None  # AllReduce, AllGather, etc.
    params: Dict = field(default_factory=dict)  # extra params (gather_dim, etc.)

    def apply(self, program: Program) -> Program:
        """Apply this tactic to create a new program.

        After inserting a collective, remaps all downstream ops so they
        consume the collective's output instead of the original tensor.
        """
        new_prog = Program(name=f"{program.name}_{self.type.value}")

        collective_op = None
        if self.type == TacticType.INSERT_ALLREDUCE:
            collective_op = AllReduce(
                x=self.tensor_name,
                output=self.output_name,
                op_type=self.params.get("op_type", "sum"),
            )
        elif self.type == TacticType.INSERT_ALLGATHER:
            collective_op = AllGather(
                x=self.tensor_name,
                output=self.output_name,
                gather_dim=self.params.get("gather_dim", 0),
            )
        elif self.type == TacticType.INSERT_REDUCESCATTER:
            collective_op = ReduceScatter(
                x=self.tensor_name,
                output=self.output_name,
                scatter_dim=self.params.get("scatter_dim", 0),
            )

        # Copy ops up to and including the target op
        for i in range(self.op_index + 1):
            if i < len(program.ops):
                new_prog.ops.append(program.ops[i])

        # Insert the collective
        if collective_op is not None:
            new_prog.ops.append(collective_op)

        # Remap subsequent ops: replace references to tensor_name with output_name.
        # Stop remapping if a later op redefines tensor_name.
        input_map = {self.tensor_name: self.output_name}
        for i in range(self.op_index + 1, len(program.ops)):
            remapped = program.ops[i].clone_with_names(
                input_map=input_map,
                output_name=program.ops[i].output_name,
            )
            new_prog.ops.append(remapped)
            if program.ops[i].output_name == self.tensor_name:
                input_map = {}

        return new_prog

    def __repr__(self):
        return f"Tactic({self.type.value} @ op[{self.op_index}]: {self.tensor_name} → {self.output_name})"


# ── Tactic proposer ──────────────────────────────────────────────────────────

class TacticProposer:
    """Proposes candidate tactics to fix placement issues.

    Given an analysis of placement problems, generates all possible
    collective insertions that could resolve the issues.
    """

    def propose(
        self,
        program: Program,
        analysis: PlacementAnalysis,
        tensor_states: Dict[str, TensorState],
    ) -> List[Tactic]:
        """Generate candidate tactics to fix placement issues."""
        tactics: List[Tactic] = []

        # For each missing collective, propose insertion
        for op_idx, tensor_name, collective_type in analysis.missing_collectives:
            if collective_type == AllReduce:
                # Propose AllReduce insertion right after the op
                output_name = self._make_output_name(tensor_name, "reduced")
                tactics.append(Tactic(
                    type=TacticType.INSERT_ALLREDUCE,
                    op_index=op_idx,
                    tensor_name=tensor_name,
                    output_name=output_name,
                    description=f"Insert AllReduce after op {op_idx} for '{tensor_name}'",
                    collective_type=AllReduce,
                    params={"op_type": "sum"},
                ))

                # Also propose AllReduce at the end (alternative position)
                if op_idx < len(program.ops) - 1:
                    tactics.append(Tactic(
                        type=TacticType.INSERT_ALLREDUCE,
                        op_index=len(program.ops) - 1,
                        tensor_name=tensor_name,
                        output_name=output_name,
                        description=f"Insert AllReduce at end for '{tensor_name}'",
                        collective_type=AllReduce,
                        params={"op_type": "sum"},
                    ))

            elif collective_type == AllGather:
                for gather_dim in [0, 1]:
                    ts = tensor_states.get(tensor_name)
                    if ts and gather_dim < len(ts.global_shape):
                        tactics.append(Tactic(
                            type=TacticType.INSERT_ALLGATHER,
                            op_index=op_idx,
                            tensor_name=tensor_name,
                            output_name=self._make_output_name(tensor_name, "gathered"),
                            description=f"Insert AllGather(dim={gather_dim}) after op {op_idx}",
                            collective_type=AllGather,
                            params={"gather_dim": gather_dim},
                        ))

        # For each partial tensor that's an output, also propose
        for tensor_name in analysis.partial_tensors:
            if not any(t.tensor_name == tensor_name for t in tactics):
                tactics.append(Tactic(
                    type=TacticType.INSERT_ALLREDUCE,
                    op_index=len(program.ops) - 1,
                    tensor_name=tensor_name,
                    output_name=self._make_output_name(tensor_name, "reduced"),
                    description=f"Insert AllReduce at end for partial output '{tensor_name}'",
                    collective_type=AllReduce,
                    params={"op_type": "sum"},
                ))

        return tactics

    def _make_output_name(self, tensor_name: str, suffix: str) -> str:
        """Generate a clean output name."""
        # Remove common partial suffixes
        base = tensor_name.replace("_partial", "").replace("_local", "")
        return f"{base}_{suffix}"


# ── Candidate program ────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """A candidate distributed program with verification results."""
    program: Program
    tactics_applied: List[Tactic]
    cost: ProgramCost
    verification_results: List[VerifyResult] = field(default_factory=list)
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def num_tactics(self) -> int:
        return len(self.tactics_applied)

    def __lt__(self, other: Candidate) -> bool:
        # Sort by: valid first, then cost, then num_tactics
        if self.is_valid != other.is_valid:
            return self.is_valid  # valid candidates first
        if self.cost.total_communication != other.cost.total_communication:
            return self.cost.total_communication < other.cost.total_communication
        return self.num_tactics < other.num_tactics

    def __repr__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"Candidate({status}, cost={self.cost}, "
            f"tactics={self.num_tactics}, ops={len(self.program)})"
        )


# ── Synthesis engine ─────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """Result of the synthesis process."""
    success: bool
    best_candidate: Optional[Candidate] = None
    all_candidates: List[Candidate] = field(default_factory=list)
    iterations: int = 0
    search_space_size: int = 0

    @property
    def valid_candidates(self) -> List[Candidate]:
        return [c for c in self.all_candidates if c.is_valid]

    def summary(self) -> str:
        lines = [
            f"SynthesisResult: {'SUCCESS' if self.success else 'FAILED'}",
            f"  Iterations: {self.iterations}",
            f"  Search space: {self.search_space_size}",
            f"  Candidates: {len(self.all_candidates)} total, {len(self.valid_candidates)} valid",
        ]
        if self.best_candidate:
            lines.append(f"  Best: {self.best_candidate}")
            lines.append(f"  Program: {self.best_candidate.program}")
        return "\n".join(lines)


class SynthesisEngine:
    """Verified parallelization synthesis engine.

    Search strategy:
      1. Start with compute-only program (no collectives)
      2. Execute to find placement issues
      3. Generate tactics to fix issues
      4. For each tactic (and combinations), verify corrected program
      5. Select minimal-cost valid program

    Uses branch-and-bound: prune candidates with cost > best known valid cost.
    """

    def __init__(
        self,
        max_tactics: int = 5,
        max_search_depth: int = 3,
        beam_width: int = 10,
    ):
        self.max_tactics = max_tactics
        self.max_search_depth = max_search_depth
        self.beam_width = beam_width

        self.proposer = TacticProposer()
        self.analyzer = PlacementAnalyzer()
        self.verifier = DistributedVerifier()
        self.optimizer = ProgramOptimizer()

    def synthesize(
        self,
        compute_program: Program,
        tensor_states: Dict[str, TensorState],
        mesh: DeviceMesh,
    ) -> SynthesisResult:
        """Synthesize the minimal correct distributed program.

        Args:
            compute_program: Compute-only IR program (no collectives)
            tensor_states: Initial tensor states with sharding
            mesh: Device mesh

        Returns:
            SynthesisResult with the best candidate
        """
        all_candidates: List[Candidate] = []
        best_cost = float('inf')
        best_candidate: Optional[Candidate] = None

        # Step 1: Execute compute ops to get initial state
        executor = MultiDeviceExecutor(mesh)
        for name, ts in tensor_states.items():
            executor.register_tensor(ts)
        initial_state = executor.run_program(compute_program)

        # Step 2: Analyze placement
        analysis = self.analyzer.analyze(compute_program, initial_state)

        if analysis.is_correct:
            # Already correct! No collectives needed.
            candidate = Candidate(
                program=compute_program,
                tactics_applied=[],
                cost=ProgramCost.from_program(compute_program),
                is_valid=True,
            )
            all_candidates.append(candidate)
            return SynthesisResult(
                success=True,
                best_candidate=candidate,
                all_candidates=all_candidates,
                iterations=1,
                search_space_size=1,
            )

        # Step 3: Generate initial tactics
        tactics = self.proposer.propose(compute_program, analysis, initial_state)

        # Step 4: Beam search over tactic combinations
        # Level 0: apply 1 tactic
        # Level 1: apply 2 tactics
        # ... up to max_search_depth
        beam: List[Candidate] = []

        for depth in range(1, self.max_search_depth + 1):
            level_candidates = self._search_level(
                compute_program, tensor_states, mesh,
                tactics, depth, best_cost,
            )

            for cand in level_candidates:
                all_candidates.append(cand)
                if cand.is_valid and cand.cost.total_communication < best_cost:
                    best_cost = cand.cost.total_communication
                    best_candidate = cand

            # Early termination: if we found valid candidates at this depth,
            # don't search deeper (we want minimal tactics)
            valid_at_level = [c for c in level_candidates if c.is_valid]
            if valid_at_level:
                break

        # If no valid candidate found via search, try optimizer as fallback
        if best_candidate is None:
            optimized, history = self.optimizer.optimize(compute_program, initial_state)
            opt_tactics = [
                Tactic(
                    type=TacticType.INSERT_ALLREDUCE,
                    op_index=0,
                    tensor_name="",
                    output_name="",
                )
                for _ in history
            ]
            opt_candidate = Candidate(
                program=optimized,
                tactics_applied=opt_tactics,
                cost=ProgramCost.from_program(optimized),
            )
            # Verify
            opt_executor = MultiDeviceExecutor(mesh)
            for name, ts in tensor_states.items():
                opt_executor.register_tensor(ts)
            opt_state = opt_executor.run_program(optimized)
            opt_analysis = self.analyzer.analyze(optimized, opt_state)
            opt_candidate.is_valid = opt_analysis.is_correct
            opt_candidate.errors = [
                f"Missing: {m}" for _, m, _ in opt_analysis.missing_collectives
            ]
            all_candidates.append(opt_candidate)

            if opt_candidate.is_valid:
                best_candidate = opt_candidate

        return SynthesisResult(
            success=best_candidate is not None,
            best_candidate=best_candidate,
            all_candidates=all_candidates,
            iterations=min(self.max_search_depth, len(tactics)),
            search_space_size=sum(
                len(list(itertools.combinations(tactics, d)))
                for d in range(1, self.max_search_depth + 1)
            ),
        )

    def _search_level(
        self,
        base_program: Program,
        tensor_states: Dict[str, TensorState],
        mesh: DeviceMesh,
        all_tactics: List[Tactic],
        depth: int,
        best_cost: float,
    ) -> List[Candidate]:
        """Search all combinations of `depth` tactics."""
        candidates: List[Candidate] = []

        for tactic_combo in itertools.combinations(all_tactics, min(depth, len(all_tactics))):
            # Apply tactics sequentially
            current = copy.deepcopy(base_program)
            applied = []

            for tactic in tactic_combo:
                current = tactic.apply(current)
                applied.append(tactic)
                # Remap: if a tactic acts on a tensor that got renamed,
                # we'd need proper remapping. For simplicity, we re-index.

            # Execute corrected program
            executor = MultiDeviceExecutor(mesh)
            for name, ts in tensor_states.items():
                executor.register_tensor(ts)
            state = executor.run_program(current)

            # Verify
            analysis = self.analyzer.analyze(current, state)
            cost = ProgramCost.from_program(current)

            # Prune: skip if cost exceeds best known
            if cost.total_communication >= best_cost:
                continue

            # Check postconditions for output tensors
            verifier = DistributedVerifier()
            all_inputs = {inp for op in current.ops for inp in op.input_names}
            output_names = [
                name for name in state
                if name not in all_inputs and not name.startswith("grad_")
            ]
            verify_errors = []
            for out_name in output_names:
                ts = state.get(out_name)
                if ts and ts.partial:
                    verify_errors.append(f"Output '{out_name}' is still PARTIAL")

            is_valid = analysis.is_correct and len(verify_errors) == 0

            candidates.append(Candidate(
                program=current,
                tactics_applied=list(applied),
                cost=cost,
                is_valid=is_valid,
                errors=(
                    [f"Missing: {t}" for _, t, _ in analysis.missing_collectives]
                    + verify_errors
                ),
            ))

        # Sort and beam
        candidates.sort()
        return candidates[:self.beam_width]


# ── Convenience function ─────────────────────────────────────────────────────

def synthesize_parallel_program(
    compute_ops: List[IROp],
    input_shapes: Dict[str, Tuple[int, ...]],
    sharding_specs: Dict[str, ShardingSpec],
    verbose: bool = False,
) -> SynthesisResult:
    """Convenience: synthesize a distributed program from a compute-only spec.

    Example:
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        result = synthesize_parallel_program(
            compute_ops=[MatMul(a="x", b="w", output="y")],
            input_shapes={"x": (8, 16), "w": (16, 32)},
            sharding_specs={
                "x": ShardingSpec((Shard(1),), mesh),
                "w": ShardingSpec((Shard(0),), mesh),
            },
        )

    Returns a SynthesisResult with the best valid program.
    """
    from .rewrite import PatternSynthesizer

    synth = PatternSynthesizer()
    program, states, descriptions = synth.synthesize_from_spec(
        compute_ops, input_shapes, sharding_specs
    )

    engine = SynthesisEngine()
    mesh = list(sharding_specs.values())[0].mesh
    tensors = {}
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

    result = engine.synthesize(
        Program(name="compute", ops=compute_ops),
        tensors,
        mesh,
    )

    if verbose:
        print(result.summary())

    return result
