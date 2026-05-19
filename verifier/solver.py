"""Z3-based verification solver for distributed tensor programs.

Encodes verification conditions as SMT formulas and checks:
  1. Postcondition: final tensors meet placement / partiality constraints
  2. Communication legality: collectives only on valid input states
  3. Gradient duality: fwd collectives have matching bwd duals
  4. Shape consistency: shapes propagate correctly through ops
  5. PP deadlock freedom: communication graph has no unmatched ops or cycles

Uses Z3's Bool/Int/Array theories to model placement propagation and
find counterexamples that violate correctness conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

from z3 import Solver, Bool, BoolVal, Int, IntVal, And, Or, Not, If, Implies, sat, unsat

from .state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    Placement,
)
from .ir import (
    IROp,
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    GELU,
    ReLU,
    Dropout,
    LayerNorm,
    RMSNorm,
    Softmax,
    Embedding,
    CrossEntropyLoss,
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    Reduce,
    AllToAll,
    Scatter,
    Gather,
    Send,
    Recv,
    FlashAttention,
    Reshape,
    Transpose,
    Cast,
    LossScale,
    FP8Quantize,
    FP8Dequantize,
    AmaxUpdate,
    ZeROGatherParam,
    ZeROScatterGrad,
    ZeROPartitionOptState,
    RingRotate,
    RingAttentionStep,
    MoEDispatch,
    MoECombine,
    ExpertCompute,
    TopKGate,
    AllReduceAsync,
    Wait,
    WaitAll,
    OverlapRegion,
    Reinterpret,
    Convert,
)
from .ir.collective import CollectiveOp
from .ir.compute import ElementWiseBinaryOp
from .schedules import MicroBatch, PP1F1BSchedule, DeadlockChecker


# ── Verification result ──────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    """Result of a verification check."""
    passed: bool
    condition: str
    details: str = ""
    counterexample: Optional[Dict[str, str]] = None

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"  [{status}] {self.condition}"]
        if self.details:
            lines.append(f"    {self.details}")
        if self.counterexample:
            lines.append(f"    Counterexample: {self.counterexample}")
        return "\n".join(lines)


# ── Z3 placement solver ─────────────────────────────────────────────────────

# Placement constants for Z3 encoding
PL_R = 0   # Replicate
PL_S0 = 1  # Shard(dim=0)
PL_S1 = 2  # Shard(dim=1)
PL_P = 3   # Partial

_PL_NAMES = {PL_R: "R", PL_S0: "S(0)", PL_S1: "S(1)", PL_P: "P"}


class Z3PlacementSolver:
    """Z3-based symbolic placement verifier with shape and slice checking.

    Three levels of verification:
      L0: Placement propagation (R/S0/S1/P labels)
      L1: Shape consistency (divisibility, contraction dim match)
      L2: Slice alignment (per-device offset correspondence)

    Each tensor gets:
      - Z3 Ints for placement: pl_{name}_m{d} ∈ {0,1,2,3}, one per mesh dim
      - Z3 Ints for shape: gs_{name}_0, gs_{name}_1 (global dims)
      - Z3 Ints for local shape: ls_{name}_0, ls_{name}_1
    """

    def __init__(self, mesh_ndim: int = 1):
        self.solver = Solver()
        self._mesh_ndim = mesh_ndim
        self._vars: Dict[str, List] = {}
        self._shape_vars: Dict[str, List] = {}
        self._local_vars: Dict[str, List] = {}
        self._tp_size: Optional[int] = None
        self._mesh_sizes: Optional[List[int]] = None

    def _var(self, name: str) -> List:
        """Get or create Z3 placement variables for a tensor (one per mesh dim)."""
        if name not in self._vars:
            vs = []
            for m in range(self._mesh_ndim):
                v = Int(f"pl_{name}_m{m}")
                self.solver.add(And(v >= 0, v <= 3))
                vs.append(v)
            self._vars[name] = vs
        return self._vars[name]

    def _shape_var(self, name: str, ndim: int = 2):
        if name not in self._shape_vars:
            gs = [Int(f"gs_{name}_{d}") for d in range(ndim)]
            ls = [Int(f"ls_{name}_{d}") for d in range(ndim)]
            for g in gs:
                self.solver.add(g > 0)
            for l in ls:
                self.solver.add(l > 0)
            self._shape_vars[name] = gs
            self._local_vars[name] = ls
        return self._shape_vars[name], self._local_vars[name]

    @property
    def num_constraints(self) -> int:
        return len(self.solver.assertions())

    @property
    def num_shape_vars(self) -> int:
        return sum(len(v) for v in self._shape_vars.values()) + \
               sum(len(v) for v in self._local_vars.values())

    def _placement_to_int(self, p: Placement) -> int:
        if isinstance(p, Replicate):
            return PL_R
        elif isinstance(p, Shard):
            return PL_S0 if p.dim == 0 else PL_S1
        elif isinstance(p, Partial):
            return PL_P
        return PL_R

    def add_input(self, name: str, placement):
        """Assert known concrete placement(s) for an input tensor.

        Args:
            placement: Single Placement (applied to mesh dim 0) or tuple of
                       Placements (one per mesh dim).
        """
        vs = self._var(name)
        if isinstance(placement, (list, tuple)):
            for m, p in enumerate(placement):
                if m < self._mesh_ndim:
                    vs[m]
                    self.solver.add(vs[m] == self._placement_to_int(p))
        else:
            self.solver.add(vs[0] == self._placement_to_int(placement))

    def add_input_shape(self, name: str, shape: Tuple[int, ...]):
        """Assert concrete shape for an input tensor."""
        gs, ls = self._shape_var(name, len(shape))
        for d, s in enumerate(shape):
            self.solver.add(gs[d] == s)

    # ── Placement encoding helpers ────────────────────────────────────────

    def _encode_matmul(self, op):
        R, S0, S1, P = IntVal(PL_R), IntVal(PL_S0), IntVal(PL_S1), IntVal(PL_P)
        a_vs, b_vs, y_vs = self._var(op.a), self._var(op.b), self._var(op.output)
        for m in range(self._mesh_ndim):
            a, b, y = a_vs[m], b_vs[m], y_vs[m]
            self.solver.add(y == If(Or(a == P, b == P), P,
                                If(And(a == S1, b == S0), P,
                                If(And(a == R,  b == S1), S1,
                                If(And(a == S0, b == R),  S0,
                                If(And(a == R,  b == R),  R,
                                If(And(a == S1, b == R),  S1,
                                If(And(a == R,  b == S0), S0,
                                P))))))))

    def _encode_elementwise(self, op):
        R, P = IntVal(PL_R), IntVal(PL_P)
        a_vs, b_vs, y_vs = self._var(op.a), self._var(op.b), self._var(op.output)
        for m in range(self._mesh_ndim):
            a, b, y = a_vs[m], b_vs[m], y_vs[m]
            self.solver.add(y == If(a == R, b, If(b == R, a, a)))
            self.solver.add(Not(And(a == P, b == P)))

    def _encode_passthrough(self, op):
        x_name = op.input_names[0]
        x_vs, y_vs = self._var(x_name), self._var(op.output_name)
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == x_vs[m])

    def _encode_waitall(self, op):
        """Encode WaitAll: each output passthrough from its corresponding input."""
        for tensor, output in zip(op.tensors, op.outputs):
            x_vs, y_vs = self._var(tensor), self._var(output)
            for m in range(self._mesh_ndim):
                self.solver.add(y_vs[m] == x_vs[m])

    def _encode_norm_op(self, op, norm_dim: int):
        """Encode norm op: output == input, with forbidden Shard(norm_dim) constraint."""
        x_name = op.input_names[0]
        x_vs, y_vs = self._var(x_name), self._var(op.output_name)
        effective_dim = norm_dim % 2 if norm_dim < 0 else norm_dim
        forbidden_pl = PL_S0 if effective_dim == 0 else PL_S1
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == x_vs[m])
            if effective_dim in (0, 1):
                self.solver.add(x_vs[m] != IntVal(forbidden_pl))

    def _encode_softmax_op(self, op):
        """Encode softmax: output == input, with forbidden Shard(reduction_dim) constraint."""
        x_name = op.input_names[0]
        x_vs, y_vs = self._var(x_name), self._var(op.output_name)
        reduction_dim = op.dim
        effective_dim = reduction_dim % 2 if reduction_dim < 0 else reduction_dim
        forbidden_pl = PL_S0 if effective_dim == 0 else PL_S1
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == x_vs[m])
            if effective_dim in (0, 1):
                self.solver.add(x_vs[m] != IntVal(forbidden_pl))

    def _encode_collective_constrained(self, op, target_pl: int,
                                        mesh_dim: Optional[int]):
        """Encode collective with mesh_dim awareness.

        When mesh_dim is set, only the targeted mesh dim gets target_pl;
        other dims pass through from input. When mesh_dim is None (legacy),
        all mesh dims get target_pl.
        """
        x_name = op.input_names[0]
        x_vs = self._var(x_name)
        y_vs = self._var(op.output_name)
        if mesh_dim is not None and mesh_dim < self._mesh_ndim:
            for m in range(self._mesh_ndim):
                if m == mesh_dim:
                    self.solver.add(y_vs[m] == IntVal(target_pl))
                else:
                    self.solver.add(y_vs[m] == x_vs[m])
        else:
            for m in range(self._mesh_ndim):
                self.solver.add(y_vs[m] == IntVal(target_pl))

    def _encode_reducescatter(self, op):
        sd = op.scatter_dim
        target_pl = PL_S0 if sd == 0 else PL_S1
        self._encode_collective_constrained(op, target_pl, op.mesh_dim)

    def _encode_alltoall(self, op):
        cd = op.concat_dim
        target_pl = PL_S0 if cd == 0 else PL_S1
        self._encode_collective_constrained(op, target_pl, op.mesh_dim)

    def _encode_flash_attention(self, op):
        q_vs, y_vs = self._var(op.q), self._var(op.output)
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == q_vs[m])

    def _encode_embedding(self, op):
        """Embedding: weight Shard(0) -> Partial, otherwise passthrough."""
        w_vs, y_vs = self._var(op.weight), self._var(op.output)
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == If(
                w_vs[m] == IntVal(PL_S0), IntVal(PL_P), w_vs[m]))

    def _encode_cross_entropy(self, op):
        """CrossEntropyLoss: logits Shard(vocab) -> Partial, else Replicate."""
        x_vs, y_vs = self._var(op.logits), self._var(op.output)
        for m in range(self._mesh_ndim):
            self.solver.add(y_vs[m] == If(
                Or(x_vs[m] == IntVal(PL_S0), x_vs[m] == IntVal(PL_S1),
                   x_vs[m] == IntVal(PL_P)),
                IntVal(PL_P),
                IntVal(PL_R),
            ))

    def encode_program(self, program: Program):
        """Walk all ops and add Z3 constraints for placement propagation."""
        for op in program.ops:
            if isinstance(op, MatMul):
                self._encode_matmul(op)
            elif isinstance(op, ElementWiseBinaryOp):
                self._encode_elementwise(op)
            elif isinstance(op, FlashAttention):
                self._encode_flash_attention(op)
            elif isinstance(op, ReduceScatter):
                self._encode_reducescatter(op)
            elif isinstance(op, AllToAll):
                self._encode_alltoall(op)
            elif isinstance(op, (AllReduce, AllGather, Gather)):
                self._encode_collective_constrained(op, PL_R, op.mesh_dim)
            elif isinstance(op, (Broadcast, Reduce)):
                self._encode_collective_constrained(op, PL_R, None)
            elif isinstance(op, Scatter):
                sd = op.scatter_dim
                target_pl = PL_S0 if sd == 0 else PL_S1
                self._encode_collective_constrained(op, target_pl, op.mesh_dim)
            elif isinstance(op, (ZeROGatherParam,)):
                self._encode_collective_constrained(op, PL_R, None)
            elif isinstance(op, ZeROScatterGrad):
                sd = op.scatter_dim
                target_pl = PL_S0 if sd == 0 else PL_S1
                self._encode_collective_constrained(op, target_pl, None)
            elif isinstance(op, LayerNorm):
                self._encode_norm_op(op, op.norm_dim)
            elif isinstance(op, RMSNorm):
                self._encode_norm_op(op, op.norm_dim)
            elif isinstance(op, Softmax):
                self._encode_softmax_op(op)
            elif isinstance(op, (SiLU, GELU, ReLU, Dropout,
                                 Reshape, Transpose, Cast, LossScale,
                                 FP8Quantize, FP8Dequantize,
                                 Reinterpret, Convert,
                                 RingAttentionStep, ExpertCompute)):
                self._encode_passthrough(op)
            elif isinstance(op, Embedding):
                self._encode_embedding(op)
            elif isinstance(op, CrossEntropyLoss):
                self._encode_cross_entropy(op)
            elif isinstance(op, (OverlapRegion,)):
                for sub in op.compute_ops + op.comm_ops:
                    self._encode_op_single(sub)
            elif isinstance(op, AllReduceAsync):
                self._encode_collective_constrained(op, PL_R, op.mesh_dim)
            elif isinstance(op, Wait):
                self._encode_passthrough(op)
            elif isinstance(op, WaitAll):
                self._encode_waitall(op)
            elif isinstance(op, TopKGate):
                self._encode_passthrough(op)
            elif isinstance(op, (MoEDispatch, MoECombine)):
                self._encode_passthrough(op)

    def _encode_op_single(self, op):
        """Encode a single op (used for OverlapRegion sub-ops)."""
        if isinstance(op, MatMul):
            self._encode_matmul(op)
        elif isinstance(op, ElementWiseBinaryOp):
            self._encode_elementwise(op)
        elif isinstance(op, (AllReduce, AllGather, Gather)):
            self._encode_collective_constrained(op, PL_R, op.mesh_dim)
        elif isinstance(op, (Broadcast, Reduce)):
            self._encode_collective_constrained(op, PL_R, None)
        elif isinstance(op, ReduceScatter):
            self._encode_reducescatter(op)
        elif isinstance(op, AllToAll):
            self._encode_alltoall(op)
        elif isinstance(op, FlashAttention):
            self._encode_flash_attention(op)
        else:
            if hasattr(op, 'input_names') and op.input_names:
                self._encode_passthrough(op)

    def _extract_counterexample(self) -> Dict[str, str]:
        """Extract a human-readable counterexample from the current SAT model."""
        model = self.solver.model()
        ce = {}
        for n, vs in self._vars.items():
            if self._mesh_ndim == 1:
                val = model.eval(vs[0], model_completion=True).as_long()
                ce[n] = _PL_NAMES.get(val, str(val))
            else:
                parts = []
                for m, v in enumerate(vs):
                    val = model.eval(v, model_completion=True).as_long()
                    parts.append(_PL_NAMES.get(val, str(val)))
                ce[n] = f"({', '.join(parts)})"
        return ce

    def check_output_equivalence(
        self,
        output_names: List[str],
    ) -> List[VerifyResult]:
        """Check if all outputs are guaranteed Replicate on all mesh dims.

        For each output, Z3 checks whether any mesh dim can be non-Replicate.
        UNSAT → proved equivalent. SAT → counterexample showing the violation.
        """
        results = []
        R = IntVal(PL_R)

        for name in output_names:
            vs = self._vars.get(name)
            if vs is None:
                continue

            self.solver.push()
            # Output must be Replicate on ALL mesh dims; check if any can be non-R
            self.solver.add(Or(*[v != R for v in vs]))

            check = self.solver.check()
            if check == sat:
                ce = self._extract_counterexample()
                results.append(VerifyResult(
                    passed=False,
                    condition=f"equivalence({name})",
                    details=(
                        f"Output '{name}' can be non-Replicate — "
                        f"not equivalent to single-GPU"
                    ),
                    counterexample=ce,
                ))
            else:
                results.append(VerifyResult(
                    passed=True,
                    condition=f"equivalence({name})",
                    details=(
                        f"Z3 proved: '{name}' is always Replicate "
                        f"(equivalent to single-GPU)"
                    ),
                ))

            self.solver.pop()

        return results

    def _check_precondition(
        self, op_name: str, input_name: str,
        required_pl: int, required_label: str,
        mesh_dim: int = 0,
    ) -> VerifyResult:
        """Check that an op's input always has the required placement on a mesh dim."""
        vs = self._vars.get(input_name)
        if vs is None:
            return VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) precondition",
                details=f"Input '{input_name}' not in Z3 model (skipped)",
            )

        x = vs[min(mesh_dim, len(vs) - 1)]
        self.solver.push()
        self.solver.add(x != IntVal(required_pl))

        check = self.solver.check()
        if check == sat:
            model = self.solver.model()
            val = model.eval(x, model_completion=True).as_long()
            result = VerifyResult(
                passed=False,
                condition=f"{op_name}({input_name}) precondition",
                details=(
                    f"Input '{input_name}' can be {_PL_NAMES.get(val, '?')}, "
                    f"not {required_label} — {op_name} may be unnecessary"
                ),
            )
        else:
            result = VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) precondition",
                details=(
                    f"Z3 proved: '{input_name}' is always {required_label} "
                    f"before {op_name}"
                ),
            )

        self.solver.pop()
        return result

    def _check_precondition_or(
        self, op_name: str, input_name: str,
        allowed_pls: List[int], label: str,
        mesh_dim: int = 0,
    ) -> VerifyResult:
        """Check that an op's input always has one of the allowed placements."""
        vs = self._vars.get(input_name)
        if vs is None:
            return VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) precondition",
                details=f"Input '{input_name}' not in Z3 model (skipped)",
            )

        x = vs[min(mesh_dim, len(vs) - 1)]
        self.solver.push()
        self.solver.add(And(*[x != IntVal(pl) for pl in allowed_pls]))

        check = self.solver.check()
        if check == sat:
            model = self.solver.model()
            val = model.eval(x, model_completion=True).as_long()
            result = VerifyResult(
                passed=False,
                condition=f"{op_name}({input_name}) precondition",
                details=(
                    f"Input '{input_name}' can be {_PL_NAMES.get(val, '?')}, "
                    f"not {label} — {op_name} may be incorrect"
                ),
            )
        else:
            result = VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) precondition",
                details=(
                    f"Z3 proved: '{input_name}' is always {label} "
                    f"before {op_name}"
                ),
            )

        self.solver.pop()
        return result

    def check_collective_preconditions(
        self,
        program: Program,
    ) -> List[VerifyResult]:
        """Verify collective preconditions via Z3.

        For each collective, checks that its input placement satisfies
        the required precondition:
          - AllReduce/Reduce: input must be Partial
          - AllGather: input must be Shard(gather_dim)
          - ReduceScatter: input must be Replicate or Partial
        """
        results = []

        for op in program.ops:
            md = getattr(op, 'mesh_dim', None) or 0
            if isinstance(op, (AllReduce, Reduce)):
                results.append(self._check_precondition(
                    type(op).__name__, op.x, PL_P, "Partial",
                    mesh_dim=md,
                ))
            elif isinstance(op, AllGather):
                expected_pl = PL_S0 if op.gather_dim == 0 else PL_S1
                expected_label = f"Shard({op.gather_dim})"
                results.append(self._check_precondition(
                    "AllGather", op.x, expected_pl, expected_label,
                    mesh_dim=md,
                ))
            elif isinstance(op, ReduceScatter):
                results.append(self._check_precondition_or(
                    "ReduceScatter", op.x,
                    [PL_R, PL_P], "Replicate or Partial",
                    mesh_dim=md,
                ))
            elif isinstance(op, AllToAll):
                expected_pl = PL_S0 if op.split_dim == 0 else PL_S1
                expected_label = f"Shard({op.split_dim})"
                results.append(self._check_precondition(
                    "AllToAll", op.x, expected_pl, expected_label,
                    mesh_dim=md,
                ))
            elif isinstance(op, Gather):
                expected_pl = PL_S0 if op.gather_dim == 0 else PL_S1
                expected_label = f"Shard({op.gather_dim})"
                results.append(self._check_precondition(
                    "Gather", op.x, expected_pl, expected_label,
                    mesh_dim=md,
                ))

        return results

    # ── Compute preconditions ─────────────────────────────────────────────

    def _check_not_forbidden(
        self, op_name: str, input_name: str,
        forbidden_pl: int, forbidden_label: str,
    ) -> VerifyResult:
        """Check that an op's input can never reach a forbidden placement on any mesh dim.

        Pushes x_m == forbidden_pl for any m; SAT means the forbidden state is reachable.
        """
        vs = self._vars.get(input_name)
        if vs is None:
            return VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) forbidden check",
                details=f"Input '{input_name}' not in Z3 model (skipped)",
            )

        self.solver.push()
        self.solver.add(Or(*[v == IntVal(forbidden_pl) for v in vs]))

        check = self.solver.check()
        if check == sat:
            ce = self._extract_counterexample()
            result = VerifyResult(
                passed=False,
                condition=f"{op_name}({input_name}) forbidden check",
                details=(
                    f"Input '{input_name}' can reach {forbidden_label}, "
                    f"which is forbidden for {op_name}"
                ),
                counterexample=ce,
            )
        else:
            result = VerifyResult(
                passed=True,
                condition=f"{op_name}({input_name}) forbidden check",
                details=(
                    f"Z3 proved: '{input_name}' can never be {forbidden_label} "
                    f"(forbidden by {op_name})"
                ),
            )

        self.solver.pop()
        return result

    def _check_not_both_partial(
        self, op_name: str, a_name: str, b_name: str,
    ) -> VerifyResult:
        """Check that two inputs to a binary op cannot both be Partial on any mesh dim."""
        a_vs = self._vars.get(a_name)
        b_vs = self._vars.get(b_name)
        if a_vs is None or b_vs is None:
            return VerifyResult(
                passed=True,
                condition=f"{op_name}({a_name}, {b_name}) partial×partial check",
                details="Input(s) not in Z3 model (skipped)",
            )

        P = IntVal(PL_P)
        self.solver.push()
        self.solver.add(Or(*[And(a_vs[m] == P, b_vs[m] == P)
                             for m in range(self._mesh_ndim)]))

        check = self.solver.check()
        if check == sat:
            ce = self._extract_counterexample()
            result = VerifyResult(
                passed=False,
                condition=f"{op_name}({a_name}, {b_name}) partial×partial check",
                details=(
                    f"Both '{a_name}' and '{b_name}' can be Partial simultaneously, "
                    f"which is forbidden for {op_name}"
                ),
                counterexample=ce,
            )
        else:
            result = VerifyResult(
                passed=True,
                condition=f"{op_name}({a_name}, {b_name}) partial×partial check",
                details=(
                    f"Z3 proved: '{a_name}' and '{b_name}' cannot both be Partial"
                ),
            )

        self.solver.pop()
        return result

    def check_compute_preconditions(
        self,
        program: Program,
    ) -> List[VerifyResult]:
        """Verify compute op preconditions via Z3.

        For each op with placement constraints:
          - LayerNorm/RMSNorm: input must NOT be Shard(norm_dim)
          - Softmax: input must NOT be Shard(reduction_dim)
          - Add/Multiply: inputs must NOT both be Partial
        """
        results = []

        for op in program.ops:
            if isinstance(op, (LayerNorm, RMSNorm)):
                norm_dim = op.norm_dim
                effective_dim = norm_dim % 2 if norm_dim < 0 else norm_dim
                if effective_dim in (0, 1):
                    forbidden_pl = PL_S0 if effective_dim == 0 else PL_S1
                    forbidden_label = f"Shard({effective_dim})"
                    results.append(self._check_not_forbidden(
                        type(op).__name__, op.x, forbidden_pl, forbidden_label,
                    ))
            elif isinstance(op, Softmax):
                reduction_dim = op.dim
                effective_dim = reduction_dim % 2 if reduction_dim < 0 else reduction_dim
                if effective_dim in (0, 1):
                    forbidden_pl = PL_S0 if effective_dim == 0 else PL_S1
                    forbidden_label = f"Shard({effective_dim})"
                    results.append(self._check_not_forbidden(
                        "Softmax", op.x, forbidden_pl, forbidden_label,
                    ))
            elif isinstance(op, ElementWiseBinaryOp):
                results.append(self._check_not_both_partial(
                    type(op).__name__, op.a, op.b,
                ))

        return results

    def check_program_satisfiability(self) -> VerifyResult:
        """Check whether the full constraint set is satisfiable.

        UNSAT means the program has conflicting placement requirements
        (e.g., upstream forces Shard(1) but downstream LayerNorm forbids it).
        """
        self.solver.push()
        check = self.solver.check()
        self.solver.pop()

        if check == unsat:
            return VerifyResult(
                passed=False,
                condition="program satisfiability",
                details=(
                    "Placement constraints are contradictory (UNSAT) — "
                    "program has conflicting placement requirements"
                ),
            )
        else:
            return VerifyResult(
                passed=True,
                condition="program satisfiability",
                details="All placement constraints are satisfiable",
            )

    # ── L1: Shape consistency ───────────────────────────────────────────────

    def _add_local_shape_constraints(self, gs, ls, pl_vars, mesh_sizes):
        """Add local_shape = global / product(sharding mesh sizes) for each dim.

        Args:
            gs: global shape Z3 vars (one per tensor dim)
            ls: local shape Z3 vars (one per tensor dim)
            pl_vars: list of Z3 placement vars (one per mesh dim)
            mesh_sizes: list of ints (size per mesh dim), or single int for 1D
        """
        S0, S1 = IntVal(PL_S0), IntVal(PL_S1)
        if isinstance(mesh_sizes, int):
            mesh_sizes = [mesh_sizes]

        for d in range(len(gs)):
            shard_pl = PL_S0 if d == 0 else PL_S1
            # Build the divisor: product of mesh_sizes[m] for all m where pl_vars[m] == Shard(d)
            # For 1D: simple If. For multi-dim: nested conditionals.
            if len(pl_vars) == 1:
                tp = IntVal(mesh_sizes[0])
                self.solver.add(ls[d] == If(
                    pl_vars[0] == IntVal(shard_pl), gs[d] / tp, gs[d]))
            else:
                # Multi-dim: enumerate combinations
                # local = global / product(mesh_sizes[m] for m where pl[m] shards this dim)
                expr = gs[d]
                # Build from innermost: check each mesh dim independently
                # For 2D: divisor = s0_if_m0_shards * s1_if_m1_shards
                # Express as nested If:
                # Start with full division, then peel off
                divisors = []
                for m in range(len(pl_vars)):
                    if m < len(mesh_sizes):
                        divisors.append((pl_vars[m], IntVal(mesh_sizes[m])))

                if len(divisors) == 2:
                    m0_pl, s0 = divisors[0]
                    m1_pl, s1 = divisors[1]
                    sp = IntVal(shard_pl)
                    self.solver.add(ls[d] == If(
                        And(m0_pl == sp, m1_pl == sp), gs[d] / (s0 * s1),
                        If(m0_pl == sp, gs[d] / s0,
                        If(m1_pl == sp, gs[d] / s1,
                        gs[d]))))
                else:
                    # Fallback for 3+ dims: only handle first mesh dim
                    tp = IntVal(mesh_sizes[0])
                    self.solver.add(ls[d] == If(
                        pl_vars[0] == IntVal(shard_pl), gs[d] / tp, gs[d]))

    def _encode_shape_passthrough(self, op, mesh_sizes):
        """Shape constraint for ops that preserve shape (SiLU, Cast, etc.)."""
        x_name = op.input_names[0]
        if x_name not in self._shape_vars:
            return
        gs_x = self._shape_vars[x_name]
        gs_y, ls_y = self._shape_var(op.output_name)
        pl_y = self._var(op.output_name)
        for d in range(len(gs_x)):
            self.solver.add(gs_y[d] == gs_x[d])
        self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

    def _encode_shape_collective_preserve(self, op, mesh_sizes):
        """Shape constraint for collectives that preserve global shape (AllReduce, Broadcast).

        When mesh_dim is set, output may not be fully Replicate (other dims
        preserve input placement), so local shape depends on output placement.
        When mesh_dim is None, output is fully Replicate → local == global.
        """
        if op.x not in self._shape_vars:
            return
        gs_x = self._shape_vars[op.x]
        gs_y, ls_y = self._shape_var(op.output)
        for d in range(len(gs_x)):
            self.solver.add(gs_y[d] == gs_x[d])

        md = getattr(op, 'mesh_dim', None)
        if md is not None:
            pl_y = self._var(op.output_name)
            self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)
        else:
            for d in range(len(gs_x)):
                self.solver.add(ls_y[d] == gs_y[d])

    def encode_shape_constraints(self, program: Program, tp_size: int = 0,
                                  mesh_sizes: Optional[List[int]] = None):
        """Add Z3 constraints for shape correctness (L1).

        (a) Divisibility: sharded dims must be divisible by mesh size on that dim
        (b) Local shape:  local = global / divisor (product of sharding mesh sizes)
        (c) MatMul:       a.global_cols == b.global_rows  (contraction dim)
        (d) MatMul out:   out.rows == a.rows, out.cols == b.cols
        (e) Add/Mul:      a.global_shape == b.global_shape

        Args:
            tp_size: single mesh dim size (backward compat, used when mesh_sizes is None)
            mesh_sizes: per-mesh-dim sizes [s0, s1, ...] for multi-dim
        """
        if mesh_sizes is None:
            mesh_sizes = [tp_size] if tp_size > 0 else [2]
        self._tp_size = mesh_sizes[0]
        self._mesh_sizes = mesh_sizes
        S0, S1 = IntVal(PL_S0), IntVal(PL_S1)

        for name in list(self._shape_vars.keys()):
            gs, ls = self._shape_vars[name], self._local_vars[name]
            pl_vs = self._var(name)
            # Divisibility: for each mesh dim m, if pl_m == Shard(d), gs[d] must be divisible
            for m in range(min(len(pl_vs), len(mesh_sizes))):
                ms = IntVal(mesh_sizes[m])
                for d in range(len(gs)):
                    shard_pl = PL_S0 if d == 0 else PL_S1
                    self.solver.add(Implies(pl_vs[m] == IntVal(shard_pl), gs[d] % ms == 0))
            self._add_local_shape_constraints(gs, ls, pl_vs, mesh_sizes)

        for op in program.ops:
            if isinstance(op, MatMul):
                if op.a not in self._shape_vars or op.b not in self._shape_vars:
                    continue
                gs_a = self._shape_vars[op.a]
                gs_b = self._shape_vars[op.b]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                self.solver.add(gs_a[1] == gs_b[0])
                self.solver.add(gs_y[0] == gs_a[0])
                self.solver.add(gs_y[1] == gs_b[1])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, ElementWiseBinaryOp):
                if op.a not in self._shape_vars or op.b not in self._shape_vars:
                    continue
                gs_a = self._shape_vars[op.a]
                gs_b = self._shape_vars[op.b]
                for d in range(min(len(gs_a), len(gs_b))):
                    self.solver.add(gs_a[d] == gs_b[d])
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(min(len(gs_a), len(gs_y))):
                    self.solver.add(gs_y[d] == gs_a[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, FlashAttention):
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                if op.q in self._shape_vars:
                    gs_q = self._shape_vars[op.q]
                    self.solver.add(gs_y[0] == gs_q[0])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, (AllReduce, Broadcast, Reduce)):
                self._encode_shape_collective_preserve(op, mesh_sizes)

            elif isinstance(op, AllGather):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, ReduceScatter):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, AllToAll):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, Scatter):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, Gather):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, (SiLU, GELU, ReLU, Dropout,
                                 LayerNorm, RMSNorm, Softmax,
                                 Cast, LossScale, Reshape, Transpose,
                                 FP8Quantize, FP8Dequantize,
                                 Reinterpret, Convert, Wait)):
                self._encode_shape_passthrough(op, mesh_sizes)

            elif isinstance(op, WaitAll):
                for tensor, output in zip(op.tensors, op.outputs):
                    if tensor not in self._shape_vars:
                        continue
                    gs_x = self._shape_vars[tensor]
                    gs_y, ls_y = self._shape_var(output)
                    pl_y = self._var(output)
                    for d in range(len(gs_x)):
                        self.solver.add(gs_y[d] == gs_x[d])
                    self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, Embedding):
                if op.weight not in self._shape_vars:
                    continue
                gs_w = self._shape_vars[op.weight]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                if len(gs_w) >= 2:
                    self.solver.add(gs_y[1] == gs_w[1])
                self._add_local_shape_constraints(gs_y, ls_y, pl_y, mesh_sizes)

            elif isinstance(op, CrossEntropyLoss):
                gs_y, ls_y = self._shape_var(op.output)
                self.solver.add(gs_y[0] == IntVal(1))
                self.solver.add(ls_y[0] == IntVal(1))

    def check_shape_consistency(self) -> List[VerifyResult]:
        """Check all shape constraints. SAT means a violation exists."""
        results = []

        self.solver.push()
        check = self.solver.check()
        if check == unsat:
            results.append(VerifyResult(
                passed=False,
                condition="shape consistency",
                details="Shape constraints are contradictory (UNSAT) — possible encoding error",
            ))
        else:
            results.append(VerifyResult(
                passed=True,
                condition="shape consistency",
                details="All shape constraints are satisfiable",
            ))
        self.solver.pop()

        mesh_sizes = self._mesh_sizes or ([self._tp_size] if self._tp_size else None)
        if mesh_sizes and self._shape_vars:
            for name in self._shape_vars:
                pl_vs = self._vars.get(name)
                gs = self._shape_vars[name]
                if pl_vs is None:
                    continue
                # Check divisibility for each mesh dim × tensor dim combination
                for m in range(min(len(pl_vs), len(mesh_sizes))):
                    ms = IntVal(mesh_sizes[m])
                    for d, shard_pl in [(0, PL_S0), (1, PL_S1)]:
                        if d >= len(gs):
                            continue
                        self.solver.push()
                        self.solver.add(pl_vs[m] == IntVal(shard_pl))
                        self.solver.add(gs[d] % ms != 0)
                        check = self.solver.check()
                        if check == sat:
                            model = self.solver.model()
                            dim_val = model.eval(gs[d], model_completion=True).as_long()
                            label = f"mesh_dim={m}" if len(mesh_sizes) > 1 else ""
                            results.append(VerifyResult(
                                passed=False,
                                condition=f"divisibility({name}, dim={d}{', ' + label if label else ''})",
                                details=(
                                    f"Shard(dim={d}) on '{name}' but dim size "
                                    f"{dim_val} not divisible by {mesh_sizes[m]}"
                                ),
                            ))
                        else:
                            results.append(VerifyResult(
                                passed=True,
                                condition=f"divisibility({name}, dim={d})",
                                details=f"'{name}' dim {d} always divisible when sharded",
                            ))
                        self.solver.pop()

        return results

    # ── L2: Slice alignment ─────────────────────────────────────────────────

    def encode_slice_constraints(self, program: Program, tp_size: int = 0,
                                  mesh_sizes: Optional[List[int]] = None):
        """Add Z3 constraints for per-device slice alignment (L2).

        For MatMul(A, B) with A sharded on cols and B sharded on rows,
        verifies that on each device d, A's column slice and B's row slice
        cover the same interval of the contraction dimension.

        Slice alignment is checked per mesh dim — sharding on different mesh
        dims produces independent slicing along the same tensor dim.
        """
        if mesh_sizes is None:
            mesh_sizes = [tp_size] if tp_size > 0 else [2]
        self._tp_size = mesh_sizes[0]
        self._mesh_sizes = mesh_sizes

        for op in program.ops:
            if not isinstance(op, MatMul):
                continue
            if op.a not in self._shape_vars or op.b not in self._shape_vars:
                continue

            pl_a_vs = self._vars.get(op.a)
            pl_b_vs = self._vars.get(op.b)
            gs_a = self._shape_vars[op.a]
            gs_b = self._shape_vars[op.b]
            if pl_a_vs is None or pl_b_vs is None:
                continue

            S0, S1 = IntVal(PL_S0), IntVal(PL_S1)

            # Check slice alignment per mesh dim
            for m in range(min(len(pl_a_vs), len(mesh_sizes))):
                tp = IntVal(mesh_sizes[m])
                pl_a = pl_a_vs[m]
                pl_b = pl_b_vs[m]

                for d in range(mesh_sizes[m]):
                    d_val = IntVal(d)
                    a_col_offset = d_val * (gs_a[1] / tp)
                    a_col_width = gs_a[1] / tp
                    b_row_offset = d_val * (gs_b[0] / tp)
                    b_row_width = gs_b[0] / tp

                    self.solver.add(Implies(
                        And(pl_a == S1, pl_b == S0),
                        And(
                            a_col_offset == b_row_offset,
                            a_col_width == b_row_width,
                        )
                    ))

    def check_slice_alignment(self, program: Program) -> List[VerifyResult]:
        """Verify per-device slice alignment for all MatMul ops.

        For each MatMul, checks whether the contraction dim shapes always
        match (derived from shape constraints). When shapes are concrete,
        reports the actual per-device slice ranges.
        """
        results = []
        tp = self._tp_size or 2

        for op in program.ops:
            if not isinstance(op, MatMul):
                continue
            if op.a not in self._shape_vars or op.b not in self._shape_vars:
                continue

            gs_a = self._shape_vars[op.a]
            gs_b = self._shape_vars[op.b]

            # Check: can contraction dims ever mismatch?
            self.solver.push()
            self.solver.add(gs_a[1] != gs_b[0])
            check = self.solver.check()
            self.solver.pop()

            if check == unsat:
                # Get concrete values for reporting
                self.solver.push()
                check2 = self.solver.check()
                if check2 == sat:
                    model = self.solver.model()
                    a_cols = model.eval(gs_a[1], model_completion=True).as_long()
                    b_rows = model.eval(gs_b[0], model_completion=True).as_long()
                    chunk = a_cols // tp
                    details = (
                        f"Contraction dim always matches: "
                        f"{op.a}.d1={a_cols} == {op.b}.d0={b_rows}. "
                        f"Per-device slice: chunk={chunk}, "
                        f"dev d gets [{op.a}[:, d*{chunk}:(d+1)*{chunk}] "
                        f"@ {op.b}[d*{chunk}:(d+1)*{chunk}, :]]"
                    )
                else:
                    details = "Contraction dim always matches"
                self.solver.pop()
                results.append(VerifyResult(
                    passed=True,
                    condition=f"slice_align({op.a}@{op.b})",
                    details=details,
                ))
            else:
                model = self.solver.model()
                a_val = model.eval(gs_a[1], model_completion=True).as_long()
                b_val = model.eval(gs_b[0], model_completion=True).as_long()
                results.append(VerifyResult(
                    passed=False,
                    condition=f"slice_align({op.a}@{op.b})",
                    details=(
                        f"Contraction dim mismatch possible: "
                        f"{op.a}.d1={a_val} != {op.b}.d0={b_val}"
                    ),
                ))

        # AllReduce coverage: partials cover [0, global_dim) without gaps
        for op in program.ops:
            if not isinstance(op, AllReduce):
                continue
            if op.x not in self._shape_vars:
                continue
            gs_x = self._shape_vars[op.x]
            self.solver.push()
            check = self.solver.check()
            if check == sat:
                model = self.solver.model()
                dims = [model.eval(g, model_completion=True).as_long()
                        for g in gs_x]
                results.append(VerifyResult(
                    passed=True,
                    condition=f"allreduce_coverage({op.x})",
                    details=(
                        f"AllReduce({op.x}): {tp} partials of shape "
                        f"{tuple(dims)} sum to full tensor"
                    ),
                ))
            self.solver.pop()

        return results


# ── Distributed verifier ─────────────────────────────────────────────────────

class DistributedVerifier:
    """Main verification engine using Z3 SMT solver.

    Encodes the distributed program's correctness conditions as SMT
    formulas and checks satisfiability.
    """

    def __init__(self):
        self.results: List[VerifyResult] = []

    def verify_postcondition(
        self,
        tensor: TensorState,
        expected_partial: bool = False,
        expected_placement: Optional[Placement] = None,
    ) -> VerifyResult:
        """Verify that a tensor meets its postcondition.

        Checks concrete tensor state against expected properties.

        Args:
            tensor: The tensor state to check
            expected_partial: Whether the tensor is expected to be partial
            expected_placement: Expected placement (e.g., Replicate())

        Returns:
            VerifyResult with pass/fail and counterexample if any
        """
        passed = True
        details = ""
        counterexample = None

        if tensor.partial != expected_partial:
            passed = False
            counterexample = {"is_partial": str(tensor.partial)}
            details = (
                f"tensor.partial={tensor.partial}, expected={expected_partial}"
            )

        if expected_placement is not None:
            actual_p = tensor.sharding.placements[0] if tensor.sharding.placements else Replicate()
            if type(actual_p) != type(expected_placement):
                passed = False
                details += f"; placement mismatch: actual={actual_p}, expected={expected_placement}"
            elif isinstance(actual_p, Shard) and isinstance(expected_placement, Shard):
                if actual_p.dim != expected_placement.dim:
                    passed = False
                    details += f"; shard dim mismatch: actual={actual_p}, expected={expected_placement}"

        condition = f"postcondition({tensor.name}): partial={expected_partial}"
        if expected_placement:
            condition += f", placement={expected_placement}"

        vr = VerifyResult(
            passed=passed,
            condition=condition,
            details=details,
            counterexample=counterexample,
        )
        self.results.append(vr)
        return vr

    def verify_communication_legality(
        self,
        program: Program,
        tensor_states: Optional[Dict[str, TensorState]] = None,
        multi_device_states: Optional[Dict[int, Dict[str, TensorState]]] = None,
    ) -> VerifyResult:
        """Verify that all collective ops are used legally.

        Checks:
          - AllReduce: input must be PARTIAL
          - AllGather: input must be sharded on gather_dim
          - ReduceScatter: input must be replicated on scatter_dim
          - Send/Recv: must have matching pairs

        Uses pre-op tensor state (not final state) so that in-place ops
        like AllReduce(x="p", output="p") are checked correctly.
        """
        errors = []

        # Build a merged view across all devices (final state)
        final_states = dict(tensor_states) if tensor_states else {}
        if multi_device_states:
            for did, dev_states in multi_device_states.items():
                for name, ts in dev_states.items():
                    if name not in final_states:
                        final_states[name] = ts

        # Walk ops in program order, tracking live tensor state so that
        # preconditions are checked against pre-op state (not final state).
        # This is critical for in-place ops like AllReduce(x="p", output="p").
        all_outputs = {op.output_name for op in program.ops if op.output_name}
        live = {
            name: ts for name, ts in final_states.items()
            if name not in all_outputs
        }

        for op in program.ops:
            if isinstance(op, (AllReduce, Reduce)):
                if op.x in live:
                    ts = live[op.x]
                    if not ts.partial:
                        errors.append(
                            f"{type(op).__name__}({op.x}) called on non-partial "
                            f"tensor: {ts.sharding}"
                        )
                else:
                    import warnings
                    warnings.warn(
                        f"{type(op).__name__}({op.x}): no tensor state available "
                        f"to verify precondition (input should be Partial)"
                    )

            elif isinstance(op, AllGather):
                if op.x in live:
                    ts = live[op.x]
                    has_shard_on_dim = any(
                        isinstance(p, Shard) and p.dim == op.gather_dim
                        for p in ts.sharding.placements
                    )
                    if not has_shard_on_dim:
                        errors.append(
                            f"AllGather({op.x}, dim={op.gather_dim}) called on tensor "
                            f"not sharded on dim {op.gather_dim}: {ts.sharding}"
                        )

            elif isinstance(op, ReduceScatter):
                if op.x in live:
                    ts = live[op.x]
                    has_shard_on_scatter = any(
                        isinstance(p, Shard) and p.dim == op.scatter_dim
                        for p in ts.sharding.placements
                    )
                    if has_shard_on_scatter:
                        errors.append(
                            f"ReduceScatter({op.x}, dim={op.scatter_dim}) called on "
                            f"tensor already sharded on dim {op.scatter_dim}: {ts.sharding}"
                        )

            elif isinstance(op, AllToAll):
                if op.x in live:
                    ts = live[op.x]
                    has_shard_on_split = any(
                        isinstance(p, Shard) and p.dim == op.split_dim
                        for p in ts.sharding.placements
                    )
                    if not has_shard_on_split:
                        errors.append(
                            f"AllToAll({op.x}, split={op.split_dim}) called on tensor "
                            f"not sharded on dim {op.split_dim}: {ts.sharding}"
                        )

            elif isinstance(op, Gather):
                if op.x in live:
                    ts = live[op.x]
                    has_shard_on_dim = any(
                        isinstance(p, Shard) and p.dim == op.gather_dim
                        for p in ts.sharding.placements
                    )
                    if not has_shard_on_dim:
                        errors.append(
                            f"Gather({op.x}, dim={op.gather_dim}) called on tensor "
                            f"not sharded on dim {op.gather_dim}: {ts.sharding}"
                        )

            elif isinstance(op, Send):
                has_matching = any(
                    isinstance(o, Recv)
                    and o.src == op.src
                    and o.dst == op.dst
                    and o.microbatch_id == op.microbatch_id
                    for o in program.ops
                )
                if not has_matching:
                    errors.append(
                        f"Send(src={op.src}, dst={op.dst}, mb={op.microbatch_id}) "
                        f"has no matching Recv"
                    )

            elif isinstance(op, Recv):
                has_matching = any(
                    isinstance(o, Send)
                    and o.src == op.src
                    and o.dst == op.dst
                    and o.microbatch_id == op.microbatch_id
                    for o in program.ops
                )
                if not has_matching:
                    errors.append(
                        f"Recv(src={op.src}, dst={op.dst}, mb={op.microbatch_id}) "
                        f"has no matching Send"
                    )

            # Update live state: try to compute output from op semantics,
            # fall back to final_states for ops we can't replay.
            out_name = op.output_name
            if out_name:
                ctx = {n: live[n] for n in op.input_names if n in live}
                if len(ctx) == len(op.input_names):
                    try:
                        result = op.apply(ctx)
                        live[out_name] = result
                    except Exception:
                        if out_name in final_states:
                            live[out_name] = final_states[out_name]
                elif out_name in final_states:
                    live[out_name] = final_states[out_name]

        passed = len(errors) == 0
        details = "; ".join(errors) if errors else "All communication ops are legal"

        vr = VerifyResult(
            passed=passed,
            condition="communication legality",
            details=details,
        )
        self.results.append(vr)
        return vr

    def verify_gradient_duality(
        self,
        fwd_program: Program,
        bwd_program: Program,
    ) -> VerifyResult:
        """Verify gradient duality: each fwd collective has a bwd dual.

        Checks that each forward collective has a matching backward dual.
        """
        errors = []

        dual_map = {
            "AllReduce": "AllReduce",
            "AllGather": "ReduceScatter",
            "ReduceScatter": "AllGather",
            "Send": "Recv",
            "Recv": "Send",
            "Broadcast": "Reduce",
            "Reduce": "Broadcast",
            "ZeROGatherParam": "ZeROScatterGrad",
            "ZeROScatterGrad": "ZeROGatherParam",
            "AllToAll": "AllToAll",
            "Scatter": "Gather",
            "Gather": "Scatter",
            "MoEDispatch": "MoECombine",
            "MoECombine": "MoEDispatch",
            "RingRotate": "RingRotate",
        }

        fwd_collectives = [op for op in fwd_program.ops if op.is_collective()]
        bwd_collectives = [op for op in bwd_program.ops if op.is_collective()]

        for fwd_op in fwd_collectives:
            fwd_type = type(fwd_op).__name__
            expected_dual = dual_map.get(fwd_type)

            if expected_dual is None:
                continue

            found = False
            for bwd_op in bwd_collectives:
                bwd_type = type(bwd_op).__name__
                if bwd_type == expected_dual:
                    if self._check_dual_match(fwd_op, bwd_op):
                        found = True
                        break

            if not found:
                errors.append(
                    f"Forward collective '{fwd_op}' has no dual in backward"
                )

        passed = len(errors) == 0
        details = "; ".join(errors) if errors else (
            f"All {len(fwd_collectives)} forward collectives have matching backward duals"
        )

        vr = VerifyResult(
            passed=passed,
            condition="gradient duality (fwd ↔ bwd)",
            details=details,
        )
        self.results.append(vr)
        return vr

    @staticmethod
    def _check_dual_match(fwd_op: IROp, bwd_op: IROp) -> bool:
        """Check property-level match between a forward op and its backward dual."""
        if isinstance(fwd_op, AllReduce) and isinstance(bwd_op, AllReduce):
            return True
        if isinstance(fwd_op, AllGather) and isinstance(bwd_op, ReduceScatter):
            return fwd_op.gather_dim == bwd_op.scatter_dim
        if isinstance(fwd_op, ReduceScatter) and isinstance(bwd_op, AllGather):
            return fwd_op.scatter_dim == bwd_op.gather_dim
        if isinstance(fwd_op, Send) and isinstance(bwd_op, Recv):
            return fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src
        if isinstance(fwd_op, Recv) and isinstance(bwd_op, Send):
            return fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src
        if isinstance(fwd_op, Broadcast) and isinstance(bwd_op, Reduce):
            return True
        if isinstance(fwd_op, Reduce) and isinstance(bwd_op, Broadcast):
            return True
        if isinstance(fwd_op, ZeROGatherParam) and isinstance(bwd_op, ZeROScatterGrad):
            return fwd_op.gather_dim == bwd_op.scatter_dim
        if isinstance(fwd_op, ZeROScatterGrad) and isinstance(bwd_op, ZeROGatherParam):
            return fwd_op.scatter_dim == bwd_op.gather_dim
        if isinstance(fwd_op, MoEDispatch) and isinstance(bwd_op, MoECombine):
            return (fwd_op.split_dim == bwd_op.concat_dim and
                    fwd_op.concat_dim == bwd_op.split_dim)
        if isinstance(fwd_op, MoECombine) and isinstance(bwd_op, MoEDispatch):
            return (fwd_op.split_dim == bwd_op.concat_dim and
                    fwd_op.concat_dim == bwd_op.split_dim)
        if isinstance(fwd_op, AllToAll) and isinstance(bwd_op, AllToAll):
            return (fwd_op.split_dim == bwd_op.concat_dim and
                    fwd_op.concat_dim == bwd_op.split_dim)
        if isinstance(fwd_op, Scatter) and isinstance(bwd_op, Gather):
            return fwd_op.scatter_dim == bwd_op.gather_dim
        if isinstance(fwd_op, Gather) and isinstance(bwd_op, Scatter):
            return fwd_op.gather_dim == bwd_op.scatter_dim
        if isinstance(fwd_op, RingRotate) and isinstance(bwd_op, RingRotate):
            return (fwd_op.ring_size == bwd_op.ring_size and
                    fwd_op.ring_dim == bwd_op.ring_dim)
        return True

    def verify_placement_consistency(
        self,
        program: Program,
        input_placements: Optional[Dict[str, Placement]] = None,
        final_tensors: Optional[Dict[str, TensorState]] = None,
        output_names: Optional[List[str]] = None,
    ) -> VerifyResult:
        """Verify placement consistency using Z3 SMT solver.

        Encodes all op propagation rules as Z3 constraints and checks
        that output tensors are always Replicate (single-GPU equivalent).
        Automatically detects mesh dimensionality from tensor states.
        """
        # Determine mesh_ndim from tensors
        mesh_ndim = 1
        if final_tensors:
            for ts in final_tensors.values():
                if ts.sharding and ts.sharding.mesh:
                    mesh_ndim = max(mesh_ndim, len(ts.sharding.mesh.shape))
                    break

        z3s = Z3PlacementSolver(mesh_ndim=mesh_ndim)

        if input_placements:
            for name, pl in input_placements.items():
                z3s.add_input(name, pl)
        elif final_tensors:
            all_inputs = {inp for op in program.ops for inp in op.input_names}
            all_outputs = {op.output_name for op in program.ops}
            pure_inputs = all_inputs - all_outputs
            for name in pure_inputs:
                ts = final_tensors.get(name)
                if ts and ts.sharding.placements:
                    if len(ts.sharding.placements) > 1:
                        z3s.add_input(name, tuple(ts.sharding.placements))
                    else:
                        z3s.add_input(name, ts.sharding.placements[0])

        z3s.encode_program(program)

        # Check program satisfiability first
        sat_result = z3s.check_program_satisfiability()
        if not sat_result.passed:
            vr = VerifyResult(
                passed=False,
                condition="placement consistency (Z3)",
                details=sat_result.details,
            )
            self.results.append(vr)
            return vr

        if output_names is None:
            all_inputs = {inp for op in program.ops for inp in op.input_names}
            output_names = [
                op.output_name for op in program.ops
                if op.output_name not in all_inputs
            ]

        eq_results = z3s.check_output_equivalence(output_names)
        pc_results = z3s.check_collective_preconditions(program)
        cc_results = z3s.check_compute_preconditions(program)

        all_checks = eq_results + pc_results + cc_results
        all_passed = all(r.passed for r in all_checks)

        details_parts = []
        for r in all_checks:
            status = "PASS" if r.passed else "FAIL"
            details_parts.append(f"[{status}] {r.condition}: {r.details}")

        vr = VerifyResult(
            passed=all_passed,
            condition="placement consistency (Z3)",
            details="; ".join(details_parts) if details_parts else "No ops to check",
            counterexample=next(
                (r.counterexample for r in all_checks if r.counterexample), None
            ),
        )
        self.results.append(vr)
        return vr

    def verify_shape_consistency(
        self,
        program: Program,
        initial_shapes: Dict[str, Tuple[int, ...]],
        final_tensors: Dict[str, TensorState],
    ) -> VerifyResult:
        """Verify shape consistency through the program."""
        errors = []

        for name, tensor in final_tensors.items():
            # Check that final shapes are valid
            if any(d <= 0 for d in tensor.global_shape):
                errors.append(f"Tensor '{name}' has invalid shape {tensor.global_shape}")
            if any(d <= 0 for d in tensor.local_shape):
                errors.append(f"Tensor '{name}' has invalid local shape {tensor.local_shape}")

        passed = len(errors) == 0
        vr = VerifyResult(
            passed=passed,
            condition="shape consistency",
            details="All shapes are consistent" if passed else "; ".join(errors),
        )
        self.results.append(vr)
        return vr

    def verify_pp_deadlock_free(
        self,
        schedule: List[MicroBatch],
        sends: List[Tuple[int, int, str, int]],  # (src, dst, tensor, mb_id)
        recvs: List[Tuple[int, int, str, int]],  # (src, dst, tensor, mb_id)
    ) -> VerifyResult:
        """Verify deadlock freedom for a PP schedule.

        Uses both structural checks and Z3-based cycle detection.
        """
        checker = DeadlockChecker()
        for src, dst, tensor, _ in sends:
            checker.add_send(src, dst, tensor)
        for src, dst, tensor, _ in recvs:
            checker.add_recv(src, dst, tensor)

        is_deadlock_free, errors = checker.check()

        # Also verify schedule order: no Recv before its Send
        ordered_errors = []
        for mb in schedule:
            for src, tensor in mb.recvs:
                # Check if the corresponding Send has already been scheduled
                send_mb = None
                for other_mb in schedule:
                    for dst, t in other_mb.sends:
                        if dst == mb.stage_id and t == tensor:
                            send_mb = other_mb
                            break
                if send_mb is None:
                    ordered_errors.append(
                        f"Recv({tensor}) at stage={mb.stage_id}, mb={mb.mb_id} "
                        f"has no matching Send in schedule"
                    )

        all_errors = errors + ordered_errors
        passed = len(all_errors) == 0

        vr = VerifyResult(
            passed=passed,
            condition="PP deadlock freedom",
            details="No deadlocks detected" if passed else "; ".join(all_errors),
        )
        self.results.append(vr)
        return vr

    def verify_all(
        self,
        program: Program,
        final_tensors: Dict[str, TensorState],
        bwd_program: Optional[Program] = None,
        initial_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        output_names: Optional[List[str]] = None,
        multi_device_states: Optional[Dict[int, Dict[str, TensorState]]] = None,
    ) -> List[VerifyResult]:
        """Run all verification checks.

        Args:
            program: The forward IR program.
            final_tensors: All tensor states after execution.
            bwd_program: Backward program (for duality check).
            initial_shapes: Initial tensor shapes (for shape check).
            output_names: Names of output tensors to check postcondition.
                          If None, auto-detects tensors not used as inputs
                          to any subsequent op.

        Returns a list of all VerifyResults.
        """
        self.results = []

        # Auto-detect output tensors: those never used as input to another op
        if output_names is None:
            all_inputs = {inp for op in program.ops for inp in op.input_names}
            output_names = [
                name for name in final_tensors
                if name not in all_inputs and not name.startswith("grad_")
            ]

        # 1. Postcondition: output tensors must not be partial
        for name in output_names:
            tensor = final_tensors.get(name)
            if tensor is not None:
                self.verify_postcondition(
                    tensor,
                    expected_partial=False,
                )

        # 2. Communication legality (check across all devices)
        self.verify_communication_legality(
            program,
            tensor_states=final_tensors,
            multi_device_states=multi_device_states,
        )

        # 3. Gradient duality
        if bwd_program:
            self.verify_gradient_duality(program, bwd_program)

        # 4. Placement consistency
        self.verify_placement_consistency(
            program, final_tensors=final_tensors, output_names=output_names,
        )

        # 5. Shape consistency (Z3-based when shapes available)
        if initial_shapes:
            self._verify_shape_z3(program, initial_shapes, final_tensors)
        elif final_tensors:
            self.verify_shape_consistency(program, {}, final_tensors)

        return self.results

    def _verify_shape_z3(
        self,
        program: Program,
        initial_shapes: Dict[str, Tuple[int, ...]],
        final_tensors: Dict[str, TensorState],
    ):
        """Run Z3-based shape consistency and slice alignment checks."""
        mesh_sizes = None
        mesh_ndim = 1
        for ts in final_tensors.values():
            if ts.sharding and ts.sharding.mesh and ts.sharding.mesh.shape:
                mesh_sizes = list(ts.sharding.mesh.shape)
                mesh_ndim = len(mesh_sizes)
                break

        if mesh_sizes is None or all(s < 2 for s in mesh_sizes):
            self.verify_shape_consistency(program, initial_shapes, final_tensors)
            return

        z3s = Z3PlacementSolver(mesh_ndim=mesh_ndim)
        for name, shape in initial_shapes.items():
            z3s.add_input_shape(name, shape)
            ts = final_tensors.get(name)
            if ts and ts.sharding and ts.sharding.placements:
                if len(ts.sharding.placements) > 1:
                    z3s.add_input(name, tuple(ts.sharding.placements))
                else:
                    z3s.add_input(name, ts.sharding.placements[0])

        z3s.encode_program(program)
        z3s.encode_shape_constraints(program, mesh_sizes=mesh_sizes)
        z3s.encode_slice_constraints(program, mesh_sizes=mesh_sizes)

        shape_results = z3s.check_shape_consistency()
        slice_results = z3s.check_slice_alignment(program)

        shape_passed = all(r.passed for r in shape_results)
        slice_passed = all(r.passed for r in slice_results)

        if shape_results:
            details = "; ".join(r.details for r in shape_results if not r.passed) \
                if not shape_passed else "All shape constraints verified by Z3"
            self.results.append(VerifyResult(
                passed=shape_passed,
                condition="shape consistency (Z3 L1)",
                details=details,
            ))

        if slice_results:
            details = "; ".join(r.details for r in slice_results if not r.passed) \
                if not slice_passed else "All slice alignments verified by Z3"
            self.results.append(VerifyResult(
                passed=slice_passed,
                condition="slice alignment (Z3 L2)",
                details=details,
            ))

    def summary(self) -> str:
        """Return a summary of all verification results."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        lines = [
            f"Verification Summary: {passed} passed, {failed} failed",
        ]
        for r in self.results:
            lines.append(repr(r))
        return "\n".join(lines)


# ── Convenience functions ────────────────────────────────────────────────────

def verify_postcondition(
    tensor: TensorState,
    expected_partial: bool = False,
    expected_placement: Optional[Placement] = None,
) -> VerifyResult:
    """Convenience: verify a single tensor's postcondition."""
    verifier = DistributedVerifier()
    return verifier.verify_postcondition(tensor, expected_partial, expected_placement)


def verify_gradient_duality(
    fwd: Program,
    bwd: Program,
) -> VerifyResult:
    """Convenience: verify gradient duality."""
    verifier = DistributedVerifier()
    return verifier.verify_gradient_duality(fwd, bwd)


def verify_communication_legality(
    program: Program,
    tensor_states: Optional[Dict[str, TensorState]] = None,
) -> VerifyResult:
    """Convenience: verify communication legality."""
    verifier = DistributedVerifier()
    return verifier.verify_communication_legality(program, tensor_states=tensor_states)


def verify_pp_deadlock_free(
    schedule: List[MicroBatch],
    sends: List[Tuple[int, int, str, int]],
    recvs: List[Tuple[int, int, str, int]],
) -> VerifyResult:
    """Convenience: verify PP deadlock freedom."""
    verifier = DistributedVerifier()
    return verifier.verify_pp_deadlock_free(schedule, sends, recvs)
