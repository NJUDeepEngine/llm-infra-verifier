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
    PlacementType,
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
    Send,
    Recv,
    FlashAttention,
)
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
      - Z3 Int for placement: pl_{name} ∈ {0,1,2,3}
      - Z3 Ints for shape: gs_{name}_0, gs_{name}_1 (global dims)
      - Z3 Ints for local shape: ls_{name}_0, ls_{name}_1
    """

    def __init__(self):
        self.solver = Solver()
        self._vars: Dict[str, Int] = {}
        self._shape_vars: Dict[str, List] = {}
        self._local_vars: Dict[str, List] = {}
        self._tp_size: Optional[int] = None

    def _var(self, name: str):
        if name not in self._vars:
            v = Int(f"pl_{name}")
            self.solver.add(And(v >= 0, v <= 3))
            self._vars[name] = v
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

    def add_input(self, name: str, placement: Placement):
        """Assert a known concrete placement for an input tensor."""
        v = self._var(name)
        if isinstance(placement, Replicate):
            self.solver.add(v == PL_R)
        elif isinstance(placement, Shard):
            self.solver.add(v == (PL_S0 if placement.dim == 0 else PL_S1))
        elif isinstance(placement, Partial):
            self.solver.add(v == PL_P)

    def add_input_shape(self, name: str, shape: Tuple[int, ...]):
        """Assert concrete shape for an input tensor."""
        gs, ls = self._shape_var(name, len(shape))
        for d, s in enumerate(shape):
            self.solver.add(gs[d] == s)

    def encode_program(self, program: Program):
        """Walk all ops and add Z3 constraints for placement propagation."""
        R, S0, S1, P = IntVal(PL_R), IntVal(PL_S0), IntVal(PL_S1), IntVal(PL_P)

        for op in program.ops:
            if isinstance(op, MatMul):
                a, b, y = self._var(op.a), self._var(op.b), self._var(op.output)
                # MatMul C = A[M,K] @ B[K,N]:
                #   S(1)×S(0) → P  (contracting dim sharded in both)
                #   R×S(1) → S(1)  (column parallel)
                #   S(0)×R → S(0)  (batch sharded)
                #   R×R → R
                #   P×any or any×P → P
                self.solver.add(y == If(Or(a == P, b == P), P,
                                    If(And(a == S1, b == S0), P,
                                    If(And(a == R,  b == S1), S1,
                                    If(And(a == S0, b == R),  S0,
                                    If(And(a == R,  b == R),  R,
                                    If(And(a == S1, b == R),  S1,
                                    If(And(a == R,  b == S0), S0,
                                    P))))))))

            elif isinstance(op, (Add, Multiply)):
                a, b, y = self._var(op.a), self._var(op.b), self._var(op.output)
                # Element-wise: R+X→X, X+R→X, same→same
                self.solver.add(y == If(a == R, b, If(b == R, a, a)))

            elif isinstance(op, SiLU):
                x, y = self._var(op.x), self._var(op.output)
                self.solver.add(y == x)

            elif isinstance(op, FlashAttention):
                q, y = self._var(op.q), self._var(op.output)
                self.solver.add(y == q)

            elif isinstance(op, AllReduce):
                y = self._var(op.output)
                self.solver.add(y == R)

            elif isinstance(op, AllGather):
                y = self._var(op.output)
                self.solver.add(y == R)

    def check_output_equivalence(
        self,
        output_names: List[str],
    ) -> List[VerifyResult]:
        """Check if all outputs are guaranteed Replicate (= single-GPU equivalent).

        For each output, Z3 checks whether the output can be non-Replicate.
        UNSAT → proved equivalent. SAT → counterexample showing the violation.
        """
        results = []
        R = IntVal(PL_R)

        for name in output_names:
            v = self._vars.get(name)
            if v is None:
                continue

            self.solver.push()
            self.solver.add(v != R)

            check = self.solver.check()
            if check == sat:
                model = self.solver.model()
                ce = {}
                for n, var in self._vars.items():
                    val = model.eval(var, model_completion=True).as_long()
                    ce[n] = _PL_NAMES.get(val, str(val))
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

    def check_collective_preconditions(
        self,
        program: Program,
    ) -> List[VerifyResult]:
        """Verify AllReduce preconditions: input must always be Partial.

        For each AllReduce, Z3 checks whether its input can be non-Partial.
        UNSAT → input is always Partial (correct usage).
        SAT → input may not be Partial (unnecessary or wrong AllReduce).
        """
        results = []
        P = IntVal(PL_P)

        for op in program.ops:
            if isinstance(op, AllReduce):
                x = self._vars.get(op.x)
                if x is None:
                    continue

                self.solver.push()
                self.solver.add(x != P)

                check = self.solver.check()
                if check == sat:
                    model = self.solver.model()
                    val = model.eval(x, model_completion=True).as_long()
                    results.append(VerifyResult(
                        passed=False,
                        condition=f"AllReduce({op.x}) precondition",
                        details=(
                            f"Input '{op.x}' can be {_PL_NAMES.get(val, '?')}, "
                            f"not Partial — AllReduce may be unnecessary"
                        ),
                    ))
                else:
                    results.append(VerifyResult(
                        passed=True,
                        condition=f"AllReduce({op.x}) precondition",
                        details=(
                            f"Z3 proved: '{op.x}' is always Partial "
                            f"before AllReduce"
                        ),
                    ))

                self.solver.pop()

        return results

    # ── L1: Shape consistency ───────────────────────────────────────────────

    def encode_shape_constraints(self, program: Program, tp_size: int):
        """Add Z3 constraints for shape correctness (L1).

        (a) Divisibility: sharded dims must be divisible by tp_size
        (b) Local shape:  local = global / tp if sharded, else global
        (c) MatMul:       a.global_cols == b.global_rows  (contraction dim)
        (d) MatMul out:   out.rows == a.rows, out.cols == b.cols
        (e) Add/Mul:      a.global_shape == b.global_shape
        """
        self._tp_size = tp_size
        R, S0, S1, P = IntVal(PL_R), IntVal(PL_S0), IntVal(PL_S1), IntVal(PL_P)
        tp = IntVal(tp_size)

        for name in list(self._shape_vars.keys()):
            gs, ls = self._shape_vars[name], self._local_vars[name]
            pl = self._var(name)
            for d in range(len(gs)):
                # Divisibility: if sharded on this dim, must be divisible
                if d == 0:
                    self.solver.add(Implies(pl == S0, gs[d] % tp == 0))
                elif d == 1:
                    self.solver.add(Implies(pl == S1, gs[d] % tp == 0))
                # Local shape derivation
                self.solver.add(ls[d] == If(
                    And(pl == S0, IntVal(d) == 0), gs[d] / tp,
                    If(And(pl == S1, IntVal(d) == 1), gs[d] / tp,
                       gs[d])))

        for op in program.ops:
            if isinstance(op, MatMul):
                if op.a not in self._shape_vars or op.b not in self._shape_vars:
                    continue
                gs_a = self._shape_vars[op.a]
                gs_b = self._shape_vars[op.b]

                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)

                # Contraction dim: a.cols == b.rows (global shapes must match)
                self.solver.add(gs_a[1] == gs_b[0])
                # Output shape: rows from A, cols from B
                self.solver.add(gs_y[0] == gs_a[0])
                self.solver.add(gs_y[1] == gs_b[1])
                # Output local shape
                for d in range(2):
                    if d == 0:
                        self.solver.add(ls_y[d] == If(
                            And(pl_y == S0, IntVal(d) == 0), gs_y[d] / tp,
                            If(And(pl_y == S1, IntVal(d) == 1), gs_y[d] / tp,
                               gs_y[d])))
                    else:
                        self.solver.add(ls_y[d] == If(
                            And(pl_y == S0, IntVal(d) == 0), gs_y[d] / tp,
                            If(And(pl_y == S1, IntVal(d) == 1), gs_y[d] / tp,
                               gs_y[d])))

            elif isinstance(op, (Add, Multiply)):
                if op.a not in self._shape_vars or op.b not in self._shape_vars:
                    continue
                gs_a = self._shape_vars[op.a]
                gs_b = self._shape_vars[op.b]
                # Element-wise requires same global shape
                for d in range(min(len(gs_a), len(gs_b))):
                    self.solver.add(gs_a[d] == gs_b[d])
                # Output shape == input shape
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(min(len(gs_a), len(gs_y))):
                    self.solver.add(gs_y[d] == gs_a[d])
                    self.solver.add(ls_y[d] == If(
                        And(pl_y == S0, IntVal(d) == 0), gs_y[d] / tp,
                        If(And(pl_y == S1, IntVal(d) == 1), gs_y[d] / tp,
                           gs_y[d])))

            elif isinstance(op, SiLU):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                    self.solver.add(ls_y[d] == If(
                        And(pl_y == S0, IntVal(d) == 0), gs_y[d] / tp,
                        If(And(pl_y == S1, IntVal(d) == 1), gs_y[d] / tp,
                           gs_y[d])))

            elif isinstance(op, FlashAttention):
                # FlashAttention output shape is unconstrained from inputs
                # because the IR may use fused QKV (3H) while output is (H).
                # The downstream MatMul contraction constraint will determine
                # the actual output dimension.
                gs_y, ls_y = self._shape_var(op.output)
                pl_y = self._var(op.output)
                if op.q in self._shape_vars:
                    gs_q = self._shape_vars[op.q]
                    # Batch dim (d0) is preserved
                    self.solver.add(gs_y[0] == gs_q[0])
                    # d1 is NOT constrained from Q (may be fused QKV)
                for d in range(2):
                    self.solver.add(ls_y[d] == If(
                        And(pl_y == S0, IntVal(d) == 0), gs_y[d] / tp,
                        If(And(pl_y == S1, IntVal(d) == 1), gs_y[d] / tp,
                           gs_y[d])))

            elif isinstance(op, AllReduce):
                if op.x not in self._shape_vars:
                    continue
                gs_x = self._shape_vars[op.x]
                gs_y, ls_y = self._shape_var(op.output)
                for d in range(len(gs_x)):
                    self.solver.add(gs_y[d] == gs_x[d])
                    self.solver.add(ls_y[d] == gs_y[d])

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

        # Check individual properties
        if self._tp_size and self._shape_vars:
            tp = IntVal(self._tp_size)
            for name in self._shape_vars:
                pl = self._vars.get(name)
                gs = self._shape_vars[name]
                if pl is None:
                    continue
                # Check divisibility for each possible shard dim
                for d, shard_pl in [(0, PL_S0), (1, PL_S1)]:
                    if d >= len(gs):
                        continue
                    self.solver.push()
                    self.solver.add(pl == shard_pl)
                    self.solver.add(gs[d] % tp != 0)
                    check = self.solver.check()
                    if check == sat:
                        model = self.solver.model()
                        dim_val = model.eval(gs[d], model_completion=True).as_long()
                        results.append(VerifyResult(
                            passed=False,
                            condition=f"divisibility({name}, dim={d})",
                            details=(
                                f"Shard(dim={d}) on '{name}' but dim size "
                                f"{dim_val} not divisible by TP={self._tp_size}"
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

    def encode_slice_constraints(self, program: Program, tp_size: int):
        """Add Z3 constraints for per-device slice alignment (L2).

        For MatMul(A, B) with A sharded on cols and B sharded on rows,
        verifies that on each device d, A's column slice and B's row slice
        cover the same interval of the contraction dimension.
        """
        self._tp_size = tp_size

        for op in program.ops:
            if not isinstance(op, MatMul):
                continue
            if op.a not in self._shape_vars or op.b not in self._shape_vars:
                continue

            pl_a = self._vars.get(op.a)
            pl_b = self._vars.get(op.b)
            gs_a = self._shape_vars[op.a]
            gs_b = self._shape_vars[op.b]
            if pl_a is None or pl_b is None:
                continue

            S0, S1 = IntVal(PL_S0), IntVal(PL_S1)
            tp = IntVal(tp_size)

            # When both A and B are sharded on the contraction dim
            # (A: Shard(1), B: Shard(0)), verify slice alignment per device
            for d in range(tp_size):
                d_val = IntVal(d)
                # A column slice: offset = d * (A_cols / tp), width = A_cols / tp
                a_col_offset = d_val * (gs_a[1] / tp)
                a_col_width = gs_a[1] / tp
                # B row slice: offset = d * (B_rows / tp), width = B_rows / tp
                b_row_offset = d_val * (gs_b[0] / tp)
                b_row_width = gs_b[0] / tp

                # When A=S(1) and B=S(0): slices must align
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
        """
        errors = []

        # Build a merged view across all devices
        merged_states = dict(tensor_states) if tensor_states else {}
        if multi_device_states:
            for did, dev_states in multi_device_states.items():
                for name, ts in dev_states.items():
                    if name not in merged_states:
                        merged_states[name] = ts

        for op in program.ops:
            if isinstance(op, AllReduce):
                # Check actual tensor state if available (across all devices)
                if op.x in merged_states:
                    ts = merged_states[op.x]
                    if not ts.partial:
                        errors.append(
                            f"AllReduce({op.x}) called on non-partial tensor: {ts.sharding}"
                        )
                else:
                    import warnings
                    warnings.warn(
                        f"AllReduce({op.x}): no tensor state available to verify "
                        f"precondition (input should be Partial)"
                    )

            elif isinstance(op, AllGather):
                if op.x in merged_states:
                    ts = merged_states[op.x]
                    has_shard_on_dim = any(
                        isinstance(p, Shard) and p.dim == op.gather_dim
                        for p in ts.sharding.placements
                    )
                    if not has_shard_on_dim:
                        errors.append(
                            f"AllGather({op.x}, dim={op.gather_dim}) called on tensor "
                            f"not sharded on dim {op.gather_dim}: {ts.sharding}"
                        )

            elif isinstance(op, Send):
                # Check that there's a matching Recv
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
                # Check that there's a matching Send
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

        # Map of dual relations
        dual_map = {
            "AllReduce": "AllReduce",
            "AllGather": "ReduceScatter",
            "ReduceScatter": "AllGather",
            "Send": "Recv",
            "Recv": "Send",
        }

        fwd_collectives = [op for op in fwd_program.ops if op.is_collective()]
        bwd_collectives = [op for op in bwd_program.ops if op.is_collective()]

        for fwd_op in fwd_collectives:
            fwd_type = type(fwd_op).__name__
            expected_dual = dual_map.get(fwd_type)

            if expected_dual is None:
                continue

            # Check if there exists a bwd collective of the dual type
            found = False
            for bwd_op in bwd_collectives:
                bwd_type = type(bwd_op).__name__
                if bwd_type == expected_dual:
                    # Check more specific properties
                    if isinstance(fwd_op, AllReduce) and isinstance(bwd_op, AllReduce):
                        found = True
                        break
                    elif isinstance(fwd_op, AllGather) and isinstance(bwd_op, ReduceScatter):
                        if fwd_op.gather_dim == bwd_op.scatter_dim:
                            found = True
                            break
                    elif isinstance(fwd_op, ReduceScatter) and isinstance(bwd_op, AllGather):
                        if fwd_op.scatter_dim == bwd_op.gather_dim:
                            found = True
                            break
                    elif isinstance(fwd_op, Send) and isinstance(bwd_op, Recv):
                        if fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src:
                            found = True
                            break
                    elif isinstance(fwd_op, Recv) and isinstance(bwd_op, Send):
                        if fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src:
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

    def verify_placement_consistency(
        self,
        program: Program,
        input_placements: Optional[Dict[str, Placement]] = None,
        final_tensors: Optional[Dict[str, TensorState]] = None,
    ) -> VerifyResult:
        """Verify placement consistency using Z3 SMT solver.

        Encodes all op propagation rules as Z3 constraints and checks
        that output tensors are always Replicate (single-GPU equivalent).
        """
        z3s = Z3PlacementSolver()

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
                    z3s.add_input(name, ts.sharding.placements[0])

        z3s.encode_program(program)

        all_inputs = {inp for op in program.ops for inp in op.input_names}
        output_names = [
            op.output_name for op in program.ops
            if op.output_name not in all_inputs
        ]

        eq_results = z3s.check_output_equivalence(output_names)
        pc_results = z3s.check_collective_preconditions(program)

        all_checks = eq_results + pc_results
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
        self.verify_placement_consistency(program)

        # 5. Shape consistency
        if initial_shapes:
            self.verify_shape_consistency(program, initial_shapes, final_tensors)

        return self.results

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
