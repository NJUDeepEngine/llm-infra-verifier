"""Formal autograd engine for distributed IR programs.

Implements VJP (vector-Jacobian product) for each IR op and verifies
gradient computation correctness:
  1. Structural correctness: fwd ↔ bwd op duality
  2. Placement correctness: gradients have matching placements
  3. Collective duality: each forward collective has a backward dual
  4. Shape consistency: grad shapes match forward input shapes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import copy

from .state import (
    TensorState,
    Shard,
    Replicate,
    Partial,
    ShardingSpec,
    DeviceMesh,
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
    Reshape,
    Transpose,
    FlashAttention,
    ZeROGatherParam,
    ZeROScatterGrad,
    RingRotate,
    MoEDispatch,
    MoECombine,
)


# ── Gradient tape entry ──────────────────────────────────────────────────────

@dataclass
class TapeEntry:
    """A single entry in the autograd tape."""
    op: IROp
    input_names: List[str]
    output_name: str
    saved_tensors: Dict[str, TensorState]  # snapshot of tensors at op time

    def __repr__(self):
        return f"TapeEntry({self.op}, inputs={self.input_names})"


# ── Gradient check result ────────────────────────────────────────────────────

@dataclass
class GradientCheckResult:
    """Result of gradient computation verification."""
    passed: bool
    fwd_ops: int
    bwd_ops: int
    collective_pairs: List[Tuple[IROp, IROp]]  # (fwd_collective, bwd_dual)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_collectives_fwd(self) -> int:
        return len([f for f, _ in self.collective_pairs])

    @property
    def num_collectives_bwd(self) -> int:
        return len([b for _, b in self.collective_pairs])

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"GradientCheckResult: {status}",
            f"  Forward ops:  {self.fwd_ops}",
            f"  Backward ops: {self.bwd_ops}",
            f"  Collective pairs: {len(self.collective_pairs)}",
        ]
        for fwd, bwd in self.collective_pairs:
            lines.append(f"    {fwd}  ←dual→  {bwd}")
        for err in self.errors:
            lines.append(f"  ERROR: {err}")
        for warn in self.warnings:
            lines.append(f"  WARN: {warn}")
        return "\n".join(lines)


# ── Autograd engine ──────────────────────────────────────────────────────────

class AutogradEngine:
    """Formal autograd engine for distributed IR.

    Records operations on a tape during forward execution, then replays
    the tape in reverse applying VJP rules to generate the backward program.

    Also verifies:
      - Gradient shapes / placements match forward inputs
      - Every forward collective has a corresponding backward dual
    """

    def __init__(self):
        self.tape: List[TapeEntry] = []
        self.gradient_map: Dict[str, TensorState] = {}

    def record(
        self,
        op: IROp,
        tensor_states: Dict[str, TensorState],
    ):
        """Record an operation on the tape."""
        saved = {}
        for name in op.input_names:
            if name in tensor_states:
                saved[name] = copy.deepcopy(tensor_states[name])
        if op.output_name in tensor_states:
            saved[op.output_name] = copy.deepcopy(tensor_states[op.output_name])

        entry = TapeEntry(
            op=op,
            input_names=list(op.input_names),
            output_name=op.output_name,
            saved_tensors=saved,
        )
        self.tape.append(entry)

    def generate_backward(
        self,
        loss_tensor_name: str,
    ) -> Program:
        """Generate the backward program from the tape.

        Walks the tape in reverse order, applying VJP rules to compute
        gradients for all tensors that require them.
        """
        bwd_program = Program(name="backward")

        # Initialize loss gradient
        self.gradient_map[loss_tensor_name] = TensorState(
            name=f"grad_{loss_tensor_name}",
            global_shape=(),
            local_shape=(),
            sharding=ShardingSpec(
                placements=(Replicate(),),
                mesh=DeviceMesh(shape=(1,), dim_names=("tp",)),
            ),
            expr="1.0",  # ∂L/∂L = 1
        )

        # Walk tape in reverse
        for entry in reversed(self.tape):
            grad_output_name = entry.output_name
            grad_output = self.gradient_map.get(entry.output_name)

            if grad_output is None:
                # This tensor doesn't need a gradient
                continue

            # Compute VJP
            grad_inputs = entry.op.vjp(entry.saved_tensors, grad_output)

            # Add gradient contributions to gradient_map
            for input_name, grad_tensor in grad_inputs.items():
                if input_name in self.gradient_map:
                    # Accumulate gradients (add)
                    existing = self.gradient_map[input_name]
                    self.gradient_map[input_name] = TensorState(
                        name=existing.name,
                        global_shape=existing.global_shape,
                        local_shape=existing.local_shape,
                        sharding=existing.sharding,
                        expr=f"({existing.expr} + {grad_tensor.expr})"
                        if existing.expr and grad_tensor.expr
                        else "",
                    )
                else:
                    self.gradient_map[input_name] = grad_tensor

            # Add bwd ops for collectives
            if entry.op.is_collective():
                bwd_op = self._generate_dual_collective(entry, grad_inputs)
                if bwd_op:
                    bwd_program.add(bwd_op)

        return bwd_program

    def _generate_dual_collective(
        self,
        entry: TapeEntry,
        grad_inputs: Dict[str, TensorState],
    ) -> Optional[IROp]:
        """Generate the dual collective op for backward."""
        op = entry.op

        if isinstance(op, AllReduce):
            # AllReduce is self-dual
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return AllReduce(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_reduced",
                    op_type=op.op_type,
                )

        elif isinstance(op, AllGather):
            # Dual is ReduceScatter
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return ReduceScatter(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_scattered",
                    scatter_dim=op.gather_dim,
                )

        elif isinstance(op, ReduceScatter):
            # Dual is AllGather
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return AllGather(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_gathered",
                    gather_dim=op.scatter_dim,
                )

        elif isinstance(op, Send):
            # Dual is Recv (direction reversed)
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return Recv(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_received",
                    src=op.dst,  # reversed
                    dst=op.src,  # reversed
                    stage=op.stage,
                    microbatch_id=op.microbatch_id,
                )

        elif isinstance(op, Recv):
            # Dual is Send (direction reversed)
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return Send(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_sent",
                    src=op.dst,  # reversed
                    dst=op.src,  # reversed
                    stage=op.stage,
                    microbatch_id=op.microbatch_id,
                )

        elif isinstance(op, ZeROGatherParam):
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return ZeROScatterGrad(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_scattered",
                    scatter_dim=op.gather_dim,
                    zero_stage=op.zero_stage,
                )

        elif isinstance(op, ZeROScatterGrad):
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return ZeROGatherParam(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_gathered",
                    gather_dim=op.scatter_dim,
                    zero_stage=op.zero_stage,
                )

        elif isinstance(op, RingRotate):
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return RingRotate(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_rotated",
                    ring_size=op.ring_size,
                    ring_dim=op.ring_dim,
                )

        elif isinstance(op, MoEDispatch):
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return MoECombine(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_combined",
                    num_experts=op.num_experts,
                    split_dim=op.concat_dim,
                    concat_dim=op.split_dim,
                )

        elif isinstance(op, MoECombine):
            input_name = op.input_names[0]
            grad_tensor = grad_inputs.get(input_name)
            if grad_tensor:
                return MoEDispatch(
                    x=grad_tensor.name,
                    output=f"{grad_tensor.name}_dispatched",
                    num_experts=op.num_experts,
                    split_dim=op.concat_dim,
                    concat_dim=op.split_dim,
                )

        return None

    def verify_gradient_correctness(
        self,
        fwd_program: Program,
        bwd_program: Program,
    ) -> GradientCheckResult:
        """Verify gradient computation correctness.

        Checks:
          1. Every forward collective has a backward dual
          2. Gradient placements match forward input placements
          3. Gradient shapes match forward input shapes
          4. No missing/extra collectives
        """
        errors = []
        warnings = []
        collective_pairs = []

        # Collect forward collectives
        fwd_collectives = {
            op.output_name: op
            for op in fwd_program.ops
            if op.is_collective()
        }

        # Collect backward collectives
        bwd_collectives = {
            op.output_name: op
            for op in bwd_program.ops
            if op.is_collective()
        }

        # Check duality: each fwd collective should have a bwd dual
        for fwd_name, fwd_op in fwd_collectives.items():
            found_dual = False
            for bwd_op in bwd_collectives.values():
                if self._is_dual(fwd_op, bwd_op):
                    collective_pairs.append((fwd_op, bwd_op))
                    found_dual = True
                    break

            if not found_dual:
                errors.append(
                    f"Forward collective '{fwd_op}' has no backward dual"
                )

        # Check that every bwd collective corresponds to a fwd collective
        paired_bwd = {id(b) for _, b in collective_pairs}
        for bwd_op in bwd_collectives.values():
            if id(bwd_op) not in paired_bwd:
                warnings.append(
                    f"Backward collective '{bwd_op}' has no forward counterpart"
                )

        # Check gradient placements
        for entry in self.tape:
            for input_name in entry.input_names:
                if input_name in self.gradient_map:
                    grad = self.gradient_map[input_name]
                    fwd_input = entry.saved_tensors.get(input_name)
                    if fwd_input:
                        # Gradient should have same sharding as forward input
                        if grad.sharding != fwd_input.sharding:
                            errors.append(
                                f"Gradient placement mismatch for '{input_name}': "
                                f"grad={grad.sharding}, fwd_input={fwd_input.sharding}"
                            )

                        # Gradient shape should match forward input
                        if grad.global_shape != fwd_input.global_shape:
                            errors.append(
                                f"Gradient shape mismatch for '{input_name}': "
                                f"grad={grad.global_shape}, fwd={fwd_input.global_shape}"
                            )

        passed = len(errors) == 0
        return GradientCheckResult(
            passed=passed,
            fwd_ops=len(fwd_program),
            bwd_ops=len(bwd_program),
            collective_pairs=collective_pairs,
            errors=errors,
            warnings=warnings,
        )

    def _is_dual(self, fwd_op: IROp, bwd_op: IROp) -> bool:
        """Check if bwd_op is the dual of fwd_op (SPMD gradient duality rules).

        SPMD duality (from spmd_types DESIGN.md):
          Collective:  AllReduce ↔ AllReduce, AllGather ↔ ReduceScatter, Send ↔ Recv
          SPMD types:  R↔P, I↔I, V↔V  (fixed by autograd)
          Reinterpret: reverse the type transition
          Convert:     reverse the type transition
        """
        # Collective ops
        if isinstance(fwd_op, AllReduce) and isinstance(bwd_op, AllReduce):
            return True
        if isinstance(fwd_op, AllGather) and isinstance(bwd_op, ReduceScatter):
            return fwd_op.gather_dim == bwd_op.scatter_dim
        if isinstance(fwd_op, ReduceScatter) and isinstance(bwd_op, AllGather):
            return fwd_op.scatter_dim == bwd_op.gather_dim

        # P2P
        if isinstance(fwd_op, Send) and isinstance(bwd_op, Recv):
            return fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src
        if isinstance(fwd_op, Recv) and isinstance(bwd_op, Send):
            return fwd_op.src == bwd_op.dst and fwd_op.dst == bwd_op.src

        # ZeRO
        if isinstance(fwd_op, ZeROGatherParam) and isinstance(bwd_op, ZeROScatterGrad):
            return fwd_op.gather_dim == bwd_op.scatter_dim
        if isinstance(fwd_op, ZeROScatterGrad) and isinstance(bwd_op, ZeROGatherParam):
            return fwd_op.scatter_dim == bwd_op.gather_dim

        # Ring
        if isinstance(fwd_op, RingRotate) and isinstance(bwd_op, RingRotate):
            return fwd_op.ring_size == bwd_op.ring_size

        # MoE
        if isinstance(fwd_op, MoEDispatch) and isinstance(bwd_op, MoECombine):
            return (fwd_op.split_dim == bwd_op.concat_dim and
                    fwd_op.concat_dim == bwd_op.split_dim)
        if isinstance(fwd_op, MoECombine) and isinstance(bwd_op, MoEDispatch):
            return (fwd_op.split_dim == bwd_op.concat_dim and
                    fwd_op.concat_dim == bwd_op.split_dim)

        # SPMD type manipulation
        from .ir import Reinterpret, Convert
        if isinstance(fwd_op, Reinterpret) and isinstance(bwd_op, Reinterpret):
            # Backward reverses the type transition
            return (fwd_op.src_type == bwd_op.dst_type and
                    fwd_op.dst_type == bwd_op.src_type)
        if isinstance(fwd_op, Convert) and isinstance(bwd_op, Convert):
            return (fwd_op.src_type == bwd_op.dst_type and
                    fwd_op.dst_type == bwd_op.src_type)

        return False

    def get_gradient(self, tensor_name: str) -> Optional[TensorState]:
        """Get the computed gradient for a tensor."""
        return self.gradient_map.get(f"grad_{tensor_name}") or self.gradient_map.get(tensor_name)

    def reset(self):
        """Reset the autograd engine."""
        self.tape.clear()
        self.gradient_map.clear()
