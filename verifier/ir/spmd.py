from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

from .base import IROp
from ..state import TensorState, LocalSPMDType

if TYPE_CHECKING:
    pass


@dataclass
class Reinterpret(IROp):
    """SPMD reinterpret: change local type WITHOUT communication.

    Changes the local SPMD type without changing the local tensor data.
    May change semantic denotation (e.g., R->P scales value by mesh size).

    Valid transitions (no expert_mode):
      R -> V, R -> P, V -> P, P -> V
    Expert-only transitions:
      R -> I, V -> R
    """
    x: str
    output: str
    src_type: LocalSPMDType
    dst_type: LocalSPMDType
    expert_mode: bool = False

    _VALID_TRANSITIONS = {
        (LocalSPMDType.REPLICATE, LocalSPMDType.VARYING),
        (LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL),
        (LocalSPMDType.VARYING, LocalSPMDType.PARTIAL),
        (LocalSPMDType.PARTIAL, LocalSPMDType.VARYING),
    }
    _EXPERT_TRANSITIONS = {
        (LocalSPMDType.REPLICATE, LocalSPMDType.INVARIANT),
        (LocalSPMDType.VARYING, LocalSPMDType.REPLICATE),
    }

    def __post_init__(self):
        key = (self.src_type, self.dst_type)
        if key in self._EXPERT_TRANSITIONS and not self.expert_mode:
            raise ValueError(
                f"Reinterpret({self.src_type.value}->{self.dst_type.value}) "
                f"requires expert_mode=True"
            )
        valid = self._VALID_TRANSITIONS | self._EXPERT_TRANSITIONS
        if key not in valid:
            raise ValueError(
                f"Invalid Reinterpret: {self.src_type.value}->{self.dst_type.value}"
            )

    @property
    def input_names(self) -> List[str]: return [self.x]
    @property
    def output_name(self) -> str: return self.output

    def propagate_spmd_type(self, input_types):
        return self.dst_type

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        ts = ctx[self.x]
        result = ts.with_local_type(self.dst_type)
        result.name = self.output
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {self.x: grad_output.with_local_type(self.src_type)}

    def is_collective(self) -> bool: return False
    def is_p2p(self) -> bool: return False
    def clone_with_names(self, im, on):
        return Reinterpret(im.get(self.x, self.x), on, self.src_type, self.dst_type, self.expert_mode)
    def __repr__(self):
        em = " [expert]" if self.expert_mode else ""
        return f"Reinterpret({self.x}, {self.src_type.value}->{self.dst_type.value}) -> {self.output}{em}"


@dataclass
class Convert(IROp):
    """SPMD convert: change local type WITHOUT communication, WITH data ops.

    Preserves semantic meaning but changes local data (e.g., R->P zeros
    non-rank-0 so that sum yields original). No communication, but local
    tensor may change.

    Valid transitions:
      R -> P (zero non-rank-0), P -> R (scale by N), I -> R, I -> V, I -> P
    """
    x: str
    output: str
    src_type: LocalSPMDType
    dst_type: LocalSPMDType
    expert_mode: bool = False

    _VALID_TRANSITIONS = {
        (LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL),
        (LocalSPMDType.PARTIAL, LocalSPMDType.REPLICATE),
        (LocalSPMDType.INVARIANT, LocalSPMDType.REPLICATE),
        (LocalSPMDType.INVARIANT, LocalSPMDType.VARYING),
        (LocalSPMDType.INVARIANT, LocalSPMDType.PARTIAL),
        (LocalSPMDType.VARYING, LocalSPMDType.PARTIAL),
        (LocalSPMDType.PARTIAL, LocalSPMDType.VARYING),
    }

    def __post_init__(self):
        if (self.src_type, self.dst_type) not in self._VALID_TRANSITIONS:
            raise ValueError(
                f"Invalid Convert: {self.src_type.value}->{self.dst_type.value}"
            )

    @property
    def input_names(self) -> List[str]: return [self.x]
    @property
    def output_name(self) -> str: return self.output

    def propagate_spmd_type(self, input_types):
        return self.dst_type

    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        ts = ctx[self.x]
        result = ts.with_local_type(self.dst_type)
        result.name = self.output
        ctx[self.output] = result
        return result

    def vjp(self, ctx, grad_output):
        return {self.x: grad_output.with_local_type(self.src_type)}

    def is_collective(self) -> bool: return False
    def is_p2p(self) -> bool: return False
    def clone_with_names(self, im, on):
        return Convert(im.get(self.x, self.x), on, self.src_type, self.dst_type, self.expert_mode)
    def __repr__(self):
        return f"Convert({self.x}, {self.src_type.value}->{self.dst_type.value}) -> {self.output}"


class SPMDGuard:
    """Runtime SPMD type validation. Enforces SPMD typing rules.

    Key rules from spmd_types DESIGN.md:
      1. Partial * Partial is FORBIDDEN (doesn't distribute over pending sum)
      2. Varying cannot be implicitly AllReduced (must reinterpret V->P first)
      3. AllReduce(R) is an error (R has no pending sum)
      4. Invariant gradient needs no AllReduce (I->I duality)
    """

    @staticmethod
    def check_multiply(a: TensorState, b: TensorState, op_name: str = "Multiply"):
        """SPMD rule: Partial * Partial is FORBIDDEN."""
        if a.is_partial and b.is_partial:
            raise ValueError(
                f"SPMD violation in {op_name}: Partial * Partial is forbidden. "
                f"'{a.name}' and '{b.name}' are both PARTIAL. "
                f"Consider: (a*b) != a_local*b_local after AllReduce. "
                f"AllReduce one operand first. (spmd_types DESIGN.md)"
            )

    @staticmethod
    def check_allreduce_input(x: TensorState):
        """AllReduce requires PARTIAL or VARYING (with reinterpret) input."""
        if x.is_replicate:
            raise ValueError(
                f"SPMD violation: AllReduce on REPLICATE tensor '{x.name}'. "
                f"REPLICATE has no pending sum. Use Reinterpret(R->P) first if intentional."
            )
        if x.is_invariant:
            raise ValueError(
                f"SPMD violation: AllReduce on INVARIANT tensor '{x.name}'. "
                f"INVARIANT gradient is already identical across ranks -- no AllReduce needed."
            )

    @staticmethod
    def check_allgather_input(x: TensorState):
        """AllGather requires VARYING input."""
        if not x.is_varying:
            raise ValueError(
                f"SPMD violation: AllGather requires VARYING input, "
                f"got {x.local_type.value} for '{x.name}'"
            )

    @staticmethod
    def check_reduce_scatter_input(x: TensorState):
        """ReduceScatter requires PARTIAL input."""
        if not x.is_partial:
            raise ValueError(
                f"SPMD violation: ReduceScatter requires PARTIAL input, "
                f"got {x.local_type.value} for '{x.name}'"
            )
