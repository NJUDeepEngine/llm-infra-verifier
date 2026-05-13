from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..state import TensorState, LocalSPMDType


class SPMDConsistencyError(ValueError):
    """Raised when SPMD type propagation disagrees with placement derivation."""
    pass


class IROp(ABC):
    """Abstract base for all IR operations."""

    def __init__(self):
        self._id = id(self)

    @abstractmethod
    def apply(self, ctx: Dict[str, TensorState]) -> TensorState:
        """Execute forward, returning the output TensorState.

        ctx is a mutable dict of {name: TensorState} representing the
        current symbolic state of all tensors.
        """
        ...

    @abstractmethod
    def vjp(
        self,
        ctx: Dict[str, TensorState],
        grad_output: TensorState,
    ) -> Dict[str, TensorState]:
        """Compute VJP, returning {input_name: grad_tensor}.

        ctx contains the tensor states at the time this op was executed
        (including saved tensors for the backward).
        """
        ...

    def is_collective(self) -> bool:
        """Whether this op involves cross-device communication (collective or P2P)."""
        return False

    def is_p2p(self) -> bool:
        """Whether this op is point-to-point (Send/Recv)."""
        return False

    def is_communication(self) -> bool:
        """Whether this op involves any inter-device communication."""
        return self.is_collective() or self.is_p2p()

    def is_async(self) -> bool:
        """Whether this op is asynchronous (returns before completion)."""
        return False

    def is_sync(self) -> bool:
        """Whether this op is a synchronization point (Wait/WaitAll)."""
        return False

    def propagate_spmd_type(
        self, input_types: Dict[str, LocalSPMDType],
    ) -> Optional[LocalSPMDType]:
        """Compute output SPMD type from input SPMD types.

        Returns None to indicate "derive from placement" (default).
        Override in subclasses where SPMD type propagation has independent rules.
        """
        return None

    def apply_checked(self, ctx: Dict[str, TensorState]) -> TensorState:
        """Apply the op and cross-validate SPMD type against placement derivation."""
        result = self.apply(ctx)

        input_types = {}
        for name in self.input_names:
            ts = ctx.get(name)
            if ts is not None and ts.local_type is not None:
                input_types[name] = ts.local_type

        spmd_type = self.propagate_spmd_type(input_types)
        if spmd_type is not None and result is not None:
            placement_derived = result.local_type
            if placement_derived != spmd_type:
                raise SPMDConsistencyError(
                    f"SPMD/placement mismatch in {type(self).__name__}: "
                    f"placement-derived={placement_derived.value}, "
                    f"SPMD-propagated={spmd_type.value}. "
                    f"Op: {self!r}, inputs={input_types}"
                )
        return result

    @property
    @abstractmethod
    def input_names(self) -> List[str]:
        """Names of input tensors."""
        ...

    @property
    @abstractmethod
    def output_name(self) -> str:
        """Name of the output tensor."""
        ...

    @abstractmethod
    def clone_with_names(self, input_map: Dict[str, str], output_name: str) -> IROp:
        """Clone this op with renamed inputs/output."""
        ...
