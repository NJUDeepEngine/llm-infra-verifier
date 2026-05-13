"""SPMD local types and placement specifications.

LocalSPMDType encodes the four SPMD states (R/I/V/P) with fixed
forward↔backward duality. Shard, Replicate, Partial describe how
tensor dimensions map onto mesh axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class LocalSPMDType(Enum):
    """SPMD local types per mesh axis (from meta-pytorch/spmd_types DESIGN.md).

    Four states with fixed forward↔backward duality:

        REPLICATE (R): data same across ranks  →  gradient is PARTIAL
        INVARIANT (I): data same across ranks  →  gradient is INVARIANT (no comm)
        VARYING  (V): data differs per rank   →  gradient is VARYING
        PARTIAL  (P): pending sum across ranks →  gradient is REPLICATE

    Duality: R↔P, I↔I, V↔V. These are fixed by autograd.
    """
    REPLICATE = "R"
    INVARIANT = "I"
    VARYING = "V"
    PARTIAL = "P"

    def gradient_type(self) -> "LocalSPMDType":
        """Return the gradient's local type (backward dual)."""
        dual = {
            LocalSPMDType.REPLICATE: LocalSPMDType.PARTIAL,
            LocalSPMDType.INVARIANT: LocalSPMDType.INVARIANT,
            LocalSPMDType.VARYING: LocalSPMDType.VARYING,
            LocalSPMDType.PARTIAL: LocalSPMDType.REPLICATE,
        }
        return dual[self]


@dataclass(frozen=True)
class Shard:
    """Tensor dimension `dim` is sharded across a mesh axis.

    This is a GLOBAL property — it describes how shards reassemble.
    In SPMD terms: the PartitionSpec entry for this dimension.
    """
    dim: int

    def __repr__(self):
        return f"Shard({self.dim})"


class PlacementType(Enum):
    REPLICATE = "replicate"
    SHARD = "shard"
    PARTIAL = "partial"


@dataclass(frozen=True)
class Replicate:
    """Tensor is replicated across all devices in the mesh dimension."""

    def __repr__(self):
        return "Replicate()"


@dataclass(frozen=True)
class Partial:
    """Tensor is partially reduced — needs AllReduce to become Replicate."""

    def __repr__(self):
        return "Partial()"


Placement = Union[Shard, Replicate, Partial]
