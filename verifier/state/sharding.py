"""Sharding specification and tensor slice computation.

ShardingSpec describes how a tensor is distributed across a device mesh.
compute_local_shape / compute_tensor_slices derive per-device views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .placement import Placement, Shard, Partial
from .device import DeviceMesh


@dataclass
class ShardingSpec:
    """Full sharding specification for a tensor on a device mesh.

    `placements` has one entry per mesh dimension, describing how the tensor
    is distributed along that dimension.
    """
    placements: Tuple[Placement, ...]
    mesh: DeviceMesh

    def __post_init__(self):
        if len(self.placements) != self.mesh.ndim:
            raise ValueError(
                f"placements length {len(self.placements)} != mesh ndim {self.mesh.ndim}"
            )

    @property
    def partial(self) -> bool:
        return any(isinstance(p, Partial) for p in self.placements)

    def get_shard_dims(self) -> Dict[int, List[int]]:
        """Return {tensor_dim: [mesh_dims]} for all Shard placements.

        Supports double-sharding: a tensor dim sharded across multiple mesh axes.
        """
        result: Dict[int, List[int]] = {}
        for mesh_dim, p in enumerate(self.placements):
            if isinstance(p, Shard):
                result.setdefault(p.dim, []).append(mesh_dim)
        return result

    def __repr__(self):
        placements_str = ", ".join(repr(p) for p in self.placements)
        return f"ShardingSpec(({placements_str},), mesh={self.mesh})"


def compute_local_shape(
    global_shape: Tuple[int, ...],
    spec: ShardingSpec,
) -> Tuple[int, ...]:
    """Compute the local shape on one device given a sharding spec."""
    shape = list(global_shape)
    for mesh_dim, p in enumerate(spec.placements):
        if isinstance(p, Shard):
            mesh_size = spec.mesh.shape[mesh_dim]
            if shape[p.dim] % mesh_size != 0:
                raise ValueError(
                    f"Dimension {p.dim} size {shape[p.dim]} not divisible "
                    f"by mesh size {mesh_size}"
                )
            shape[p.dim] //= mesh_size
    return tuple(shape)


@dataclass(frozen=True)
class TensorSlice:
    """What slice of a global tensor a specific device holds.

    For Shard(dim=0) on device 1 of TP=2, global_shape=(8, 16):
      offsets=(4, 0), local_shape=(4, 16)
      → this device holds rows [4:8], all columns [0:16]
    """
    device_id: int
    global_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    offsets: Tuple[int, ...]

    @property
    def ranges(self) -> Tuple[Tuple[int, int], ...]:
        """(start, end) per dimension."""
        return tuple((o, o + s) for o, s in zip(self.offsets, self.local_shape))

    def range_str(self) -> str:
        """Human-readable slice notation, e.g. '[4:8, 0:16]'."""
        parts = []
        for dim, (start, end) in enumerate(self.ranges):
            if start == 0 and end == self.global_shape[dim]:
                parts.append(":")
            else:
                parts.append(f"{start}:{end}")
        return "[" + ", ".join(parts) + "]"

    def __repr__(self):
        ranges = self.ranges
        parts = [f"{s}:{e}" for s, e in ranges]
        return f"Slice(dev{self.device_id}, [{', '.join(parts)}])"


def compute_tensor_slices(
    global_shape: Tuple[int, ...],
    spec: ShardingSpec,
) -> Dict[int, TensorSlice]:
    """Compute the slice each device holds for a tensor.

    Returns {device_id: TensorSlice} for all devices in the mesh.
    """
    mesh = spec.mesh
    local_shape = compute_local_shape(global_shape, spec)

    result = {}
    for device_id in mesh.device_ids:
        coords = mesh.device_to_coord(device_id)
        offsets = [0] * len(global_shape)
        for mesh_dim, p in enumerate(spec.placements):
            if isinstance(p, Shard):
                chunk = global_shape[p.dim] // mesh.shape[mesh_dim]
                offsets[p.dim] = coords[mesh_dim] * chunk

        result[device_id] = TensorSlice(
            device_id=device_id,
            global_shape=global_shape,
            local_shape=local_shape,
            offsets=tuple(offsets),
        )
    return result
