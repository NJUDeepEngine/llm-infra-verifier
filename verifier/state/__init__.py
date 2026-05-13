"""Core state definitions for distributed tensor verification.

TensorState tracks per-device tensor metadata: placement, sharding spec,
symbolic expression, and autograd/pipeline annotations. The design follows
DTensor semantics with explicit per-dimension sharding.

DeviceTopology models the physical GPU hardware graph (nodes + links).
DeviceMesh maps logical parallelism dimensions onto physical devices.
TensorSlice tracks which slice of a global tensor each device holds.
"""

from .placement import (
    LocalSPMDType,
    Shard,
    PlacementType,
    Replicate,
    Partial,
    Placement,
)

from .device import (
    DeviceNode,
    Link,
    DeviceTopology,
    DeviceMesh,
)

from .sharding import (
    ShardingSpec,
    AccessPattern,
    compute_local_shape,
    TensorSlice,
    compute_tensor_slices,
)

from .tensor import (
    TensorState,
)

__all__ = [
    # placement
    "LocalSPMDType", "Shard", "PlacementType", "Replicate", "Partial", "Placement",
    # device
    "DeviceNode", "Link", "DeviceTopology", "DeviceMesh",
    # sharding
    "ShardingSpec", "AccessPattern", "compute_local_shape",
    "TensorSlice", "compute_tensor_slices",
    # tensor
    "TensorState",
]
