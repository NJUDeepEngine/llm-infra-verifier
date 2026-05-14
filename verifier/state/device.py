"""Device topology and mesh abstractions.

DeviceTopology models the physical GPU hardware graph (nodes + links).
DeviceMesh maps logical parallelism dimensions onto physical devices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DeviceNode:
    """A single GPU device in the cluster."""
    device_id: int

    def __repr__(self):
        return f"GPU({self.device_id})"


@dataclass
class Link:
    """A communication link between two devices."""
    src: int
    dst: int
    link_type: str = "NVLink"
    bandwidth_gbps: float = 300.0

    def __repr__(self):
        return f"{self.src}↔{self.dst}({self.link_type})"


@dataclass
class DeviceTopology:
    """Physical GPU topology: nodes (devices) and edges (links).

    Models the hardware graph that computation is placed onto.
    Each node is a GPU; each link represents direct connectivity
    (NVLink, PCIe, or network).
    """
    nodes: List[DeviceNode] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)

    def __post_init__(self):
        self._adj: Dict[int, Set[int]] = {}
        for n in self.nodes:
            self._adj[n.device_id] = set()
        for link in self.links:
            self._adj.setdefault(link.src, set()).add(link.dst)
            self._adj.setdefault(link.dst, set()).add(link.src)

    def device_ids(self) -> List[int]:
        return [n.device_id for n in self.nodes]

    def neighbors(self, device_id: int) -> List[int]:
        return list(self._adj.get(device_id, []))

    def are_connected(self, d1: int, d2: int) -> bool:
        return d2 in self._adj.get(d1, set())

    def all_connected(self, device_ids: List[int]) -> bool:
        """Check if all given devices form a fully connected group."""
        for d1, d2 in combinations(device_ids, 2):
            if d2 not in self._adj.get(d1, set()):
                return False
        return True

    def get_link(self, d1: int, d2: int) -> Optional[Link]:
        for link in self.links:
            if (link.src == d1 and link.dst == d2) or \
               (link.src == d2 and link.dst == d1):
                return link
        return None

    @staticmethod
    def fully_connected(n_devices: int, link_type: str = "NVLink",
                        bandwidth_gbps: float = 300.0) -> DeviceTopology:
        """Create a fully connected topology (e.g., NVLink mesh within a node)."""
        nodes = [DeviceNode(i) for i in range(n_devices)]
        links = []
        for i, j in combinations(range(n_devices), 2):
            links.append(Link(i, j, link_type, bandwidth_gbps))
        return DeviceTopology(nodes, links)

    def __repr__(self):
        return (f"DeviceTopology({len(self.nodes)} GPUs, "
                f"{len(self.links)} links)")


@dataclass
class DeviceMesh:
    """Multi-dimensional device topology.

    Maps logical parallelism dimensions onto physical devices.
    Optionally backed by a DeviceTopology that models hardware connectivity.

    Example:
        topo = DeviceTopology.fully_connected(8)
        mesh = DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"),
                          topology=topo)
        # 2 TP groups × 4 DP groups = 8 devices
    """
    shape: Tuple[int, ...]
    dim_names: Tuple[str, ...]
    device_ids: Optional[List[int]] = None
    topology: Optional[DeviceTopology] = None

    def __post_init__(self):
        if len(self.shape) != len(self.dim_names):
            raise ValueError(
                f"shape {self.shape} and dim_names {self.dim_names} must have same length"
            )
        if self.device_ids is None:
            self.device_ids = list(range(self.num_devices))
        if len(self.device_ids) != self.num_devices:
            raise ValueError(
                f"device_ids length {len(self.device_ids)} != "
                f"num_devices {self.num_devices}"
            )
        self._id_to_flat: Dict[int, int] = {
            did: flat for flat, did in enumerate(self.device_ids)
        }

    @property
    def num_devices(self) -> int:
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_submesh(self, dim_name: str) -> Tuple[int, int]:
        """Return (size, index) for a named mesh dimension."""
        idx = self.dim_names.index(dim_name)
        return self.shape[idx], idx

    def coord_to_device(self, *coords: int) -> int:
        """Map logical mesh coordinates to a physical device ID.

        For mesh shape (2,4), coord (1,2) → flat index 1*4+2=6 → device_ids[6].
        """
        if len(coords) != self.ndim:
            raise ValueError(f"Expected {self.ndim} coords, got {len(coords)}")
        for i, c in enumerate(coords):
            if not (0 <= c < self.shape[i]):
                raise ValueError(
                    f"Coordinate {c} out of range [0, {self.shape[i]}) "
                    f"for mesh dimension '{self.dim_names[i]}'"
                )
        flat = 0
        for i, c in enumerate(coords):
            flat = flat * self.shape[i] + c
        return self.device_ids[flat]

    def device_to_coord(self, device_id: int) -> Tuple[int, ...]:
        """Map a physical device ID back to logical mesh coordinates."""
        flat = self._id_to_flat.get(device_id)
        if flat is None:
            raise ValueError(f"Device {device_id} not in mesh")
        coords = []
        for dim_size in reversed(self.shape):
            coords.append(flat % dim_size)
            flat //= dim_size
        return tuple(reversed(coords))

    def devices_in_group(self, mesh_dim: int, index: int) -> List[int]:
        """All device IDs that share a communication group along mesh_dim.

        E.g., for mesh (2,4), mesh_dim=0, index=0:
          returns devices at coords (0,0),(0,1),(0,2),(0,3).
        """
        return [
            did for did in self.device_ids
            if self.device_to_coord(did)[mesh_dim] == index
        ]

    def validate_topology(self) -> List[str]:
        """Check that each communication group has full connectivity."""
        if self.topology is None:
            return []
        errors = []
        for dim in range(self.ndim):
            for idx in range(self.shape[dim]):
                group = self.devices_in_group(dim, idx)
                if not self.topology.all_connected(group):
                    errors.append(
                        f"Communication group {self.dim_names[dim]}[{idx}] "
                        f"devices {group} not fully connected"
                    )
        return errors

    def __repr__(self):
        return f"DeviceMesh(shape={self.shape}, dim_names={self.dim_names})"
