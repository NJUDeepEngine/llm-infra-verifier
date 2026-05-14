"""Distributed IR operations with forward placement semantics and VJP rules.

Each op encodes:
  - forward:  placement propagation (how sharding flows through compute)
  - vjp:      vector-Jacobian product for backward pass
  - constraints: Z3-encodable legality conditions
"""

from .base import IROp, SPMDConsistencyError

from .compute import (
    ElementWiseBinaryOp,
    MatMul,
    Add,
    Multiply,
    SiLU,
    FlashAttention,
)

from .collective import (
    CollectiveOp,
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    Reduce,
    AllToAll,
    Scatter,
    Gather,
)

from .p2p import (
    Send,
    Recv,
    SendAsync,
    RecvAsync,
)

from .async_ops import (
    Handle,
    Stream,
    DEFAULT_STREAM,
    COMM_STREAM,
    COMPUTE_STREAM,
    AllReduceAsync,
    Wait,
    WaitAll,
    OverlapRegion,
)

from .shape import (
    Reshape,
    Transpose,
)

from .spmd import (
    Reinterpret,
    Convert,
    SPMDGuard,
)

from .precision import (
    Cast,
    LossScale,
    DtypeGuard,
)

from .zero import (
    ZeROGatherParam,
    ZeROScatterGrad,
    ZeROPartitionOptState,
)

from .cp import (
    RingRotate,
    RingAttentionStep,
    RingAttention,
)

from .moe import (
    TopKGate,
    MoEDispatch,
    MoECombine,
    ExpertCompute,
)

from .program import (
    Program,
    ir_to_str,
)

__all__ = [
    # base
    "IROp", "SPMDConsistencyError",
    # compute
    "ElementWiseBinaryOp", "MatMul", "Add", "Multiply", "SiLU", "FlashAttention",
    # collective
    "CollectiveOp", "AllReduce", "AllGather", "ReduceScatter",
    "Broadcast", "Reduce", "AllToAll", "Scatter", "Gather",
    # p2p
    "Send", "Recv", "SendAsync", "RecvAsync",
    # async
    "Handle", "Stream", "DEFAULT_STREAM", "COMM_STREAM", "COMPUTE_STREAM",
    "AllReduceAsync", "Wait", "WaitAll", "OverlapRegion",
    # shape
    "Reshape", "Transpose",
    # spmd
    "Reinterpret", "Convert", "SPMDGuard",
    # precision
    "Cast", "LossScale", "DtypeGuard",
    # zero
    "ZeROGatherParam", "ZeROScatterGrad", "ZeROPartitionOptState",
    # cp
    "RingRotate", "RingAttentionStep", "RingAttention",
    # moe
    "TopKGate", "MoEDispatch", "MoECombine", "ExpertCompute",
    # program
    "Program", "ir_to_str",
]
