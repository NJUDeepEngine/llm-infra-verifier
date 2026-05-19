"""Tests for the TIR lifter: TIR blocks → distributed IR → verification.

Covers block classification, matmul/elementwise/attention lifting,
backward generation, and end-to-end lift+execute+verify_all().
"""

import warnings

import pytest

from verifier.state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
)
from verifier.ir import (
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    AllReduce,
    AllGather,
    FlashAttention,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier
from verifier.temporal import verify_temporal
from verifier.tir_lifter import (
    TIRLifter,
    TIRFunc,
    TIRBlock,
    TIRBlockAxis,
    TIRBufferRegion,
    TIRVar,
    TIRGrid,
    LiftResult,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_tp_mesh(tp_size=2):
    return DeviceMesh(shape=(tp_size,), dim_names=("tp",))


def make_matmul_block(a, b, y, B, H, O):
    """Standard matmul block: Y[i,j] += A[i,k] * B[k,j]."""
    i = TIRVar("i")
    j = TIRVar("j")
    k = TIRVar("k")
    return TIRBlock(
        name="matmul",
        axes=[
            TIRBlockAxis(var=i, type="S", extent=B),
            TIRBlockAxis(var=j, type="S", extent=O),
            TIRBlockAxis(var=k, type="R", extent=H),
        ],
        reads=[
            TIRBufferRegion(buffer=a, indices=["i", "k"]),
            TIRBufferRegion(buffer=b, indices=["k", "j"]),
        ],
        writes=[TIRBufferRegion(buffer=y, indices=["i", "j"])],
        body=f"{y}[i,j] += {a}[i,k] * {b}[k,j]",
    )


def make_elementwise_block(x, y, body="silu"):
    """Elementwise block: Y[i,j] = f(X[i,j])."""
    i = TIRVar("i")
    j = TIRVar("j")
    return TIRBlock(
        name="elementwise",
        axes=[
            TIRBlockAxis(var=i, type="S", extent=8),
            TIRBlockAxis(var=j, type="S", extent=32),
        ],
        reads=[TIRBufferRegion(buffer=x, indices=["i", "j"])],
        writes=[TIRBufferRegion(buffer=y, indices=["i", "j"])],
        body=body,
    )


def make_binary_block(a, b, y, body="+"):
    """Binary elementwise block: Y[i,j] = A[i,j] op B[i,j]."""
    i = TIRVar("i")
    j = TIRVar("j")
    return TIRBlock(
        name="binary",
        axes=[
            TIRBlockAxis(var=i, type="S", extent=8),
            TIRBlockAxis(var=j, type="S", extent=32),
        ],
        reads=[
            TIRBufferRegion(buffer=a, indices=["i", "j"]),
            TIRBufferRegion(buffer=b, indices=["i", "j"]),
        ],
        writes=[TIRBufferRegion(buffer=y, indices=["i", "j"])],
        body=f"{y}[i,j] = {a}[i,j] {body} {b}[i,j]",
    )


def make_attention_block(q, k, v, o, B=8, S=16, D=16):
    """Attention block: O = softmax(Q@K^T) @ V."""
    i = TIRVar("i")
    j = TIRVar("j")
    d = TIRVar("d")
    return TIRBlock(
        name="attention",
        axes=[
            TIRBlockAxis(var=i, type="S", extent=B),
            TIRBlockAxis(var=j, type="S", extent=S),
            TIRBlockAxis(var=d, type="R", extent=D),
        ],
        reads=[
            TIRBufferRegion(buffer=q, indices=["i", "j"]),
            TIRBufferRegion(buffer=k, indices=["i", "j"]),
            TIRBufferRegion(buffer=v, indices=["i", "j"]),
        ],
        writes=[TIRBufferRegion(buffer=o, indices=["i", "j"])],
        body=f"{o} = softmax({q} @ {k}^T) @ {v}",
    )


def make_lifter(mesh, specs):
    return TIRLifter(sharding_specs=specs)


# ── Block Classification ───────────────────────────────────────────────────


class TestBlockClassification:
    def test_matmul(self):
        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        mesh = make_tp_mesh()
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        lifter = TIRLifter(sharding_specs={"X": spec})
        assert lifter._classify_block(block) == "matmul"

    def test_elementwise(self):
        block = make_elementwise_block("X", "Y", body="silu")
        mesh = make_tp_mesh()
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        lifter = TIRLifter(sharding_specs={"X": spec})
        assert lifter._classify_block(block) == "elementwise"

    def test_attention(self):
        block = make_attention_block("Q", "K", "V", "O")
        mesh = make_tp_mesh()
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        lifter = TIRLifter(sharding_specs={"Q": spec})
        assert lifter._classify_block(block) == "attention"

    def test_generic(self):
        """Block with 3 non-QKV reads and 2 reduce axes → generic."""
        i = TIRVar("i")
        j = TIRVar("j")
        k1 = TIRVar("k1")
        k2 = TIRVar("k2")
        block = TIRBlock(
            name="generic",
            axes=[
                TIRBlockAxis(var=i, type="S", extent=8),
                TIRBlockAxis(var=j, type="S", extent=8),
                TIRBlockAxis(var=k1, type="R", extent=16),
                TIRBlockAxis(var=k2, type="R", extent=16),
            ],
            reads=[
                TIRBufferRegion(buffer="A", indices=["i", "k1"]),
                TIRBufferRegion(buffer="B", indices=["k1", "k2"]),
                TIRBufferRegion(buffer="C", indices=["k2", "j"]),
            ],
            writes=[TIRBufferRegion(buffer="D", indices=["i", "j"])],
        )
        mesh = make_tp_mesh()
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        lifter = TIRLifter(sharding_specs={"A": spec})
        assert lifter._classify_block(block) == "generic"


# ── Matmul Lifting ──────────────────────────────────────────────────────────


class TestMatmulLifting:
    def test_sharded_reduce_inserts_allreduce(self):
        """Row parallel: X Shard(1), W Shard(0) → AllReduce inserted."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": s1, "W": s0})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 1
        assert isinstance(result.collectives_inserted[0], AllReduce)
        assert len(result.fwd_program.ops) >= 2

        y = result.tensors.get("Y")
        assert y is not None
        assert not y.partial

    def test_column_parallel_no_allreduce(self):
        """Column parallel: X Replicate, W Shard(1) → no AllReduce."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": rep, "W": s1})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 0
        assert len(result.fwd_program.ops) == 1
        assert isinstance(result.fwd_program.ops[0], MatMul)

    def test_both_replicate_no_collective(self):
        """Both Replicate → no collective, output Replicate."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": rep, "W": rep})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 0
        y = result.tensors.get("Y")
        assert y is not None
        assert not y.partial


# ── Elementwise Lifting ─────────────────────────────────────────────────────


class TestElementwiseLifting:
    def test_silu(self):
        """Unary SiLU block → SiLU op, preserves placement, no collective."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

        block = make_elementwise_block("X", "Y", body="silu(X)")
        tir_func = TIRFunc(
            name="silu",
            buffers={"X": (8, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": s1})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 0
        assert len(result.fwd_program.ops) == 1
        assert isinstance(result.fwd_program.ops[0], SiLU)

    def test_binary_add(self):
        """Binary add block → Add op, no collective."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

        block = make_binary_block("A", "B", "Y", body="+")
        tir_func = TIRFunc(
            name="add",
            buffers={"A": (8, 32), "B": (8, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"A": s1, "B": s1})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 0
        assert len(result.fwd_program.ops) == 1
        assert isinstance(result.fwd_program.ops[0], Add)


# ── Attention Lifting ───────────────────────────────────────────────────────


class TestAttentionLifting:
    def test_no_ring_all_replicate(self):
        """Q/K/V all Replicate → FlashAttention, no AllReduce."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        block = make_attention_block("Q", "K", "V", "O", B=8, S=16, D=16)
        tir_func = TIRFunc(
            name="attn",
            buffers={"Q": (8, 16), "K": (8, 16), "V": (8, 16), "O": (8, 16)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"Q": rep, "K": rep, "V": rep})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 0
        fa_ops = [op for op in result.fwd_program.ops if isinstance(op, FlashAttention)]
        assert len(fa_ops) == 1

    def test_cp_ring_k_sharded(self):
        """K Shard(1), Q Replicate → FlashAttention + AllReduce (CP ring)."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

        block = make_attention_block("Q", "K", "V", "O", B=8, S=16, D=16)
        tir_func = TIRFunc(
            name="attn_cp",
            buffers={"Q": (8, 16), "K": (8, 16), "V": (8, 16), "O": (8, 16)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"Q": rep, "K": s1, "V": s1})
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 1
        assert isinstance(result.collectives_inserted[0], AllReduce)


# ── Backward Generation ────────────────────────────────────────────────────


class TestBackwardGeneration:
    def test_matmul_allreduce_backward_has_dual(self):
        """Forward AllReduce → backward also has AllReduce (self-dual)."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": s1, "W": s0})
        result = lifter.lift(tir_func)

        bwd_allreduce = [op for op in result.bwd_program.ops if isinstance(op, AllReduce)]
        assert len(bwd_allreduce) >= 1


# ── E2E: Lift → Execute → Verify ───────────────────────────────────────────


class TestE2ELiftAndVerify:
    def test_linear_lift_execute_verify_all(self):
        """TIR matmul → lift → executor → verify_all() all pass."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": s1, "W": s0})
        result = lifter.lift(tir_func)

        executor = MultiDeviceExecutor(mesh)
        for ts in result.tensors.values():
            executor.register_tensor(ts)
        state = executor.run_program(result.fwd_program)

        y = state.get("Y")
        assert y is not None
        assert not y.partial

        verifier = DistributedVerifier()
        results = verifier.verify_all(result.fwd_program, state)
        for vr in results:
            assert vr.passed, f"{vr.condition} failed: {vr.details}"

    def test_mlp_multi_block(self):
        """Multi-block MLP: matmul+silu+matmul+multiply+matmul → 1 AllReduce."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        blocks = [
            make_matmul_block("X", "W_gate", "gate_raw", 8, 16, 32),
            make_elementwise_block("gate_raw", "gate", body="silu(gate_raw)"),
            make_matmul_block("X", "W_up", "up", 8, 16, 32),
            make_binary_block("gate", "up", "h"),
            make_matmul_block("h", "W_down", "Y", 8, 32, 16),
        ]
        tir_func = TIRFunc(
            name="mlp",
            buffers={
                "X": (8, 16), "W_gate": (16, 32), "gate_raw": (8, 32),
                "gate": (8, 32), "W_up": (16, 32), "up": (8, 32),
                "h": (8, 32), "W_down": (32, 16), "Y": (8, 16),
            },
            blocks=blocks,
        )
        lifter = TIRLifter(sharding_specs={
            "X": rep, "W_gate": s1, "W_up": s1, "W_down": s0,
        })
        result = lifter.lift(tir_func)

        assert len(result.collectives_inserted) == 1
        assert isinstance(result.collectives_inserted[0], AllReduce)

        executor = MultiDeviceExecutor(mesh)
        for ts in result.tensors.values():
            executor.register_tensor(ts)
        state = executor.run_program(result.fwd_program)

        y = state.get("Y")
        assert y is not None
        assert not y.partial

        verifier = DistributedVerifier()
        results = verifier.verify_all(result.fwd_program, state)
        for vr in results:
            assert vr.passed, f"{vr.condition} failed: {vr.details}"

    def test_lifted_program_temporal_safe(self):
        """Lifted program has no async ops → temporal safety trivially passes."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        block = make_matmul_block("X", "W", "Y", 8, 16, 32)
        tir_func = TIRFunc(
            name="linear",
            buffers={"X": (8, 16), "W": (16, 32), "Y": (8, 32)},
            blocks=[block],
        )
        lifter = TIRLifter(sharding_specs={"X": s1, "W": s0})
        result = lifter.lift(tir_func)

        tr = verify_temporal(result.fwd_program)
        assert tr.is_safe, f"Expected safe: {tr.summary()}"
