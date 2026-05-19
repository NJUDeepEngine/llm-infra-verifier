"""Comprehensive tests for MultiDeviceExecutor dispatch and state management.

Covers: per-device execution, P2P communication, async ops, collective ops,
shape ops, slice propagation, state management APIs, and real LLM training
scenarios (TP, PP, DP, compute-comm overlap).
"""

import pytest
import copy

from verifier.state import (
    TensorState,
    ShardingSpec,
    DeviceMesh,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
    TensorSlice,
)
from verifier.ir import (
    MatMul, Add, Multiply, SiLU, FlashAttention,
    AllReduce, AllGather, ReduceScatter,
    Send, Recv, SendAsync, RecvAsync,
    AllReduceAsync, Wait, WaitAll, OverlapRegion,
    Reshape, Transpose,
    Reinterpret, Convert,
    Cast, LossScale, FP8Quantize, FP8Dequantize, AmaxUpdate,
    RingRotate, RingAttentionStep, RingAttention,
    Program,
)
from verifier.executor import MultiDeviceExecutor, SliceRule


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mesh(n, dim_name="tp"):
    return DeviceMesh(shape=(n,), dim_names=(dim_name,))


def _mesh2d(tp=2, dp=4):
    return DeviceMesh(shape=(tp, dp), dim_names=("tp", "dp"))


def _make_tensor(name, shape, spec, dtype=None, expr="", requires_grad=False):
    local = compute_local_shape(shape, spec)
    return TensorState(
        name=name, global_shape=shape, local_shape=local,
        sharding=spec, dtype=dtype, expr=expr, requires_grad=requires_grad,
    )


def _rep(mesh):
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape), mesh=mesh)


def _shard(mesh, dim=0, mesh_dim=0):
    placements = [Replicate() for _ in mesh.shape]
    placements[mesh_dim] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _partial(mesh, mesh_dim=0):
    placements = [Replicate() for _ in mesh.shape]
    placements[mesh_dim] = Partial()
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


# ── P2P Communication ────────────────────────────────────────────────────────


class TestP2PExecution:
    """Tests for Send/Recv execution paths (pipeline parallelism)."""

    def test_send_moves_tensor_to_dst(self):
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("act", (8, 16), spec, expr="activation")
        exe.register_tensor(t)

        prog = Program(name="pp_send")
        prog.add(Send(x="act", output="act_sent", src=0, dst=1, stage=1, microbatch_id=0))
        exe.run_program(prog)

        sent = exe.get_tensor("act_sent", device_id=1)
        assert sent is not None
        assert sent.global_shape == (8, 16)
        assert sent.stage == 1
        assert sent.microbatch_id == 0

    def test_send_not_on_src_raises(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        prog = Program(name="bad_send")
        prog.add(Send(x="missing", output="out", src=0, dst=1, stage=1, microbatch_id=0))
        with pytest.raises(ValueError, match="not found on source device"):
            exe.run_program(prog)

    def test_recv_after_send(self):
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec, expr="x_data")
        exe.register_tensor(t)

        prog = Program(name="pp_send_recv")
        prog.add(Send(x="x", output="x_transit", src=0, dst=2, stage=1, microbatch_id=0))
        prog.add(Recv(x="x_transit", output="x_received", src=0, dst=2, stage=2, microbatch_id=0))
        exe.run_program(prog)

        received = exe.get_tensor("x_received", device_id=2)
        assert received is not None
        assert received.global_shape == (4, 8)
        assert received.stage == 2

    def test_recv_without_send_raises(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        prog = Program(name="bad_recv")
        prog.add(Recv(x="ghost", output="out", src=0, dst=1, stage=1, microbatch_id=0))
        with pytest.raises(ValueError, match="no matching Send"):
            exe.run_program(prog)

    def test_pipeline_two_stages(self):
        """2-stage pipeline: stage0 computes, sends to stage1."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)

        x = _make_tensor("x", (4, 8), spec, expr="input")
        w0 = _make_tensor("w0", (8, 16), spec, expr="weight0")
        exe.register_tensor(x)
        exe.register_tensor(w0)

        prog = Program(name="2stage_pp")
        prog.add(MatMul(a="x", b="w0", output="h0"))
        prog.add(Send(x="h0", output="h0_sent", src=0, dst=1, stage=1, microbatch_id=0))
        exe.run_program(prog)

        h0_on_dev1 = exe.get_tensor("h0_sent", device_id=1)
        assert h0_on_dev1 is not None
        assert h0_on_dev1.global_shape == (4, 16)


# ── Async Operations ─────────────────────────────────────────────────────────


class TestAsyncExecution:
    """Tests for async ops: AllReduceAsync, Wait, SendAsync, RecvAsync, OverlapRegion."""

    def test_allreduce_async_then_wait(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _partial(mesh)
        t = _make_tensor("x", (4, 8), spec, expr="partial_sum")
        exe.register_tensor(t)

        prog = Program(name="async_ar")
        prog.add(AllReduceAsync(x="x", output="x_async", handle="h0"))
        prog.add(Wait(handle="h0", tensor="x_async", output="x_done"))
        exe.run_program(prog)

        for did in range(2):
            result = exe.get_tensor("x_done", did)
            assert result is not None
            assert result.global_shape == (4, 8)

    def test_send_async_recv_async(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("act", (4, 8), spec, dtype="bf16", expr="act")
        exe.register_tensor(t)

        prog = Program(name="async_p2p")
        prog.add(SendAsync(
            x="act", output="act_inflight", handle="h1",
            src=0, dst=1, stage=0, microbatch_id=0,
        ))
        exe.run_program(prog)

        inflight = exe.get_tensor("act_inflight", device_id=1)
        assert inflight is not None
        assert inflight.dtype == "bf16"

    def test_overlap_region(self):
        """Compute-comm overlap: matmul on compute stream, allreduce on comm stream."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec_s = _shard(mesh, dim=1)
        spec_p = _partial(mesh)

        x = _make_tensor("x", (4, 8), _rep(mesh), expr="x")
        w = _make_tensor("w", (8, 16), spec_s, expr="w")
        p = _make_tensor("prev_partial", (4, 4), spec_p, expr="prev")
        exe.register_tensor(x)
        exe.register_tensor(w)
        exe.register_tensor(p)

        overlap = OverlapRegion(
            compute_ops=[MatMul(a="x", b="w", output="y")],
            comm_ops=[AllReduce(x="prev_partial", output="prev_reduced")],
        )
        prog = Program(name="overlap")
        prog.add(overlap)
        exe.run_program(prog)

        for did in range(2):
            y = exe.get_tensor("y", did)
            assert y is not None
            reduced = exe.get_tensor("prev_reduced", did)
            assert reduced is not None

    def test_wait_all(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _partial(mesh)
        t1 = _make_tensor("a", (4, 8), spec, expr="a")
        t2 = _make_tensor("b", (4, 8), spec, expr="b")
        exe.register_tensor(t1)
        exe.register_tensor(t2)

        prog = Program(name="wait_all")
        prog.add(AllReduceAsync(x="a", output="a_async", handle="h0"))
        prog.add(AllReduceAsync(x="b", output="b_async", handle="h1"))
        prog.add(WaitAll(
            handles=("h0", "h1"),
            tensors=("a_async", "b_async"),
            outputs=("a_done", "b_done"),
        ))
        exe.run_program(prog)

        for did in range(2):
            a_done = exe.get_tensor("a_done", did)
            assert a_done is not None


# ── Collective Operations ────────────────────────────────────────────────────


class TestCollectiveExecution:
    """Tests for AllGather, ReduceScatter, AllReduce through executor."""

    def test_allgather_shard_to_replicate(self):
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (16, 8), spec, expr="sharded_x")
        exe.register_tensor(t)

        prog = Program(name="ag")
        prog.add(AllGather(x="x", output="x_full", gather_dim=0))
        exe.run_program(prog)

        for did in range(4):
            result = exe.get_tensor("x_full", did)
            assert result is not None
            assert result.local_shape == (16, 8)
            assert any(isinstance(p, Replicate) for p in result.sharding.placements)

    def test_reducescatter_replicate_to_shard(self):
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (16, 8), spec, expr="full_x")
        exe.register_tensor(t)

        prog = Program(name="rs")
        prog.add(ReduceScatter(x="x", output="x_shard", scatter_dim=0))
        exe.run_program(prog)

        for did in range(4):
            result = exe.get_tensor("x_shard", did)
            assert result is not None
            assert result.local_shape == (4, 8)

    def test_allreduce_partial_to_replicate(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _partial(mesh)
        t = _make_tensor("grad", (8, 16), spec, expr="partial_grad")
        exe.register_tensor(t)

        prog = Program(name="ar")
        prog.add(AllReduce(x="grad", output="grad_full"))
        exe.run_program(prog)

        for did in range(2):
            result = exe.get_tensor("grad_full", did)
            assert result is not None
            assert not result.partial

    def test_allgather_then_reducescatter_roundtrip(self):
        """AllGather followed by ReduceScatter should return to original sharding."""
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (16, 8), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="ag_rs")
        prog.add(AllGather(x="x", output="x_full", gather_dim=0))
        prog.add(ReduceScatter(x="x_full", output="x_back", scatter_dim=0))
        exe.run_program(prog)

        for did in range(4):
            result = exe.get_tensor("x_back", did)
            assert result is not None
            assert result.local_shape == (4, 8)


# ── Shape Operations ──────────────────────────────────────────────────────────


class TestShapeExecution:
    """Tests for Reshape and Transpose through executor."""

    def test_reshape_on_mesh(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (8, 16), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="reshape")
        prog.add(Reshape(x="x", output="y", new_shape=(8, 4, 4)))
        exe.run_program(prog)

        for did in range(2):
            result = exe.get_tensor("y", did)
            assert result is not None
            assert result.global_shape == (8, 4, 4)

    def test_transpose_on_mesh(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (8, 16), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="transpose")
        prog.add(Transpose(x="x", output="y", dim0=0, dim1=1))
        exe.run_program(prog)

        for did in range(2):
            result = exe.get_tensor("y", did)
            assert result is not None
            assert result.global_shape == (16, 8)

    def test_reshape_preserves_dtype(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec, dtype="bf16", expr="x")
        exe.register_tensor(t)

        prog = Program(name="reshape_dtype")
        prog.add(Reshape(x="x", output="y", new_shape=(2, 2, 8)))
        exe.run_program(prog)

        result = exe.get_tensor("y", 0)
        assert result.dtype == "bf16"


# ── Slice Propagation ────────────────────────────────────────────────────────


class TestSlicePropagation:
    """Tests for slice rule correctness after executor refactor."""

    def test_matmul_slice_rule(self):
        """MatMul: output slice = (A.row_offset, B.col_offset)."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec_s0 = _shard(mesh, dim=0)
        spec_s1 = _shard(mesh, dim=1)

        x = _make_tensor("x", (8, 16), spec_s0, expr="x")
        w = _make_tensor("w", (16, 32), spec_s1, expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="mm_slice")
        prog.add(MatMul(a="x", b="w", output="y"))
        exe.run_program(prog)

        slices = exe.final_slices()
        s0 = slices[0].get("y")
        s1 = slices[1].get("y")
        assert s0 is not None
        assert s0.offsets[0] == 0
        assert s1 is not None
        assert s1.offsets[0] == 4  # second half of rows

    def test_allreduce_zero_offset_slice(self):
        """AllReduce: output slice has all-zero offsets (full tensor on each device)."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _partial(mesh)
        t = _make_tensor("x", (4, 8), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="ar_slice")
        prog.add(AllReduce(x="x", output="y"))
        exe.run_program(prog)

        slices = exe.final_slices()
        for did in range(2):
            s = slices[did].get("y")
            assert s is not None
            assert s.offsets == (0, 0)

    def test_unary_inherits_first_input_slice(self):
        """Unary ops (SiLU, Cast, etc.) inherit slice from input."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (8, 16), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="unary_slice")
        prog.add(SiLU(x="x", output="y"))
        exe.run_program(prog)

        slices = exe.final_slices()
        sx = slices[0].get("x")
        sy = slices[0].get("y")
        assert sx is not None and sy is not None
        assert sy.offsets == sx.offsets

    def test_elementwise_inherit_any_slice(self):
        """Element-wise ops inherit slice from whichever input has one."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec_s = _shard(mesh, dim=0)
        spec_r = _rep(mesh)

        a = _make_tensor("a", (8, 16), spec_s, expr="a")
        b = _make_tensor("b", (8, 16), spec_r, expr="b")
        exe.register_tensor(a)
        exe.register_tensor(b)

        prog = Program(name="ew_slice")
        prog.add(Add(a="a", b="b", output="c"))
        exe.run_program(prog)

        slices = exe.final_slices()
        sa = slices[0].get("a")
        sc = slices[0].get("c")
        assert sa is not None and sc is not None
        assert sc.offsets == sa.offsets


# ── State Management API ─────────────────────────────────────────────────────


class TestStateManagement:
    """Tests for executor state management methods."""

    def test_get_all_devices_tensor(self):
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (16, 8), spec, expr="x")
        exe.register_tensor(t)

        all_tensors = exe.get_all_devices_tensor("x")
        assert len(all_tensors) == 4
        for did, ts in all_tensors.items():
            assert ts.name == "x"
            assert ts.local_shape == (4, 8)

    def test_get_all_devices_tensor_missing(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        result = exe.get_all_devices_tensor("nonexistent")
        assert result == {}

    def test_reset_devices(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec)
        exe.register_tensor(t)

        assert exe.get_tensor("x", 0) is not None
        exe.reset_devices()
        assert exe.get_tensor("x", 0) is None

    def test_state_snapshot(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="snap")
        prog.add(SiLU(x="x", output="y"))
        prog.add(SiLU(x="y", output="z"))
        exe.run_program(prog)

        snap0 = exe.state_snapshot(0)
        assert "y" in snap0[0]
        assert "z" not in snap0[0]

        snap1 = exe.state_snapshot(1)
        assert "z" in snap1[0]

    def test_state_snapshot_out_of_range(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        with pytest.raises(IndexError):
            exe.state_snapshot(0)

    def test_op_history(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec, expr="x")
        exe.register_tensor(t)

        prog = Program(name="hist")
        prog.add(SiLU(x="x", output="y"))
        prog.add(SiLU(x="y", output="z"))
        exe.run_program(prog)

        assert len(exe.op_history) == 2
        assert isinstance(exe.op_history[0], SiLU)
        assert isinstance(exe.op_history[1], SiLU)

    def test_final_state(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _rep(mesh)
        t = _make_tensor("x", (4, 8), spec)
        exe.register_tensor(t)

        prog = Program(name="fs")
        prog.add(SiLU(x="x", output="y"))
        exe.run_program(prog)

        fs = exe.final_state()
        assert 0 in fs and 1 in fs
        assert "y" in fs[0]
        assert "y" in fs[1]

    def test_final_slices(self):
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)
        spec = _shard(mesh, dim=0)
        t = _make_tensor("x", (8, 16), spec)
        exe.register_tensor(t)

        slices = exe.final_slices()
        assert 0 in slices and 1 in slices
        assert "x" in slices[0]


# ── Real LLM Training Scenarios ──────────────────────────────────────────────


class TestLLMScenarios:
    """End-to-end tests modeling real distributed LLM training patterns."""

    def test_megatron_column_parallel_linear(self):
        """Megatron column-parallel: weight sharded on dim=1, output is Shard(1)."""
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (32, 4096), _rep(mesh), expr="x")
        w = _make_tensor("w", (4096, 16384), _shard(mesh, dim=1), expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="col_parallel")
        prog.add(MatMul(a="x", b="w", output="h"))
        exe.run_program(prog)

        for did in range(4):
            h = exe.get_tensor("h", did)
            assert h is not None
            assert h.local_shape == (32, 4096)
            assert h.global_shape == (32, 16384)

    def test_megatron_row_parallel_linear(self):
        """Megatron row-parallel: weight sharded on dim=0, output is Partial, needs AllReduce."""
        mesh = _mesh(4)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (32, 16384), _shard(mesh, dim=1), expr="x")
        w = _make_tensor("w", (16384, 4096), _shard(mesh, dim=0), expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="row_parallel")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        prog.add(AllReduce(x="y_partial", output="y"))
        exe.run_program(prog)

        for did in range(4):
            y = exe.get_tensor("y", did)
            assert y is not None
            assert y.local_shape == (32, 4096)
            assert not y.partial

    def test_swiglu_mlp_block(self):
        """SwiGLU MLP: gate_proj(S1) -> SiLU -> up_proj(S1) -> multiply -> down_proj(S0) -> AllReduce."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (8, 512), _rep(mesh), expr="x")
        w_gate = _make_tensor("w_gate", (512, 1024), _shard(mesh, dim=1), expr="wg")
        w_up = _make_tensor("w_up", (512, 1024), _shard(mesh, dim=1), expr="wu")
        w_down = _make_tensor("w_down", (1024, 512), _shard(mesh, dim=0), expr="wd")
        exe.register_tensor(x)
        exe.register_tensor(w_gate)
        exe.register_tensor(w_up)
        exe.register_tensor(w_down)

        prog = Program(name="swiglu")
        prog.add(MatMul(a="x", b="w_gate", output="gate_raw"))
        prog.add(SiLU(x="gate_raw", output="gate"))
        prog.add(MatMul(a="x", b="w_up", output="up"))
        prog.add(Multiply(a="gate", b="up", output="h"))
        prog.add(MatMul(a="h", b="w_down", output="y_partial"))
        prog.add(AllReduce(x="y_partial", output="y"))
        exe.run_program(prog)

        for did in range(2):
            y = exe.get_tensor("y", did)
            assert y is not None
            assert y.global_shape == (8, 512)
            assert not y.partial

    def test_data_parallel_allreduce_gradients(self):
        """Data parallelism: each device has full model, AllReduce gradients."""
        mesh = _mesh(4, dim_name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        grad = _make_tensor("grad_w", (1024, 1024), _partial(mesh), dtype="fp32", expr="grad")
        exe.register_tensor(grad)

        prog = Program(name="dp_allreduce")
        prog.add(AllReduce(x="grad_w", output="grad_w_synced"))
        exe.run_program(prog)

        for did in range(4):
            synced = exe.get_tensor("grad_w_synced", did)
            assert synced is not None
            assert not synced.partial
            assert synced.dtype == "fp32"

    def test_zero_stage3_fwd_gather_bwd_scatter(self):
        """ZeRO-3: gather params before fwd, scatter grads after bwd."""
        from verifier.ir import ZeROGatherParam, ZeROScatterGrad

        mesh = _mesh(4, dim_name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        w_shard = _make_tensor("w", (4096, 4096), _shard(mesh, dim=0, mesh_dim=0), expr="w")
        exe.register_tensor(w_shard)

        prog = Program(name="zero3")
        prog.add(ZeROGatherParam(x="w", output="w_full", gather_dim=0))
        exe.run_program(prog)

        for did in range(4):
            w_full = exe.get_tensor("w_full", did)
            assert w_full is not None
            assert w_full.local_shape == (4096, 4096)

    def test_fp8_forward_pipeline(self):
        """FP8 training: quantize -> matmul -> dequantize -> amax update."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (32, 4096), _rep(mesh), dtype="fp32", expr="x")
        w = _make_tensor("w", (4096, 4096), _shard(mesh, dim=1), dtype="fp32", expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="fp8_fwd")
        prog.add(FP8Quantize(x="x", output="x_q", scale_expr="sx",
                             src_dtype="fp32", dst_dtype="fp8e4m3"))
        prog.add(FP8Quantize(x="w", output="w_q", scale_expr="sw",
                             src_dtype="fp32", dst_dtype="fp8e4m3"))
        prog.add(MatMul(a="x_q", b="w_q", output="y_q"))
        prog.add(FP8Dequantize(x="y_q", output="y", scale_expr="sy",
                               src_dtype="fp8e4m3", dst_dtype="fp32"))
        prog.add(AmaxUpdate(x="x_q", output="amax_x", tensor_name="x"))
        exe.run_program(prog)

        for did in range(2):
            y = exe.get_tensor("y", did)
            assert y is not None
            assert y.dtype == "fp32"
            assert y.fp8_scale_expr is None

            amax = exe.get_tensor("amax_x", did)
            assert amax is not None
            assert amax.dtype == "fp32"
            assert amax.global_shape == (1,)

    def test_tp_attention_block(self):
        """TP attention: QKV column-parallel -> FlashAttention -> output row-parallel."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (8, 512), _rep(mesh), expr="x")
        wq = _make_tensor("wq", (512, 512), _shard(mesh, dim=1), expr="wq")
        wk = _make_tensor("wk", (512, 512), _shard(mesh, dim=1), expr="wk")
        wv = _make_tensor("wv", (512, 512), _shard(mesh, dim=1), expr="wv")
        wo = _make_tensor("wo", (512, 512), _shard(mesh, dim=0), expr="wo")
        exe.register_tensor(x)
        exe.register_tensor(wq)
        exe.register_tensor(wk)
        exe.register_tensor(wv)
        exe.register_tensor(wo)

        prog = Program(name="tp_attn")
        prog.add(MatMul(a="x", b="wq", output="q"))
        prog.add(MatMul(a="x", b="wk", output="k"))
        prog.add(MatMul(a="x", b="wv", output="v"))
        prog.add(FlashAttention(q="q", k="k", v="v", output="attn_out"))
        prog.add(MatMul(a="attn_out", b="wo", output="o_partial"))
        prog.add(AllReduce(x="o_partial", output="o"))
        exe.run_program(prog)

        for did in range(2):
            o = exe.get_tensor("o", did)
            assert o is not None
            assert o.global_shape == (8, 512)
            assert not o.partial

    def test_2d_mesh_tp_dp(self):
        """2D mesh: TP on dim 0, DP on dim 1. Weight sharded on TP, grad reduced on DP."""
        mesh = _mesh2d(tp=2, dp=2)
        exe = MultiDeviceExecutor(mesh=mesh)

        spec_tp_shard = ShardingSpec(
            placements=(Shard(dim=1), Replicate()), mesh=mesh
        )
        spec_rep = _rep(mesh)

        x = _make_tensor("x", (8, 512), spec_rep, expr="x")
        w = _make_tensor("w", (512, 1024), spec_tp_shard, expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="2d_mesh")
        prog.add(MatMul(a="x", b="w", output="h"))
        exe.run_program(prog)

        for did in range(4):
            h = exe.get_tensor("h", did)
            assert h is not None
            assert h.global_shape == (8, 1024)

    def test_mixed_precision_training_loop(self):
        """Mixed precision: cast to fp16, compute, cast back, loss scale."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make_tensor("x", (8, 512), _rep(mesh), dtype="fp32", expr="x")
        w = _make_tensor("w", (512, 512), _shard(mesh, dim=1), dtype="fp32", expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="amp")
        prog.add(Cast(x="x", output="x16", src_dtype="fp32", dst_dtype="fp16"))
        prog.add(Cast(x="w", output="w16", src_dtype="fp32", dst_dtype="fp16"))
        prog.add(MatMul(a="x16", b="w16", output="y16"))
        prog.add(Cast(x="y16", output="y32", src_dtype="fp16", dst_dtype="fp32"))
        prog.add(LossScale(x="y32", output="y_scaled", scale=65536.0, direction="scale"))
        exe.run_program(prog)

        for did in range(2):
            y = exe.get_tensor("y_scaled", did)
            assert y is not None
            assert y.dtype == "fp32"

    def test_pipeline_parallel_microbatches(self):
        """PP with 2 microbatches on 2 stages: interleaved Send/Recv."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        mb0 = _make_tensor("mb0", (4, 8), _rep(mesh), expr="mb0")
        mb1 = _make_tensor("mb1", (4, 8), _rep(mesh), expr="mb1")
        w0 = _make_tensor("w0", (8, 16), _rep(mesh), expr="w0")
        exe.register_tensor(mb0)
        exe.register_tensor(mb1)
        exe.register_tensor(w0)

        prog = Program(name="pp_1f1b")
        # Stage 0: compute mb0, send to stage 1
        prog.add(MatMul(a="mb0", b="w0", output="h0_mb0"))
        prog.add(Send(x="h0_mb0", output="h0_mb0_sent", src=0, dst=1, stage=1, microbatch_id=0))
        # Stage 0: compute mb1, send to stage 1
        prog.add(MatMul(a="mb1", b="w0", output="h0_mb1"))
        prog.add(Send(x="h0_mb1", output="h0_mb1_sent", src=0, dst=1, stage=1, microbatch_id=1))
        exe.run_program(prog)

        # Both microbatches arrived at device 1
        h0 = exe.get_tensor("h0_mb0_sent", device_id=1)
        h1 = exe.get_tensor("h0_mb1_sent", device_id=1)
        assert h0 is not None and h0.microbatch_id == 0
        assert h1 is not None and h1.microbatch_id == 1
        assert h0.global_shape == (4, 16)


# ── Strict mode tests ──────────────────────────────────────────────────────────


class TestStrictMode:
    """Strict mode raises on missing inputs instead of warn-and-skip."""

    def _make_executor(self, strict: bool):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        exe = MultiDeviceExecutor(mesh, strict=strict)
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(8, 16),
            sharding=spec, expr="x",
        )
        exe.register_tensor(x)
        return exe

    def test_strict_raises_on_missing_input(self):
        """strict=True raises ValueError when an op's input is missing."""
        exe = self._make_executor(strict=True)
        prog = Program("bad", ops=[
            MatMul(a="x", b="w_missing", output="y"),
        ])
        with pytest.raises(ValueError, match="w_missing"):
            exe.run_program(prog)

    def test_loose_warns_on_missing_input(self):
        """strict=False (default) warns and skips, no exception."""
        exe = self._make_executor(strict=False)
        prog = Program("bad", ops=[
            MatMul(a="x", b="w_missing", output="y"),
        ])
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = exe.run_program(prog)
            assert any("w_missing" in str(warning.message) for warning in w)
        assert "y" not in state

    def test_strict_cascading_missing(self):
        """strict=True catches the first missing input in a chain."""
        exe = self._make_executor(strict=True)
        prog = Program("chain", ops=[
            MatMul(a="x", b="w_missing", output="y"),
            Add(a="y", b="x", output="z"),
        ])
        with pytest.raises(ValueError, match="w_missing"):
            exe.run_program(prog)

