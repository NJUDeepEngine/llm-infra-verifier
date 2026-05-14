"""Tests for ZeRO extension: ZeROGatherParam, ZeROScatterGrad, ZeROPartitionOptState."""

import pytest

from verifier.state import (
    TensorState,
    LocalSPMDType,
    ShardingSpec,
    DeviceMesh,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
)
from verifier.ir import (
    ZeROGatherParam,
    ZeROScatterGrad,
    ZeROPartitionOptState,
    MatMul,
    Program,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine


def _mesh4():
    return DeviceMesh(shape=(4,), dim_names=("dp",))


def _mesh_2d():
    return DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"))


def _rep_spec(mesh=None):
    mesh = mesh or _mesh4()
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape), mesh=mesh)


def _shard_spec(dim=0, mesh=None):
    mesh = mesh or _mesh4()
    placements = [Replicate() for _ in mesh.shape]
    placements[0] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _partial_spec(mesh=None):
    mesh = mesh or _mesh4()
    placements = [Replicate() for _ in mesh.shape]
    placements[0] = Partial()
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _make_tensor(name, shape=(16, 8), spec=None, dtype=None, expr=""):
    spec = spec or _rep_spec()
    local = compute_local_shape(shape, spec)
    return TensorState(
        name=name, global_shape=shape, local_shape=local,
        sharding=spec, dtype=dtype, expr=expr,
    )


def _assert_grad_matches_fwd(grad: TensorState, fwd: TensorState):
    assert grad.global_shape == fwd.global_shape
    assert grad.local_shape == fwd.local_shape
    assert grad.sharding.placements == fwd.sharding.placements


# ── ZeROGatherParam ───────────────────────────────────────────────────────


class TestZeROGatherParam:
    def test_gather_shard_to_replicate(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("param", spec=spec)
        ctx = {"param": t}
        op = ZeROGatherParam(x="param", output="param_full", gather_dim=0)
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)
        assert result.global_shape == t.global_shape
        assert result.local_shape == t.global_shape  # fully gathered

    def test_gather_only_transforms_matching_dim(self):
        """Shard(dim=1) should NOT be gathered if gather_dim=0."""
        spec = _shard_spec(dim=1)
        t = _make_tensor("param", spec=spec)
        ctx = {"param": t}
        op = ZeROGatherParam(x="param", output="y", gather_dim=0)
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 1 for p in result.sharding.placements)

    def test_gather_stage_validation(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("param", spec=spec)
        ctx = {"param": t}
        op = ZeROGatherParam(x="param", output="y", gather_dim=0, zero_stage=2)
        with pytest.raises(ValueError, match="stage >= 3"):
            op.apply(ctx)

    def test_gather_is_collective(self):
        op = ZeROGatherParam(x="x", output="y", gather_dim=0)
        assert op.is_collective() is True

    def test_gather_vjp_shape_matches_fwd_input(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("param", spec=spec, expr="param")
        ctx = {"param": t}
        op = ZeROGatherParam(x="param", output="y", gather_dim=0)
        grad_out = _make_tensor("grad_y")
        grads = op.vjp(ctx, grad_out)
        _assert_grad_matches_fwd(grads["param"], t)
        assert "ZeROScatterGrad" in grads["param"].expr

    def test_gather_clone_with_names(self):
        op = ZeROGatherParam(x="a", output="b", gather_dim=0, zero_stage=3)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.gather_dim == 0 and op2.zero_stage == 3

    def test_gather_spmd_type(self):
        op = ZeROGatherParam(x="x", output="y", gather_dim=0)
        assert op.propagate_spmd_type({"x": LocalSPMDType.VARYING}) == LocalSPMDType.REPLICATE


# ── ZeROScatterGrad ──────────────────────────────────────────────────────


class TestZeROScatterGrad:
    def test_scatter_replicate_to_shard_dim0(self):
        spec = _rep_spec()
        t = _make_tensor("grad", spec=spec)
        ctx = {"grad": t}
        op = ZeROScatterGrad(x="grad", output="grad_shard", scatter_dim=0)
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in result.sharding.placements)
        assert result.local_shape[0] == t.global_shape[0] // 4  # mesh size 4

    def test_scatter_replicate_to_shard_dim1(self):
        spec = _rep_spec()
        t = _make_tensor("grad", spec=spec)
        ctx = {"grad": t}
        op = ZeROScatterGrad(x="grad", output="grad_shard", scatter_dim=1)
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 1 for p in result.sharding.placements)
        assert result.local_shape[1] == t.global_shape[1] // 4

    def test_scatter_partial_to_shard(self):
        spec = _partial_spec()
        t = _make_tensor("grad", spec=spec)
        ctx = {"grad": t}
        op = ZeROScatterGrad(x="grad", output="grad_shard", scatter_dim=0)
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in result.sharding.placements)

    def test_scatter_stage_validation(self):
        t = _make_tensor("grad")
        ctx = {"grad": t}
        op = ZeROScatterGrad(x="grad", output="y", scatter_dim=0, zero_stage=1)
        with pytest.raises(ValueError, match="stage >= 2"):
            op.apply(ctx)

    def test_scatter_is_collective(self):
        op = ZeROScatterGrad(x="x", output="y", scatter_dim=0)
        assert op.is_collective() is True

    def test_scatter_vjp_shape_matches_fwd_input(self):
        t = _make_tensor("grad", expr="grad")
        ctx = {"grad": t}
        op = ZeROScatterGrad(x="grad", output="y", scatter_dim=0)
        grad_out = _make_tensor("grad_y")
        grads = op.vjp(ctx, grad_out)
        _assert_grad_matches_fwd(grads["grad"], t)
        assert "ZeROGatherParam" in grads["grad"].expr

    def test_scatter_clone_with_names(self):
        op = ZeROScatterGrad(x="a", output="b", scatter_dim=1, zero_stage=3)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.scatter_dim == 1 and op2.zero_stage == 3


# ── ZeROPartitionOptState ────────────────────────────────────────────────


class TestZeROPartitionOptState:
    def test_partition_replicate_to_shard(self):
        spec = _rep_spec()
        t = _make_tensor("opt_state", spec=spec)
        ctx = {"opt_state": t}
        op = ZeROPartitionOptState(x="opt_state", output="opt_shard", partition_dim=0)
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in result.sharding.placements)
        assert result.zero_stage == 1

    def test_partition_local_shape_changes(self):
        mesh = _mesh4()
        spec = _rep_spec(mesh)
        t = _make_tensor("opt", shape=(16, 8), spec=spec)
        ctx = {"opt": t}
        op = ZeROPartitionOptState(x="opt", output="opt_s", partition_dim=0)
        result = op.apply(ctx)
        assert result.local_shape == (4, 8)  # 16/4=4

    def test_partition_dim1(self):
        mesh = _mesh4()
        spec = _rep_spec(mesh)
        t = _make_tensor("opt", shape=(16, 8), spec=spec)
        ctx = {"opt": t}
        op = ZeROPartitionOptState(x="opt", output="opt_s", partition_dim=1)
        result = op.apply(ctx)
        assert result.local_shape == (16, 2)  # 8/4=2

    def test_partition_is_not_collective(self):
        op = ZeROPartitionOptState(x="x", output="y", partition_dim=0)
        assert op.is_collective() is False

    def test_partition_vjp_is_empty(self):
        t = _make_tensor("opt")
        ctx = {"opt": t}
        op = ZeROPartitionOptState(x="opt", output="y", partition_dim=0)
        grad_out = _make_tensor("grad_y")
        assert op.vjp(ctx, grad_out) == {}

    def test_partition_clone_with_names(self):
        op = ZeROPartitionOptState(x="a", output="b", partition_dim=0, zero_stage=1)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.partition_dim == 0

    def test_partition_propagates_zero_stage(self):
        spec = _rep_spec()
        t = _make_tensor("opt", spec=spec)
        ctx = {"opt": t}
        op = ZeROPartitionOptState(x="opt", output="y", partition_dim=0, zero_stage=2)
        result = op.apply(ctx)
        assert result.zero_stage == 2


# ── Executor integration ───────────────────────────────────────────────────


class TestZeROExecutor:
    def test_gather_scatter_roundtrip(self):
        mesh = _mesh4()
        spec = _shard_spec(dim=0, mesh=mesh)
        param = _make_tensor("param", shape=(16, 8), spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(param)

        prog = Program(name="zero_roundtrip")
        prog.add(ZeROGatherParam(x="param", output="param_full", gather_dim=0))
        prog.add(ZeROScatterGrad(x="param_full", output="param_shard", scatter_dim=0))
        exe.run_program(prog)

        full = exe.get_tensor("param_full", 0)
        assert all(isinstance(p, Replicate) for p in full.sharding.placements)
        assert full.local_shape == (16, 8)

        shard = exe.get_tensor("param_shard", 0)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in shard.sharding.placements)
        assert shard.local_shape == (4, 8)

    def test_partition_opt_state(self):
        mesh = _mesh4()
        spec = _rep_spec(mesh)
        opt = _make_tensor("momentum", shape=(16, 8), spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(opt)

        prog = Program(name="zero_opt")
        prog.add(ZeROPartitionOptState(x="momentum", output="momentum_shard", partition_dim=0))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("momentum_shard", did)
            assert result.local_shape == (4, 8)

    def test_stage3_gather_compute_scatter_pipeline(self):
        """Stage 3: gather sharded param → matmul → scatter gradient back."""
        mesh = _mesh4()
        spec_shard = _shard_spec(dim=0, mesh=mesh)
        spec_rep = _rep_spec(mesh)

        param = _make_tensor("w", shape=(16, 8), spec=spec_shard)
        x = _make_tensor("x", shape=(4, 16), spec=spec_rep)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(param)
        exe.register_tensor(x)

        prog = Program(name="stage3")
        prog.add(ZeROGatherParam(x="w", output="w_full", gather_dim=0))
        prog.add(MatMul(a="x", b="w_full", output="y"))
        prog.add(ZeROScatterGrad(x="y", output="y_shard", scatter_dim=0))
        exe.run_program(prog)

        w_full = exe.get_tensor("w_full", 0)
        assert all(isinstance(p, Replicate) for p in w_full.sharding.placements)

        y = exe.get_tensor("y", 0)
        assert y.global_shape == (4, 8)

        y_shard = exe.get_tensor("y_shard", 0)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in y_shard.sharding.placements)

    def test_zero_on_2d_mesh_tp_dp(self):
        """ZeRO gather on DP dim should preserve TP dim placement."""
        mesh = _mesh_2d()  # (2, 4) = (tp, dp)
        # TP: Shard(dim=1), DP: Shard(dim=0) → column-parallel + ZeRO stage 3
        spec = ShardingSpec(placements=(Shard(dim=1), Shard(dim=0)), mesh=mesh)
        param = _make_tensor("w", shape=(16, 8), spec=spec)
        # local: dim1 /= 2 (tp), dim0 /= 4 (dp) → (4, 4)
        assert param.local_shape == (4, 4)

        ctx = {"w": param}
        op = ZeROGatherParam(x="w", output="w_dp_full", gather_dim=0)
        result = op.apply(ctx)
        # DP dim gathered → Replicate, TP dim unchanged → Shard(1)
        assert result.sharding.placements == (Shard(dim=1), Replicate())
        assert result.local_shape == (16, 4)  # dim0 restored, dim1 still TP-sharded


# ── Autograd duality ──────────────────────────────────────────────────────


class TestZeROAutograd:
    def test_gather_scatter_duality(self):
        engine = AutogradEngine()
        assert engine._is_dual(
            ZeROGatherParam(x="p", output="pf", gather_dim=0),
            ZeROScatterGrad(x="g", output="gs", scatter_dim=0),
        )

    def test_scatter_gather_duality(self):
        engine = AutogradEngine()
        assert engine._is_dual(
            ZeROScatterGrad(x="g", output="gs", scatter_dim=1),
            ZeROGatherParam(x="gg", output="gf", gather_dim=1),
        )

    def test_dim_mismatch_not_dual(self):
        engine = AutogradEngine()
        assert not engine._is_dual(
            ZeROGatherParam(x="p", output="pf", gather_dim=0),
            ZeROScatterGrad(x="g", output="gs", scatter_dim=1),
        )

    def test_generate_dual_collective_zero(self):
        """Full autograd flow: record ZeROGatherParam → generate backward → get ZeROScatterGrad."""
        mesh = _mesh4()
        spec_shard = _shard_spec(dim=0, mesh=mesh)
        spec_rep = _rep_spec(mesh)
        param = _make_tensor("w", shape=(16, 8), spec=spec_shard, expr="w")

        op = ZeROGatherParam(x="w", output="w_full", gather_dim=0)
        ctx = {"w": param}
        result = op.apply(ctx)

        engine = AutogradEngine()
        engine.record(op, ctx)

        bwd = engine.generate_backward("w_full")
        bwd_ops = bwd.ops
        assert len(bwd_ops) == 1
        assert isinstance(bwd_ops[0], ZeROScatterGrad)
        assert bwd_ops[0].scatter_dim == 0
