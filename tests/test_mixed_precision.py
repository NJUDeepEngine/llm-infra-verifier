"""Tests for mixed precision (AMP) extension: Cast, LossScale, DtypeGuard."""

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
    Cast,
    LossScale,
    DtypeGuard,
    MatMul,
    AllReduce,
    Program,
)
from verifier.executor import MultiDeviceExecutor


def _mesh2():
    return DeviceMesh(shape=(2,), dim_names=("tp",))


def _mesh_2d():
    return DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"))


def _rep_spec(mesh=None):
    mesh = mesh or _mesh2()
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape), mesh=mesh)


def _shard_spec(dim=0, mesh=None):
    mesh = mesh or _mesh2()
    placements = [Replicate() for _ in mesh.shape]
    placements[0] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _partial_spec(mesh=None):
    mesh = mesh or _mesh2()
    placements = [Replicate() for _ in mesh.shape]
    placements[0] = Partial()
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _make_tensor(name, shape=(4, 8), spec=None, dtype=None, expr=""):
    spec = spec or _rep_spec()
    local = compute_local_shape(shape, spec)
    return TensorState(
        name=name, global_shape=shape, local_shape=local,
        sharding=spec, dtype=dtype, expr=expr,
    )


def _assert_grad_matches_fwd(grad: TensorState, fwd: TensorState):
    """VJP invariant: gradient must match forward input's shape and sharding."""
    assert grad.global_shape == fwd.global_shape, (
        f"grad shape {grad.global_shape} != fwd shape {fwd.global_shape}"
    )
    assert grad.local_shape == fwd.local_shape, (
        f"grad local_shape {grad.local_shape} != fwd {fwd.local_shape}"
    )
    assert grad.sharding.placements == fwd.sharding.placements, (
        f"grad placements {grad.sharding.placements} != fwd {fwd.sharding.placements}"
    )


# ── TensorState dtype field ────────────────────────────────────────────────


class TestTensorStateDtype:
    def test_default_dtype_is_none(self):
        t = _make_tensor("x")
        assert t.dtype is None
        assert t.is_fp32 is True

    def test_fp16_dtype(self):
        t = _make_tensor("x", dtype="fp16")
        assert t.is_fp16 is True
        assert t.is_fp32 is False
        assert t.is_bf16 is False

    def test_bf16_dtype(self):
        t = _make_tensor("x", dtype="bf16")
        assert t.is_bf16 is True
        assert t.is_fp16 is False
        assert t.is_fp32 is False

    def test_explicit_fp32(self):
        t = _make_tensor("x", dtype="fp32")
        assert t.is_fp32 is True
        assert t.is_fp16 is False

    def test_with_dtype_returns_copy(self):
        t = _make_tensor("x", dtype="fp32")
        t2 = t.with_dtype("fp16")
        assert t2.dtype == "fp16"
        assert t.dtype == "fp32"
        assert t2.name == t.name
        assert t2.global_shape == t.global_shape

    def test_dtype_in_hash(self):
        t1 = _make_tensor("x", dtype="fp32")
        t2 = _make_tensor("x", dtype="fp16")
        t3 = _make_tensor("x", dtype="fp32")
        assert hash(t1) != hash(t2)
        assert hash(t1) == hash(t3)


# ── Cast op ────────────────────────────────────────────────────────────────


class TestCast:
    def test_cast_fp32_to_fp16(self):
        t = _make_tensor("x", dtype="fp32", expr="x")
        ctx = {"x": t}
        op = Cast(x="x", output="x_fp16", src_dtype="fp32", dst_dtype="fp16")
        result = op.apply(ctx)
        assert result.dtype == "fp16"
        assert result.name == "x_fp16"
        assert result.global_shape == t.global_shape
        assert result.local_shape == t.local_shape
        assert result.sharding.placements == t.sharding.placements

    def test_cast_preserves_sharding(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("x", dtype="fp32", spec=spec)
        ctx = {"x": t}
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="bf16")
        result = op.apply(ctx)
        assert result.sharding.placements == spec.placements
        assert result.dtype == "bf16"
        assert result.local_shape == t.local_shape

    def test_cast_none_dtype_input_accepted(self):
        """dtype=None should be accepted (not checked against src_dtype)."""
        t = _make_tensor("x", dtype=None)
        ctx = {"x": t}
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="fp16")
        result = op.apply(ctx)
        assert result.dtype == "fp16"

    def test_cast_dtype_mismatch_raises(self):
        t = _make_tensor("x", dtype="fp16")
        ctx = {"x": t}
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="bf16")
        with pytest.raises(ValueError, match="expected src_dtype=fp32"):
            op.apply(ctx)

    def test_cast_stored_in_ctx(self):
        t = _make_tensor("x", dtype="fp32")
        ctx = {"x": t}
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="fp16")
        op.apply(ctx)
        assert "y" in ctx
        assert ctx["y"].dtype == "fp16"

    def test_cast_vjp_reverses_dtype_and_preserves_shape(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("x", dtype="fp32", spec=spec, expr="x")
        ctx = {"x": t}
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="fp16")
        grad_out = _make_tensor("grad_y", dtype="fp16", spec=spec)
        grads = op.vjp(ctx, grad_out)
        grad_x = grads["x"]
        assert grad_x.dtype == "fp32"
        _assert_grad_matches_fwd(grad_x, t)

    def test_cast_clone_with_names(self):
        op = Cast(x="a", output="b", src_dtype="fp32", dst_dtype="fp16")
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.src_dtype == "fp32" and op2.dst_dtype == "fp16"

    def test_cast_spmd_passthrough(self):
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="fp16")
        assert op.propagate_spmd_type({"x": LocalSPMDType.VARYING}) == LocalSPMDType.VARYING
        assert op.propagate_spmd_type({"x": LocalSPMDType.PARTIAL}) == LocalSPMDType.PARTIAL
        assert op.propagate_spmd_type({"x": LocalSPMDType.REPLICATE}) == LocalSPMDType.REPLICATE

    def test_cast_not_collective(self):
        op = Cast(x="x", output="y", src_dtype="fp32", dst_dtype="fp16")
        assert op.is_collective() is False


# ── LossScale op ───────────────────────────────────────────────────────────


class TestLossScale:
    def test_scale_direction(self):
        t = _make_tensor("loss", shape=(1,), expr="loss")
        ctx = {"loss": t}
        op = LossScale(x="loss", output="scaled_loss", scale=1024.0, direction="scale")
        result = op.apply(ctx)
        assert result.global_shape == t.global_shape
        assert result.sharding.placements == t.sharding.placements
        assert "* 1024" in result.expr

    def test_unscale_direction(self):
        t = _make_tensor("grad", shape=(4, 8), expr="grad")
        ctx = {"grad": t}
        op = LossScale(x="grad", output="unscaled", scale=1024.0, direction="unscale")
        result = op.apply(ctx)
        assert result.global_shape == t.global_shape
        assert "/ 1024" in result.expr

    def test_lossscale_empty_expr(self):
        t = _make_tensor("x", shape=(4, 8))
        ctx = {"x": t}
        op = LossScale(x="x", output="y", scale=512.0, direction="scale")
        result = op.apply(ctx)
        assert result.expr == ""
        assert result.global_shape == t.global_shape

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="must be 'scale' or 'unscale'"):
            LossScale(x="x", output="y", scale=1.0, direction="invalid")

    def test_lossscale_preserves_shape_and_sharding(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("x", spec=spec)
        ctx = {"x": t}
        op = LossScale(x="x", output="y", scale=512.0, direction="scale")
        result = op.apply(ctx)
        assert result.global_shape == t.global_shape
        assert result.local_shape == t.local_shape
        assert result.sharding.placements == spec.placements

    def test_lossscale_vjp_reverses_and_preserves_shape(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("x", spec=spec, expr="x")
        ctx = {"x": t}
        grad_out = _make_tensor("grad_y", spec=spec, expr="grad_y")
        op = LossScale(x="x", output="y", scale=1024.0, direction="scale")
        grads = op.vjp(ctx, grad_out)
        grad_x = grads["x"]
        _assert_grad_matches_fwd(grad_x, t)
        assert "/ 1024" in grad_x.expr

    def test_lossscale_vjp_unscale_reverses_to_scale(self):
        t = _make_tensor("x", expr="x")
        ctx = {"x": t}
        grad_out = _make_tensor("grad_y", expr="grad_y")
        op = LossScale(x="x", output="y", scale=256.0, direction="unscale")
        grads = op.vjp(ctx, grad_out)
        assert "* 256" in grads["x"].expr

    def test_lossscale_clone_with_names(self):
        op = LossScale(x="a", output="b", scale=256.0, direction="unscale")
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.scale == 256.0 and op2.direction == "unscale"


# ── DtypeGuard ─────────────────────────────────────────────────────────────


class TestDtypeGuard:
    def test_allreduce_fp16_warning(self):
        t = _make_tensor("x", dtype="fp16")
        msg = DtypeGuard.check_allreduce_dtype(t)
        assert msg is not None
        assert "overflow" in msg.lower()

    def test_allreduce_bf16_no_warning(self):
        t = _make_tensor("x", dtype="bf16")
        assert DtypeGuard.check_allreduce_dtype(t) is None

    def test_allreduce_fp32_no_warning(self):
        t = _make_tensor("x", dtype="fp32")
        assert DtypeGuard.check_allreduce_dtype(t) is None

    def test_allreduce_none_dtype_no_warning(self):
        t = _make_tensor("x")
        assert DtypeGuard.check_allreduce_dtype(t) is None

    def test_matmul_dtype_mismatch_fp16_fp32(self):
        a = _make_tensor("a", dtype="fp16")
        b = _make_tensor("b", dtype="fp32")
        msg = DtypeGuard.check_matmul_dtype_match(a, b)
        assert msg is not None
        assert "mismatch" in msg.lower()

    def test_matmul_dtype_mismatch_fp16_bf16(self):
        a = _make_tensor("a", dtype="fp16")
        b = _make_tensor("b", dtype="bf16")
        msg = DtypeGuard.check_matmul_dtype_match(a, b)
        assert msg is not None

    def test_matmul_same_dtype_no_warning(self):
        a = _make_tensor("a", dtype="fp16")
        b = _make_tensor("b", dtype="fp16")
        assert DtypeGuard.check_matmul_dtype_match(a, b) is None

    def test_matmul_both_none_no_warning(self):
        a = _make_tensor("a")
        b = _make_tensor("b")
        assert DtypeGuard.check_matmul_dtype_match(a, b) is None

    def test_collective_preserves_dtype_ok(self):
        a = _make_tensor("a", dtype="fp16")
        b = _make_tensor("b", dtype="fp16")
        assert DtypeGuard.check_collective_preserves_dtype(a, b) is None

    def test_collective_changes_dtype_error(self):
        a = _make_tensor("a", dtype="fp16")
        b = _make_tensor("b", dtype="fp32")
        msg = DtypeGuard.check_collective_preserves_dtype(a, b)
        assert msg is not None
        assert "dtype" in msg.lower()


# ── Executor integration ───────────────────────────────────────────────────


class TestMixedPrecisionExecutor:
    def test_cast_on_mesh(self):
        mesh = _mesh2()
        spec = _rep_spec(mesh)
        t = _make_tensor("x", dtype="fp32", spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(t)

        prog = Program(name="cast_test")
        prog.add(Cast(x="x", output="x_fp16", src_dtype="fp32", dst_dtype="fp16"))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("x_fp16", did)
            assert result is not None
            assert result.dtype == "fp16"
            assert result.global_shape == t.global_shape

    def test_cast_roundtrip_preserves_sharding(self):
        mesh = _mesh2()
        spec = _shard_spec(dim=0, mesh=mesh)
        t = _make_tensor("x", dtype="fp32", spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(t)

        prog = Program(name="roundtrip")
        prog.add(Cast(x="x", output="x16", src_dtype="fp32", dst_dtype="fp16"))
        prog.add(Cast(x="x16", output="x32", src_dtype="fp16", dst_dtype="fp32"))
        exe.run_program(prog)

        result = exe.get_tensor("x32", 0)
        assert result.dtype == "fp32"
        assert result.sharding.placements == spec.placements
        assert result.local_shape == t.local_shape

    def test_loss_scale_on_mesh(self):
        mesh = _mesh2()
        spec = _rep_spec(mesh)
        t = _make_tensor("loss", shape=(1,), spec=spec, expr="loss")

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(t)

        prog = Program(name="scale_test")
        prog.add(LossScale(x="loss", output="scaled", scale=1024.0, direction="scale"))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("scaled", did)
            assert result is not None
            assert "1024" in result.expr
            assert result.global_shape == (1,)

    def test_cast_then_matmul_pipeline(self):
        """fp32 weight → Cast fp16 → MatMul → output should be fp16."""
        mesh = _mesh2()
        spec = _rep_spec(mesh)
        x = _make_tensor("x", shape=(4, 8), spec=spec, dtype="fp16", expr="x")
        w = _make_tensor("w", shape=(8, 16), spec=spec, dtype="fp32", expr="w")

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="cast_matmul")
        prog.add(Cast(x="w", output="w16", src_dtype="fp32", dst_dtype="fp16"))
        prog.add(MatMul(a="x", b="w16", output="y"))
        exe.run_program(prog)

        w16 = exe.get_tensor("w16", 0)
        assert w16.dtype == "fp16"
        y = exe.get_tensor("y", 0)
        assert y is not None
        assert y.global_shape == (4, 16)

    def test_cast_on_2d_mesh(self):
        """Cast should preserve per-dim placements on a 2D (TP, DP) mesh."""
        mesh = _mesh_2d()  # (2, 4)
        spec = ShardingSpec(placements=(Shard(dim=1), Replicate()), mesh=mesh)
        t = _make_tensor("x", shape=(4, 8), spec=spec, dtype="fp32")

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(t)

        prog = Program(name="cast_2d")
        prog.add(Cast(x="x", output="x16", src_dtype="fp32", dst_dtype="fp16"))
        exe.run_program(prog)

        result = exe.get_tensor("x16", 0)
        assert result.dtype == "fp16"
        assert result.sharding.placements == (Shard(dim=1), Replicate())
        assert result.local_shape == t.local_shape
