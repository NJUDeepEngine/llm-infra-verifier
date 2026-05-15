"""Tests for CP Ring Attention extension: RingRotate, RingAttentionStep, RingAttention."""

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
    RingRotate,
    RingAttentionStep,
    RingAttention,
    Add,
    AllReduce,
    Program,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine


def _mesh(n=4):
    return DeviceMesh(shape=(n,), dim_names=("cp",))


def _mesh_2d():
    return DeviceMesh(shape=(2, 4), dim_names=("tp", "cp"))


def _rep_spec(mesh=None):
    mesh = mesh or _mesh()
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape), mesh=mesh)


def _shard_spec(dim=0, mesh=None):
    mesh = mesh or _mesh()
    placements = [Replicate() for _ in mesh.shape]
    placements[0] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _make_tensor(name, shape=(4, 16, 64), spec=None, ring_step=None, expr=""):
    spec = spec or _rep_spec()
    local = compute_local_shape(shape, spec)
    return TensorState(
        name=name, global_shape=shape, local_shape=local, sharding=spec,
        ring_step=ring_step, expr=expr,
    )


def _assert_grad_matches_fwd(grad: TensorState, fwd: TensorState):
    assert grad.global_shape == fwd.global_shape
    assert grad.local_shape == fwd.local_shape
    assert grad.sharding.placements == fwd.sharding.placements


# ── RingRotate ─────────────────────────────────────────────────────────────


class TestRingRotate:
    def test_rotate_increments_ring_step(self):
        t = _make_tensor("k", ring_step=0)
        ctx = {"k": t}
        op = RingRotate(x="k", output="k_rot", ring_size=4)
        result = op.apply(ctx)
        assert result.ring_step == 1
        assert result.name == "k_rot"

    def test_rotate_from_none_ring_step(self):
        t = _make_tensor("k")
        assert t.ring_step is None
        ctx = {"k": t}
        op = RingRotate(x="k", output="k_rot", ring_size=4)
        result = op.apply(ctx)
        assert result.ring_step == 1

    def test_rotate_preserves_shape_and_sharding(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("k", spec=spec)
        ctx = {"k": t}
        op = RingRotate(x="k", output="k_rot", ring_size=4)
        result = op.apply(ctx)
        assert result.global_shape == t.global_shape
        assert result.local_shape == t.local_shape
        assert result.sharding.placements == spec.placements

    def test_rotate_is_collective(self):
        op = RingRotate(x="x", output="y", ring_size=4)
        assert op.is_collective() is True

    def test_rotate_with_handle(self):
        t = _make_tensor("k")
        ctx = {"k": t}
        op = RingRotate(x="k", output="k_rot", ring_size=4, handle="h0")
        result = op.apply(ctx)
        assert result._async_handle == "h0"

    def test_rotate_vjp_shape_matches(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("k", spec=spec, expr="k")
        ctx = {"k": t}
        op = RingRotate(x="k", output="k_rot", ring_size=4)
        grad_out = _make_tensor("grad_k_rot", spec=spec)
        grads = op.vjp(ctx, grad_out)
        _assert_grad_matches_fwd(grads["k"], t)
        assert "RingRotateReverse" in grads["k"].expr

    def test_rotate_clone_with_names(self):
        op = RingRotate(x="a", output="b", ring_size=4, ring_dim=1)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.ring_size == 4 and op2.ring_dim == 1

    def test_rotate_repr(self):
        op = RingRotate(x="k", output="k_rot", ring_size=4, ring_dim=0)
        r = repr(op)
        assert "ring=4" in r and "dim=0" in r


# ── RingAttentionStep ──────────────────────────────────────────────────────


class TestRingAttentionStep:
    def test_step_produces_partial_on_seq_shard(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        shard_seq = _shard_spec(dim=1, mesh=mesh)

        q = _make_tensor("q", spec=rep)
        k = _make_tensor("k", spec=shard_seq)
        v = _make_tensor("v", spec=shard_seq)

        ctx = {"q": q, "k": k, "v": v}
        op = RingAttentionStep(
            q="q", k="k", v="v", output="attn_0",
            ring_step=0, ring_size=4,
        )
        result = op.apply(ctx)
        assert any(isinstance(p, Partial) for p in result.sharding.placements)
        assert result.ring_step == 0
        assert result.global_shape == q.global_shape

    def test_step_replicated_inputs_produce_replicate(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        q = _make_tensor("q", spec=rep)
        k = _make_tensor("k", spec=rep)
        v = _make_tensor("v", spec=rep)

        ctx = {"q": q, "k": k, "v": v}
        op = RingAttentionStep(
            q="q", k="k", v="v", output="out",
            ring_step=2, ring_size=4,
        )
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)
        assert result.ring_step == 2

    def test_step_vjp_shape_matches_all_inputs(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        shard_seq = _shard_spec(dim=1, mesh=mesh)
        q = _make_tensor("q", spec=rep, expr="q")
        k = _make_tensor("k", spec=shard_seq, expr="k")
        v = _make_tensor("v", spec=shard_seq, expr="v")

        ctx = {"q": q, "k": k, "v": v}
        op = RingAttentionStep(
            q="q", k="k", v="v", output="out",
            ring_step=1, ring_size=4,
        )
        grad_out = _make_tensor("grad_out", spec=rep)
        grads = op.vjp(ctx, grad_out)

        assert set(grads.keys()) == {"q", "k", "v"}
        _assert_grad_matches_fwd(grads["q"], q)
        _assert_grad_matches_fwd(grads["k"], k)
        _assert_grad_matches_fwd(grads["v"], v)
        assert grads["q"].ring_step == 1

    def test_step_clone_with_names(self):
        op = RingAttentionStep(
            q="q", k="k", v="v", output="o",
            ring_step=0, ring_size=4, causal=True,
        )
        op2 = op.clone_with_names({"q": "q2", "k": "k2"}, "o2")
        assert op2.q == "q2" and op2.k == "k2" and op2.v == "v"
        assert op2.causal is True and op2.ring_step == 0


# ── RingAttention (composite) ─────────────────────────────────────────────


class TestRingAttention:
    def test_apply_produces_replicate(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        q = _make_tensor("q", spec=rep)
        k = _make_tensor("k", spec=rep)
        v = _make_tensor("v", spec=rep)

        ctx = {"q": q, "k": k, "v": v}
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=4)
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)
        assert result.global_shape == q.global_shape

    def test_is_collective(self):
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=4)
        assert op.is_collective() is True

    def test_expand_op_counts(self):
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=4)
        expanded = op.expand()

        step_ops = [o for o in expanded if isinstance(o, RingAttentionStep)]
        rotate_ops = [o for o in expanded if isinstance(o, RingRotate)]
        add_ops = [o for o in expanded if isinstance(o, Add)]
        reduce_ops = [o for o in expanded if isinstance(o, AllReduce)]

        assert len(step_ops) == 4
        assert len(rotate_ops) == 6  # (ring_size-1) * 2 (K+V)
        assert len(add_ops) == 3    # ring_size - 1
        assert len(reduce_ops) == 1

    def test_expand_op_counts_ring3(self):
        """Verify counts scale with ring_size, not hardcoded to 4."""
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=3)
        expanded = op.expand()

        steps = [o for o in expanded if isinstance(o, RingAttentionStep)]
        rotates = [o for o in expanded if isinstance(o, RingRotate)]
        adds = [o for o in expanded if isinstance(o, Add)]

        assert len(steps) == 3
        assert len(rotates) == 4  # (3-1)*2
        assert len(adds) == 2     # 3-1

    def test_expand_ring_steps_sequential(self):
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=3)
        expanded = op.expand()
        steps = [o for o in expanded if isinstance(o, RingAttentionStep)]
        assert [s.ring_step for s in steps] == [0, 1, 2]

    def test_expand_last_op_is_allreduce_to_output(self):
        op = RingAttention(q="q", k="k", v="v", output="final_out", ring_size=2)
        expanded = op.expand()
        assert isinstance(expanded[-1], AllReduce)
        assert expanded[-1].output == "final_out"

    def test_vjp_shape_matches_all_inputs(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        shard_seq = _shard_spec(dim=1, mesh=mesh)
        q = _make_tensor("q", spec=rep, expr="q")
        k = _make_tensor("k", spec=shard_seq, expr="k")
        v = _make_tensor("v", spec=shard_seq, expr="v")

        ctx = {"q": q, "k": k, "v": v}
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=4)
        grad_out = _make_tensor("grad_out")
        grads = op.vjp(ctx, grad_out)

        assert set(grads.keys()) == {"q", "k", "v"}
        _assert_grad_matches_fwd(grads["q"], q)
        _assert_grad_matches_fwd(grads["k"], k)
        _assert_grad_matches_fwd(grads["v"], v)

    def test_clone_with_names(self):
        op = RingAttention(q="q", k="k", v="v", output="o", ring_size=4, causal=True)
        op2 = op.clone_with_names({"q": "q2"}, "o2")
        assert op2.q == "q2" and op2.output == "o2"
        assert op2.ring_size == 4 and op2.causal is True

    def test_spmd_type(self):
        op = RingAttention(q="q", k="k", v="v", output="o", ring_size=4)
        assert op.propagate_spmd_type({}) == LocalSPMDType.REPLICATE


# ── Executor integration ───────────────────────────────────────────────────


class TestCPRingExecutor:
    def test_ring_rotate_on_mesh(self):
        mesh = _mesh()
        spec = _rep_spec(mesh)
        k = _make_tensor("k", spec=spec, ring_step=0)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(k)

        prog = Program(name="rotate_test")
        prog.add(RingRotate(x="k", output="k_rot", ring_size=4))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("k_rot", did)
            assert result is not None
            assert result.ring_step == 1
            assert result.global_shape == k.global_shape

    def test_ring_rotate_data_source_verification(self):
        """Tag each device's tensor uniquely, verify (did-1)%ring_size source after rotation."""
        mesh = _mesh(4)
        ring_size = 4
        spec = _rep_spec(mesh)
        k = _make_tensor("k", spec=spec, ring_step=0)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(k)

        for did in range(mesh.num_devices):
            exe.devices[did].tensors["k"].expr = f"k_dev{did}"

        prog = Program(name="rotate_trace")
        prog.add(RingRotate(x="k", output="k_rot", ring_size=ring_size))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            expected_src = (did - 1) % ring_size
            result = exe.get_tensor("k_rot", did)
            assert result.expr == f"k_dev{expected_src}", (
                f"device {did}: expected data from dev{expected_src}, "
                f"got expr='{result.expr}'"
            )

    def test_ring_rotate_ring_size_less_than_devices(self):
        """ring_size=3 on 6-device mesh: wrap uses ring_size, not num_devices."""
        mesh = _mesh(6)
        ring_size = 3
        spec = _rep_spec(mesh)
        k = _make_tensor("k", spec=spec, ring_step=0)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(k)

        for did in range(mesh.num_devices):
            exe.devices[did].tensors["k"].expr = f"k_dev{did}"

        prog = Program(name="rotate_small_ring")
        prog.add(RingRotate(x="k", output="k_rot", ring_size=ring_size))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            expected_src = (did - 1) % ring_size
            result = exe.get_tensor("k_rot", did)
            assert result.expr == f"k_dev{expected_src}", (
                f"device {did}: ring_size={ring_size}, expected src dev{expected_src}, "
                f"got '{result.expr}'"
            )

    def test_multi_step_ring_rotate(self):
        """Rotate K through 3 steps on 4-device ring, verify ring_step accumulates."""
        mesh = _mesh(4)
        spec = _rep_spec(mesh)
        k = _make_tensor("k", spec=spec, ring_step=0)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(k)

        for did in range(mesh.num_devices):
            exe.devices[did].tensors["k"].expr = f"k_dev{did}"

        prog = Program(name="multi_rotate")
        prog.add(RingRotate(x="k", output="k1", ring_size=4))
        prog.add(RingRotate(x="k1", output="k2", ring_size=4))
        prog.add(RingRotate(x="k2", output="k3", ring_size=4))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("k3", did)
            assert result.ring_step == 3
            expected_src = (did - 3) % 4
            assert result.expr == f"k_dev{expected_src}"

    def test_ring_attention_expand_on_executor(self):
        """RingAttention executed via expand() on a real 4-device mesh."""
        mesh = _mesh(4)
        rep = _rep_spec(mesh)
        shard_seq = _shard_spec(dim=1, mesh=mesh)

        q = _make_tensor("q", spec=rep, expr="q")
        k = _make_tensor("k", spec=shard_seq, expr="k")
        v = _make_tensor("v", spec=shard_seq, expr="v")

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(q)
        exe.register_tensor(k)
        exe.register_tensor(v)

        prog = Program(name="ring_attn")
        prog.add(RingAttention(q="q", k="k", v="v", output="out", ring_size=4))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("out", did)
            assert result is not None
            assert all(isinstance(p, Replicate) for p in result.sharding.placements)
            assert result.global_shape == q.global_shape


# ── Autograd duality ──────────────────────────────────────────────────────


class TestCPRingAutograd:
    def test_ring_rotate_self_dual(self):
        engine = AutogradEngine()
        assert engine._is_dual(
            RingRotate(x="k", output="kr", ring_size=4),
            RingRotate(x="gk", output="gkr", ring_size=4),
        )

    def test_ring_rotate_size_mismatch_not_dual(self):
        engine = AutogradEngine()
        assert not engine._is_dual(
            RingRotate(x="k", output="kr", ring_size=4),
            RingRotate(x="gk", output="gkr", ring_size=8),
        )

    def test_generate_dual_collective_ring(self):
        """Record RingRotate → generate backward → get reverse RingRotate."""
        mesh = _mesh()
        spec = _rep_spec(mesh)
        k = _make_tensor("k", spec=spec, ring_step=0, expr="k")

        op = RingRotate(x="k", output="k_rot", ring_size=4)
        ctx = {"k": k}
        result = op.apply(ctx)

        engine = AutogradEngine()
        engine.record(op, ctx)

        bwd = engine.generate_backward("k_rot")
        bwd_ops = bwd.ops
        assert len(bwd_ops) == 1
        assert isinstance(bwd_ops[0], RingRotate)
        assert bwd_ops[0].ring_size == 4

    def test_ring_rotate_dim_mismatch_not_dual(self):
        """ring_dim differs → not dual, even if ring_size matches."""
        engine = AutogradEngine()
        assert not engine._is_dual(
            RingRotate(x="k", output="kr", ring_size=4, ring_dim=0),
            RingRotate(x="gk", output="gkr", ring_size=4, ring_dim=1),
        )


# ── dtype preservation ────────────────────────────────────────────────────


class TestCPRingDtypePreservation:
    """Regression: Ring ops must propagate dtype from input."""

    def test_ring_attention_step_preserves_dtype(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        q = _make_tensor("q", spec=rep)
        k = _make_tensor("k", spec=rep)
        v = _make_tensor("v", spec=rep)
        q.dtype = "bf16"
        k.dtype = "bf16"
        v.dtype = "bf16"
        ctx = {"q": q, "k": k, "v": v}
        op = RingAttentionStep(
            q="q", k="k", v="v", output="out",
            ring_step=0, ring_size=4,
        )
        result = op.apply(ctx)
        assert result.dtype == "bf16"

    def test_ring_attention_preserves_dtype(self):
        mesh = _mesh()
        rep = _rep_spec(mesh)
        q = _make_tensor("q", spec=rep)
        k = _make_tensor("k", spec=rep)
        v = _make_tensor("v", spec=rep)
        q.dtype = "fp16"
        k.dtype = "fp16"
        v.dtype = "fp16"
        ctx = {"q": q, "k": k, "v": v}
        op = RingAttention(q="q", k="k", v="v", output="out", ring_size=4)
        result = op.apply(ctx)
        assert result.dtype == "fp16"
