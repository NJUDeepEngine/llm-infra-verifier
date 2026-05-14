"""Tests for MoE Routing extension: TopKGate, MoEDispatch, MoECombine, ExpertCompute."""

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
    TopKGate,
    MoEDispatch,
    MoECombine,
    ExpertCompute,
    Program,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine


def _mesh4():
    return DeviceMesh(shape=(4,), dim_names=("ep",))


def _mesh_2d():
    return DeviceMesh(shape=(2, 4), dim_names=("tp", "ep"))


def _rep_spec(mesh=None):
    mesh = mesh or _mesh4()
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape), mesh=mesh)


def _shard_spec(dim=0, mesh=None, mesh_dim=0):
    mesh = mesh or _mesh4()
    placements = [Replicate() for _ in mesh.shape]
    placements[mesh_dim] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _make_tensor(name, shape=(32, 128), spec=None, expr=""):
    spec = spec or _rep_spec()
    local = compute_local_shape(shape, spec)
    return TensorState(
        name=name, global_shape=shape, local_shape=local,
        sharding=spec, expr=expr,
    )


def _assert_grad_matches_fwd(grad: TensorState, fwd: TensorState):
    assert grad.global_shape == fwd.global_shape
    assert grad.local_shape == fwd.local_shape
    assert grad.sharding.placements == fwd.sharding.placements


# ── TopKGate ──────────────────────────────────────────────────────────────


class TestTopKGate:
    def test_gate_produces_two_outputs(self):
        x = _make_tensor("tokens", shape=(32, 128))
        gw = _make_tensor("gate_w", shape=(128, 8))

        ctx = {"tokens": x, "gate_w": gw}
        op = TopKGate(
            x="tokens", gate_weight="gate_w",
            output="gate_scores", indices_output="gate_indices",
            num_experts=8, top_k=2,
        )
        result = op.apply(ctx)
        assert result.name == "gate_scores"
        assert result.global_shape == x.global_shape
        assert "gate_indices" in ctx
        indices = ctx["gate_indices"]
        assert indices.global_shape == (32, 2)
        assert indices.sharding.placements == x.sharding.placements

    def test_gate_preserves_sharding(self):
        spec = _shard_spec(dim=0)
        x = _make_tensor("tokens", shape=(32, 128), spec=spec)
        gw = _make_tensor("gate_w", shape=(128, 8), spec=spec)

        ctx = {"tokens": x, "gate_w": gw}
        op = TopKGate(
            x="tokens", gate_weight="gate_w",
            output="scores", indices_output="indices",
            num_experts=8,
        )
        result = op.apply(ctx)
        assert result.sharding.placements == spec.placements
        assert result.local_shape == x.local_shape

    def test_gate_annotates_num_experts(self):
        x = _make_tensor("tokens")
        gw = _make_tensor("gate_w", shape=(128, 16))

        ctx = {"tokens": x, "gate_w": gw}
        op = TopKGate(
            x="tokens", gate_weight="gate_w",
            output="scores", indices_output="indices",
            num_experts=16, top_k=3,
        )
        result = op.apply(ctx)
        assert result.num_experts == 16
        assert ctx["indices"].num_experts == 16

    def test_gate_vjp_shape_matches_input(self):
        spec = _shard_spec(dim=0)
        x = _make_tensor("tokens", spec=spec, expr="tokens")
        gw = _make_tensor("gate_w", shape=(128, 8), spec=spec)
        ctx = {"tokens": x, "gate_w": gw}
        op = TopKGate(
            x="tokens", gate_weight="gate_w",
            output="scores", indices_output="indices",
            num_experts=8,
        )
        grad_out = _make_tensor("grad_scores", spec=spec)
        grads = op.vjp(ctx, grad_out)
        assert "tokens" in grads
        _assert_grad_matches_fwd(grads["tokens"], x)

    def test_gate_clone_with_names(self):
        op = TopKGate(
            x="a", gate_weight="gw", output="b", indices_output="idx",
            num_experts=8, top_k=2, capacity_factor=1.5,
        )
        op2 = op.clone_with_names({"a": "c", "gw": "gw2"}, "d")
        assert op2.x == "c" and op2.gate_weight == "gw2"
        assert op2.output == "d" and op2.num_experts == 8
        assert op2.top_k == 2 and op2.capacity_factor == 1.5

    def test_gate_spmd_passthrough(self):
        op = TopKGate(
            x="x", gate_weight="gw", output="y", indices_output="idx",
            num_experts=8,
        )
        assert op.propagate_spmd_type({"x": LocalSPMDType.VARYING}) == LocalSPMDType.VARYING
        assert op.propagate_spmd_type({"x": LocalSPMDType.REPLICATE}) == LocalSPMDType.REPLICATE

    def test_gate_not_collective(self):
        op = TopKGate(
            x="x", gate_weight="gw", output="y", indices_output="idx",
            num_experts=8,
        )
        assert op.is_collective() is False


# ── MoEDispatch ───────────────────────────────────────────────────────────


class TestMoEDispatch:
    def test_dispatch_shard_transform(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("tokens", spec=spec)
        ctx = {"tokens": t}
        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        )
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 1 for p in result.sharding.placements)
        assert not any(isinstance(p, Shard) and p.dim == 0 for p in result.sharding.placements)

    def test_dispatch_replicate_input_unchanged(self):
        """Replicate has no Shard(split_dim) to transform → stays Replicate."""
        spec = _rep_spec()
        t = _make_tensor("tokens", spec=spec)
        ctx = {"tokens": t}
        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        )
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)

    def test_dispatch_annotates_metadata(self):
        t = _make_tensor("tokens")
        ctx = {"tokens": t}
        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=8, split_dim=0, concat_dim=1,
            expert_capacity=64,
        )
        result = op.apply(ctx)
        assert result.num_experts == 8
        assert result.expert_capacity == 64

    def test_dispatch_no_capacity_by_default(self):
        t = _make_tensor("tokens")
        ctx = {"tokens": t}
        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        )
        result = op.apply(ctx)
        assert result.expert_capacity is None

    def test_dispatch_is_collective(self):
        op = MoEDispatch(x="x", output="y", num_experts=4, split_dim=0, concat_dim=1)
        assert op.is_collective() is True

    def test_dispatch_vjp_shape_matches_input(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("tokens", spec=spec, expr="tokens")
        ctx = {"tokens": t}
        op = MoEDispatch(
            x="tokens", output="y", num_experts=4, split_dim=0, concat_dim=1,
        )
        grad_out = _make_tensor("grad_y", spec=spec)
        grads = op.vjp(ctx, grad_out)
        _assert_grad_matches_fwd(grads["tokens"], t)
        assert "MoECombine" in grads["tokens"].expr

    def test_dispatch_clone_with_names(self):
        op = MoEDispatch(
            x="a", output="b", num_experts=4, split_dim=0, concat_dim=1,
            expert_capacity=32,
        )
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.split_dim == 0 and op2.concat_dim == 1
        assert op2.expert_capacity == 32


# ── MoECombine ────────────────────────────────────────────────────────────


class TestMoECombine:
    def test_combine_shard_transform(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("expert_out", spec=spec)
        ctx = {"expert_out": t}
        op = MoECombine(
            x="expert_out", output="combined",
            num_experts=4, split_dim=1, concat_dim=0,
        )
        result = op.apply(ctx)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in result.sharding.placements)
        assert not any(isinstance(p, Shard) and p.dim == 1 for p in result.sharding.placements)

    def test_combine_is_collective(self):
        op = MoECombine(x="x", output="y", num_experts=4, split_dim=0, concat_dim=1)
        assert op.is_collective() is True

    def test_combine_vjp_shape_matches_input(self):
        spec = _shard_spec(dim=1)
        t = _make_tensor("expert_out", spec=spec, expr="expert_out")
        ctx = {"expert_out": t}
        op = MoECombine(
            x="expert_out", output="y", num_experts=4, split_dim=1, concat_dim=0,
        )
        grad_out = _make_tensor("grad_y", spec=spec)
        grads = op.vjp(ctx, grad_out)
        _assert_grad_matches_fwd(grads["expert_out"], t)
        assert "MoEDispatch" in grads["expert_out"].expr

    def test_combine_clone_with_names(self):
        op = MoECombine(x="a", output="b", num_experts=4, split_dim=1, concat_dim=0)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.split_dim == 1 and op2.concat_dim == 0


# ── ExpertCompute ─────────────────────────────────────────────────────────


class TestExpertCompute:
    def test_expert_compute_passthrough(self):
        t = _make_tensor("tokens")
        ctx = {"tokens": t}
        op = ExpertCompute(x="tokens", output="expert_out", expert_id=0, num_experts=4)
        result = op.apply(ctx)
        assert result.global_shape == t.global_shape
        assert result.local_shape == t.local_shape
        assert result.sharding.placements == t.sharding.placements
        assert result.expert_id == 0
        assert result.num_experts == 4

    def test_expert_compute_with_sharded_input(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("tokens", spec=spec)
        ctx = {"tokens": t}
        op = ExpertCompute(x="tokens", output="out", expert_id=2, num_experts=8)
        result = op.apply(ctx)
        assert result.sharding.placements == spec.placements
        assert result.local_shape == t.local_shape
        assert result.expert_id == 2

    def test_expert_compute_not_collective(self):
        op = ExpertCompute(x="x", output="y", expert_id=0, num_experts=4)
        assert op.is_collective() is False

    def test_expert_compute_vjp_shape_matches(self):
        spec = _shard_spec(dim=0)
        t = _make_tensor("tokens", spec=spec, expr="tokens")
        ctx = {"tokens": t}
        op = ExpertCompute(x="tokens", output="y", expert_id=2, num_experts=8)
        grad_out = _make_tensor("grad_y", spec=spec)
        grads = op.vjp(ctx, grad_out)
        assert "tokens" in grads
        _assert_grad_matches_fwd(grads["tokens"], t)
        assert "expert2" in grads["tokens"].expr

    def test_expert_compute_clone_with_names(self):
        op = ExpertCompute(x="a", output="b", expert_id=1, num_experts=4)
        op2 = op.clone_with_names({"a": "c"}, "d")
        assert op2.x == "c" and op2.output == "d"
        assert op2.expert_id == 1 and op2.num_experts == 4

    def test_expert_compute_spmd_passthrough(self):
        op = ExpertCompute(x="x", output="y", expert_id=0, num_experts=4)
        assert op.propagate_spmd_type({"x": LocalSPMDType.VARYING}) == LocalSPMDType.VARYING
        assert op.propagate_spmd_type({"x": LocalSPMDType.PARTIAL}) == LocalSPMDType.PARTIAL


# ── Executor integration ───────────────────────────────────────────────────


class TestMoEExecutor:
    def test_dispatch_combine_roundtrip(self):
        mesh = _mesh4()
        spec = _shard_spec(dim=0, mesh=mesh)
        tokens = _make_tensor("tokens", shape=(32, 128), spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(tokens)

        prog = Program(name="moe_roundtrip")
        prog.add(MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        ))
        prog.add(MoECombine(
            x="dispatched", output="combined",
            num_experts=4, split_dim=1, concat_dim=0,
        ))
        exe.run_program(prog)

        dispatched = exe.get_tensor("dispatched", 0)
        assert any(isinstance(p, Shard) and p.dim == 1 for p in dispatched.sharding.placements)
        assert dispatched.local_shape[1] == 128 // 4

        combined = exe.get_tensor("combined", 0)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in combined.sharding.placements)
        assert combined.local_shape[0] == 32 // 4

    def test_topk_gate_on_mesh(self):
        mesh = _mesh4()
        spec = _rep_spec(mesh)
        tokens = _make_tensor("tokens", shape=(32, 128), spec=spec)
        gw = _make_tensor("gate_w", shape=(128, 8), spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(tokens)
        exe.register_tensor(gw)

        prog = Program(name="gate_test")
        prog.add(TopKGate(
            x="tokens", gate_weight="gate_w",
            output="scores", indices_output="indices",
            num_experts=8, top_k=2,
        ))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            scores = exe.get_tensor("scores", did)
            indices = exe.get_tensor("indices", did)
            assert scores is not None
            assert scores.global_shape == (32, 128)
            assert indices is not None
            assert indices.global_shape == (32, 2)

    def test_expert_compute_on_mesh(self):
        mesh = _mesh4()
        spec = _rep_spec(mesh)
        tokens = _make_tensor("tokens", spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(tokens)

        prog = Program(name="expert_test")
        prog.add(ExpertCompute(
            x="tokens", output="expert_out", expert_id=0, num_experts=4,
        ))
        exe.run_program(prog)

        for did in range(mesh.num_devices):
            result = exe.get_tensor("expert_out", did)
            assert result.expert_id == 0
            assert result.num_experts == 4
            assert result.global_shape == tokens.global_shape

    def test_full_moe_pipeline(self):
        """Gate → Dispatch → ExpertCompute → Combine end-to-end."""
        mesh = _mesh4()
        spec = _shard_spec(dim=0, mesh=mesh)
        tokens = _make_tensor("tokens", shape=(32, 128), spec=spec)
        gw = _make_tensor("gate_w", shape=(128, 4), spec=spec)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(tokens)
        exe.register_tensor(gw)

        prog = Program(name="full_moe")
        prog.add(TopKGate(
            x="tokens", gate_weight="gate_w",
            output="scores", indices_output="indices",
            num_experts=4, top_k=1,
        ))
        prog.add(MoEDispatch(
            x="scores", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        ))
        prog.add(ExpertCompute(
            x="dispatched", output="expert_out",
            expert_id=0, num_experts=4,
        ))
        prog.add(MoECombine(
            x="expert_out", output="combined",
            num_experts=4, split_dim=1, concat_dim=0,
        ))
        exe.run_program(prog)

        scores = exe.get_tensor("scores", 0)
        assert scores is not None

        dispatched = exe.get_tensor("dispatched", 0)
        assert any(isinstance(p, Shard) and p.dim == 1 for p in dispatched.sharding.placements)

        expert_out = exe.get_tensor("expert_out", 0)
        assert expert_out.expert_id == 0

        combined = exe.get_tensor("combined", 0)
        assert any(isinstance(p, Shard) and p.dim == 0 for p in combined.sharding.placements)

    def test_moe_dispatch_on_2d_mesh(self):
        """On (TP, EP) mesh, dispatch should only transform the EP dim."""
        mesh = _mesh_2d()  # (2, 4) = (tp, ep)
        # TP: Shard(dim=1), EP: Shard(dim=0) → column-parallel + expert-parallel
        spec = ShardingSpec(placements=(Shard(dim=1), Shard(dim=0)), mesh=mesh)
        tokens = _make_tensor("tokens", shape=(32, 128), spec=spec)
        assert tokens.local_shape == (8, 64)  # dim0/4, dim1/2

        ctx = {"tokens": tokens}
        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        )
        result = op.apply(ctx)
        # EP dim: Shard(0) with split_dim=0 → Shard(1) with concat_dim=1
        # TP dim: Shard(1) is not Shard(split_dim=0) → unchanged
        assert result.sharding.placements[0] == Shard(dim=1)  # TP preserved
        assert result.sharding.placements[1] == Shard(dim=1)  # EP transformed


# ── Autograd duality ──────────────────────────────────────────────────────


class TestMoEAutograd:
    def test_dispatch_combine_duality(self):
        engine = AutogradEngine()
        assert engine._is_dual(
            MoEDispatch(x="t", output="d", num_experts=4, split_dim=0, concat_dim=1),
            MoECombine(x="g", output="c", num_experts=4, split_dim=1, concat_dim=0),
        )

    def test_combine_dispatch_duality(self):
        engine = AutogradEngine()
        assert engine._is_dual(
            MoECombine(x="t", output="c", num_experts=4, split_dim=1, concat_dim=0),
            MoEDispatch(x="g", output="d", num_experts=4, split_dim=0, concat_dim=1),
        )

    def test_dim_mismatch_not_dual(self):
        engine = AutogradEngine()
        assert not engine._is_dual(
            MoEDispatch(x="t", output="d", num_experts=4, split_dim=0, concat_dim=1),
            MoECombine(x="g", output="c", num_experts=4, split_dim=0, concat_dim=1),
        )

    def test_generate_dual_collective_dispatch(self):
        """Record MoEDispatch → generate backward → get MoECombine."""
        mesh = _mesh4()
        spec = _shard_spec(dim=0, mesh=mesh)
        tokens = _make_tensor("tokens", shape=(32, 128), spec=spec, expr="tokens")

        op = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=4, split_dim=0, concat_dim=1,
        )
        ctx = {"tokens": tokens}
        result = op.apply(ctx)

        engine = AutogradEngine()
        engine.record(op, ctx)

        bwd = engine.generate_backward("dispatched")
        bwd_ops = bwd.ops
        assert len(bwd_ops) == 1
        assert isinstance(bwd_ops[0], MoECombine)
        assert bwd_ops[0].split_dim == 1  # reversed
        assert bwd_ops[0].concat_dim == 0
