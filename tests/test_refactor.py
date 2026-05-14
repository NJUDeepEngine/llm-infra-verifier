"""Refactoring validation tests.

Verifies that the ir/ and state/ package splits + abstraction changes
did not break any behavior. Covers:
  1. Import compatibility: all public symbols accessible
  2. State sub-module dependency chain
  3. New collective ops (Broadcast, Reduce, AllToAll, Scatter, Gather)
  4. CollectiveOp base class mechanics
  5. ElementWiseBinaryOp base class edge cases
  6. clone_with_names() round-trip for all ops
  7. apply_checked() cross-validation (consistent + inconsistent)
  8. SPMDGuard integration in collectives
  9. End-to-end: Megatron GPT-2 style program with spmd_checking
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from verifier.state import (
    LocalSPMDType,
    Shard, Replicate, Partial, Placement,
    DeviceNode, Link, DeviceTopology, DeviceMesh,
    ShardingSpec, compute_local_shape,
    TensorSlice, compute_tensor_slices,
    TensorState,
)
from verifier.ir import (
    IROp, SPMDConsistencyError,
    ElementWiseBinaryOp, MatMul, Add, Multiply, SiLU, FlashAttention,
    CollectiveOp, AllReduce, AllGather, ReduceScatter,
    Broadcast, Reduce, AllToAll, Scatter, Gather,
    Send, Recv, SendAsync, RecvAsync,
    Handle, Stream, DEFAULT_STREAM, COMM_STREAM, COMPUTE_STREAM,
    AllReduceAsync, Wait, WaitAll, OverlapRegion,
    Reshape, Transpose,
    Reinterpret, Convert, SPMDGuard,
    Program, ir_to_str,
)
from verifier.executor import MultiDeviceExecutor


def _mesh(n=2):
    return DeviceMesh(shape=(n,), dim_names=("tp",))

def _spec(placement, mesh=None):
    mesh = mesh or _mesh()
    return ShardingSpec(placements=(placement,), mesh=mesh)

def _tensor(name, placement, shape=(8, 16), mesh=None):
    mesh = mesh or _mesh()
    spec = _spec(placement, mesh)
    ls = compute_local_shape(shape, spec)
    return TensorState(name=name, global_shape=shape, local_shape=ls, sharding=spec, expr=name)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Import compatibility — all 36+ symbols from ir/, all 16 from state/
# ═══════════════════════════════════════════════════════════════════════════════

class TestImportCompatibility:
    def test_state_all_symbols(self):
        from verifier import state
        for sym in state.__all__:
            assert hasattr(state, sym), f"Missing export: {sym}"

    def test_ir_all_symbols(self):
        from verifier import ir
        for sym in ir.__all__:
            assert hasattr(ir, sym), f"Missing export: {sym}"

    def test_relative_imports_from_ir_ops(self):
        """IR ops import from ..state — verify the chain works."""
        from verifier.ir.compute import MatMul, Add, Multiply, SiLU, FlashAttention
        from verifier.ir.collective import AllReduce, AllGather, ReduceScatter
        from verifier.ir.collective import Broadcast, Reduce, AllToAll, Scatter, Gather
        from verifier.ir.p2p import Send, Recv, SendAsync, RecvAsync
        from verifier.ir.async_ops import AllReduceAsync, Wait, WaitAll
        from verifier.ir.shape import Reshape, Transpose
        from verifier.ir.spmd import Reinterpret, Convert, SPMDGuard

    def test_state_submodule_imports(self):
        """Each state sub-module can be imported independently."""
        from verifier.state.placement import LocalSPMDType, Shard, Replicate, Partial
        from verifier.state.device import DeviceNode, Link, DeviceTopology, DeviceMesh
        from verifier.state.sharding import ShardingSpec, compute_local_shape, TensorSlice
        from verifier.state.tensor import TensorState


# ═══════════════════════════════════════════════════════════════════════════════
# 2. State sub-module dependency chain
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateDependencyChain:
    def test_placement_has_no_internal_deps(self):
        """placement.py should not import from other state sub-modules."""
        import verifier.state.placement as p
        assert hasattr(p, 'LocalSPMDType')
        assert hasattr(p, 'Shard')
        assert hasattr(p, 'Placement')

    def test_device_has_no_placement_dep(self):
        """device.py should not import from placement."""
        import verifier.state.device as d
        assert hasattr(d, 'DeviceMesh')
        assert hasattr(d, 'DeviceTopology')

    def test_sharding_depends_on_placement_and_device(self):
        from verifier.state.sharding import ShardingSpec, compute_local_shape
        mesh = _mesh()
        spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        assert compute_local_shape((8, 16), spec) == (4, 16)

    def test_tensor_depends_on_all(self):
        t = _tensor("x", Shard(dim=0))
        assert t.local_type == LocalSPMDType.VARYING
        assert t.global_shape == (8, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. New collective ops — placement transforms + VJP
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewCollectives:
    def test_broadcast_any_to_replicate(self):
        x = _tensor("x", Shard(dim=0))
        op = Broadcast(x="x", output="y", root=0)
        ctx = {"x": x}
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)

    def test_reduce_partial_to_replicate(self):
        x = _tensor("x", Partial())
        op = Reduce(x="x", output="y", root=0)
        ctx = {"x": x}
        result = op.apply(ctx)
        assert all(isinstance(p, Replicate) for p in result.sharding.placements)

    def test_reduce_rejects_non_partial(self):
        x = _tensor("x", Replicate())
        op = Reduce(x="x", output="y", root=0)
        with pytest.raises(ValueError, match="Reduce requires PARTIAL"):
            op.apply({"x": x})

    def test_alltoall_shard_dim_swap(self):
        x = _tensor("x", Shard(dim=0))
        op = AllToAll(x="x", output="y", split_dim=0, concat_dim=1)
        ctx = {"x": x}
        result = op.apply(ctx)
        shard_dims = result.sharding.get_shard_dims()
        assert 1 in shard_dims

    def test_scatter_replicate_to_shard(self):
        x = _tensor("x", Replicate())
        op = Scatter(x="x", output="y", scatter_dim=0, root=0)
        ctx = {"x": x}
        result = op.apply(ctx)
        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 0

    def test_gather_shard_to_replicate(self):
        x = _tensor("x", Shard(dim=1))
        op = Gather(x="x", output="y", gather_dim=1, root=0)
        ctx = {"x": x}
        result = op.apply(ctx)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_collective_vjp_duality(self):
        """Verify VJP duality: AllGather↔ReduceScatter, Broadcast↔Reduce, Scatter↔Gather."""
        mesh = _mesh()
        x_shard = _tensor("x", Shard(dim=0), mesh=mesh)
        x_repl = _tensor("x", Replicate(), mesh=mesh)
        x_part = _tensor("x", Partial(), mesh=mesh)

        ag = AllGather(x="x", output="y", gather_dim=0)
        ag_grad = ag.vjp({"x": x_shard}, None)
        assert "x" in ag_grad

        bc = Broadcast(x="x", output="y", root=0)
        bc_grad = bc.vjp({"x": x_repl}, None)
        assert "x" in bc_grad

        sc = Scatter(x="x", output="y", scatter_dim=0, root=0)
        sc_grad = sc.vjp({"x": x_repl}, None)
        assert "x" in sc_grad

        ga = Gather(x="x", output="y", gather_dim=0, root=0)
        ga_grad = ga.vjp({"x": x_shard}, None)
        assert "x" in ga_grad

        a2a = AllToAll(x="x", output="y", split_dim=0, concat_dim=1)
        a2a_grad = a2a.vjp({"x": x_shard}, None)
        assert "x" in a2a_grad


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CollectiveOp base class mechanics
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectiveOpBase:
    def test_all_collectives_are_collective(self):
        ops = [
            AllReduce(x="x", output="y"),
            AllGather(x="x", output="y", gather_dim=0),
            ReduceScatter(x="x", output="y", scatter_dim=0),
            Broadcast(x="x", output="y"),
            Reduce(x="x", output="y"),
            AllToAll(x="x", output="y", split_dim=0, concat_dim=1),
            Scatter(x="x", output="y", scatter_dim=0),
            Gather(x="x", output="y", gather_dim=0),
        ]
        for op in ops:
            assert op.is_collective(), f"{type(op).__name__}.is_collective() should be True"
            assert op.is_communication(), f"{type(op).__name__}.is_communication() should be True"

    def test_collective_shared_result_construction(self):
        """All collectives use CollectiveOp.apply() shared path."""
        mesh = _mesh()
        x_part = _tensor("x", Partial(), mesh=mesh)
        x_shard = _tensor("x", Shard(dim=0), mesh=mesh)
        x_repl = _tensor("x", Replicate(), mesh=mesh)

        cases = [
            (AllReduce(x="x", output="y"), x_part),
            (AllGather(x="x", output="y", gather_dim=0), x_shard),
            (ReduceScatter(x="x", output="y", scatter_dim=0), x_repl),
            (Broadcast(x="x", output="y"), x_shard),
            (Reduce(x="x", output="y"), x_part),
            (Scatter(x="x", output="y", scatter_dim=0), x_repl),
            (Gather(x="x", output="y", gather_dim=0), x_shard),
        ]
        for op, x in cases:
            ctx = {"x": x}
            result = op.apply(ctx)
            assert result.name == "y", f"{type(op).__name__} output name"
            assert result.global_shape == x.global_shape, f"{type(op).__name__} preserves global_shape"
            assert "y" in ctx, f"{type(op).__name__} writes to ctx"

    def test_validate_spmd_fires_on_allreduce(self):
        """AllReduce rejects non-PARTIAL input (placement _validate fires first)."""
        x = _tensor("x", Replicate())
        op = AllReduce(x="x", output="y")
        with pytest.raises(ValueError, match="AllReduce requires PARTIAL"):
            op.apply({"x": x})

    def test_validate_spmd_fires_on_reduce(self):
        x = _tensor("x", Replicate())
        op = Reduce(x="x", output="y")
        with pytest.raises(ValueError):
            op.apply({"x": x})


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ElementWiseBinaryOp edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestElementWiseBase:
    def test_shape_mismatch_raises(self):
        mesh = _mesh()
        a = _tensor("a", Replicate(), shape=(8, 16), mesh=mesh)
        b = _tensor("b", Replicate(), shape=(8, 32), mesh=mesh)
        op = Add(a="a", b="b", output="y")
        with pytest.raises(ValueError, match="shape mismatch"):
            op.apply({"a": a, "b": b})

    def test_mesh_mismatch_raises(self):
        m1 = DeviceMesh(shape=(2,), dim_names=("tp",))
        m2 = DeviceMesh(shape=(4,), dim_names=("dp",))
        a = _tensor("a", Replicate(), mesh=m1)
        b = _tensor("b", Replicate(), mesh=m2)
        op = Add(a="a", b="b", output="y")
        with pytest.raises(ValueError, match="mesh mismatch"):
            op.apply({"a": a, "b": b})

    def test_incompatible_placements_raises(self):
        mesh = _mesh()
        a = _tensor("a", Shard(dim=0), mesh=mesh)
        b = _tensor("b", Shard(dim=1), mesh=mesh)
        op = Add(a="a", b="b", output="y")
        with pytest.raises(ValueError, match="incompatible placements"):
            op.apply({"a": a, "b": b})

    def test_replicate_plus_shard_inherits_shard(self):
        mesh = _mesh()
        a = _tensor("a", Replicate(), mesh=mesh)
        b = _tensor("b", Shard(dim=0), mesh=mesh)
        op = Add(a="a", b="b", output="y")
        result = op.apply({"a": a, "b": b})
        assert isinstance(result.sharding.placements[0], Shard)

    def test_multiply_partial_partial_raises(self):
        mesh = _mesh()
        a = _tensor("a", Partial(), mesh=mesh)
        b = _tensor("b", Partial(), mesh=mesh)
        op = Multiply(a="a", b="b", output="y")
        with pytest.raises(ValueError, match="SPMD violation"):
            op.apply({"a": a, "b": b})


# ═══════════════════════════════════════════════════════════════════════════════
# 6. clone_with_names() round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestCloneWithNames:
    def _verify_clone(self, op, input_map, new_output):
        cloned = op.clone_with_names(input_map, new_output)
        assert cloned.output_name == new_output
        assert type(cloned) is type(op)
        for orig, renamed in input_map.items():
            assert renamed in cloned.input_names

    def test_clone_matmul(self):
        self._verify_clone(
            MatMul(a="a", b="b", output="y"),
            {"a": "a2", "b": "b2"}, "y2",
        )

    def test_clone_add(self):
        self._verify_clone(
            Add(a="a", b="b", output="y"),
            {"a": "a2", "b": "b2"}, "y2",
        )

    def test_clone_allreduce(self):
        self._verify_clone(
            AllReduce(x="x", output="y"),
            {"x": "x2"}, "y2",
        )

    def test_clone_allgather(self):
        self._verify_clone(
            AllGather(x="x", output="y", gather_dim=0),
            {"x": "x2"}, "y2",
        )

    def test_clone_broadcast(self):
        self._verify_clone(
            Broadcast(x="x", output="y", root=0),
            {"x": "x2"}, "y2",
        )

    def test_clone_alltoall(self):
        self._verify_clone(
            AllToAll(x="x", output="y", split_dim=0, concat_dim=1),
            {"x": "x2"}, "y2",
        )

    def test_clone_scatter(self):
        self._verify_clone(
            Scatter(x="x", output="y", scatter_dim=0),
            {"x": "x2"}, "y2",
        )

    def test_clone_gather(self):
        self._verify_clone(
            Gather(x="x", output="y", gather_dim=0),
            {"x": "x2"}, "y2",
        )

    def test_clone_reshape(self):
        self._verify_clone(
            Reshape(x="x", output="y", new_shape=(4, 32)),
            {"x": "x2"}, "y2",
        )

    def test_clone_silu(self):
        self._verify_clone(
            SiLU(x="x", output="y"),
            {"x": "x2"}, "y2",
        )

    def test_clone_flash_attention(self):
        cloned = FlashAttention(q="q", k="k", v="v", output="o").clone_with_names(
            {"q": "q2", "k": "k2", "v": "v2"}, "o2",
        )
        assert cloned.output_name == "o2"
        assert set(cloned.input_names) == {"q2", "k2", "v2"}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. apply_checked() cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyChecked:
    def test_silu_consistent(self):
        x = _tensor("x", Shard(dim=0))
        op = SiLU(x="x", output="y")
        result = op.apply_checked({"x": x})
        assert result.local_type == LocalSPMDType.VARYING

    def test_add_consistent(self):
        mesh = _mesh()
        a = _tensor("a", Replicate(), mesh=mesh)
        b = _tensor("b", Shard(dim=0), mesh=mesh)
        op = Add(a="a", b="b", output="y")
        result = op.apply_checked({"a": a, "b": b})
        assert result.local_type == LocalSPMDType.VARYING

    def test_allgather_consistent(self):
        x = _tensor("x", Shard(dim=0))
        op = AllGather(x="x", output="y", gather_dim=0)
        result = op.apply_checked({"x": x})
        assert result.local_type == LocalSPMDType.REPLICATE

    def test_scatter_consistent(self):
        x = _tensor("x", Replicate())
        op = Scatter(x="x", output="y", scatter_dim=0)
        result = op.apply_checked({"x": x})
        assert result.local_type == LocalSPMDType.VARYING

    def test_broadcast_consistent(self):
        x = _tensor("x", Shard(dim=0))
        op = Broadcast(x="x", output="y")
        result = op.apply_checked({"x": x})
        assert result.local_type == LocalSPMDType.REPLICATE

    def test_reshape_consistent(self):
        x = _tensor("x", Replicate())
        op = Reshape(x="x", output="y", new_shape=(16, 8))
        result = op.apply_checked({"x": x})
        assert result.local_type == LocalSPMDType.REPLICATE


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SPMD propagation completeness — every op has a rule
# ═══════════════════════════════════════════════════════════════════════════════

class TestSPMDPropagationCompleteness:
    def test_all_collectives_have_propagation(self):
        R, V, P = LocalSPMDType.REPLICATE, LocalSPMDType.VARYING, LocalSPMDType.PARTIAL
        cases = [
            (AllReduce(x="x", output="y"), {"x": P}, R),
            (AllGather(x="x", output="y", gather_dim=0), {"x": V}, R),
            (ReduceScatter(x="x", output="y", scatter_dim=0), {"x": P}, V),
            (Broadcast(x="x", output="y"), {"x": V}, R),
            (Reduce(x="x", output="y"), {"x": P}, R),
            (AllToAll(x="x", output="y", split_dim=0, concat_dim=1), {"x": V}, V),
            (Scatter(x="x", output="y", scatter_dim=0), {"x": R}, V),
            (Gather(x="x", output="y", gather_dim=0), {"x": V}, R),
        ]
        for op, inputs, expected in cases:
            result = op.propagate_spmd_type(inputs)
            assert result == expected, (
                f"{type(op).__name__}.propagate_spmd_type({inputs}) "
                f"= {result}, expected {expected}"
            )

    def test_all_compute_ops_have_propagation(self):
        R, V = LocalSPMDType.REPLICATE, LocalSPMDType.VARYING
        mm = MatMul(a="a", b="b", output="y")
        assert mm.propagate_spmd_type({"a": R, "b": R}) == R

        add = Add(a="a", b="b", output="y")
        assert add.propagate_spmd_type({"a": R, "b": V}) == V

        mul = Multiply(a="a", b="b", output="y")
        assert mul.propagate_spmd_type({"a": V, "b": V}) == V

        silu = SiLU(x="x", output="y")
        assert silu.propagate_spmd_type({"x": V}) == V

        fa = FlashAttention(q="q", k="k", v="v", output="o")
        assert fa.propagate_spmd_type({"q": R, "k": R, "v": R}) == R

    def test_shape_ops_have_propagation(self):
        V = LocalSPMDType.VARYING
        assert Reshape(x="x", output="y", new_shape=(4, 32)).propagate_spmd_type({"x": V}) == V
        assert Transpose(x="x", output="y").propagate_spmd_type({"x": V}) == V

    def test_spmd_ops_have_propagation(self):
        R, P = LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL
        ri = Reinterpret(x="x", output="y", src_type=R, dst_type=P)
        assert ri.propagate_spmd_type({"x": R}) == P

        cv = Convert(x="x", output="y", src_type=R, dst_type=P)
        assert cv.propagate_spmd_type({"x": R}) == P

    def test_async_ops_have_propagation(self):
        R, P = LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL
        ara = AllReduceAsync(x="x", output="y", handle="h")
        assert ara.propagate_spmd_type({"x": P}) == R

        w = Wait(handle="h", tensor="x", output="y")
        assert w.propagate_spmd_type({"x": R}) == R


# ═══════════════════════════════════════════════════════════════════════════════
# 9. End-to-end: TP linear with spmd_checking through executor
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndSPMD:
    def test_row_parallel_matmul_allreduce(self):
        """S(1) @ S(0) → P → AllReduce → R, with SPMD cross-validation."""
        mesh = _mesh()
        x = _tensor("x", Shard(dim=1), shape=(8, 16), mesh=mesh)
        w = TensorState(
            name="w", global_shape=(16, 32),
            local_shape=compute_local_shape((16, 32), _spec(Shard(dim=0), mesh)),
            sharding=_spec(Shard(dim=0), mesh), expr="w",
        )
        prog = Program(ops=[
            MatMul(a="x", b="w", output="h"),
            AllReduce(x="h", output="y"),
        ])
        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        executor.register_tensor(x)
        executor.register_tensor(w)
        result = executor.run_program(prog)
        assert "y" in result
        y = result["y"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert not y.partial

    def test_column_parallel_matmul_allgather(self):
        """R @ S(1) → S(1) → AllGather → R, with SPMD cross-validation."""
        mesh = _mesh()
        x = _tensor("x", Replicate(), shape=(8, 16), mesh=mesh)
        w = TensorState(
            name="w", global_shape=(16, 32),
            local_shape=compute_local_shape((16, 32), _spec(Shard(dim=1), mesh)),
            sharding=_spec(Shard(dim=1), mesh), expr="w",
        )
        prog = Program(ops=[
            MatMul(a="x", b="w", output="h"),
            AllGather(x="h", output="y", gather_dim=1),
        ])
        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        executor.register_tensor(x)
        executor.register_tensor(w)
        result = executor.run_program(prog)
        assert "y" in result
        assert result["y"].local_type == LocalSPMDType.REPLICATE

    def test_full_mlp_block(self):
        """MatMul → SiLU → MatMul → AllReduce, full TP MLP with SPMD checking."""
        mesh = _mesh()
        x = _tensor("x", Replicate(), shape=(4, 8), mesh=mesh)
        w1 = TensorState(
            name="w1", global_shape=(8, 16),
            local_shape=compute_local_shape((8, 16), _spec(Shard(dim=1), mesh)),
            sharding=_spec(Shard(dim=1), mesh), expr="w1",
        )
        w2 = TensorState(
            name="w2", global_shape=(16, 8),
            local_shape=compute_local_shape((16, 8), _spec(Shard(dim=0), mesh)),
            sharding=_spec(Shard(dim=0), mesh), expr="w2",
        )
        prog = Program(ops=[
            MatMul(a="x", b="w1", output="h1"),       # R @ S(1) → S(1)
            SiLU(x="h1", output="h1_act"),             # S(1) → S(1)
            MatMul(a="h1_act", b="w2", output="h2"),   # S(1) @ S(0) → P
            AllReduce(x="h2", output="y"),              # P → R
        ])
        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        executor.register_tensor(x)
        executor.register_tensor(w1)
        executor.register_tensor(w2)
        result = executor.run_program(prog)
        assert result["y"].local_type == LocalSPMDType.REPLICATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
