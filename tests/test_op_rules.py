"""Systematic placement-rule tests for every IR op.

For each op, tests cover:
  - Every placement input→output branch in apply()
  - Output shape (global & local) derivation
  - Error paths (invalid inputs raise ValueError)
  - VJP grad shape/placement matching
"""

import pytest
from verifier.state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    LocalSPMDType,
    compute_local_shape,
)
from verifier.ir import (
    MatMul, Add, Multiply, SiLU, FlashAttention,
    AllReduce, AllGather, ReduceScatter, Broadcast, Reduce,
    AllToAll, Scatter, Gather,
    Send, Recv, SendAsync, RecvAsync,
    AllReduceAsync, Wait, WaitAll, OverlapRegion,
    Reshape, Transpose,
    Reinterpret, Convert,
    SPMDConsistencyError,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def mesh1d(tp=2):
    return DeviceMesh(shape=(tp,), dim_names=("tp",))

def mesh2d(tp=2, dp=2):
    return DeviceMesh(shape=(dp, tp), dim_names=("dp", "tp"))

def T(name, shape, placements, mesh):
    """Create a TensorState with given placements tuple."""
    spec = ShardingSpec(placements=placements, mesh=mesh)
    return TensorState(
        name=name,
        global_shape=shape,
        local_shape=compute_local_shape(shape, spec),
        sharding=spec,
        expr=name,
        requires_grad=True,
    )

def placement_of(ts, dim=0):
    """Return the placement at mesh dim."""
    return ts.sharding.placements[dim]

def is_shard(ts, tensor_dim, mesh_dim=0):
    p = placement_of(ts, mesh_dim)
    return isinstance(p, Shard) and p.dim == tensor_dim

def is_replicate(ts, mesh_dim=0):
    return isinstance(placement_of(ts, mesh_dim), Replicate)

def is_partial(ts, mesh_dim=0):
    return isinstance(placement_of(ts, mesh_dim), Partial)


# ═════════════════════════════════════════════════════════════════════════════
# MatMul
# ═════════════════════════════════════════════════════════════════════════════

class TestMatMulRules:
    M = mesh1d(2)
    S = (8, 4)   # A shape
    BS = (4, 6)  # B shape

    def _run(self, pa, pb):
        a = T("a", self.S, (pa,), self.M)
        b = T("b", self.BS, (pb,), self.M)
        ctx = {"a": a, "b": b}
        return MatMul(a="a", b="b", output="y").apply(ctx)

    # ── placement combos ────────────────────────────────────────────────

    def test_r_r(self):
        y = self._run(Replicate(), Replicate())
        assert is_replicate(y)
        assert y.global_shape == (8, 6)
        assert y.local_shape == (8, 6)

    def test_r_s1(self):
        """Column parallel: R × S(1) → S(1)."""
        y = self._run(Replicate(), Shard(1))
        assert is_shard(y, 1)
        assert y.local_shape == (8, 3)

    def test_s0_r(self):
        """Data/row parallel: S(0) × R → S(0)."""
        y = self._run(Shard(0), Replicate())
        assert is_shard(y, 0)
        assert y.local_shape == (4, 6)

    def test_s1_s0(self):
        """Contraction sharded both sides → Partial."""
        y = self._run(Shard(1), Shard(0))
        assert is_partial(y)

    def test_r_s0(self):
        """B rows sharded, A no dim=1 shard → inherits A's Replicate."""
        y = self._run(Replicate(), Shard(0))
        assert is_replicate(y)

    def test_s0_s1(self):
        """A=S(0) overwritten by B=S(1) → S(1)."""
        y = self._run(Shard(0), Shard(1))
        assert is_shard(y, 1)

    def test_s0_s0(self):
        """Both S(0), no contraction trigger → S(0)."""
        y = self._run(Shard(0), Shard(0))
        assert is_shard(y, 0)

    def test_s1_s1(self):
        """Both S(1) → S(1)."""
        y = self._run(Shard(1), Shard(1))
        assert is_shard(y, 1)

    def test_s1_r(self):
        """A=S(1), B=R → S(1) unchanged."""
        y = self._run(Shard(1), Replicate())
        assert is_shard(y, 1)

    # ── shape derivation ────────────────────────────────────────────────

    def test_output_global_shape(self):
        y = self._run(Replicate(), Replicate())
        assert y.global_shape == (8, 6)

    def test_output_local_shape_shard0(self):
        y = self._run(Shard(0), Replicate())
        assert y.global_shape == (8, 6)
        assert y.local_shape == (4, 6)

    # ── VJP ─────────────────────────────────────────────────────────────

    def test_vjp_shapes_match_inputs(self):
        a = T("a", self.S, (Shard(0),), self.M)
        b = T("b", self.BS, (Replicate(),), self.M)
        ctx = {"a": a, "b": b}
        op = MatMul(a="a", b="b", output="y")
        out = op.apply(ctx)
        grads = op.vjp(ctx, out)
        assert grads["a"].global_shape == a.global_shape
        assert grads["a"].local_shape == a.local_shape
        assert grads["a"].sharding.placements == a.sharding.placements
        assert grads["b"].global_shape == b.global_shape
        assert grads["b"].local_shape == b.local_shape
        assert grads["b"].sharding.placements == b.sharding.placements


# ═════════════════════════════════════════════════════════════════════════════
# ElementWise (Add / Multiply)
# ═════════════════════════════════════════════════════════════════════════════

class TestElementWiseRules:
    M = mesh1d(2)
    S = (8, 4)

    def _run_add(self, pa, pb):
        a = T("a", self.S, (pa,), self.M)
        b = T("b", self.S, (pb,), self.M)
        return Add(a="a", b="b", output="y").apply({"a": a, "b": b})

    def _run_mul(self, pa, pb):
        a = T("a", self.S, (pa,), self.M)
        b = T("b", self.S, (pb,), self.M)
        return Multiply(a="a", b="b", output="y").apply({"a": a, "b": b})

    # ── valid combos ────────────────────────────────────────────────────

    def test_r_r(self):
        y = self._run_add(Replicate(), Replicate())
        assert is_replicate(y)

    def test_r_s0(self):
        y = self._run_add(Replicate(), Shard(0))
        assert is_shard(y, 0)

    def test_s0_r(self):
        y = self._run_add(Shard(0), Replicate())
        assert is_shard(y, 0)

    def test_s0_s0(self):
        y = self._run_add(Shard(0), Shard(0))
        assert is_shard(y, 0)

    def test_r_p(self):
        y = self._run_add(Replicate(), Partial())
        assert is_partial(y)

    def test_p_r(self):
        y = self._run_add(Partial(), Replicate())
        assert is_partial(y)

    def test_p_p(self):
        y = self._run_add(Partial(), Partial())
        assert is_partial(y)

    def test_s1_s1(self):
        y = self._run_add(Shard(1), Shard(1))
        assert is_shard(y, 1)

    # ── error combos ────────────────────────────────────────────────────

    def test_s0_s1_incompatible(self):
        with pytest.raises(ValueError, match="incompatible"):
            self._run_add(Shard(0), Shard(1))

    def test_s0_p_incompatible(self):
        with pytest.raises(ValueError, match="incompatible"):
            self._run_add(Shard(0), Partial())

    def test_p_s0_incompatible(self):
        with pytest.raises(ValueError, match="incompatible"):
            self._run_add(Partial(), Shard(0))

    def test_shape_mismatch(self):
        a = T("a", (8, 4), (Replicate(),), self.M)
        b = T("b", (8, 6), (Replicate(),), self.M)
        with pytest.raises(ValueError, match="shape mismatch"):
            Add(a="a", b="b", output="y").apply({"a": a, "b": b})

    def test_mesh_mismatch(self):
        m2 = mesh1d(4)
        a = T("a", self.S, (Replicate(),), self.M)
        b = T("b", self.S, (Replicate(),), m2)
        with pytest.raises(ValueError, match="mesh mismatch"):
            Add(a="a", b="b", output="y").apply({"a": a, "b": b})

    # ── Multiply: Partial × Partial forbidden ───────────────────────────

    def test_multiply_p_p_forbidden(self):
        with pytest.raises(ValueError, match="Partial"):
            self._run_mul(Partial(), Partial())

    def test_multiply_r_s0_ok(self):
        y = self._run_mul(Replicate(), Shard(0))
        assert is_shard(y, 0)

    # ── VJP ─────────────────────────────────────────────────────────────

    def test_add_vjp_shapes(self):
        a = T("a", self.S, (Shard(0),), self.M)
        b = T("b", self.S, (Shard(0),), self.M)
        ctx = {"a": a, "b": b}
        op = Add(a="a", b="b", output="y")
        out = op.apply(ctx)
        grads = op.vjp(ctx, out)
        assert grads["a"].global_shape == a.global_shape
        assert grads["b"].global_shape == b.global_shape

    def test_multiply_vjp_shapes(self):
        a = T("a", self.S, (Shard(0),), self.M)
        b = T("b", self.S, (Shard(0),), self.M)
        ctx = {"a": a, "b": b}
        op = Multiply(a="a", b="b", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, T("grad_y", self.S, (Shard(0),), self.M))
        assert grads["a"].global_shape == a.global_shape
        assert grads["a"].sharding.placements == a.sharding.placements
        assert grads["b"].global_shape == b.global_shape


# ═════════════════════════════════════════════════════════════════════════════
# SiLU
# ═════════════════════════════════════════════════════════════════════════════

class TestSiLURules:
    M = mesh1d(2)
    S = (8, 4)

    def _run(self, p):
        x = T("x", self.S, (p,), self.M)
        return SiLU(x="x", output="y").apply({"x": x})

    def test_replicate(self):
        y = self._run(Replicate())
        assert is_replicate(y)
        assert y.global_shape == self.S
        assert y.local_shape == self.S

    def test_shard0(self):
        y = self._run(Shard(0))
        assert is_shard(y, 0)
        assert y.local_shape == (4, 4)

    def test_shard1(self):
        y = self._run(Shard(1))
        assert is_shard(y, 1)
        assert y.local_shape == (8, 2)

    def test_partial(self):
        y = self._run(Partial())
        assert is_partial(y)

    def test_vjp(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = SiLU(x="x", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert grads["x"].global_shape == x.global_shape
        assert grads["x"].sharding.placements == x.sharding.placements


# ═════════════════════════════════════════════════════════════════════════════
# FlashAttention
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionRules:
    M = mesh1d(2)
    S = (8, 4)  # (seq, head_dim)

    def _run(self, pq, pk, pv=None):
        pv = pv or pk
        q = T("q", self.S, (pq,), self.M)
        k = T("k", self.S, (pk,), self.M)
        v = T("v", self.S, (pv,), self.M)
        ctx = {"q": q, "k": k, "v": v}
        return FlashAttention(q="q", k="k", v="v", output="o").apply(ctx)

    def test_all_replicate(self):
        o = self._run(Replicate(), Replicate())
        assert is_replicate(o)
        assert o.global_shape == self.S

    def test_k_shard1_q_replicate(self):
        """K head-dim sharded, Q not → Partial."""
        o = self._run(Replicate(), Shard(1))
        assert is_partial(o)

    def test_both_shard1(self):
        """Both Q and K head-dim sharded → stays Shard(1)."""
        o = self._run(Shard(1), Shard(1))
        assert is_shard(o, 1)

    def test_q_shard0_batch(self):
        """Q batch-sharded, K replicate → S(0)."""
        o = self._run(Shard(0), Replicate())
        assert is_shard(o, 0)

    def test_q_shard0_k_shard1(self):
        """Q=S(0), K=S(1): Q lacks S(1) → Partial on that mesh dim."""
        o = self._run(Shard(0), Shard(1))
        assert is_partial(o)

    def test_output_shape_equals_q(self):
        o = self._run(Replicate(), Replicate())
        assert o.global_shape == self.S

    def test_vjp(self):
        q = T("q", self.S, (Shard(0),), self.M)
        k = T("k", self.S, (Replicate(),), self.M)
        v = T("v", self.S, (Replicate(),), self.M)
        ctx = {"q": q, "k": k, "v": v}
        op = FlashAttention(q="q", k="k", v="v", output="o")
        out = op.apply(ctx)
        grads = op.vjp(ctx, out)
        assert grads["q"].global_shape == q.global_shape
        assert grads["q"].sharding.placements == q.sharding.placements
        assert grads["k"].global_shape == k.global_shape
        assert grads["v"].global_shape == v.global_shape


# ═════════════════════════════════════════════════════════════════════════════
# AllReduce
# ═════════════════════════════════════════════════════════════════════════════

class TestAllReduceRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_partial_to_replicate(self):
        x = T("x", self.S, (Partial(),), self.M)
        y = AllReduce(x="x", output="y").apply({"x": x})
        assert is_replicate(y)
        assert y.global_shape == self.S
        assert y.local_shape == self.S

    def test_rejects_replicate(self):
        x = T("x", self.S, (Replicate(),), self.M)
        with pytest.raises(ValueError, match="PARTIAL"):
            AllReduce(x="x", output="y").apply({"x": x})

    def test_rejects_shard(self):
        x = T("x", self.S, (Shard(0),), self.M)
        with pytest.raises(ValueError, match="PARTIAL"):
            AllReduce(x="x", output="y").apply({"x": x})

    def test_vjp_is_self_dual(self):
        x = T("x", self.S, (Partial(),), self.M)
        ctx = {"x": x}
        op = AllReduce(x="x", output="y")
        out = op.apply(ctx)
        grads = op.vjp(ctx, out)
        assert "AllReduce" in grads["x"].expr

    def test_preserves_non_partial_dims_2d(self):
        """On 2D mesh: (Partial, Shard(0)) → (Replicate, Shard(0))."""
        m = mesh2d(tp=2, dp=2)
        x = T("x", self.S, (Partial(), Shard(0)), m)
        y = AllReduce(x="x", output="y").apply({"x": x})
        assert is_replicate(y, mesh_dim=0)
        assert is_shard(y, 0, mesh_dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# AllGather
# ═════════════════════════════════════════════════════════════════════════════

class TestAllGatherRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_shard_to_replicate(self):
        x = T("x", self.S, (Shard(0),), self.M)
        y = AllGather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_replicate(y)
        assert y.local_shape == self.S

    def test_shard_other_dim_unchanged(self):
        """S(1) with gather_dim=0 → no change (no S(0) to gather)."""
        x = T("x", self.S, (Shard(1),), self.M)
        y = AllGather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_shard(y, 1)

    def test_replicate_noop(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = AllGather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_replicate(y)

    def test_partial_noop(self):
        """Partial has no Shard to gather → unchanged."""
        x = T("x", self.S, (Partial(),), self.M)
        y = AllGather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_partial(y)

    def test_vjp_dual_is_reduce_scatter(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = AllGather(x="x", output="y", gather_dim=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "ReduceScatter" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# ReduceScatter
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceScatterRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_replicate_to_shard(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = ReduceScatter(x="x", output="y", scatter_dim=0).apply({"x": x})
        assert is_shard(y, 0)
        assert y.local_shape == (4, 4)

    def test_partial_to_shard(self):
        x = T("x", self.S, (Partial(),), self.M)
        y = ReduceScatter(x="x", output="y", scatter_dim=0).apply({"x": x})
        assert is_shard(y, 0)
        assert y.local_shape == (4, 4)

    def test_shard_input_no_change(self):
        """All-Shard input has no R/P to replace → passes through."""
        x = T("x", self.S, (Shard(0),), self.M)
        y = ReduceScatter(x="x", output="y", scatter_dim=1).apply({"x": x})
        assert is_shard(y, 0)

    def test_scatter_dim1(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = ReduceScatter(x="x", output="y", scatter_dim=1).apply({"x": x})
        assert is_shard(y, 1)
        assert y.local_shape == (8, 2)

    def test_vjp_dual_is_allgather(self):
        x = T("x", self.S, (Replicate(),), self.M)
        ctx = {"x": x}
        op = ReduceScatter(x="x", output="y", scatter_dim=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "AllGather" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Broadcast
# ═════════════════════════════════════════════════════════════════════════════

class TestBroadcastRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_replicate_stays(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = Broadcast(x="x", output="y").apply({"x": x})
        assert is_replicate(y)

    def test_shard_to_replicate(self):
        x = T("x", self.S, (Shard(0),), self.M)
        y = Broadcast(x="x", output="y").apply({"x": x})
        assert is_replicate(y)
        assert y.local_shape == self.S

    def test_partial_to_replicate(self):
        x = T("x", self.S, (Partial(),), self.M)
        y = Broadcast(x="x", output="y").apply({"x": x})
        assert is_replicate(y)

    def test_2d_all_replicate(self):
        m = mesh2d()
        x = T("x", self.S, (Shard(0), Partial()), m)
        y = Broadcast(x="x", output="y").apply({"x": x})
        assert is_replicate(y, 0)
        assert is_replicate(y, 1)

    def test_vjp_dual_is_reduce(self):
        x = T("x", self.S, (Replicate(),), self.M)
        ctx = {"x": x}
        op = Broadcast(x="x", output="y", root=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Reduce" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Reduce
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_partial_to_replicate(self):
        x = T("x", self.S, (Partial(),), self.M)
        y = Reduce(x="x", output="y").apply({"x": x})
        assert is_replicate(y)

    def test_rejects_replicate(self):
        x = T("x", self.S, (Replicate(),), self.M)
        with pytest.raises(ValueError, match="PARTIAL"):
            Reduce(x="x", output="y").apply({"x": x})

    def test_rejects_shard(self):
        x = T("x", self.S, (Shard(0),), self.M)
        with pytest.raises(ValueError, match="PARTIAL"):
            Reduce(x="x", output="y").apply({"x": x})

    def test_vjp_dual_is_broadcast(self):
        x = T("x", self.S, (Partial(),), self.M)
        ctx = {"x": x}
        op = Reduce(x="x", output="y", root=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Broadcast" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# AllToAll
# ═════════════════════════════════════════════════════════════════════════════

class TestAllToAllRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_shard_dim_swap(self):
        """S(0) with split=0, concat=1 → S(1)."""
        x = T("x", self.S, (Shard(0),), self.M)
        y = AllToAll(x="x", output="y", split_dim=0, concat_dim=1).apply({"x": x})
        assert is_shard(y, 1)

    def test_no_match_unchanged(self):
        """S(1) with split=0 → no match, stays S(1)."""
        x = T("x", self.S, (Shard(1),), self.M)
        y = AllToAll(x="x", output="y", split_dim=0, concat_dim=1).apply({"x": x})
        assert is_shard(y, 1)

    def test_replicate_unchanged(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = AllToAll(x="x", output="y", split_dim=0, concat_dim=1).apply({"x": x})
        assert is_replicate(y)

    def test_vjp_reverses_dims(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = AllToAll(x="x", output="y", split_dim=0, concat_dim=1)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "1->0" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Scatter
# ═════════════════════════════════════════════════════════════════════════════

class TestScatterRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_replicate_to_shard(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = Scatter(x="x", output="y", scatter_dim=0).apply({"x": x})
        assert is_shard(y, 0)
        assert y.local_shape == (4, 4)

    def test_shard_input_unchanged(self):
        """No Replicate to replace → passes through."""
        x = T("x", self.S, (Shard(0),), self.M)
        y = Scatter(x="x", output="y", scatter_dim=1).apply({"x": x})
        assert is_shard(y, 0)

    def test_partial_input_unchanged(self):
        """Partial is not Replicate → no replacement."""
        x = T("x", self.S, (Partial(),), self.M)
        y = Scatter(x="x", output="y", scatter_dim=0).apply({"x": x})
        assert is_partial(y)

    def test_vjp_dual_is_gather(self):
        x = T("x", self.S, (Replicate(),), self.M)
        ctx = {"x": x}
        op = Scatter(x="x", output="y", scatter_dim=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Gather" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Gather
# ═════════════════════════════════════════════════════════════════════════════

class TestGatherRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_shard_to_replicate(self):
        x = T("x", self.S, (Shard(0),), self.M)
        y = Gather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_replicate(y)
        assert y.local_shape == self.S

    def test_shard_other_dim_unchanged(self):
        x = T("x", self.S, (Shard(1),), self.M)
        y = Gather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_shard(y, 1)

    def test_replicate_noop(self):
        x = T("x", self.S, (Replicate(),), self.M)
        y = Gather(x="x", output="y", gather_dim=0).apply({"x": x})
        assert is_replicate(y)

    def test_vjp_dual_is_scatter(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = Gather(x="x", output="y", gather_dim=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Scatter" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Reshape
# ═════════════════════════════════════════════════════════════════════════════

class TestReshapeRules:
    M = mesh1d(2)

    def test_replicate_shape_change(self):
        x = T("x", (8, 4), (Replicate(),), self.M)
        y = Reshape(x="x", output="y", new_shape=(4, 8)).apply({"x": x})
        assert y.global_shape == (4, 8)
        assert y.local_shape == (4, 8)
        assert is_replicate(y)

    def test_shard0_preserved(self):
        x = T("x", (8, 4), (Shard(0),), self.M)
        y = Reshape(x="x", output="y", new_shape=(8, 2)).apply({"x": x})
        assert y.global_shape == (8, 2)
        assert y.local_shape == (4, 2)
        assert is_shard(y, 0)

    def test_partial_preserved(self):
        x = T("x", (8, 4), (Partial(),), self.M)
        y = Reshape(x="x", output="y", new_shape=(4, 8)).apply({"x": x})
        assert is_partial(y)

    def test_vjp_restores_original_shape(self):
        x = T("x", (8, 4), (Shard(0),), self.M)
        ctx = {"x": x}
        op = Reshape(x="x", output="y", new_shape=(4, 8))
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert grads["x"].global_shape == (8, 4)


# ═════════════════════════════════════════════════════════════════════════════
# Transpose
# ═════════════════════════════════════════════════════════════════════════════

class TestTransposeRules:
    M = mesh1d(2)

    def test_replicate_swap_shape(self):
        x = T("x", (8, 4), (Replicate(),), self.M)
        y = Transpose(x="x", output="y").apply({"x": x})
        assert y.global_shape == (4, 8)
        assert y.local_shape == (4, 8)

    def test_shard0_becomes_shard1(self):
        x = T("x", (8, 4), (Shard(0),), self.M)
        y = Transpose(x="x", output="y").apply({"x": x})
        assert y.global_shape == (4, 8)
        assert is_shard(y, 1)
        assert y.local_shape == (4, 4)

    def test_shard1_becomes_shard0(self):
        x = T("x", (8, 4), (Shard(1),), self.M)
        y = Transpose(x="x", output="y").apply({"x": x})
        assert y.global_shape == (4, 8)
        assert is_shard(y, 0)
        assert y.local_shape == (2, 8)

    def test_partial_unchanged(self):
        x = T("x", (8, 4), (Partial(),), self.M)
        y = Transpose(x="x", output="y").apply({"x": x})
        assert y.global_shape == (4, 8)
        assert is_partial(y)

    def test_vjp_restores_original(self):
        x = T("x", (8, 4), (Shard(0),), self.M)
        ctx = {"x": x}
        op = Transpose(x="x", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert grads["x"].global_shape == (8, 4)
        assert grads["x"].sharding.placements == x.sharding.placements


# ═════════════════════════════════════════════════════════════════════════════
# Send / Recv
# ═════════════════════════════════════════════════════════════════════════════

class TestSendRecvRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_send_preserves_placement(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        y = Send(x="x", output="y", src=0, dst=1, stage=0, microbatch_id=0).apply(ctx)
        assert is_shard(y, 0)
        assert y.stage == 1
        assert y.microbatch_id == 0

    def test_recv_preserves_placement(self):
        x = T("x", self.S, (Replicate(),), self.M)
        ctx = {"x": x}
        y = Recv(x="x", output="y", src=0, dst=1, stage=1, microbatch_id=0).apply(ctx)
        assert is_replicate(y)
        assert y.stage == 1

    def test_send_vjp_mentions_recv(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = Send(x="x", output="y", src=0, dst=1, stage=0, microbatch_id=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Recv" in grads["x"].expr

    def test_recv_vjp_mentions_send(self):
        x = T("x", self.S, (Shard(0),), self.M)
        ctx = {"x": x}
        op = Recv(x="x", output="y", src=0, dst=1, stage=1, microbatch_id=0)
        op.apply(ctx)
        grads = op.vjp(ctx, x)
        assert "Send" in grads["x"].expr


# ═════════════════════════════════════════════════════════════════════════════
# Async ops
# ═════════════════════════════════════════════════════════════════════════════

class TestAsyncRules:
    M = mesh1d(2)
    S = (8, 4)

    def test_allreduce_async_partial_to_replicate(self):
        x = T("x", self.S, (Partial(),), self.M)
        y = AllReduceAsync(
            x="x", output="y", handle="h0"
        ).apply({"x": x})
        assert is_replicate(y)
        assert y._async_handle == "h0"

    def test_allreduce_async_rejects_non_partial(self):
        x = T("x", self.S, (Replicate(),), self.M)
        with pytest.raises(ValueError, match="PARTIAL"):
            AllReduceAsync(x="x", output="y", handle="h0").apply({"x": x})

    def test_wait_clears_handle(self):
        x = T("x", self.S, (Replicate(),), self.M)
        x._async_handle = "h0"
        ctx = {"x": x}
        y = Wait(handle="h0", tensor="x", output="y").apply(ctx)
        assert y._async_handle is None
        assert is_replicate(y)
        assert y.global_shape == self.S

    def test_waitall_clears_handles(self):
        x1 = T("x1", self.S, (Shard(0),), self.M)
        x1._async_handle = "h1"
        x2 = T("x2", self.S, (Replicate(),), self.M)
        x2._async_handle = "h2"
        ctx = {"x1": x1, "x2": x2}
        op = WaitAll(
            handles=("h1", "h2"),
            tensors=("x1", "x2"),
            outputs=("y1", "y2"),
        )
        op.apply(ctx)
        assert ctx["y1"]._async_handle is None
        assert ctx["y2"]._async_handle is None
        assert is_shard(ctx["y1"], 0)
        assert is_replicate(ctx["y2"])

    def test_send_async_sets_handle(self):
        x = T("x", self.S, (Shard(0),), self.M)
        y = SendAsync(
            x="x", output="y", handle="h0",
            src=0, dst=1, stage=0, microbatch_id=0,
        ).apply({"x": x})
        assert y._async_handle == "h0"
        assert is_shard(y, 0)


# ═════════════════════════════════════════════════════════════════════════════
# Reinterpret
# ═════════════════════════════════════════════════════════════════════════════

class TestReinterpretRules:

    def _make(self, src, dst, expert=False):
        return Reinterpret(x="x", output="y",
                           src_type=src, dst_type=dst, expert_mode=expert)

    # ── valid transitions ───────────────────────────────────────────────

    def test_r_to_v(self):
        op = self._make(LocalSPMDType.REPLICATE, LocalSPMDType.VARYING)
        assert op.propagate_spmd_type({"x": LocalSPMDType.REPLICATE}) == LocalSPMDType.VARYING

    def test_r_to_p(self):
        op = self._make(LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL)
        assert op.propagate_spmd_type({"x": LocalSPMDType.REPLICATE}) == LocalSPMDType.PARTIAL

    def test_v_to_p(self):
        op = self._make(LocalSPMDType.VARYING, LocalSPMDType.PARTIAL)
        assert op.propagate_spmd_type({}) == LocalSPMDType.PARTIAL

    def test_p_to_v(self):
        op = self._make(LocalSPMDType.PARTIAL, LocalSPMDType.VARYING)
        assert op.propagate_spmd_type({}) == LocalSPMDType.VARYING

    # ── expert transitions ──────────────────────────────────────────────

    def test_r_to_i_requires_expert(self):
        with pytest.raises(ValueError, match="expert_mode"):
            self._make(LocalSPMDType.REPLICATE, LocalSPMDType.INVARIANT)

    def test_r_to_i_with_expert(self):
        op = self._make(LocalSPMDType.REPLICATE, LocalSPMDType.INVARIANT, expert=True)
        assert op.propagate_spmd_type({}) == LocalSPMDType.INVARIANT

    def test_v_to_r_requires_expert(self):
        with pytest.raises(ValueError, match="expert_mode"):
            self._make(LocalSPMDType.VARYING, LocalSPMDType.REPLICATE)

    def test_v_to_r_with_expert(self):
        op = self._make(LocalSPMDType.VARYING, LocalSPMDType.REPLICATE, expert=True)
        assert op.propagate_spmd_type({}) == LocalSPMDType.REPLICATE

    # ── invalid transitions ─────────────────────────────────────────────

    def test_p_to_r_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            self._make(LocalSPMDType.PARTIAL, LocalSPMDType.REPLICATE)

    def test_i_to_v_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            self._make(LocalSPMDType.INVARIANT, LocalSPMDType.VARYING)

    # ── apply changes local_type ────────────────────────────────────────

    def test_apply_changes_type(self):
        m = mesh1d(2)
        x = T("x", (8, 4), (Replicate(),), m)
        op = self._make(LocalSPMDType.REPLICATE, LocalSPMDType.VARYING)
        result = op.apply({"x": x})
        assert result.local_type == LocalSPMDType.VARYING


# ═════════════════════════════════════════════════════════════════════════════
# Convert
# ═════════════════════════════════════════════════════════════════════════════

class TestConvertRules:

    def _make(self, src, dst):
        return Convert(x="x", output="y", src_type=src, dst_type=dst)

    # ── valid transitions ───────────────────────────────────────────────

    def test_r_to_p(self):
        op = self._make(LocalSPMDType.REPLICATE, LocalSPMDType.PARTIAL)
        assert op.propagate_spmd_type({}) == LocalSPMDType.PARTIAL

    def test_p_to_r(self):
        op = self._make(LocalSPMDType.PARTIAL, LocalSPMDType.REPLICATE)
        assert op.propagate_spmd_type({}) == LocalSPMDType.REPLICATE

    def test_i_to_r(self):
        op = self._make(LocalSPMDType.INVARIANT, LocalSPMDType.REPLICATE)
        assert op.propagate_spmd_type({}) == LocalSPMDType.REPLICATE

    def test_i_to_v(self):
        op = self._make(LocalSPMDType.INVARIANT, LocalSPMDType.VARYING)
        assert op.propagate_spmd_type({}) == LocalSPMDType.VARYING

    def test_i_to_p(self):
        op = self._make(LocalSPMDType.INVARIANT, LocalSPMDType.PARTIAL)
        assert op.propagate_spmd_type({}) == LocalSPMDType.PARTIAL

    def test_v_to_p(self):
        op = self._make(LocalSPMDType.VARYING, LocalSPMDType.PARTIAL)
        assert op.propagate_spmd_type({}) == LocalSPMDType.PARTIAL

    def test_p_to_v(self):
        op = self._make(LocalSPMDType.PARTIAL, LocalSPMDType.VARYING)
        assert op.propagate_spmd_type({}) == LocalSPMDType.VARYING

    # ── invalid transitions ─────────────────────────────────────────────

    def test_r_to_v_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            self._make(LocalSPMDType.REPLICATE, LocalSPMDType.VARYING)

    def test_v_to_r_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            self._make(LocalSPMDType.VARYING, LocalSPMDType.REPLICATE)

    # ── apply changes local_type ────────────────────────────────────────

    def test_apply_changes_type(self):
        m = mesh1d(2)
        x = T("x", (8, 4), (Partial(),), m)
        op = self._make(LocalSPMDType.PARTIAL, LocalSPMDType.REPLICATE)
        result = op.apply({"x": x})
        assert result.local_type == LocalSPMDType.REPLICATE
