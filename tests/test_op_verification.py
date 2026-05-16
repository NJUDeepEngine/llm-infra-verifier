"""How to verify a distributed operator — end-to-end examples.

Each test class demonstrates one verification approach:

  1. SingleOpPlacement   — apply() an op, inspect output placement/shape
  2. SPMDTypeChecking    — apply_checked() catches type mismatches
  3. VJPCorrectness      — vjp() produces gradients with correct metadata
  4. Z3FormalProof       — Z3 solver proves properties for ALL possible inputs
  5. MultiDeviceExec     — MultiDeviceExecutor runs across a device mesh
  6. ComposedPipeline    — full verify loop: build program → execute → Z3 check
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
    compute_tensor_slices,
)
from verifier.ir import (
    MatMul,
    Add,
    Multiply,
    AllReduce,
    AllGather,
    ReduceScatter,
    Reshape,
    SiLU,
    GELU,
    ReLU,
    Dropout,
    LayerNorm,
    RMSNorm,
    Softmax,
    Embedding,
    CrossEntropyLoss,
    Program,
    ir_to_str,
    SPMDConsistencyError,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import Z3PlacementSolver, VerifyResult
from verifier.autograd import AutogradEngine


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mesh(tp=2):
    """2-device TP mesh."""
    return DeviceMesh(shape=(tp,), dim_names=("tp",))


def make_tensor(name, shape, placement, mesh):
    """Create a TensorState with derived local shape."""
    spec = ShardingSpec(placements=(placement,), mesh=mesh)
    return TensorState(
        name=name,
        global_shape=shape,
        local_shape=compute_local_shape(shape, spec),
        sharding=spec,
        expr=name,
        requires_grad=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. Single-op placement propagation
# ═════════════════════════════════════════════════════════════════════════════

class TestSingleOpPlacement:
    """Verify that apply() produces the correct output placement and shape."""

    def test_matmul_row_parallel(self):
        """Row-parallel: A=Shard(0), B=Replicate → Y=Shard(0).

        Each device holds different rows of A but the full B,
        so the output is naturally sharded on rows — no communication.
        """
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Shard(dim=0), mesh)
        b = make_tensor("B", (4, 6), Replicate(), mesh)

        ctx = {"A": a, "B": b}
        op = MatMul(a="A", b="B", output="Y")
        result = op.apply(ctx)

        assert result.global_shape == (8, 6)
        assert result.local_shape == (4, 6)
        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 0
        assert not result.partial

    def test_matmul_column_parallel(self):
        """Column-parallel: A=Replicate, B=Shard(1) → Y=Shard(1).

        Each device holds the full A but different columns of B,
        so the output is sharded on columns.
        """
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Replicate(), mesh)
        b = make_tensor("B", (4, 6), Shard(dim=1), mesh)

        ctx = {"A": a, "B": b}
        result = MatMul(a="A", b="B", output="Y").apply(ctx)

        assert result.global_shape == (8, 6)
        assert result.local_shape == (8, 3)
        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 1

    def test_matmul_contraction_produces_partial(self):
        """A=Shard(1), B=Shard(0) → Y=Partial.

        Both sides shard the contraction dim — each device computes a
        partial result that must be AllReduced.
        """
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Shard(dim=1), mesh)
        b = make_tensor("B", (4, 6), Shard(dim=0), mesh)

        ctx = {"A": a, "B": b}
        result = MatMul(a="A", b="B", output="Y").apply(ctx)

        assert result.partial
        assert isinstance(result.sharding.placements[0], Partial)

    def test_add_inherits_shard_from_non_replicate(self):
        """Add: Replicate + Shard(0) → Shard(0).

        The replicate input conforms to whatever the other input's
        placement is.
        """
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Replicate(), mesh)
        b = make_tensor("B", (8, 4), Shard(dim=0), mesh)

        ctx = {"A": a, "B": b}
        result = Add(a="A", b="B", output="Y").apply(ctx)

        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 0

    def test_allreduce_partial_to_replicate(self):
        """AllReduce: Partial → Replicate.

        Sums partial results across devices. This is the only valid
        input state for AllReduce.
        """
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Partial(), mesh)

        ctx = {"X": x}
        result = AllReduce(x="X", output="Y").apply(ctx)

        assert isinstance(result.sharding.placements[0], Replicate)
        assert not result.partial

    def test_allreduce_rejects_non_partial(self):
        """AllReduce on Replicate input is an error — no pending sum."""
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Replicate(), mesh)

        ctx = {"X": x}
        with pytest.raises(ValueError, match="PARTIAL"):
            AllReduce(x="X", output="Y").apply(ctx)

    def test_allgather_shard_to_replicate(self):
        """AllGather: Shard(0) → Replicate.

        Gathers the sharded dim so every device has the full tensor.
        """
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Shard(dim=0), mesh)

        ctx = {"X": x}
        result = AllGather(x="X", output="Y", gather_dim=0).apply(ctx)

        assert isinstance(result.sharding.placements[0], Replicate)
        assert result.local_shape == (8, 4)

    def test_reduce_scatter_to_shard(self):
        """ReduceScatter: Partial → Shard(0).

        Reduces then scatters — output is sharded.
        """
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Partial(), mesh)

        ctx = {"X": x}
        result = ReduceScatter(x="X", output="Y", scatter_dim=0).apply(ctx)

        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 0
        assert result.local_shape == (4, 4)


# ═════════════════════════════════════════════════════════════════════════════
# 2. SPMD type checking (apply_checked)
# ═════════════════════════════════════════════════════════════════════════════

class TestSPMDTypeChecking:
    """apply_checked() cross-validates SPMD type propagation vs placement."""

    def test_consistent_matmul(self):
        """When SPMD type and placement agree, apply_checked succeeds."""
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Replicate(), mesh)
        b = make_tensor("B", (4, 6), Replicate(), mesh)

        ctx = {"A": a, "B": b}
        result = MatMul(a="A", b="B", output="Y").apply_checked(ctx)

        assert result.local_type == LocalSPMDType.REPLICATE

    def test_multiply_partial_partial_is_forbidden(self):
        """SPMD rule: Partial * Partial is semantically wrong.

        (a+b) * (c+d) != (a*c) + (b*d), so element-wise multiply
        on two PARTIAL tensors cannot produce a valid result.
        """
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Partial(), mesh)
        b = make_tensor("B", (8, 4), Partial(), mesh)

        ctx = {"A": a, "B": b}
        with pytest.raises(ValueError, match="Partial .* Partial"):
            Multiply(a="A", b="B", output="Y").apply(ctx)

    def test_silu_preserves_type(self):
        """Element-wise unary ops preserve SPMD type."""
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Shard(dim=0), mesh)

        ctx = {"X": x}
        result = SiLU(x="X", output="Y").apply_checked(ctx)

        assert result.local_type == LocalSPMDType.VARYING

    def test_allreduce_produces_replicate(self):
        """AllReduce: Partial → Replicate at the SPMD type level."""
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Partial(), mesh)

        ctx = {"X": x}
        result = AllReduce(x="X", output="Y").apply_checked(ctx)

        assert result.local_type == LocalSPMDType.REPLICATE


# ═════════════════════════════════════════════════════════════════════════════
# 3. VJP (gradient) correctness
# ═════════════════════════════════════════════════════════════════════════════

class TestVJPCorrectness:
    """Verify that vjp() produces gradients with correct shape and placement."""

    def test_matmul_grad_shapes_match_inputs(self):
        """MatMul VJP: grad_A.shape == A.shape, grad_B.shape == B.shape."""
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Shard(dim=0), mesh)
        b = make_tensor("B", (4, 6), Replicate(), mesh)

        ctx = {"A": a, "B": b}
        op = MatMul(a="A", b="B", output="Y")
        result = op.apply(ctx)

        grad_out = result.with_name("grad_Y")
        grads = op.vjp(ctx, grad_out)

        assert grads["A"].global_shape == a.global_shape
        assert grads["B"].global_shape == b.global_shape
        assert grads["A"].local_shape == a.local_shape
        assert grads["B"].local_shape == b.local_shape

    def test_matmul_grad_placements_match_inputs(self):
        """Gradient placements should match the corresponding input."""
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Shard(dim=0), mesh)
        b = make_tensor("B", (4, 6), Replicate(), mesh)

        ctx = {"A": a, "B": b}
        op = MatMul(a="A", b="B", output="Y")
        op.apply(ctx)

        grad_out = a.with_name("grad_Y")
        grads = op.vjp(ctx, grad_out)

        assert grads["A"].sharding.placements == a.sharding.placements
        assert grads["B"].sharding.placements == b.sharding.placements

    def test_add_grad_is_identity(self):
        """Add VJP: grad flows through unchanged to both inputs."""
        mesh = make_mesh(tp=2)
        a = make_tensor("A", (8, 4), Shard(dim=0), mesh)
        b = make_tensor("B", (8, 4), Shard(dim=0), mesh)

        ctx = {"A": a, "B": b}
        op = Add(a="A", b="B", output="Y")
        result = op.apply(ctx)

        grad_out = result.with_name("grad_Y")
        grads = op.vjp(ctx, grad_out)

        assert grads["A"].global_shape == a.global_shape
        assert grads["B"].global_shape == b.global_shape

    def test_allreduce_is_self_dual(self):
        """AllReduce VJP: backward of AllReduce is AllReduce."""
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Partial(), mesh)

        ctx = {"X": x}
        op = AllReduce(x="X", output="Y")
        result = op.apply(ctx)

        grad_out = result.with_name("grad_Y")
        grads = op.vjp(ctx, grad_out)

        assert "AllReduce" in grads["X"].expr

    def test_allgather_dual_is_reduce_scatter(self):
        """AllGather VJP: backward is ReduceScatter."""
        mesh = make_mesh(tp=2)
        x = make_tensor("X", (8, 4), Shard(dim=0), mesh)

        ctx = {"X": x}
        op = AllGather(x="X", output="Y", gather_dim=0)
        result = op.apply(ctx)

        grad_out = result.with_name("grad_Y")
        grads = op.vjp(ctx, grad_out)

        assert "ReduceScatter" in grads["X"].expr


# ═════════════════════════════════════════════════════════════════════════════
# 4. Z3 formal verification
# ═════════════════════════════════════════════════════════════════════════════

class TestZ3FormalProof:
    """Use Z3 to prove properties hold for ALL possible placements."""

    def test_tp_row_parallel_output_is_replicate(self):
        """Prove: row-parallel MatMul + AllReduce always yields Replicate.

        Row-parallel TP pattern (Megatron-LM second half of MLP):
          x is Shard(1) — hidden dim sharded from previous column-parallel layer
          w is Shard(0) — weight rows sharded to match
          Y_partial = x @ w  → contraction dim sharded on both sides → Partial
          AllReduce(Y_partial) → Replicate

        Z3 proves output is Replicate for any concrete shape.
        """
        program = Program("tp_row_parallel", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])

        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        results = solver.check_output_equivalence(["y"])
        assert all(r.passed for r in results)

    def test_allreduce_precondition_verified(self):
        """Prove: AllReduce input is always Partial in the row-parallel pattern."""
        program = Program("tp_row_parallel", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])

        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        results = solver.check_collective_preconditions(program)
        assert all(r.passed for r in results)

    def test_missing_allreduce_detected(self):
        """Detect: without AllReduce, row-parallel output is NOT Replicate."""
        program = Program("broken_tp", ops=[
            MatMul(a="x", b="w", output="y"),
        ])

        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        results = solver.check_output_equivalence(["y"])
        assert any(not r.passed for r in results)

    def test_shape_consistency(self):
        """Z3 verifies shapes propagate correctly through MatMul."""
        program = Program("shape_check", ops=[
            MatMul(a="x", b="w", output="y"),
        ])

        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.add_input_shape("x", (32, 16))
        solver.add_input_shape("w", (16, 64))
        solver.encode_program(program)
        solver.encode_shape_constraints(program, tp_size=2)

        results = solver.check_shape_consistency()
        assert results[0].passed, f"Shape consistency failed: {results[0].details}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Multi-device execution
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiDeviceExecution:
    """MultiDeviceExecutor tracks per-device state across the mesh."""

    def test_tp_row_parallel_per_device(self):
        """Run row-parallel TP on 2 devices, verify each device's state.

        Row-parallel: x is already Shard(1) from a previous column-parallel
        layer, w is Shard(0). The contraction dim is sharded on both sides,
        producing Partial → AllReduce → Replicate.
        """
        mesh = make_mesh(tp=2)
        executor = MultiDeviceExecutor(mesh=mesh)

        x_spec = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(32, 16),
            local_shape=compute_local_shape((32, 16), x_spec),
            sharding=x_spec, expr="x", requires_grad=True,
        )

        w_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        w = TensorState(
            name="w", global_shape=(16, 64),
            local_shape=compute_local_shape((16, 64), w_spec),
            sharding=w_spec, expr="w", requires_grad=True,
        )

        executor.register_tensor(x)
        executor.register_tensor(w)

        program = Program("tp_row_parallel", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])
        final = executor.run_program(program)

        y = final["y"]
        assert y.global_shape == (32, 64)
        assert isinstance(y.sharding.placements[0], Replicate)

        for did in range(2):
            dev_y = executor.get_tensor("y", device_id=did)
            assert dev_y is not None
            assert dev_y.local_shape == (32, 64)

    def test_tensor_slices(self):
        """Verify per-device slices are computed correctly."""
        mesh = make_mesh(tp=2)

        spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        slices = compute_tensor_slices((8, 4), spec)

        assert slices[0].offsets == (0, 0)
        assert slices[0].local_shape == (4, 4)
        assert slices[1].offsets == (4, 0)
        assert slices[1].local_shape == (4, 4)

    def test_spmd_checking_in_executor(self):
        """Executor with spmd_checking=True validates SPMD types."""
        mesh = make_mesh(tp=2)
        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)

        x_spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(8, 4),
            local_shape=compute_local_shape((8, 4), x_spec),
            sharding=x_spec, expr="x", requires_grad=True,
        )
        w_spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        w = TensorState(
            name="w", global_shape=(4, 6),
            local_shape=compute_local_shape((4, 6), w_spec),
            sharding=w_spec, expr="w", requires_grad=True,
        )

        executor.register_tensor(x)
        executor.register_tensor(w)

        program = Program("rr_matmul", ops=[
            MatMul(a="x", b="w", output="y"),
        ])
        final = executor.run_program(program)
        assert final["y"].local_type == LocalSPMDType.REPLICATE


# ═════════════════════════════════════════════════════════════════════════════
# 6. Composed pipeline: build → execute → Z3 verify
# ═════════════════════════════════════════════════════════════════════════════

class TestComposedPipeline:
    """Full verification loop: symbolic execution + formal proof."""

    def test_mlp_block(self):
        """Verify a Megatron-style tensor-parallel MLP block:

            Column parallel:
              x (Replicate) @ w1 (Shard(1)) → h (Shard(1))
              SiLU(h)                        → h_act (Shard(1))

            Row parallel:
              h_act (Shard(1)) @ w2 (Shard(0)) → out_partial (Partial)
              AllReduce(out_partial)             → out (Replicate)

        Step 1: Symbolic execution confirms shapes and placements
        Step 2: Z3 proves output is always Replicate
        Step 3: Z3 confirms AllReduce preconditions
        """
        mesh = make_mesh(tp=2)

        # Step 1: Symbolic execution
        executor = MultiDeviceExecutor(mesh=mesh)

        x_spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(32, 128),
            local_shape=compute_local_shape((32, 128), x_spec),
            sharding=x_spec, expr="x", requires_grad=True,
        )
        w1_spec = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        w1 = TensorState(
            name="w1", global_shape=(128, 256),
            local_shape=compute_local_shape((128, 256), w1_spec),
            sharding=w1_spec, expr="w1", requires_grad=True,
        )
        w2_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        w2 = TensorState(
            name="w2", global_shape=(256, 128),
            local_shape=compute_local_shape((256, 128), w2_spec),
            sharding=w2_spec, expr="w2", requires_grad=True,
        )

        executor.register_tensor(x)
        executor.register_tensor(w1)
        executor.register_tensor(w2)

        program = Program("tp_mlp", ops=[
            MatMul(a="x", b="w1", output="h"),
            SiLU(x="h", output="h_act"),
            MatMul(a="h_act", b="w2", output="out_partial"),
            AllReduce(x="out_partial", output="out"),
        ])

        final = executor.run_program(program)

        assert final["h"].global_shape == (32, 256)
        assert isinstance(final["h"].sharding.placements[0], Shard)
        assert final["out"].global_shape == (32, 128)
        assert isinstance(final["out"].sharding.placements[0], Replicate)

        # Step 2: Z3 proves output equivalence
        solver = Z3PlacementSolver()
        solver.add_input("x", Replicate())
        solver.add_input("w1", Shard(dim=1))
        solver.add_input("w2", Shard(dim=0))
        solver.encode_program(program)

        equiv = solver.check_output_equivalence(["out"])
        assert all(r.passed for r in equiv), (
            f"Output not proven Replicate: {equiv}"
        )

        # Step 3: Z3 confirms AllReduce preconditions
        precond = solver.check_collective_preconditions(program)
        assert all(r.passed for r in precond), (
            f"AllReduce precondition violated: {precond}"
        )

    def test_autograd_engine(self):
        """AutogradEngine verifies fwd/bwd collective duality.

        Workflow: record forward ops on tape → generate backward →
        verify_gradient_correctness checks shapes, placements, and
        collective duality (AllReduce ↔ AllReduce).
        """
        mesh = make_mesh(tp=2)

        x_spec = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(8, 4),
            local_shape=compute_local_shape((8, 4), x_spec),
            sharding=x_spec, expr="x", requires_grad=True,
        )
        w_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        w = TensorState(
            name="w", global_shape=(4, 8),
            local_shape=compute_local_shape((4, 8), w_spec),
            sharding=w_spec, expr="w", requires_grad=True,
        )

        fwd_program = Program("tp_linear", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])

        engine = AutogradEngine()

        ctx = {"x": x, "w": w}
        for op in fwd_program.ops:
            result = op.apply(ctx)
            engine.record(op, ctx)

        bwd_program = engine.generate_backward("y")
        check = engine.verify_gradient_correctness(fwd_program, bwd_program)

        assert check.passed, f"Gradient check failed: {check.errors}"
        assert check.fwd_ops == 2
        assert check.bwd_ops > 0
        assert len(check.collective_pairs) > 0

    def test_program_pretty_print(self):
        """ir_to_str produces a readable program listing."""
        program = Program("demo", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])

        text = ir_to_str(program)
        assert "MatMul" in text
        assert "AllReduce" in text
        assert "[0]" in text
        assert "[1]" in text


# ═════════════════════════════════════════════════════════════════════════════
# 7. New compute ops: activations, normalization, embedding, loss
# ═════════════════════════════════════════════════════════════════════════════

class TestNewComputeOps:
    """Verify placement propagation and sharding constraints for new ops."""

    # ── Element-wise activations ────────────────────────────────────────────

    def test_gelu_preserves_placement(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 4), Shard(dim=1), mesh)
        ctx = {"x": x}
        result = GELU(x="x", output="y").apply(ctx)
        assert result.global_shape == (8, 4)
        assert isinstance(result.sharding.placements[0], Shard)
        assert result.sharding.placements[0].dim == 1

    def test_relu_preserves_placement(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 4), Replicate(), mesh)
        ctx = {"x": x}
        result = ReLU(x="x", output="y").apply(ctx)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_dropout_preserves_placement(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 4), Partial(), mesh)
        ctx = {"x": x}
        result = Dropout(x="x", output="y", p=0.1).apply(ctx)
        assert isinstance(result.sharding.placements[0], Partial)

    def test_gelu_vjp(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 4), Shard(dim=0), mesh)
        ctx = {"x": x}
        op = GELU(x="x", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, ctx["y"])
        assert "x" in grads
        assert grads["x"].sharding == x.sharding

    # ── Normalization ops ───────────────────────────────────────────────────

    def test_layernorm_replicate_ok(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Replicate(), mesh)
        ctx = {"x": x}
        result = LayerNorm(x="x", output="y", norm_dim=-1).apply(ctx)
        assert result.global_shape == (8, 128)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_layernorm_batch_shard_ok(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=0), mesh)
        ctx = {"x": x}
        result = LayerNorm(x="x", output="y", norm_dim=1).apply(ctx)
        assert isinstance(result.sharding.placements[0], Shard)

    def test_layernorm_shard_on_norm_dim_error(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=1), mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on norm_dim"):
            LayerNorm(x="x", output="y", norm_dim=1).apply(ctx)

    def test_layernorm_negative_norm_dim_error(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=1), mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on norm_dim"):
            LayerNorm(x="x", output="y", norm_dim=-1).apply(ctx)

    def test_rmsnorm_shard_on_norm_dim_error(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=1), mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on norm_dim"):
            RMSNorm(x="x", output="y", norm_dim=-1).apply(ctx)

    def test_rmsnorm_batch_shard_ok(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=0), mesh)
        ctx = {"x": x}
        result = RMSNorm(x="x", output="y", norm_dim=-1).apply(ctx)
        assert isinstance(result.sharding.placements[0], Shard)

    def test_softmax_replicate_ok(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Replicate(), mesh)
        ctx = {"x": x}
        result = Softmax(x="x", output="y", dim=-1).apply(ctx)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_softmax_shard_on_reduction_dim_error(self):
        mesh = make_mesh(tp=2)
        x = make_tensor("x", (8, 128), Shard(dim=1), mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on dim"):
            Softmax(x="x", output="y", dim=-1).apply(ctx)

    # ── Vocab-parallel ops ──────────────────────────────────────────────────

    def test_embedding_replicate_weight(self):
        mesh = make_mesh(tp=2)
        indices = make_tensor("ids", (32,), Replicate(), mesh)
        weight = make_tensor("W_emb", (50000, 128), Replicate(), mesh)
        ctx = {"ids": indices, "W_emb": weight}
        result = Embedding(indices="ids", weight="W_emb", output="emb").apply(ctx)
        assert result.global_shape == (32, 128)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_embedding_vocab_parallel(self):
        mesh = make_mesh(tp=2)
        indices = make_tensor("ids", (32,), Replicate(), mesh)
        weight = make_tensor("W_emb", (50000, 128), Shard(dim=0), mesh)
        ctx = {"ids": indices, "W_emb": weight}
        result = Embedding(indices="ids", weight="W_emb", output="emb").apply(ctx)
        assert result.global_shape == (32, 128)
        assert isinstance(result.sharding.placements[0], Partial)

    def test_embedding_vjp(self):
        mesh = make_mesh(tp=2)
        indices = make_tensor("ids", (32,), Replicate(), mesh)
        weight = make_tensor("W_emb", (50000, 128), Shard(dim=0), mesh)
        ctx = {"ids": indices, "W_emb": weight}
        op = Embedding(indices="ids", weight="W_emb", output="emb")
        op.apply(ctx)
        grads = op.vjp(ctx, ctx["emb"])
        assert "W_emb" in grads
        assert grads["W_emb"].sharding == weight.sharding

    def test_cross_entropy_replicate_logits(self):
        mesh = make_mesh(tp=2)
        logits = make_tensor("logits", (32, 50000), Replicate(), mesh)
        targets = make_tensor("targets", (32,), Replicate(), mesh)
        ctx = {"logits": logits, "targets": targets}
        result = CrossEntropyLoss(
            logits="logits", targets="targets", output="loss",
        ).apply(ctx)
        assert result.global_shape == (1,)
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_cross_entropy_vocab_parallel(self):
        mesh = make_mesh(tp=2)
        logits = make_tensor("logits", (32, 50000), Shard(dim=1), mesh)
        targets = make_tensor("targets", (32,), Replicate(), mesh)
        ctx = {"logits": logits, "targets": targets}
        result = CrossEntropyLoss(
            logits="logits", targets="targets", output="loss", vocab_dim=-1,
        ).apply(ctx)
        assert result.global_shape == (1,)
        assert isinstance(result.sharding.placements[0], Partial)

    def test_cross_entropy_vjp(self):
        mesh = make_mesh(tp=2)
        logits = make_tensor("logits", (32, 50000), Shard(dim=1), mesh)
        targets = make_tensor("targets", (32,), Replicate(), mesh)
        ctx = {"logits": logits, "targets": targets}
        op = CrossEntropyLoss(
            logits="logits", targets="targets", output="loss",
        )
        op.apply(ctx)
        grads = op.vjp(ctx, ctx["loss"])
        assert "logits" in grads
        assert grads["logits"].sharding == logits.sharding

    # ── Clone / repr coverage ───────────────────────────────────────────────

    def test_clone_with_names(self):
        op = LayerNorm(x="x", output="y", norm_dim=-1)
        cloned = op.clone_with_names({"x": "a"}, "b")
        assert cloned.x == "a"
        assert cloned.output == "b"
        assert cloned.norm_dim == -1

    def test_repr_all_new_ops(self):
        ops = [
            GELU(x="x", output="y"),
            ReLU(x="x", output="y"),
            Dropout(x="x", output="y", p=0.1),
            LayerNorm(x="x", output="y"),
            RMSNorm(x="x", output="y"),
            Softmax(x="x", output="y"),
            Embedding(indices="ids", weight="W", output="emb"),
            CrossEntropyLoss(logits="l", targets="t", output="loss"),
        ]
        for op in ops:
            r = repr(op)
            assert type(op).__name__ in r

    # ── Multi-device execution ──────────────────────────────────────────────

    def test_executor_gelu_layernorm(self):
        mesh = make_mesh(tp=2)
        executor = MultiDeviceExecutor(mesh=mesh)

        x_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        x = TensorState(
            name="x", global_shape=(8, 128),
            local_shape=compute_local_shape((8, 128), x_spec),
            sharding=x_spec, expr="x", requires_grad=True,
        )
        executor.register_tensor(x)

        program = Program("test_new_ops", ops=[
            GELU(x="x", output="h"),
            LayerNorm(x="h", output="y", norm_dim=-1),
        ])
        final = executor.run_program(program)
        assert final["y"].global_shape == (8, 128)
        assert isinstance(final["y"].sharding.placements[0], Shard)

    def test_executor_embedding_allreduce(self):
        mesh = make_mesh(tp=2)
        executor = MultiDeviceExecutor(mesh=mesh)

        ids_spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        ids = TensorState(
            name="ids", global_shape=(32,),
            local_shape=(32,), sharding=ids_spec, expr="ids",
        )
        w_spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        w = TensorState(
            name="W", global_shape=(50000, 128),
            local_shape=compute_local_shape((50000, 128), w_spec),
            sharding=w_spec, expr="W", requires_grad=True,
        )
        executor.register_tensor(ids)
        executor.register_tensor(w)

        program = Program("vocab_parallel_embed", ops=[
            Embedding(indices="ids", weight="W", output="emb_partial"),
            AllReduce(x="emb_partial", output="emb"),
        ])
        final = executor.run_program(program)
        assert isinstance(final["emb"].sharding.placements[0], Replicate)
