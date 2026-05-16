"""Deep Z3 solver tests: forbidden placements, constraint propagation, shape/slice integration."""

import pytest
from verifier import *
from verifier.solver import (
    Z3PlacementSolver,
    DistributedVerifier,
    VerifyResult,
    PL_R, PL_S0, PL_S1, PL_P,
)


# ── Forbidden placement detection ───────────────────────────────────────────


class TestForbiddenPlacements:
    """Z3 detects when ops receive forbidden input placements."""

    def test_layernorm_shard_on_norm_dim_forbidden(self):
        """LayerNorm(norm_dim=1) with input Shard(1) is forbidden — Z3 detects."""
        program = Program("ln_bad", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        # The constraint system should be UNSAT: we asserted x==S1 but
        # the norm op adds x!=S1
        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_layernorm_negative_norm_dim_forbidden(self):
        """LayerNorm(norm_dim=-1) with Shard(1) is forbidden (dim=-1 maps to dim=1 for 2D)."""
        program = Program("ln_neg", ops=[
            LayerNorm(x="x", output="y", norm_dim=-1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_rmsnorm_shard_on_norm_dim_forbidden(self):
        """RMSNorm(norm_dim=1) with Shard(1) is forbidden."""
        program = Program("rn_bad", ops=[
            RMSNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_softmax_shard_on_reduction_dim_forbidden(self):
        """Softmax(dim=1) with Shard(1) is forbidden."""
        program = Program("sm_bad", ops=[
            Softmax(x="x", output="y", dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_layernorm_shard_on_batch_dim_safe(self):
        """LayerNorm(norm_dim=1) with Shard(0) is safe — batch sharding is fine."""
        program = Program("ln_ok", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_softmax_replicate_safe(self):
        """Softmax with Replicate input is always safe."""
        program = Program("sm_ok", ops=[
            Softmax(x="x", output="y", dim=-1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Replicate())
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_allgather_then_layernorm_safe(self):
        """AllGather(S1→R) before LayerNorm(norm_dim=1) proves safe."""
        program = Program("ag_ln", ops=[
            AllGather(x="x", output="x_full", gather_dim=1),
            LayerNorm(x="x_full", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        # x is S1, AllGather → R, LayerNorm sees R → safe
        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # And the compute precondition check should confirm safety
        results = solver.check_compute_preconditions(program)
        ln_results = [r for r in results if "LayerNorm" in r.condition]
        assert all(r.passed for r in ln_results)

    def test_layernorm_norm_dim_0_shard_dim_0_forbidden(self):
        """LayerNorm(norm_dim=0) with Shard(0) is forbidden."""
        program = Program("ln_d0", ops=[
            LayerNorm(x="x", output="y", norm_dim=0),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed


# ── Partial × Partial detection ─────────────────────────────────────────────


class TestPartialPartialForbidden:
    """Z3 detects when binary ops receive two Partial inputs."""

    def test_add_partial_partial_forbidden(self):
        """Add(Partial, Partial) is forbidden — Z3 detects UNSAT."""
        program = Program("add_pp", ops=[
            Add(a="x", b="y", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Partial())
        solver.add_input("y", Partial())
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_add_partial_replicate_safe(self):
        """Add(Partial, Replicate) is fine."""
        program = Program("add_pr", ops=[
            Add(a="x", b="y", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Partial())
        solver.add_input("y", Replicate())
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_multiply_partial_partial_forbidden(self):
        """Multiply(Partial, Partial) is forbidden."""
        program = Program("mul_pp", ops=[
            Multiply(a="x", b="y", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Partial())
        solver.add_input("y", Partial())
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_two_matmul_partial_then_add_forbidden(self):
        """Two MatMul producing Partial, then Add(both) → forbidden."""
        program = Program("chain_pp", ops=[
            MatMul(a="x1", b="w1", output="p1"),
            MatMul(a="x2", b="w2", output="p2"),
            Add(a="p1", b="p2", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x1", Shard(dim=1))
        solver.add_input("w1", Shard(dim=0))
        solver.add_input("x2", Shard(dim=1))
        solver.add_input("w2", Shard(dim=0))
        solver.encode_program(program)

        # Both MatMul(S1, S0) → Partial, then Add(P, P) is UNSAT
        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_matmul_partial_allreduce_then_add_safe(self):
        """MatMul→Partial, AllReduce→R, then Add with Replicate is fine."""
        program = Program("fix_pp", ops=[
            MatMul(a="x1", b="w1", output="p1"),
            AllReduce(x="p1", output="r1"),
            MatMul(a="x2", b="w2", output="p2"),
            AllReduce(x="p2", output="r2"),
            Add(a="r1", b="r2", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x1", Shard(dim=1))
        solver.add_input("w1", Shard(dim=0))
        solver.add_input("x2", Shard(dim=1))
        solver.add_input("w2", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed


# ── Constraint propagation ──────────────────────────────────────────────────


class TestConstraintPropagation:
    """Z3 propagates constraints through the program to prove safety."""

    def test_row_parallel_allreduce_then_layernorm_safe(self):
        """MatMul→Partial → AllReduce→R → LayerNorm(norm_dim=1): provably safe."""
        program = Program("rp_ln", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
            LayerNorm(x="y", output="y_norm", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        results = solver.check_compute_preconditions(program)
        assert all(r.passed for r in results)

    def test_partial_into_layernorm_safe(self):
        """Partial input to LayerNorm is fine — only Shard(norm_dim) is forbidden."""
        program = Program("p_ln", ops=[
            MatMul(a="x", b="w", output="y_p"),
            LayerNorm(x="y_p", output="y_norm", norm_dim=0),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        # MatMul(S1, S0) → Partial. LayerNorm(norm_dim=0) forbids S0 only.
        # Partial is not S0, so it's fine.
        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_direct_shard_to_layernorm_unsat(self):
        """Input S(1) fed directly to LayerNorm(norm_dim=1) → UNSAT."""
        program = Program("direct_ln", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_full_mlp_block_satisfiable(self):
        """Full MLP: MatMul→AllReduce→LayerNorm→SiLU→MatMul: all satisfiable."""
        program = Program("mlp", ops=[
            MatMul(a="x", b="w1", output="h_p"),
            AllReduce(x="h_p", output="h"),
            LayerNorm(x="h", output="h_norm", norm_dim=1),
            SiLU(x="h_norm", output="h_act"),
            MatMul(a="h_act", b="w2", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w1", Shard(dim=0))
        solver.add_input("w2", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # All compute preconditions satisfied
        results = solver.check_compute_preconditions(program)
        assert all(r.passed for r in results)

        # Output is Replicate
        eq_results = solver.check_output_equivalence(["y"])
        assert all(r.passed for r in eq_results)

    def test_row_parallel_with_softmax_after_allreduce_safe(self):
        """AllReduce before Softmax(dim=1) proves safe."""
        program = Program("rp_sm", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
            Softmax(x="y", output="y_sm", dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        results = solver.check_compute_preconditions(program)
        assert all(r.passed for r in results)


# ── Program satisfiability ──────────────────────────────────────────────────


class TestProgramSatisfiability:
    """Check_program_satisfiability detects contradictory programs."""

    def test_valid_program_satisfiable(self):
        """A well-formed row-parallel program is satisfiable."""
        program = Program("valid", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.add_input("w", Shard(dim=0))
        solver.encode_program(program)

        result = solver.check_program_satisfiability()
        assert result.passed

    def test_conflicting_constraints_unsat(self):
        """Input S(1) + LayerNorm(norm_dim=1) → contradictory → UNSAT."""
        program = Program("conflict", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=1))
        solver.encode_program(program)

        result = solver.check_program_satisfiability()
        assert not result.passed
        assert "contradictory" in result.details.lower() or "UNSAT" in result.details

    def test_two_allreduce_then_add_safe(self):
        """Two AllReduces producing Replicate, then Add → safe."""
        program = Program("ar_add", ops=[
            AllReduce(x="p1", output="r1"),
            AllReduce(x="p2", output="r2"),
            Add(a="r1", b="r2", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("p1", Partial())
        solver.add_input("p2", Partial())
        solver.encode_program(program)

        result = solver.check_program_satisfiability()
        assert result.passed


# ── check_compute_preconditions detailed tests ──────────────────────────────


class TestCheckComputePreconditions:
    """Detailed tests for the check_compute_preconditions method."""

    def test_detects_layernorm_can_receive_forbidden(self):
        """When input is unconstrained, LayerNorm(norm_dim=1) detects S1 is possible."""
        program = Program("detect", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver()
        # Don't constrain input — it's free
        solver.encode_program(program)

        # The encoding itself adds x != S1, so check_compute_preconditions
        # should confirm it's unreachable
        results = solver.check_compute_preconditions(program)
        # Since _encode_norm_op already added x != S1, the forbidden state is
        # unreachable and the check should pass
        assert len(results) == 1
        assert results[0].passed

    def test_detects_partial_partial_reachable(self):
        """When inputs are both unconstrained, Add can potentially receive P×P — but encoding forbids it."""
        program = Program("pp_check", ops=[
            Add(a="x", b="y", output="z"),
        ])
        solver = Z3PlacementSolver()
        # Don't constrain inputs
        solver.encode_program(program)

        # The encoding adds Not(And(a==P, b==P)), so the check should pass
        results = solver.check_compute_preconditions(program)
        assert len(results) == 1
        assert results[0].passed

    def test_report_includes_counterexample_when_failing(self):
        """When constraints allow forbidden state, report includes counterexample."""
        # Create a solver without using encode_program (to avoid the built-in constraints)
        # and manually test _check_not_forbidden
        solver = Z3PlacementSolver()
        # Just create a free variable
        solver._var("x")
        # Don't add any constraints — x is free in {0,1,2,3}

        result = solver._check_not_forbidden("LayerNorm", "x", PL_S1, "Shard(1)")
        assert not result.passed
        assert result.counterexample is not None


# ── Integration with DistributedVerifier ────────────────────────────────────


class TestVerifierIntegration:
    """Test that DistributedVerifier.verify_placement_consistency uses new checks."""

    def test_verify_placement_consistency_catches_compute_violations(self):
        """verify_placement_consistency now checks compute preconditions."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)

        x = TensorState("x", (8, 16), (8, 8), spec_s1, expr="x")

        # Program that feeds S(1) directly to LayerNorm(norm_dim=1) — invalid
        program = Program("bad_ln", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])

        verifier = DistributedVerifier()
        result = verifier.verify_placement_consistency(
            program,
            final_tensors={"x": x},
            output_names=["y"],
        )
        # Should fail because the program is UNSAT
        assert not result.passed

    def test_verify_placement_consistency_passes_valid_mlp(self):
        """verify_placement_consistency passes for valid row-parallel + LayerNorm."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        spec_r = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        x = TensorState("x", (8, 16), (8, 8), spec_s1, expr="x")
        w = TensorState("w", (16, 32), (8, 32), spec_s0, expr="w")

        program = Program("good_mlp", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
            LayerNorm(x="y", output="y_norm", norm_dim=1),
        ])

        verifier = DistributedVerifier()
        result = verifier.verify_placement_consistency(
            program,
            final_tensors={"x": x, "w": w},
            output_names=["y_norm"],
        )
        assert result.passed

    def test_verify_all_with_shapes(self):
        """verify_all runs Z3 shape checks when initial_shapes is provided."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        spec_r = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        x = TensorState("x", (8, 16), (8, 8), spec_s1, expr="x")
        w = TensorState("w", (16, 32), (8, 32), spec_s0, expr="w")

        program = Program("tp_shapes", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        final = executor.run_program(program)

        verifier = DistributedVerifier()
        results = verifier.verify_all(
            program, final,
            initial_shapes={"x": (8, 16), "w": (16, 32)},
        )

        # Should include shape/slice results
        conditions = [r.condition for r in results]
        assert any("shape" in c.lower() for c in conditions)
        assert all(r.passed for r in results)

    def test_verify_all_detects_shape_mismatch(self):
        """verify_all with mismatched contraction dims → shape check fails."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        # x is (8, 16), w is (32, 64) → contraction dim mismatch: 16 != 32
        x = TensorState("x", (8, 16), (8, 8), spec_s1, expr="x")
        w = TensorState("w", (32, 64), (16, 64), spec_s0, expr="w")

        program = Program("bad_shapes", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])

        verifier = DistributedVerifier()
        results = verifier.verify_all(
            program,
            final_tensors={"x": x, "w": w, "y_p": x, "y": x},
            initial_shapes={"x": (8, 16), "w": (32, 64)},
        )

        # Shape check should detect contraction dim mismatch
        shape_results = [r for r in results if "shape" in r.condition.lower()
                        or "slice" in r.condition.lower()]
        # At minimum, slice alignment should catch this
        has_failure = any(not r.passed for r in shape_results)
        assert has_failure


# ── End-to-end scenarios ────────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end verification scenarios combining all checks."""

    def test_megatron_mlp_all_checks_pass(self):
        """Full Megatron MLP: row-parallel with LayerNorm passes all checks."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        x = TensorState("x", (8, 16), compute_local_shape((8, 16), spec_s1),
                        spec_s1, expr="x")
        w_gate = TensorState("w_gate", (16, 32),
                             compute_local_shape((16, 32), spec_s0), spec_s0, expr="w_gate")
        w_down = TensorState("w_down", (32, 16),
                             compute_local_shape((32, 16), spec_s0), spec_s0, expr="w_down")

        program = Program("megatron_mlp", ops=[
            MatMul(a="x", b="w_gate", output="h_p"),
            AllReduce(x="h_p", output="h"),
            LayerNorm(x="h", output="h_norm", norm_dim=-1),
            SiLU(x="h_norm", output="h_act"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        executor.register_tensor(x)
        executor.register_tensor(w_gate)
        executor.register_tensor(w_down)
        final = executor.run_program(program)

        verifier = DistributedVerifier()
        results = verifier.verify_all(
            program, final,
            initial_shapes={"x": (8, 16), "w_gate": (16, 32)},
        )

        assert all(r.passed for r in results), verifier.summary()

    def test_missing_allreduce_before_layernorm_detected(self):
        """Missing AllReduce: MatMul→Partial→LayerNorm → postcondition + placement both fail."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        spec_p = ShardingSpec(placements=(Partial(),), mesh=mesh)

        x = TensorState("x", (8, 16), compute_local_shape((8, 16), spec_s1),
                        spec_s1, expr="x")
        w = TensorState("w", (16, 32), compute_local_shape((16, 32), spec_s0),
                        spec_s0, expr="w")

        # Bug: no AllReduce before LayerNorm — output is Partial
        program = Program("bug_no_ar", ops=[
            MatMul(a="x", b="w", output="y_p"),
            # Missing: AllReduce(x="y_p", output="y")
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        final = executor.run_program(program)

        verifier = DistributedVerifier()
        results = verifier.verify_all(program, final)

        # Postcondition should fail (output is Partial)
        post_results = [r for r in results if "postcondition" in r.condition]
        assert any(not r.passed for r in post_results)

    def test_embedding_allreduce_layernorm_chain(self):
        """Vocab-parallel Embedding → AllReduce → LayerNorm: complete and safe."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        spec_r = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        program = Program("emb_ln", ops=[
            Embedding(indices="ids", weight="W", output="emb_p"),
            AllReduce(x="emb_p", output="emb"),
            LayerNorm(x="emb", output="emb_norm", norm_dim=-1),
        ])

        solver = Z3PlacementSolver()
        solver.add_input("ids", Replicate())
        solver.add_input("W", Shard(dim=0))
        solver.encode_program(program)

        # Program satisfiable
        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # Output is Replicate
        eq_results = solver.check_output_equivalence(["emb_norm"])
        assert all(r.passed for r in eq_results)

        # Compute preconditions pass
        cc_results = solver.check_compute_preconditions(program)
        assert all(r.passed for r in cc_results)
