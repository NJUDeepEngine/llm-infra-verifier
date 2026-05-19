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


# ── Multi-dimensional mesh tests ──────────────────────────────────────────────


class TestMultiDimMesh:
    """Z3 solver correctly handles 2D+ mesh placements."""

    def test_2d_mesh_matmul_tp_dp(self):
        """2D mesh (tp=2, dp=4): MatMul with TP on dim1, DP on dim0.

        x: (Shard(1), Shard(0)) — split cols on TP, split rows on DP
        w: (Shard(0), Replicate) — split rows on TP, replicated on DP
        y: (Partial, Shard(0)) — partial on TP dim, still sharded on DP
        """
        program = Program("tp_dp_matmul", ops=[
            MatMul(a="x", b="w", output="y"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=1), Shard(dim=0)))
        solver.add_input("w", (Shard(dim=0), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y should be (Partial, Shard(0)) — TP dim produces Partial, DP dim passes S0
        # Check that y is NOT Replicate on all dims (it's Partial on dim 0)
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_2d_mesh_allreduce_resolves_tp_dim(self):
        """AllReduce after MatMul resolves Partial on TP dim.

        After AllReduce: y goes from (P, S0) to (R, R).
        But since AllReduce makes output R on ALL mesh dims, the DP shard is lost.
        In practice, the DP shard is on the batch dim so AllReduce on TP group only
        affects TP mesh dim. However, our Z3 model encodes AllReduce as → R per dim.
        """
        program = Program("tp_dp_ar", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=1), Shard(dim=0)))
        solver.add_input("w", (Shard(dim=0), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # AllReduce output is Replicate on all mesh dims
        eq_results = solver.check_output_equivalence(["y"])
        assert all(r.passed for r in eq_results)

    def test_2d_mesh_forbidden_layernorm(self):
        """2D mesh: LayerNorm(norm_dim=1) forbids S1 on BOTH mesh dims."""
        program = Program("2d_ln", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        # Input is S1 on mesh dim 1 (DP dim) — should be forbidden
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Replicate(), Shard(dim=1)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_2d_mesh_layernorm_shard_batch_safe(self):
        """2D mesh: LayerNorm(norm_dim=1) with Shard(0) on any mesh dim is safe."""
        program = Program("2d_ln_ok", ops=[
            LayerNorm(x="x", output="y", norm_dim=1),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=0), Shard(dim=0)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_2d_mesh_partial_partial_forbidden_per_dim(self):
        """2D mesh: Add(Partial, Partial) forbidden on each mesh dim independently."""
        program = Program("2d_add_pp", ops=[
            Add(a="x", b="y", output="z"),
        ])
        # Partial on mesh dim 0 for both — forbidden
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Replicate()))
        solver.add_input("y", (Partial(), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert not sat_result.passed

    def test_2d_mesh_partial_replicate_safe(self):
        """2D mesh: Add(Partial on m0, Replicate on m0) — safe per-dim merge."""
        program = Program("2d_add_pr", ops=[
            Add(a="x", b="y", output="z"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Shard(dim=0)))
        solver.add_input("y", (Replicate(), Shard(dim=0)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_2d_mesh_embedding_partial_on_tp(self):
        """2D mesh: Embedding with weight S(0) on TP dim produces Partial on TP."""
        program = Program("2d_emb", ops=[
            Embedding(indices="ids", weight="W", output="emb"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("ids", (Replicate(), Replicate()))
        solver.add_input("W", (Shard(dim=0), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # emb is NOT replicate on mesh dim 0 (it's Partial there)
        eq_results = solver.check_output_equivalence(["emb"])
        assert not all(r.passed for r in eq_results)

    def test_2d_mesh_reducescatter(self):
        """2D mesh: ReduceScatter produces Shard on the scatter dim per mesh dim."""
        program = Program("2d_rs", ops=[
            ReduceScatter(x="x", output="y", scatter_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Partial()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

    def test_2d_mesh_flash_attention_follows_q(self):
        """2D mesh: FlashAttention output placement follows Q on all mesh dims."""
        program = Program("2d_fa", ops=[
            FlashAttention(q="q", k="k", v="v", output="attn"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("q", (Shard(dim=0), Replicate()))
        solver.add_input("k", (Replicate(), Replicate()))
        solver.add_input("v", (Replicate(), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # attn should follow q: (S0, R) — not Replicate on all dims
        eq_results = solver.check_output_equivalence(["attn"])
        assert not all(r.passed for r in eq_results)

    def test_verify_placement_consistency_detects_mesh_ndim(self):
        """DistributedVerifier auto-detects 2D mesh and uses multi-dim solver."""
        mesh = DeviceMesh(shape=(2, 4), dim_names=("tp", "dp"))
        spec_s1_r = ShardingSpec(placements=(Shard(dim=1), Replicate()), mesh=mesh)
        spec_s0_r = ShardingSpec(placements=(Shard(dim=0), Replicate()), mesh=mesh)

        x = TensorState("x", (8, 16), (8, 8), spec_s1_r, expr="x")
        w = TensorState("w", (16, 32), (8, 32), spec_s0_r, expr="w")

        program = Program("2d_verify", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])

        verifier = DistributedVerifier()
        result = verifier.verify_placement_consistency(
            program,
            final_tensors={"x": x, "w": w},
            output_names=["y"],
        )
        assert result.passed

    def test_2d_mesh_shape_constraints(self):
        """2D mesh: shape divisibility checked per mesh dim."""
        program = Program("2d_shapes", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=1), Shard(dim=0)))
        solver.add_input("w", (Shard(dim=0), Replicate()))
        solver.add_input_shape("x", (8, 16))
        solver.add_input_shape("w", (16, 32))
        solver.encode_program(program)
        solver.encode_shape_constraints(program, mesh_sizes=[2, 4])

        shape_results = solver.check_shape_consistency()
        # All divisibility checks should pass: 16/2=8 (TP on x cols), 8/4=2 (DP on x rows)
        assert all(r.passed for r in shape_results)


# ── Mesh-dim-aware collective encoding tests ──────────────────────────────────


class TestMeshDimCollectives:
    """Z3 solver respects op.mesh_dim for targeted collective encoding."""

    def test_allreduce_mesh_dim_preserves_other_dims(self):
        """AllReduce(mesh_dim=0) on (P, S0) → (R, S0): only resolves dim 0."""
        program = Program("ar_md0", ops=[
            AllReduce(x="x", output="y", mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Shard(dim=0)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (R, S0) — NOT fully Replicate
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_allgather_mesh_dim_preserves_other_dims(self):
        """AllGather(mesh_dim=0) on (S0, S0) → (R, S0): gathers only on dim 0."""
        program = Program("ag_md0", ops=[
            AllGather(x="x", output="y", gather_dim=0, mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=0), Shard(dim=0)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (R, S0) — NOT fully Replicate
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_reducescatter_mesh_dim_targets_specific_dim(self):
        """ReduceScatter(mesh_dim=1) only transforms mesh dim 1."""
        program = Program("rs_md1", ops=[
            ReduceScatter(x="x", output="y", scatter_dim=0, mesh_dim=1),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Replicate(), Partial()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (R, S0) — dim 0 preserved as R, dim 1 transformed to S0
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_alltoall_mesh_dim(self):
        """AllToAll(mesh_dim=0) swaps dims only on mesh dim 0, preserves dim 1."""
        program = Program("a2a_md0", ops=[
            AllToAll(x="x", output="y", split_dim=0, concat_dim=1, mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=0), Shard(dim=0)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (S1, S0) — dim 0 transformed S0→S1, dim 1 preserved S0
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_allreduce_no_mesh_dim_resolves_all(self):
        """AllReduce without mesh_dim resolves Partial on ALL mesh dims (backward compat)."""
        program = Program("ar_legacy", ops=[
            AllReduce(x="x", output="y"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Partial()))
        solver.encode_program(program)

        # y should be (R, R) — fully Replicate
        eq_results = solver.check_output_equivalence(["y"])
        assert all(r.passed for r in eq_results)

    def test_tp_allreduce_preserves_dp_shard(self):
        """Real-world TP+DP: MatMul → AllReduce(mesh_dim=0) preserves DP Shard(0).

        x: (S1, S0) — col-sharded on TP, row-sharded on DP
        w: (S0, R)  — row-sharded on TP, replicated on DP
        y_p: (P, S0) — Partial on TP, S0 preserved on DP
        y: (R, S0)   — AllReduce resolves only TP dim
        """
        program = Program("tp_dp_real", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduce(x="y_p", output="y", mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=1), Shard(dim=0)))
        solver.add_input("w", (Shard(dim=0), Replicate()))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (R, S0) — NOT fully Replicate (DP shard preserved)
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_scatter_mesh_dim_preserves_other_dims(self):
        """Scatter(mesh_dim=0) only transforms dim 0, preserves dim 1."""
        program = Program("sc_md0", ops=[
            Scatter(x="x", output="y", scatter_dim=0, mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Replicate(), Shard(dim=1)))
        solver.encode_program(program)

        sat_result = solver.check_program_satisfiability()
        assert sat_result.passed

        # y is (S0, S1) — dim 0 transformed R→S0, dim 1 preserved S1
        eq_results = solver.check_output_equivalence(["y"])
        assert not all(r.passed for r in eq_results)

    def test_precondition_check_respects_mesh_dim(self):
        """Precondition for AllReduce(mesh_dim=0) checks Partial on dim 0 only."""
        program = Program("prec_md", ops=[
            AllReduce(x="x", output="y", mesh_dim=0),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        # x is Partial on dim 0 (correct for AllReduce), Shard(0) on dim 1
        solver.add_input("x", (Partial(), Shard(dim=0)))
        solver.encode_program(program)

        results = solver.check_collective_preconditions(program)
        # Should pass: dim 0 is Partial as required
        assert all(r.passed for r in results)

    def test_allgather_no_mesh_dim_preserves_other_dims(self):
        """AllGather(mesh_dim=None) on 2D mesh: only gathers Shard(gather_dim), preserves others."""
        from z3 import sat, IntVal
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=0), Shard(dim=1)))
        prog = Program("ag_nodim", ops=[
            AllGather(x="x", output="y", gather_dim=0),
        ])
        solver.encode_program(prog)

        # dim 0: input is Shard(0) which matches gather_dim=0 → output is Replicate
        solver.solver.push()
        solver.solver.add(solver._var("y")[0] != IntVal(0))
        assert solver.solver.check() != sat, "dim 0 should be Replicate"
        solver.solver.pop()

        # dim 1: input is Shard(1) which does NOT match gather_dim=0 → preserved
        solver.solver.push()
        solver.solver.add(solver._var("y")[1] != IntVal(2))  # PL_S1 = 2
        assert solver.solver.check() != sat, "dim 1 should preserve Shard(1)"
        solver.solver.pop()

    def test_gather_no_mesh_dim_preserves_other_dims(self):
        """Gather(mesh_dim=None) on 2D mesh preserves non-gathered dims."""
        from z3 import sat, IntVal
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Shard(dim=0), Replicate()))
        prog = Program("ga_nodim", ops=[
            Gather(x="x", output="y", gather_dim=0),
        ])
        solver.encode_program(prog)

        # dim 0: Shard(0) matches gather_dim=0 → Replicate
        solver.solver.push()
        solver.solver.add(solver._var("y")[0] != IntVal(0))
        assert solver.solver.check() != sat
        solver.solver.pop()

        # dim 1: Replicate → preserved as Replicate
        solver.solver.push()
        solver.solver.add(solver._var("y")[1] != IntVal(0))
        assert solver.solver.check() != sat
        solver.solver.pop()


# ── Wait / WaitAll solver encoding ──────────────────────────────────────────


class TestWaitEncoding:
    """Wait and WaitAll ops must be encoded as passthrough in the solver."""

    def test_wait_passthrough_preserves_placement(self):
        """AllReduceAsync → Wait: solver sees output as Replicate."""
        prog = Program("wait_pt", ops=[
            AllReduceAsync(x="x", output="y", handle="h1",
                           stream=Stream("comm", 0)),
            Wait(handle="h1", tensor="y", output="y_safe"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Partial())
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["y_safe"])
        assert all(r.passed for r in eq), "Wait output should be Replicate"

    def test_allreduce_async_wait_matmul_replicate(self):
        """AllReduceAsync → Wait → MatMul: final output should be Replicate."""
        prog = Program("async_wait_mm", ops=[
            AllReduceAsync(x="x", output="y", handle="h1",
                           stream=Stream("comm", 0)),
            Wait(handle="h1", tensor="y", output="y_safe"),
            MatMul(a="y_safe", b="w", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Partial())
        solver.add_input("w", Replicate())
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["z"])
        assert all(r.passed for r in eq), "MatMul(R, R) -> R"

    def test_wait_without_async_is_passthrough(self):
        """Wait on a non-async tensor just passes placement through."""
        prog = Program("wait_plain", ops=[
            MatMul(a="x", b="w", output="y"),
            Wait(handle="h_fake", tensor="y", output="y_safe"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("x", Shard(dim=0))
        solver.add_input("w", Replicate())
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["y_safe"])
        # MatMul(S0, R) -> S0, Wait passes S0 through
        assert not all(r.passed for r in eq), "y_safe should be S0, not R"

    def test_waitall_multi_output_passthrough(self):
        """WaitAll maps each input tensor's placement to corresponding output."""
        prog = Program("waitall_pt", ops=[
            AllReduceAsync(x="a", output="a_r", handle="h1",
                           stream=Stream("comm", 0)),
            AllReduceAsync(x="b", output="b_r", handle="h2",
                           stream=Stream("comm", 0)),
            WaitAll(handles=["h1", "h2"], tensors=["a_r", "b_r"],
                    outputs=["a_safe", "b_safe"]),
            MatMul(a="a_safe", b="b_safe", output="z"),
        ])
        solver = Z3PlacementSolver()
        solver.add_input("a", Partial())
        solver.add_input("b", Partial())
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["z"])
        assert all(r.passed for r in eq), "WaitAll outputs should be R"

    def test_wait_2d_mesh_preserves_both_dims(self):
        """Wait preserves placement on both mesh dims."""
        prog = Program("wait_2d", ops=[
            AllReduceAsync(x="x", output="y", handle="h1",
                           stream=Stream("comm", 0), mesh_dim=0),
            Wait(handle="h1", tensor="y", output="y_safe"),
        ])
        solver = Z3PlacementSolver(mesh_ndim=2)
        solver.add_input("x", (Partial(), Shard(dim=0)))
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["y_safe"])
        # AllReduceAsync(mesh_dim=0): P→R on dim 0, S0 preserved on dim 1
        # y_safe = (R, S0) — not fully Replicate
        assert not all(r.passed for r in eq)

    def test_waitall_shape_passthrough(self):
        """WaitAll preserves shape constraints through to outputs."""
        prog = Program("waitall_shape", ops=[
            AllReduceAsync(x="a", output="a_r", handle="h1",
                           stream=Stream("comm", 0)),
            WaitAll(handles=["h1"], tensors=["a_r"], outputs=["a_safe"]),
            MatMul(a="a_safe", b="w", output="z"),
        ])
        mesh = DeviceMesh(shape=(4,), dim_names=("tp",))
        solver = Z3PlacementSolver()
        solver.add_input("a", Partial())
        solver.add_input("w", Replicate())
        solver.encode_program(prog)
        eq = solver.check_output_equivalence(["z"])
        assert all(r.passed for r in eq)


# ── In-place op SSA renaming ───────────────────────────────────────────────


class TestInPlaceSSA:
    """Z3 placement consistency must handle in-place ops via SSA renaming."""

    def test_inplace_allreduce_passes_consistency(self):
        """AllReduce(x='p', output='p') should pass Z3 placement consistency."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        x = TensorState(
            name="x", global_shape=(8, 16),
            local_shape=compute_local_shape((8, 16), spec_s1),
            sharding=spec_s1, expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32),
            local_shape=compute_local_shape((16, 32), spec_s0),
            sharding=spec_s0, expr="w",
        )

        prog = Program("inplace_ar", ops=[
            MatMul(a="x", b="w", output="p"),
            AllReduce(x="p", output="p"),
        ])

        import warnings
        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = executor.run_program(prog)

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state, output_names=["p"])
        for r in results:
            assert r.passed, f"Failed: {r.condition} — {r.details}"

    def test_inplace_chain_matmul_allreduce_matmul(self):
        """In-place AllReduce between two MatMuls: p=matmul, p=AR(p), z=matmul(p,w2)."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        spec_r = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        x = TensorState(
            name="x", global_shape=(8, 16),
            local_shape=compute_local_shape((8, 16), spec_s1),
            sharding=spec_s1, expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32),
            local_shape=compute_local_shape((16, 32), spec_s0),
            sharding=spec_s0, expr="w",
        )
        w2 = TensorState(
            name="w2", global_shape=(32, 16),
            local_shape=(32, 16),
            sharding=spec_r, expr="w2",
        )

        prog = Program("inplace_chain", ops=[
            MatMul(a="x", b="w", output="p"),
            AllReduce(x="p", output="p"),
            MatMul(a="p", b="w2", output="z"),
        ])

        import warnings
        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        executor.register_tensor(w2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = executor.run_program(prog)

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state, output_names=["z"])
        for r in results:
            assert r.passed, f"Failed: {r.condition} — {r.details}"

    def test_ssa_rename_noop_for_distinct_names(self):
        """Programs without in-place ops should not be modified by SSA."""
        prog = Program("normal", ops=[
            MatMul(a="x", b="w", output="y_partial"),
            AllReduce(x="y_partial", output="y"),
        ])

        ssa_prog, ssa_map = DistributedVerifier._ssa_rename_program(prog)
        # No in-place ops → same program returned
        assert ssa_prog is prog
        assert ssa_map == {}

    def test_ssa_rename_creates_versions(self):
        """In-place AllReduce creates versioned names."""
        prog = Program("inplace", ops=[
            MatMul(a="x", b="w", output="p"),
            AllReduce(x="p", output="p"),
        ])

        ssa_prog, ssa_map = DistributedVerifier._ssa_rename_program(prog)
        assert ssa_prog is not prog
        # MatMul output is "p" (first definition, no version)
        assert ssa_prog.ops[0].output == "p"
        # AllReduce input is "p", output is "p__v1"
        assert ssa_prog.ops[1].x == "p"
        assert ssa_prog.ops[1].output == "p__v1"
        assert ssa_map["p"] == "p__v1"

    def test_inplace_with_downstream_consumer(self):
        """In-place AllReduce + downstream op reads the latest version."""
        prog = Program("inplace_chain", ops=[
            MatMul(a="x", b="w", output="p"),
            AllReduce(x="p", output="p"),
            MatMul(a="p", b="w2", output="z"),
        ])

        ssa_prog, ssa_map = DistributedVerifier._ssa_rename_program(prog)
        # After SSA: MatMul(x,w)->p, AllReduce(p)->p__v1, MatMul(p__v1,w2)->z
        assert ssa_prog.ops[2].a == "p__v1"

    def test_inplace_allreduce_2d_mesh(self):
        """In-place AllReduce on 2D mesh passes Z3 consistency."""
        mesh = DeviceMesh(shape=(2, 2), dim_names=("tp", "dp"))
        spec_x = ShardingSpec(
            placements=(Shard(dim=1), Replicate()), mesh=mesh,
        )
        spec_w = ShardingSpec(
            placements=(Shard(dim=0), Replicate()), mesh=mesh,
        )

        x = TensorState(
            name="x", global_shape=(8, 16),
            local_shape=compute_local_shape((8, 16), spec_x),
            sharding=spec_x, expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32),
            local_shape=compute_local_shape((16, 32), spec_w),
            sharding=spec_w, expr="w",
        )

        prog = Program("inplace_2d", ops=[
            MatMul(a="x", b="w", output="p"),
            AllReduce(x="p", output="p"),
        ])

        import warnings
        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = executor.run_program(prog)

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state, output_names=["p"])
        for r in results:
            assert r.passed, f"Failed: {r.condition} — {r.details}"
