"""Tests for the distributed tensor verifier."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from verifier.state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
)
from verifier.ir import (
    Program,
    MatMul,
    Add,
    Multiply,
    SiLU,
    AllReduce,
    AllGather,
    ReduceScatter,
    Send,
    Recv,
    FlashAttention,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine, GradientCheckResult
from verifier.solver import DistributedVerifier, VerifyResult
from verifier.schedules import PP1F1BSchedule, DeadlockChecker
from verifier.rewrite import PlacementAnalyzer, PlacementAnalysis, ProgramCost


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_row_parallel_tensors(mesh=None):
    """Create tensors for Row Parallel Linear (both sharded on reduce dim).

    X: shape (B, H), Shard on dim=1 (H dim = reduce dim for X @ W)
    W: shape (H, O), Shard on dim=0 (H dim = reduce dim for X @ W)

    MatMul(X, W) → PARTIAL → needs AllReduce → REPLICATE
    """
    if mesh is None:
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
    B, H, O = 8, 16, 32
    x = TensorState(
        name="x",
        global_shape=(B, H),
        local_shape=(B, H // 2),   # shard on dim 1 (H) by factor 2
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="x",
        requires_grad=True,
    )
    w = TensorState(
        name="w",
        global_shape=(H, O),
        local_shape=(H // 2, O),   # shard on dim 0 (H) by factor 2
        sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
        expr="w",
        requires_grad=True,
    )
    return x, w, mesh


# ── State tests ──────────────────────────────────────────────────────────────

class TestTensorState:
    def test_shard_placement(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        t = TensorState(
            name="x",
            global_shape=(8, 16),
            local_shape=(4, 16),
            sharding=spec,
            expr="x",
        )
        assert t.partial == False
        assert t.is_replicated == False
        assert t.global_shape == (8, 16)
        assert t.local_shape == (4, 16)

    def test_partial_tensor(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec = ShardingSpec(placements=(Partial(),), mesh=mesh)
        t = TensorState(
            name="y",
            global_shape=(8, 32),
            local_shape=(8, 32),
            sharding=spec,
            expr="y",
        )
        assert t.partial == True

    def test_compute_local_shape(self):
        mesh = DeviceMesh(shape=(4,), dim_names=("tp",))
        spec = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        local = compute_local_shape((16, 32), spec)
        assert local == (4, 32)

    def test_replicate_placement(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        spec = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        t = TensorState(
            name="z",
            global_shape=(8, 32),
            local_shape=(8, 32),
            sharding=spec,
            expr="z",
        )
        assert t.is_replicated == True
        assert t.partial == False


# ── IR tests ─────────────────────────────────────────────────────────────────

class TestIROps:
    def test_matmul_row_parallel_becomes_partial(self):
        """Row Parallel: X(Shard1) @ W(Shard0) → PARTIAL (both on reduce dim H)."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        # X: shard on dim=1 (H=16, reduce dim in X@W) → local (8, 8)
        # W: shard on dim=0 (H=16, reduce dim in X@W) → local (8, 32)
        a = TensorState(
            name="a", global_shape=(8, 16), local_shape=(8, 8),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh), expr="a",
        )
        b = TensorState(
            name="b", global_shape=(16, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh), expr="b",
        )

        op = MatMul(a="a", b="b", output="y")
        ctx = {"a": a, "b": b}
        result = op.apply(ctx)

        assert result.name == "y"
        assert result.global_shape == (8, 32)
        assert result.partial == True, (
            f"Both sharded on reduce dim → expected PARTIAL, got {result.sharding}"
        )

    def test_matmul_column_parallel_no_comm_in_fwd(self):
        """Column Parallel: X(Replicate) @ W(Shard1) → Shard(1) output, no comm in fwd."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        a = TensorState(
            name="a", global_shape=(8, 16), local_shape=(8, 16),
            sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh), expr="a",
        )
        b = TensorState(
            name="b", global_shape=(16, 32), local_shape=(16, 16),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh), expr="b",
        )

        op = MatMul(a="a", b="b", output="y")
        ctx = {"a": a, "b": b}
        result = op.apply(ctx)

        assert result.partial == False
        assert isinstance(result.sharding.placements[0], Shard)

    def test_allreduce_converts_partial_to_replicate(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Partial(),), mesh=mesh), expr="x",
        )

        op = AllReduce(x="x", output="y")
        ctx = {"x": x}
        result = op.apply(ctx)

        assert result.partial == False
        assert isinstance(result.sharding.placements[0], Replicate)

    def test_allreduce_requires_partial_input(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh), expr="x",
        )

        op = AllReduce(x="x", output="y")
        ctx = {"x": x}
        with pytest.raises(ValueError):
            op.apply(ctx)

    def test_matmul_replicated_no_comm(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        a = TensorState(
            name="a", global_shape=(8, 16), local_shape=(8, 16),
            sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh), expr="a",
        )
        b = TensorState(
            name="b", global_shape=(16, 32), local_shape=(16, 32),
            sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh), expr="b",
        )

        op = MatMul(a="a", b="b", output="y")
        ctx = {"a": a, "b": b}
        result = op.apply(ctx)

        assert result.partial == False
        assert isinstance(result.sharding.placements[0], Replicate)


# ── Executor tests ───────────────────────────────────────────────────────────

class TestExecutor:
    def test_execute_row_parallel(self):
        """Execute Row Parallel Linear: MatMul → AllReduce across 2 devices."""
        x, w, mesh = make_row_parallel_tensors()

        fwd = Program(name="test_tp")
        fwd.add(MatMul(a="x", b="w", output="y_partial"))
        fwd.add(AllReduce(x="y_partial", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        result = executor.run_program(fwd)

        # y_partial should exist and be partial after matmul
        y_partial = executor.get_tensor("y_partial", device_id=0)
        assert y_partial is not None
        assert y_partial.partial == True, "y_partial should be PARTIAL after MatMul"

        # y should be replicated after AllReduce
        y = executor.get_tensor("y", device_id=0)
        assert y is not None
        assert y.partial == False, "y should be REPLICATE after AllReduce"
        assert y.global_shape == (8, 32)

    def test_multi_device_state_isolation(self):
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(4, 16),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh), expr="x",
        )

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)

        dev0 = executor.get_tensor("x", device_id=0)
        dev1 = executor.get_tensor("x", device_id=1)

        assert dev0 is not None
        assert dev1 is not None
        assert dev0.local_shape == (4, 16)
        assert dev1.local_shape == (4, 16)


# ── Autograd tests ───────────────────────────────────────────────────────────

class TestAutograd:
    def test_gradient_duality_allreduce(self):
        """AllReduce is self-dual: fwd AllReduce → bwd AllReduce."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Partial(),), mesh=mesh),
            expr="x", requires_grad=True,
        )

        fwd = Program(name="test")
        fwd.add(AllReduce(x="x", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.run_program(fwd)

        autograd = AutogradEngine()
        for op in fwd.ops:
            tensor_states = executor.devices[0].tensors
            autograd.record(op, tensor_states)

        bwd = autograd.generate_backward("y")

        ar_ops = [op for op in bwd.ops if isinstance(op, AllReduce)]
        assert len(ar_ops) >= 1, f"Expected AllReduce in backward, got {bwd.ops}"

    def test_gradient_check_row_parallel(self):
        """Row Parallel: verify fwd AllReduce has bwd AllReduce dual."""
        x, w, mesh = make_row_parallel_tensors()

        fwd = Program(name="correct")
        fwd.add(MatMul(a="x", b="w", output="y_partial"))
        fwd.add(AllReduce(x="y_partial", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        executor.run_program(fwd)

        autograd = AutogradEngine()
        for op in fwd.ops:
            tensor_states = {**executor.devices[0].tensors}
            autograd.record(op, tensor_states)

        bwd = autograd.generate_backward("y")
        check = autograd.verify_gradient_correctness(fwd, bwd)

        fwd_ar = [op for op in fwd.ops if isinstance(op, AllReduce)]
        assert len(fwd_ar) == 1
        assert check.passed, f"Gradient check failed: {check.errors}"


# ── Solver tests ─────────────────────────────────────────────────────────────

class TestSolver:
    def test_postcondition_partial_fails(self):
        """A PARTIAL tensor should fail postcondition check for partial=False."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        t = TensorState(
            name="y",
            global_shape=(8, 32),
            local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Partial(),), mesh=mesh),
            expr="y",
        )

        verifier = DistributedVerifier()
        result = verifier.verify_postcondition(t, expected_partial=False)
        assert not result.passed, "Expected FAIL for partial tensor with expected_partial=False"

    def test_postcondition_replicate_passes(self):
        """A REPLICATE tensor should pass postcondition check."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        t = TensorState(
            name="y",
            global_shape=(8, 32),
            local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
            expr="y",
        )

        verifier = DistributedVerifier()
        result = verifier.verify_postcondition(t, expected_partial=False)
        assert result.passed, f"Expected PASS but got: {result.details}"

    def test_communication_legality_send_without_recv_fails(self):
        """Send without matching Recv should fail."""
        fwd = Program(name="bad")
        fwd.add(Send(x="h", output="h_sent", src=0, dst=1, stage=0, microbatch_id=0))
        # Missing Recv!

        verifier = DistributedVerifier()
        result = verifier.verify_communication_legality(fwd)
        assert not result.passed, "Expected FAIL for Send without Recv"

    def test_communication_legality_send_with_recv_passes(self):
        """Send with matching Recv should pass."""
        fwd = Program(name="good")
        fwd.add(Send(x="h", output="h_sent", src=0, dst=1, stage=0, microbatch_id=0))
        fwd.add(Recv(x="h_sent", output="h_rcvd", src=0, dst=1, stage=0, microbatch_id=0))

        verifier = DistributedVerifier()
        result = verifier.verify_communication_legality(fwd)
        assert result.passed, f"Expected PASS but got: {result.details}"


# ── Schedule tests ───────────────────────────────────────────────────────────

class TestSchedules:
    def test_1f1b_schedule_generation(self):
        sched = PP1F1BSchedule(num_stages=2, num_microbatches=4)
        schedule = sched.generate_simple()

        assert len(schedule) > 0

        n_fwd = sum(1 for m in schedule if m.op_type.value == "forward")
        n_bwd = sum(1 for m in schedule if m.op_type.value == "backward")

        assert n_fwd == 8  # 2 stages x 4 microbatches
        assert n_bwd == 8  # 2 stages x 4 microbatches

    def test_deadlock_checker_matched_send_recv(self):
        checker = DeadlockChecker()
        checker.add_send(0, 1, "h0")
        checker.add_recv(0, 1, "h0")

        is_free, errors = checker.check()
        assert is_free, f"Expected deadlock-free but got: {errors}"

    def test_deadlock_checker_unmatched_send(self):
        checker = DeadlockChecker()
        checker.add_send(0, 1, "h0")
        # Missing Recv!

        is_free, errors = checker.check()
        assert not is_free, "Expected deadlock detection for unmatched Send"

    def test_bidirectional_pp_communication_is_matchable(self):
        """Bidirectional Send/Recv in PP passes Send/Recv matching checks.

        PP commonly has device 0 → 1 and 1 → 0 communication patterns
        (e.g., forward sends activations, backward sends gradients).

        The structural deadlock checker may flag cycles in the wait-for graph,
        but the actual 1F1B schedule ordering resolves them. This test ensures
        the matching checks pass; cycle detection is a separate concern
        resolved by schedule verification.
        """
        checker = DeadlockChecker()
        # Device 0 sends activation to device 1
        checker.add_send(0, 1, "h_fwd")
        checker.add_recv(0, 1, "h_fwd")
        # Device 1 sends gradient to device 0
        checker.add_send(1, 0, "grad_h")
        checker.add_recv(1, 0, "grad_h")

        is_free, errors = checker.check()
        # Send/Recv matches pass; wait-for cycle may be reported
        # but matched communication is semantically correct
        matched_errors = [e for e in errors if "Unmatched" in e]
        assert len(matched_errors) == 0, (
            f"All Send/Recv should match, got: {matched_errors}"
        )


# ── Integration tests ────────────────────────────────────────────────────────

class TestIntegration:
    def test_tp_linear_end_to_end(self):
        """Full end-to-end: Row Parallel Linear → Execute → Autograd → Verify."""
        x, w, mesh = make_row_parallel_tensors()

        fwd = Program(name="e2e")
        fwd.add(MatMul(a="x", b="w", output="y_partial"))
        fwd.add(AllReduce(x="y_partial", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        result = executor.run_program(fwd)

        # Verify postcondition
        y = executor.get_tensor("y", device_id=0)
        verifier = DistributedVerifier()
        vr = verifier.verify_postcondition(y, expected_partial=False)
        assert vr.passed, f"Postcondition failed: {vr.details}"

        # Autograd
        autograd = AutogradEngine()
        for op in fwd.ops:
            tensor_states = {**executor.devices[0].tensors}
            autograd.record(op, tensor_states)
        bwd = autograd.generate_backward("y")

        # Verify gradient duality
        duality = verifier.verify_gradient_duality(fwd, bwd)
        assert duality.passed, f"Gradient duality failed: {duality.details}"

    def test_missing_allreduce_detected(self):
        """Verify that missing AllReduce is detected via postcondition check."""
        x, w, mesh = make_row_parallel_tensors()

        fwd = Program(name="bug")
        fwd.add(MatMul(a="x", b="w", output="y"))
        # Missing AllReduce! y should remain PARTIAL

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        result = executor.run_program(fwd)

        y = executor.get_tensor("y", device_id=0)
        assert y.partial == True, (
            f"Expected y to be PARTIAL (missing AllReduce), got {y.sharding}"
        )

        verifier = DistributedVerifier()
        vr = verifier.verify_postcondition(y, expected_partial=False)
        assert not vr.passed, "Expected verification to FAIL for missing AllReduce"


# ── Rewrite tests ────────────────────────────────────────────────────────────

class TestRewrite:
    def test_placement_analyzer_finds_missing_allreduce(self):
        """Analyzer should detect Row Parallel without AllReduce."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(8, 8),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh), expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh), expr="w",
        )

        fwd = Program(name="test")
        fwd.add(MatMul(a="x", b="w", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        state = executor.run_program(fwd)

        analyzer = PlacementAnalyzer()
        analysis = analyzer.analyze(fwd, state)

        assert not analysis.is_correct
        assert len(analysis.missing_collectives) >= 1
        assert analysis.missing_collectives[0][2] == AllReduce

    def test_placement_analyzer_passes_correct_program(self):
        """Analyzer should pass Row Parallel WITH AllReduce."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(8, 8),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh), expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh), expr="w",
        )

        fwd = Program(name="test")
        fwd.add(MatMul(a="x", b="w", output="y_partial"))
        fwd.add(AllReduce(x="y_partial", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        state = executor.run_program(fwd)

        analyzer = PlacementAnalyzer()
        analysis = analyzer.analyze(fwd, state)

        assert analysis.is_correct, f"Expected correct, got: {analysis}"

    def test_program_cost(self):
        """Cost model should rank programs correctly."""
        cheap = Program(name="cheap")
        cheap.add(MatMul(a="x", b="w", output="y"))

        expensive = Program(name="expensive")
        expensive.add(MatMul(a="x", b="w", output="y_p"))
        expensive.add(AllReduce(x="y_p", output="y"))

        cheap_cost = ProgramCost.from_program(cheap)
        expensive_cost = ProgramCost.from_program(expensive)

        assert cheap_cost.total_communication == 0
        assert expensive_cost.total_communication == 2  # AllReduce = 2x
        assert cheap_cost < expensive_cost


# ── Synthesis tests ──────────────────────────────────────────────────────────

class TestSynthesis:
    def test_tactic_proposer_generates_allreduce(self):
        """Proposer should generate AllReduce tactics for Partial outputs."""
        from verifier.synthesis import TacticProposer
        from verifier.rewrite import PlacementAnalysis

        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(8, 8),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh), expr="x",
        )
        w = TensorState(
            name="w", global_shape=(16, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh), expr="w",
        )

        fwd = Program(name="test")
        fwd.add(MatMul(a="x", b="w", output="y"))

        executor = MultiDeviceExecutor(mesh)
        executor.register_tensor(x)
        executor.register_tensor(w)
        state = executor.run_program(fwd)

        analysis = PlacementAnalysis(
            partial_tensors=["y"],
            missing_collectives=[(0, "y", AllReduce)],
            redundant_collectives=[],
            collectives_ok=[],
        )

        proposer = TacticProposer()
        tactics = proposer.propose(fwd, analysis, state)

        assert len(tactics) >= 1
        assert any(t.type.value == "insert_allreduce" for t in tactics)

    def test_synthesis_finds_valid_program(self):
        """Synthesis engine should find a valid Row Parallel program."""
        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

        compute_ops = [MatMul(a="x", b="w", output="y")]

        x = TensorState(
            name="x", global_shape=(8, 16), local_shape=(8, 8),
            sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
            expr="x", requires_grad=True,
        )
        w = TensorState(
            name="w", global_shape=(16, 32), local_shape=(8, 32),
            sharding=ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
            expr="w", requires_grad=True,
        )

        from verifier.synthesis import SynthesisEngine
        engine = SynthesisEngine(max_tactics=3, max_search_depth=2)

        compute_program = Program(name="compute")
        compute_program.add(compute_ops[0])

        tensors = {"x": x, "w": w}
        result = engine.synthesize(compute_program, tensors, mesh)

        assert result.success, f"Synthesis failed: no valid candidate found"
        assert result.best_candidate is not None
        assert result.best_candidate.is_valid

        # Best program should have AllReduce
        has_ar = any(
            isinstance(op, AllReduce) for op in result.best_candidate.program.ops
        )
        assert has_ar, "Synthesized program should contain AllReduce"


# ── LLM frontend tests ───────────────────────────────────────────────────────

class TestLLMFrontend:
    def test_parse_op_dict_matmul(self):
        """Parse a MatMul op dict."""
        from verifier.llm_frontend import parse_op_dict

        op = parse_op_dict({
            "type": "MatMul", "a": "x", "b": "w", "output": "y"
        })
        assert isinstance(op, MatMul)
        assert op.a == "x"
        assert op.b == "w"
        assert op.output == "y"

    def test_parse_op_dict_allreduce(self):
        """Parse an AllReduce op dict."""
        from verifier.llm_frontend import parse_op_dict

        op = parse_op_dict({
            "type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"
        })
        assert isinstance(op, AllReduce)
        assert op.x == "y_partial"
        assert op.output == "y"

    def test_parse_op_dict_send_recv(self):
        """Parse Send and Recv op dicts."""
        from verifier.llm_frontend import parse_op_dict

        send = parse_op_dict({
            "type": "Send", "x": "h", "output": "h_sent",
            "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0,
        })
        assert isinstance(send, Send)
        assert send.src == 0 and send.dst == 1

        recv = parse_op_dict({
            "type": "Recv", "x": "h_sent", "output": "h_rcvd",
            "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0,
        })
        assert isinstance(recv, Recv)

    def test_llm_ir_response_to_program(self):
        """Convert LLM JSON response to Program."""
        from verifier.llm_frontend import LLMIRResponse
        import json

        response = json.dumps({
            "fwd_ops": [
                {"type": "MatMul", "a": "x", "b": "w", "output": "y_partial"},
                {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"},
            ],
            "bwd_ops": [],
            "sharding": {"x": "Shard(1)", "w": "Shard(0)"},
        })

        ir = LLMIRResponse.from_json(response)
        program = ir.to_program("test")

        assert len(program.ops) == 2
        assert isinstance(program.ops[0], MatMul)
        assert isinstance(program.ops[1], AllReduce)

    def test_mock_llm_tp_linear(self):
        """Mock LLM should extract correct IR for Row Parallel Linear."""
        from verifier.llm_frontend import MockLLM, LLMIRResponse

        llm = MockLLM()
        response = llm.generate("""
        y_partial = x @ w
        y = dist.all_reduce(y_partial)
        """)

        ir = LLMIRResponse.from_json(response)
        program = ir.to_program("test")

        assert len(program.ops) == 2
        assert isinstance(program.ops[0], MatMul)
        assert isinstance(program.ops[1], AllReduce)

    def test_mock_llm_tp_mlp(self):
        """Mock LLM should extract correct IR for TP MLP."""
        from verifier.llm_frontend import MockLLM, LLMIRResponse

        llm = MockLLM()
        response = llm.generate("""
        gate = silu(x @ w_gate)
        up = x @ w_up
        h = gate * up
        y = dist.all_reduce(h @ w_down)
        """)

        ir = LLMIRResponse.from_json(response)
        program = ir.to_program("test")

        assert len(program.ops) >= 3
        op_types = [type(op).__name__ for op in program.ops]
        assert "MatMul" in op_types
        assert "AllReduce" in op_types

    def test_llm_verification_loop_tp_linear(self):
        """Full LLM + Verifier loop for Row Parallel Linear."""
        from verifier.llm_frontend import LLMVerificationLoop, MockLLM

        mesh = DeviceMesh(shape=(2,), dim_names=("tp",))
        llm = MockLLM()
        loop = LLMVerificationLoop(llm=llm, max_iterations=3)

        from verifier.state import compute_local_shape

        spec_x = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        spec_w = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)

        tensors = {
            "x": TensorState(
                name="x", global_shape=(8, 16),
                local_shape=compute_local_shape((8, 16), spec_x),
                sharding=spec_x, expr="x", requires_grad=True,
            ),
            "w": TensorState(
                name="w", global_shape=(16, 32),
                local_shape=compute_local_shape((16, 32), spec_w),
                sharding=spec_w, expr="w", requires_grad=True,
            ),
        }

        code = """
        y_partial = x @ w  # x: Shard(1), w: Shard(0)
        y = dist.all_reduce(y_partial)
        """

        result = loop.verify_code(code, mesh=mesh, tensor_states=tensors)

        # Should succeed on first try (mock LLM outputs correct IR)
        assert result.success, f"LLM verification failed: {result.errors}"
        assert result.iterations == 1


# ── Temporal verification tests ──────────────────────────────────────────────

class TestTemporal:
    def test_correct_overlap_is_safe(self):
        """Correct overlap: independent compute on COMPUTE while AR on COMM, then Wait."""
        from verifier.ir import AllReduceAsync, Wait, COMM_STREAM
        from verifier.temporal import verify_temporal

        prog = Program("correct")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        prog.add(AllReduceAsync(x="y_partial", output="y", handle="h1",
                                 op_type="sum", stream=COMM_STREAM))
        prog.add(MatMul(a="x2", b="w2", output="z_indep"))
        prog.add(Wait(handle="h1", tensor="y", output="y_safe"))
        prog.add(MatMul(a="y_safe", b="w3", output="z"))

        result = verify_temporal(prog)
        assert result.is_safe, f"Expected safe, got: {result.summary()}"

    def test_missing_wait_detected(self):
        """Missing Wait: async op but no Wait at all."""
        from verifier.ir import AllReduceAsync
        from verifier.temporal import verify_temporal

        prog = Program("missing")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        prog.add(AllReduceAsync(x="y_partial", output="y", handle="h1"))
        prog.add(MatMul(a="y", b="w2", output="z"))  # reads y without Wait!

        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_missing_waits >= 1, f"Expected missing wait, got {result.summary()}"

    def test_buffer_aliasing_detected(self):
        """Buffer aliasing: two async ops writing to the same buffer."""
        from verifier.ir import AllReduceAsync, Wait
        from verifier.temporal import verify_temporal

        prog = Program("alias")
        prog.add(MatMul(a="x", b="w1", output="p1"))
        prog.add(MatMul(a="x", b="w2", output="p2"))
        prog.add(AllReduceAsync(x="p1", output="buf", handle="h1"))
        prog.add(AllReduceAsync(x="p2", output="buf", handle="h2"))  # same buffer!
        prog.add(Wait(handle="h1", tensor="buf", output="buf1"))
        prog.add(Wait(handle="h2", tensor="buf", output="buf2"))

        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_buffer_aliases >= 1, f"Expected buffer aliasing, got {result.summary()}"

    def test_data_race_different_streams(self):
        """Data race: compute reads async output on different stream before Wait."""
        from verifier.ir import AllReduceAsync, Wait, COMM_STREAM, COMPUTE_STREAM
        from verifier.temporal import verify_temporal

        prog = Program("race")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        # AllReduceAsync on COMM stream
        prog.add(AllReduceAsync(x="y_partial", output="y", handle="h1",
                                 op_type="sum", stream=COMM_STREAM))
        # MatMul on default stream reads 'y' before Wait
        prog.add(MatMul(a="y", b="w2", output="z"))
        prog.add(Wait(handle="h1", tensor="y", output="y_safe"))

        result = verify_temporal(prog)
        assert not result.is_safe

    def test_no_wait_is_missing_wait(self):
        """Missing wait with no Wait op whatsoever."""
        from verifier.ir import AllReduceAsync
        from verifier.temporal import verify_temporal

        prog = Program("no_wait")
        prog.add(AllReduceAsync(x="x", output="y", handle="h1"))
        prog.add(MatMul(a="y", b="w", output="z"))  # uses async output, no Wait

        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_missing_waits >= 1

    def test_hb_graph_builds_edges(self):
        """HB graph should have program order + wait edges."""
        from verifier.ir import AllReduceAsync, Wait
        from verifier.temporal import TemporalGraph

        prog = Program("hb")
        prog.add(MatMul(a="x", b="w", output="y_p"))
        prog.add(AllReduceAsync(x="y_p", output="y", handle="h1"))
        prog.add(Wait(handle="h1", tensor="y", output="y_safe"))
        prog.add(MatMul(a="y_safe", b="w2", output="z"))

        graph = TemporalGraph(prog)
        # Should have: 0→1 (program order), 1→2 (wait), 2→3 (program order)
        # Plus data deps: 0→1 (write y_p → read y_p), 1→2 (write y → read y), etc.
        assert len(graph.hb_edges) >= 5, f"Expected >=5 HB edges, got {len(graph.hb_edges)}"

    def test_sync_allreduce_is_safe(self):
        """Sync AllReduce (not async) should have no temporal violations."""
        from verifier.temporal import verify_temporal

        prog = Program("sync")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        prog.add(AllReduce(x="y_partial", output="y"))  # sync AllReduce
        prog.add(MatMul(a="y", b="w2", output="z"))

        result = verify_temporal(prog)
        assert result.is_safe, f"Sync AllReduce should be safe, got {result.summary()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
