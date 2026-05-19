"""End-to-end tests for the LLM frontend pipeline.

Tests the full chain: PyTorch code → MockLLM → IR → executor → DistributedVerifier.
"""

import json
import warnings

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
    AllReduce,
    AllGather,
    FlashAttention,
)
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier
from verifier.temporal import verify_temporal
from verifier.llm_frontend import (
    LLMVerificationLoop,
    LLMIRResponse,
    MockLLM,
    parse_op_dict,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def make_tp_mesh(tp_size=2):
    return DeviceMesh(shape=(tp_size,), dim_names=("tp",))


def _ts(name, shape, spec):
    return TensorState(
        name=name,
        global_shape=shape,
        local_shape=compute_local_shape(shape, spec),
        sharding=spec,
        expr=name.lower(),
        requires_grad=True,
    )


def make_tp_linear_tensors(mesh):
    """Row-parallel: x Shard(1), w Shard(0)."""
    spec_s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
    spec_s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
    return {
        "x": _ts("x", (8, 16), spec_s1),
        "w": _ts("w", (16, 32), spec_s0),
    }


def make_tp_mlp_tensors(mesh):
    """SwiGLU MLP: x Replicate, w_gate/w_up Shard(1), w_down Shard(0)."""
    rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
    s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
    s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
    return {
        "x": _ts("x", (8, 16), rep),
        "w_gate": _ts("w_gate", (16, 32), s1),
        "w_up": _ts("w_up", (16, 32), s1),
        "w_down": _ts("w_down", (32, 16), s0),
        "y": _ts("y", (8, 16), rep),
    }


def run_pipeline(code, mesh, tensors, max_iter=3):
    """Run MockLLM → verify_code() and return (LLMVerifyResult, state_dict)."""
    llm = MockLLM()
    loop = LLMVerificationLoop(llm=llm, max_iterations=max_iter)
    result = loop.verify_code(code, mesh=mesh, tensor_states=tensors)
    if result.success and result.final_program:
        executor = MultiDeviceExecutor(mesh)
        for ts in tensors.values():
            executor.register_tensor(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = executor.run_program(result.final_program)
        return result, state
    return result, {}


# ── TP Linear E2E ───────────────────────────────────────────────────────────


class TestE2ETPLinear:
    def test_full_verify_all_passes(self):
        """Row-parallel linear: full pipeline → verify_all() all pass."""
        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        result, state = run_pipeline(
            "y_partial = x @ w\ny = dist.all_reduce(y_partial)", mesh, tensors
        )
        assert result.success
        assert result.iterations == 1

        prog = result.final_program
        assert sum(1 for op in prog.ops if isinstance(op, AllReduce)) == 1

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state)
        for vr in results:
            assert vr.passed, f"{vr.condition} failed: {vr.details}"

    def test_postcondition_and_placement_consistency(self):
        """Postcondition (not partial) and Z3 placement consistency both pass."""
        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        result, state = run_pipeline(
            "y_partial = x @ w\ny = dist.all_reduce(y_partial)", mesh, tensors
        )
        assert result.success

        verifier = DistributedVerifier()
        y = state["y"]
        pc = verifier.verify_postcondition(y, expected_partial=False)
        assert pc.passed, f"postcondition failed: {pc.details}"

        plc = verifier.verify_placement_consistency(
            result.final_program, final_tensors=state, output_names=["y"],
        )
        assert plc.passed, f"placement consistency failed: {plc.details}"

    def test_temporal_trivially_safe(self):
        """No async ops → temporal verification is trivially safe."""
        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        result, _ = run_pipeline(
            "y_partial = x @ w\ny = dist.all_reduce(y_partial)", mesh, tensors
        )
        assert result.success
        tr = verify_temporal(result.final_program)
        assert tr.is_safe, f"Expected safe: {tr.summary()}"


# ── TP MLP E2E ──────────────────────────────────────────────────────────────


class TestE2ETPMlp:
    def test_swiglu_full_verify_all_passes(self):
        """SwiGLU MLP: gate+silu+up+multiply+down+AllReduce → all checks pass."""
        mesh = make_tp_mesh()
        tensors = make_tp_mlp_tensors(mesh)
        code = (
            "gate = silu(x @ w_gate)\n"
            "up = x @ w_up\n"
            "h = gate * up\n"
            "y = dist.all_reduce(h @ w_down)"
        )
        result, state = run_pipeline(code, mesh, tensors)
        assert result.success
        assert result.iterations == 1

        prog = result.final_program
        assert len(prog.ops) == 6
        assert sum(1 for op in prog.ops if isinstance(op, AllReduce)) == 1

        gate = state.get("gate")
        assert gate is not None
        assert any(isinstance(p, Shard) and p.dim == 1 for p in gate.sharding.placements)

        y = state.get("y")
        assert y is not None
        assert not y.partial

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state)
        for vr in results:
            assert vr.passed, f"{vr.condition} failed: {vr.details}"

    def test_shape_consistency_passes(self):
        """Z3 shape consistency check passes for MLP."""
        mesh = make_tp_mesh()
        tensors = make_tp_mlp_tensors(mesh)
        code = (
            "gate = silu(x @ w_gate)\n"
            "up = x @ w_up\n"
            "h = gate * up\n"
            "y = dist.all_reduce(h @ w_down)"
        )
        result, state = run_pipeline(code, mesh, tensors)
        assert result.success

        verifier = DistributedVerifier()
        sc = verifier.verify_shape_consistency(result.final_program, {}, state)
        assert sc.passed, f"shape consistency failed: {sc.details}"


# ── Pipeline Parallelism E2E ────────────────────────────────────────────────


class TestE2EPipelineParallel:
    def test_pp_extracts_correct_ir_but_single_executor_fails(self):
        """PP extraction works but single-executor verify_code() fails.

        The MockLLM correctly extracts Send/Recv IR, but the single-device
        MultiDeviceExecutor cannot execute PP programs (Recv output only
        exists on the destination device).  verify_code() returns failure.
        We verify the extracted IR structure is correct regardless.
        """
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        tensors = {
            "x": _ts("x", (8, 16), rep),
            "w0": _ts("w0", (16, 16), rep),
            "w1": _ts("w1", (16, 16), rep),
        }
        code = "h0 = x @ w0\nsend(h0, dst=1)\nh0 = recv(src=0)\ny = h0 @ w1"

        llm = MockLLM()
        loop = LLMVerificationLoop(llm=llm, max_iterations=1)
        result = loop.verify_code(code, mesh=mesh, tensor_states=tensors)

        # verify_code() fails because strict executor can't run PP
        assert not result.success

        # But the extracted IR structure is correct
        prog = result.final_program
        assert prog is not None
        assert len(prog.ops) == 4

        from verifier.ir import Send, Recv
        sends = [op for op in prog.ops if isinstance(op, Send)]
        recvs = [op for op in prog.ops if isinstance(op, Recv)]
        assert len(sends) == 1
        assert len(recvs) == 1
        assert sends[0].dst == recvs[0].dst

        # Comm legality on the program itself passes (Send/Recv matched)
        executor = MultiDeviceExecutor(mesh)
        for ts in tensors.values():
            executor.register_tensor(ts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = executor.run_program(prog)

        verifier = DistributedVerifier()
        cl = verifier.verify_communication_legality(prog, tensor_states=state)
        assert cl.passed, f"comm legality failed: {cl.details}"


# ── Bug Detection E2E ───────────────────────────────────────────────────────


class TestE2EBugDetection:
    def test_missing_allreduce_fails_postcondition(self):
        """MatMul without AllReduce: output is Partial → postcondition fails."""
        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        prog = Program("bug", ops=[MatMul(a="x", b="w", output="y")])

        executor = MultiDeviceExecutor(mesh)
        for ts in tensors.values():
            executor.register_tensor(ts)
        state = executor.run_program(prog)

        y = state["y"]
        assert y.partial

        verifier = DistributedVerifier()
        pc = verifier.verify_postcondition(y, expected_partial=False)
        assert not pc.passed

    def test_allreduce_on_non_partial_raises(self):
        """AllReduce on Replicate tensor raises ValueError."""
        mesh = make_tp_mesh()
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)
        tensors = {"x": _ts("x", (8, 16), rep)}

        prog = Program("bad_ar", ops=[AllReduce(x="x", output="x_r")])
        executor = MultiDeviceExecutor(mesh)
        for ts in tensors.values():
            executor.register_tensor(ts)

        with pytest.raises(ValueError, match="PARTIAL|not partial"):
            executor.run_program(prog)

    def test_illegal_allgather_fails_via_loop(self):
        """AllGather on Replicate tensor: loop returns failure."""

        class IllegalLLM:
            call_count = 0
            call_history = []

            def generate(self, prompt):
                self.call_count += 1
                resp = json.dumps({
                    "fwd_ops": [{"type": "AllGather", "x": "x", "output": "y", "gather_dim": 0}],
                    "bwd_ops": [],
                    "sharding": {"x": "Replicate"},
                })
                self.call_history.append((prompt, resp))
                return resp

        mesh = make_tp_mesh()
        loop = LLMVerificationLoop(llm=IllegalLLM(), max_iterations=1)
        result = loop.verify_code("y = allgather(x)", mesh=mesh)
        assert not result.success


# ── Feedback Loop E2E ───────────────────────────────────────────────────────


class TestE2EFeedbackLoop:
    def test_stateful_mock_corrects_on_iteration_2(self):
        """First iteration: wrong IR (no AllReduce). Second: correct IR."""

        class CorrectingLLM:
            def __init__(self):
                self.call_count = 0
                self.call_history = []

            def generate(self, prompt):
                self.call_count += 1
                if self.call_count == 1:
                    resp = json.dumps({
                        "fwd_ops": [{"type": "MatMul", "a": "x", "b": "w", "output": "y"}],
                        "bwd_ops": [],
                        "sharding": {"x": "Shard(1)", "w": "Shard(0)"},
                    })
                else:
                    resp = json.dumps({
                        "fwd_ops": [
                            {"type": "MatMul", "a": "x", "b": "w", "output": "y_partial"},
                            {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"},
                        ],
                        "bwd_ops": [],
                        "sharding": {"x": "Shard(1)", "w": "Shard(0)"},
                    })
                self.call_history.append((prompt, resp))
                return resp

        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        loop = LLMVerificationLoop(llm=CorrectingLLM(), max_iterations=3)
        result = loop.verify_code(
            "y_partial = x @ w\ny = dist.all_reduce(y_partial)",
            mesh=mesh, tensor_states=tensors,
        )
        assert result.success
        assert result.iterations == 2

    def test_max_iterations_exhausted(self):
        """Always-wrong LLM exhausts max iterations."""

        class AlwaysWrongLLM:
            call_count = 0
            call_history = []

            def generate(self, prompt):
                self.call_count += 1
                resp = json.dumps({"fwd_ops": [], "bwd_ops": [], "sharding": {}})
                self.call_history.append((prompt, resp))
                return resp

        mesh = make_tp_mesh()
        loop = LLMVerificationLoop(llm=AlwaysWrongLLM(), max_iterations=2)
        result = loop.verify_code("y = x @ w", mesh=mesh)
        assert not result.success
        assert result.iterations == 2

    def test_feedback_contains_error_details(self):
        """Feedback prompt to LLM contains verifier error text."""

        class SpyLLM:
            def __init__(self):
                self.call_count = 0
                self.call_history = []

            def generate(self, prompt):
                self.call_count += 1
                resp = json.dumps({
                    "fwd_ops": [{"type": "MatMul", "a": "x", "b": "w", "output": "y"}],
                    "bwd_ops": [],
                    "sharding": {"x": "Shard(1)", "w": "Shard(0)"},
                })
                self.call_history.append((prompt, resp))
                return resp

        mesh = make_tp_mesh()
        tensors = make_tp_linear_tensors(mesh)
        loop = LLMVerificationLoop(llm=SpyLLM(), max_iterations=2)
        result = loop.verify_code("y = x @ w", mesh=mesh, tensor_states=tensors)
        assert not result.success

        assert len(loop.llm.call_history) >= 2
        feedback_prompt = loop.llm.call_history[1][0]
        assert "Missing" in feedback_prompt or "partial" in feedback_prompt.lower()


# ── Direct JSON (Attention) E2E ─────────────────────────────────────────────


class TestE2EDirectJSON:
    def test_tp_attention_via_direct_json(self):
        """TP attention: ColParallel QKV → FlashAttention → RowParallel out → AllReduce."""
        mesh = make_tp_mesh()
        s1 = ShardingSpec(placements=(Shard(dim=1),), mesh=mesh)
        s0 = ShardingSpec(placements=(Shard(dim=0),), mesh=mesh)
        rep = ShardingSpec(placements=(Replicate(),), mesh=mesh)

        tensors = {
            "x": _ts("x", (8, 16), rep),
            "W_qkv": _ts("W_qkv", (16, 48), s1),
            "W_out": _ts("W_out", (48, 16), s0),
        }

        ir_resp = LLMIRResponse(
            fwd_ops=[
                {"type": "MatMul", "a": "x", "b": "W_qkv", "output": "qkv"},
                {"type": "FlashAttention", "q": "qkv", "k": "qkv", "v": "qkv", "output": "attn"},
                {"type": "MatMul", "a": "attn", "b": "W_out", "output": "out_partial"},
                {"type": "AllReduce", "x": "out_partial", "output": "out", "op_type": "sum"},
            ],
            bwd_ops=[],
            sharding={},
        )
        prog = ir_resp.to_program("tp_attn")
        assert len(prog.ops) == 4

        executor = MultiDeviceExecutor(mesh)
        for ts in tensors.values():
            executor.register_tensor(ts)
        state = executor.run_program(prog)

        out = state["out"]
        assert not out.partial

        verifier = DistributedVerifier()
        results = verifier.verify_all(prog, state)
        for vr in results:
            assert vr.passed, f"{vr.condition} failed: {vr.details}"
