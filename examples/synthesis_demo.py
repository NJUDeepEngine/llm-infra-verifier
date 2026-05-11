"""
Demo: Verified Parallelization Synthesis.

Shows the full pipeline:
  1. Single-device compute → Execute →Find Partial tensors
  2. Tactic Proposer → Generate candidate fixes
  3. Synthesis Engine → Search over tactic combinations
  4. Verifier → Check each candidate
  5. Select → Minimal-cost correct program

Also demonstrates the LLM frontend (with mock LLM) for:
  PyTorch code → IR extraction → Verification → Feedback loop
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verifier.state import (
    TensorState, DeviceMesh, ShardingSpec, Shard, Replicate, Partial,
    compute_local_shape,
)
from verifier.ir import Program, MatMul, AllReduce, ir_to_str
from verifier.executor import MultiDeviceExecutor
from verifier.rewrite import PlacementAnalyzer, ProgramCost, PatternSynthesizer
from verifier.synthesis import SynthesisEngine, TacticProposer, synthesize_parallel_program
from verifier.solver import DistributedVerifier
from verifier.llm_frontend import (
    LLMVerificationLoop, MockLLM, PromptBuilder, LLMIRResponse,
    extract_and_verify,
)


def demo_synthesis_row_parallel():
    """Demonstrate synthesis for Row Parallel Linear."""
    print("=" * 70)
    print("  SYNTHESIS DEMO: Row Parallel Linear")
    print("=" * 70)

    # ── Input: compute-only program + sharding ──
    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    # Single-device compute: just MatMul
    compute_ops = [MatMul(a="x", b="w", output="y")]

    # Sharding: Row Parallel (both sharded on reduce dim H)
    sharding = {
        "x": ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        "w": ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
    }

    input_shapes = {"x": (8, 16), "w": (16, 32)}

    print("\n📥 Input:")
    print(f"  Mesh: {mesh}")
    print(f"  Compute: MatMul(x, w) → y")
    print(f"  Sharding: x=Shard(1), w=Shard(0)")
    print(f"  Shapes: x={input_shapes['x']}, w={input_shapes['w']}")

    # ── Step 1: Execute compute-only → find issues ──
    print("\n🔍 Step 1: Execute compute-only program...")
    executor = MultiDeviceExecutor(mesh)
    for name, shape in input_shapes.items():
        spec = sharding[name]
        local = compute_local_shape(shape, spec)
        ts = TensorState(
            name=name, global_shape=shape, local_shape=local,
            sharding=spec, expr=name.lower(), requires_grad=True,
        )
        executor.register_tensor(ts)

    compute_program = Program(name="compute_only")
    compute_program.add(compute_ops[0])
    state = executor.run_program(compute_program)

    analyzer = PlacementAnalyzer()
    analysis = analyzer.analyze(compute_program, state)
    print(f"  {analysis}")

    # ── Step 2: Propose tactics ──
    print("\n💡 Step 2: Propose tactics...")
    proposer = TacticProposer()
    tactics = proposer.propose(compute_program, analysis, state)
    for t in tactics:
        print(f"  {t}")

    # ── Step 3: Synthesize (search over tactics) ──
    print("\n🔧 Step 3: Synthesize (search)...")
    engine = SynthesisEngine(max_tactics=3, max_search_depth=2)
    tensors_dict = {
        name: TensorState(
            name=name, global_shape=shape,
            local_shape=compute_local_shape(shape, sharding[name]),
            sharding=sharding[name], expr=name.lower(), requires_grad=True,
        )
        for name, shape in input_shapes.items()
    }
    result = engine.synthesize(compute_program, tensors_dict, mesh)

    print(f"  Candidates evaluated: {len(result.all_candidates)}")
    for i, c in enumerate(result.all_candidates):
        status = "✅" if c.is_valid else "❌"
        print(f"    [{i}] {status} {c}")

    # ── Step 4: Show best result ──
    print("\n✅ Step 4: Best synthesized program:")
    if result.best_candidate:
        print(f"  Cost: {result.best_candidate.cost}")
        print(f"  Tactics: {len(result.best_candidate.tactics_applied)}")
        for t in result.best_candidate.tactics_applied:
            print(f"    - {t.description}")
        print(f"\n  Program:")
        print(f"  {ir_to_str(result.best_candidate.program)}")

        # Verify final program
        executor2 = MultiDeviceExecutor(mesh)
        for name, ts in tensors_dict.items():
            executor2.register_tensor(ts)
        final_state = executor2.run_program(result.best_candidate.program)

        verifier = DistributedVerifier()
        vr = verifier.verify_postcondition(
            final_state.get("y_reduced", final_state.get("y")),
            expected_partial=False,
        )
        print(f"  Final verification: {'PASSED' if vr.passed else 'FAILED'}")
    else:
        print("  No valid candidate found!")

    print()
    return result


def demo_synthesis_mlp():
    """Demonstrate synthesis for TP MLP."""
    print("=" * 70)
    print("  SYNTHESIS DEMO: Megatron TP MLP")
    print("=" * 70)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    # Compute-only program
    compute_ops = [
        MatMul(a="x", b="w_gate", output="gate_raw"),
        MatMul(a="x", b="w_up", output="up"),
        MatMul(a="h", b="w_down", output="y"),
    ]

    # Note: gate_raw is naturally Shard(1) from Column Parallel
    # up is naturally Shard(1)
    # h = gate * up (element-wise, same placement)
    # y is from Row Parallel: Shard(1) @ Shard(0) → Partial

    sharding = {
        "x": ShardingSpec(placements=(Replicate(),), mesh=mesh),
        "w_gate": ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        "w_up": ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        "w_down": ShardingSpec(placements=(Shard(dim=0),), mesh=mesh),
    }

    input_shapes = {
        "x": (8, 16),
        "w_gate": (16, 64),
        "w_up": (16, 64),
        "w_down": (64, 16),
    }

    print("\n📥 Input:")
    print(f"  Compute: gate=X@W_gate, up=X@W_up, y=H@W_down")
    print(f"  Sharding: X=Rep, W_gate/up=Shard(1), W_down=Shard(0)")

    # Execute
    executor = MultiDeviceExecutor(mesh)
    for name, shape in input_shapes.items():
        spec = sharding[name]
        local = compute_local_shape(shape, spec)
        ts = TensorState(
            name=name, global_shape=shape, local_shape=local,
            sharding=spec, expr=name.lower(), requires_grad=True,
        )
        executor.register_tensor(ts)

    compute_program = Program(name="compute")
    for op in compute_ops:
        compute_program.add(op)
    state = executor.run_program(compute_program)

    analysis = PlacementAnalyzer().analyze(compute_program, state)
    print(f"\n🔍 Analysis: {analysis}")

    # Synthesize
    engine = SynthesisEngine(max_tactics=3, max_search_depth=2)
    tensors_dict = {
        name: TensorState(
            name=name, global_shape=shape,
            local_shape=compute_local_shape(shape, sharding[name]),
            sharding=sharding[name], expr=name.lower(), requires_grad=True,
        )
        for name, shape in input_shapes.items()
    }
    result = engine.synthesize(compute_program, tensors_dict, mesh)

    print(f"\n✅ Best program:")
    if result.best_candidate:
        print(f"  {ir_to_str(result.best_candidate.program)}")
        print(f"  Cost: {result.best_candidate.cost}")
    else:
        print("  No valid candidate (expected: gate/up need intermediate tensors)")

    print()
    return result


def demo_llm_frontend():
    """Demonstrate the LLM frontend with mock LLM."""
    print("=" * 70)
    print("  LLM FRONTEND DEMO: PyTorch → IR → Verify → Refine")
    print("=" * 70)

    mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

    # ── Test 1: Row Parallel Linear (correct code) ──
    print("\n📝 Test 1: Row Parallel Linear (correct)")
    code_1 = """
# Row Parallel Linear forward
y_partial = x @ w  # x: Shard(1), w: Shard(0)
y = dist.all_reduce(y_partial, op=ReduceOp.SUM)
"""
    llm = MockLLM()
    result_1 = extract_and_verify(code_1, mesh=mesh, llm=llm)
    print(f"  {result_1.summary()}")

    # ── Test 2: Missing AllReduce (LLM should detect) ──
    print("\n📝 Test 2: Row Parallel with bug (missing AllReduce)")
    code_2 = """
# BUG: Row Parallel Linear without all_reduce!
y = x @ w  # x: Shard(1), w: Shard(0) — output is PARTIAL!
"""
    # For this test, manually extract and verify
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build_extraction_prompt(code_2)
    response = llm.generate(prompt)
    ir_resp = LLMIRResponse.from_json(response)
    program = ir_resp.to_program("bug_test")

    print(f"  LLM extracted ops: {[type(op).__name__ for op in program.ops]}")
    print(f"  LLM sharding: {ir_resp.sharding}")

    # The mock LLM detects "all_reduce" keyword and always includes it.
    # Let's manually test the "missing" case:
    bug_program = Program(name="bug")
    bug_program.add(MatMul(a="x", b="w", output="y"))
    bug_program_ops = [type(op).__name__ for op in bug_program.ops]
    print(f"  Bug program ops (no AllReduce): {bug_program_ops}")

    # The LLM would need to be prompted with the feedback to fix it
    feedback_prompt = prompt_builder.build_feedback_prompt(
        code_2,
        json.dumps([{"type": "MatMul", "a": "x", "b": "w", "output": "y"}]),
        ["Missing AllReduce for tensor 'y' — output is PARTIAL"],
    )
    llm.reset()
    fixed_response = llm.generate(feedback_prompt)
    fixed_ir = LLMIRResponse.from_json(fixed_response)
    fixed_program = fixed_ir.to_program("fixed")
    print(f"  After feedback, LLM fixed ops: {[type(op).__name__ for op in fixed_program.ops]}")

    # ── Test 3: Show prompt structure ──
    print("\n📋 Test 3: Prompt template preview")
    prompt_3 = prompt_builder.build_extraction_prompt(
        "y = x @ w\ny = dist.all_reduce(y)"
    )
    # Show first 300 chars
    print(f"  Prompt length: {len(prompt_3)} chars")
    print(f"  Includes IR schema: {'IR Operation Format' in prompt_3}")
    print(f"  Includes few-shot examples: {'Example 1' in prompt_3}")

    print()
    return result_1


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  DTENSOR-VERIFIER: Verified Parallelization Synthesis")
    print("=" * 70 + "\n")

    result_tp = demo_synthesis_row_parallel()
    result_mlp = demo_synthesis_mlp()
    result_llm = demo_llm_frontend()

    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Row Parallel synthesis:  {'SUCCESS' if result_tp.success else 'FAILED'}")
    print(f"  MLP synthesis:           {'SUCCESS' if result_mlp.success else 'FAILED'}")
    print(f"  LLM frontend:            {'SUCCESS' if result_llm.success else 'FAILED'}")
    print()
