"""LLM frontend for translating PyTorch/NCCL distributed code to verification IR.

Implements the "LLM proposes tactics → Verifier checks → Search/refine" loop.

Supports:
  1. Prompt templates for PyTorch → IR translation
  2. Structured output parsing (JSON → IROp)
  3. Few-shot examples of known parallel patterns
  4. Feedback loop: verification errors → LLM refinement
  5. Pluggable LLM backend (Anthropic, OpenAI, or mock for testing)

Architecture:
  PyTorch Code ──(Prompt)──> LLM ──(JSON)──> IR Parser ──> Program
       ↑                                                       │
       │                                                       │
       └──── Feedback (errors) ──── Verifier ◄─────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import json
import re

from .state import (
    TensorState,
    DeviceMesh,
    ShardingSpec,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
)
from .ir import (
    IROp,
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
    Reshape,
    Transpose,
)
from .executor import MultiDeviceExecutor
from .solver import DistributedVerifier
from .rewrite import PlacementAnalyzer


# ── LLM response model ───────────────────────────────────────────────────────

@dataclass
class LLMIRResponse:
    """Structured response from LLM containing extracted IR."""
    fwd_ops: List[dict]
    bwd_ops: List[dict]
    sharding: Dict[str, str] = field(default_factory=dict)
    mesh: Optional[dict] = None  # {"shape": [2], "dim_names": ["tp"]}
    raw_response: str = ""

    @classmethod
    def from_json(cls, json_str: str) -> LLMIRResponse:
        """Parse LLM JSON response."""
        # Extract JSON from potentially noisy LLM output
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(json_str)

        return cls(
            fwd_ops=data.get("fwd_ops", []),
            bwd_ops=data.get("bwd_ops", []),
            sharding=data.get("sharding", {}),
            mesh=data.get("mesh"),
            raw_response=json_str,
        )

    def to_program(self, name: str = "llm_extracted") -> Program:
        """Convert the fwd_ops dicts into a Program."""
        program = Program(name=name)
        for op_dict in self.fwd_ops:
            op = parse_op_dict(op_dict)
            if op:
                program.add(op)
        return program

    def to_bwd_program(self) -> Program:
        """Convert bwd_ops dicts into a Program."""
        program = Program(name="llm_extracted_bwd")
        for op_dict in self.bwd_ops:
            op = parse_op_dict(op_dict)
            if op:
                program.add(op)
        return program


def parse_op_dict(op_dict: dict) -> Optional[IROp]:
    """Parse a single op dictionary into an IROp instance.

    Expected format:
      {"type": "MatMul", "a": "x", "b": "w", "output": "y"}
      {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"}
      {"type": "Send", "x": "h", "output": "h_sent", "src": 0, "dst": 1, ...}
    """
    op_type = op_dict.get("type", "")

    if op_type == "MatMul":
        return MatMul(
            a=op_dict["a"], b=op_dict["b"],
            output=op_dict.get("output", f"{op_dict['a']}_{op_dict['b']}_mm"),
        )
    elif op_type == "Add":
        return Add(
            a=op_dict["a"], b=op_dict["b"],
            output=op_dict.get("output", f"{op_dict['a']}_{op_dict['b']}_add"),
        )
    elif op_type == "Multiply":
        return Multiply(
            a=op_dict["a"], b=op_dict["b"],
            output=op_dict.get("output", f"{op_dict['a']}_{op_dict['b']}_mul"),
        )
    elif op_type == "SiLU":
        return SiLU(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_silu"),
        )
    elif op_type == "AllReduce":
        return AllReduce(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_reduced"),
            op_type=op_dict.get("op_type", "sum"),
        )
    elif op_type == "AllGather":
        return AllGather(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_gathered"),
            gather_dim=op_dict.get("gather_dim", 0),
        )
    elif op_type == "ReduceScatter":
        return ReduceScatter(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_scattered"),
            scatter_dim=op_dict.get("scatter_dim", 0),
            op_type=op_dict.get("op_type", "sum"),
        )
    elif op_type == "Send":
        return Send(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_sent"),
            src=op_dict.get("src", 0),
            dst=op_dict.get("dst", 1),
            stage=op_dict.get("stage", 0),
            microbatch_id=op_dict.get("microbatch_id", 0),
        )
    elif op_type == "Recv":
        return Recv(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_rcvd"),
            src=op_dict.get("src", 0),
            dst=op_dict.get("dst", 1),
            stage=op_dict.get("stage", 0),
            microbatch_id=op_dict.get("microbatch_id", 0),
        )
    elif op_type == "FlashAttention":
        return FlashAttention(
            q=op_dict.get("q", "q"),
            k=op_dict.get("k", "k"),
            v=op_dict.get("v", "v"),
            output=op_dict.get("output", "o"),
            softmax_scale=op_dict.get("softmax_scale", 1.0),
            causal=op_dict.get("causal", False),
        )
    elif op_type == "Reshape":
        return Reshape(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_reshaped"),
            new_shape=tuple(op_dict.get("new_shape", ())),
        )
    elif op_type == "Transpose":
        return Transpose(
            x=op_dict["x"],
            output=op_dict.get("output", f"{op_dict['x']}_T"),
            dim0=op_dict.get("dim0", 0),
            dim1=op_dict.get("dim1", 1),
        )

    return None


# ── Prompt templates ─────────────────────────────────────────────────────────

IR_SCHEMA_DESCRIPTION = """
IR Operation Format (JSON):
  MatMul:        {"type": "MatMul", "a": "...", "b": "...", "output": "..."}
  Add:           {"type": "Add", "a": "...", "b": "...", "output": "..."}
  Multiply:      {"type": "Multiply", "a": "...", "b": "...", "output": "..."}
  SiLU:          {"type": "SiLU", "x": "...", "output": "..."}
  AllReduce:     {"type": "AllReduce", "x": "...", "output": "...", "op_type": "sum"}
  AllGather:     {"type": "AllGather", "x": "...", "output": "...", "gather_dim": 0}
  ReduceScatter: {"type": "ReduceScatter", "x": "...", "output": "...", "scatter_dim": 0}
  Send:          {"type": "Send", "x": "...", "output": "...", "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0}
  Recv:          {"type": "Recv", "x": "...", "output": "...", "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0}
  FlashAttention: {"type": "FlashAttention", "q": "...", "k": "...", "v": "...", "output": "..."}
  Reshape:       {"type": "Reshape", "x": "...", "output": "...", "new_shape": [...]}
  Transpose:     {"type": "Transpose", "x": "...", "output": "...", "dim0": 0, "dim1": 1}
"""

FEW_SHOT_EXAMPLE_TP_LINEAR = """
Example 1 — Row Parallel Linear:
PyTorch code:
```python
# Row Parallel Linear forward
y_partial = x @ w  # x: Shard(0), w: Shard(1)
y = dist.all_reduce(y_partial, op=ReduceOp.SUM)
```
IR response:
```json
{
  "fwd_ops": [
    {"type": "MatMul", "a": "x", "b": "w", "output": "y_partial"},
    {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"}
  ],
  "bwd_ops": [],
  "sharding": {"x": "Shard(0)", "w": "Shard(1)", "y": "Replicate"}
}
```
"""

FEW_SHOT_EXAMPLE_TP_MLP = """
Example 2 — Megatron TP MLP:
PyTorch code:
```python
# Column Parallel (gate): no fwd communication
gate = silu(x @ w_gate)  # x: Replicate, w_gate: Shard(1)
# Column Parallel (up): no fwd communication
up = x @ w_up  # x: Replicate, w_up: Shard(1)
h = gate * up
# Row Parallel (down): needs AllReduce
y = dist.all_reduce(h @ w_down, op=ReduceOp.SUM)  # h: Shard(1), w_down: Shard(0)
```
IR response:
```json
{
  "fwd_ops": [
    {"type": "MatMul", "a": "x", "b": "w_gate", "output": "gate_raw"},
    {"type": "SiLU", "x": "gate_raw", "output": "gate"},
    {"type": "MatMul", "a": "x", "b": "w_up", "output": "up"},
    {"type": "Multiply", "a": "gate", "b": "up", "output": "h"},
    {"type": "MatMul", "a": "h", "b": "w_down", "output": "y_partial"},
    {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"}
  ],
  "bwd_ops": [],
  "sharding": {
    "x": "Replicate",
    "w_gate": "Shard(1)",
    "w_up": "Shard(1)",
    "w_down": "Shard(0)",
    "y": "Replicate"
  }
}
```
"""

FEW_SHOT_EXAMPLE_PP = """
Example 3 — Pipeline Parallelism (2-stage, 1F1B):
PyTorch code:
```python
# Stage 0 (device 0)
h0 = layer0(x)
send(h0, dst=1)

# Stage 1 (device 1)
h0 = recv(src=0)
y = layer1(h0)
```
IR response:
```json
{
  "fwd_ops": [
    {"type": "MatMul", "a": "x", "b": "w0", "output": "h0"},
    {"type": "Send", "x": "h0", "output": "h0_sent", "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0},
    {"type": "Recv", "x": "h0_sent", "output": "h0_rcvd", "src": 0, "dst": 1, "stage": 1, "microbatch_id": 0},
    {"type": "MatMul", "a": "h0_rcvd", "b": "w1", "output": "y"}
  ],
  "bwd_ops": [],
  "sharding": {"x": "Replicate", "w0": "Replicate", "w1": "Replicate"}
}
```
"""

EXTRACTION_PROMPT = """You are a distributed tensor program analyzer. Extract the computation graph and communication operations from the given PyTorch code into a structured IR format.

Rules:
1. For each torch operation (matmul, add, multiply, silu, etc.), create a compute IR op.
2. For each distributed collective (all_reduce, all_gather, reduce_scatter, send, recv), create a communication IR op.
3. Infer the sharding/placement of each tensor from the code context.
4. If the code has backward pass, extract it separately.
5. Output ONLY valid JSON, no explanation.

{ir_schema}

{few_shot_examples}

PyTorch Code:
```python
{code}
```

Respond with JSON containing "fwd_ops", "bwd_ops", "sharding", and optionally "mesh".
"""

FEEDBACK_PROMPT = """The verifier found issues with your extracted IR. Please fix them.

Original PyTorch Code:
```python
{code}
```

Your Previous IR:
```json
{previous_ir}
```

Verification Errors:
{errors}

Please provide a corrected IR that fixes ALL of these issues.
Respond with JSON containing "fwd_ops", "bwd_ops", "sharding".
"""


# ── Prompt builder ───────────────────────────────────────────────────────────

@dataclass
class PromptBuilder:
    """Build prompts for LLM extraction and refinement."""

    include_examples: bool = True

    def build_extraction_prompt(self, code: str) -> str:
        """Build the initial extraction prompt."""
        examples = ""
        if self.include_examples:
            examples = (
                FEW_SHOT_EXAMPLE_TP_LINEAR
                + FEW_SHOT_EXAMPLE_TP_MLP
                + FEW_SHOT_EXAMPLE_PP
            )

        return EXTRACTION_PROMPT.format(
            ir_schema=IR_SCHEMA_DESCRIPTION,
            few_shot_examples=examples,
            code=code,
        )

    def build_feedback_prompt(
        self,
        code: str,
        previous_ir: str,
        errors: List[str],
    ) -> str:
        """Build a feedback/refinement prompt."""
        return FEEDBACK_PROMPT.format(
            code=code,
            previous_ir=previous_ir,
            errors="\n".join(f"  - {e}" for e in errors),
        )


# ── Mock LLM for testing ─────────────────────────────────────────────────────

class MockLLM:
    """Mock LLM that returns predefined IR for known patterns.

    In production, replace with Anthropic/OpenAI API calls.
    """

    def __init__(self):
        self.call_count = 0
        self.call_history: List[Tuple[str, str]] = []  # [(prompt, response)]

    def _extract_user_code(self, prompt: str) -> str:
        """Extract the user code section from the prompt, ignoring few-shot examples."""
        marker = "PyTorch Code:"
        idx = prompt.rfind(marker)
        if idx >= 0:
            return prompt[idx:]
        marker_lower = "pytorch code:"
        idx = prompt.lower().rfind(marker_lower)
        if idx >= 0:
            return prompt[idx:]
        return prompt

    def generate(self, prompt: str) -> str:
        """Generate a response based on known patterns in the prompt."""
        self.call_count += 1
        user_code = self._extract_user_code(prompt)
        code_lower = user_code.lower()

        has_allreduce = "all_reduce" in code_lower
        has_matmul = "matmul" in code_lower or " @ " in user_code
        has_send_recv = "send" in code_lower and "recv" in code_lower

        if "silu" in code_lower and "gate" in code_lower:
            # TP MLP pattern
            response = json.dumps({
                "fwd_ops": [
                    {"type": "MatMul", "a": "x", "b": "w_gate", "output": "gate_raw"},
                    {"type": "SiLU", "x": "gate_raw", "output": "gate"},
                    {"type": "MatMul", "a": "x", "b": "w_up", "output": "up"},
                    {"type": "Multiply", "a": "gate", "b": "up", "output": "h"},
                    {"type": "MatMul", "a": "h", "b": "w_down", "output": "y_partial"},
                    {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"},
                ],
                "bwd_ops": [],
                "sharding": {
                    "x": "Replicate", "w_gate": "Shard(1)",
                    "w_up": "Shard(1)", "w_down": "Shard(0)",
                    "y": "Replicate",
                },
            })
        elif has_send_recv:
            # PP pattern
            response = json.dumps({
                "fwd_ops": [
                    {"type": "MatMul", "a": "x", "b": "w0", "output": "h0"},
                    {"type": "Send", "x": "h0", "output": "h0_sent", "src": 0, "dst": 1, "stage": 0, "microbatch_id": 0},
                    {"type": "Recv", "x": "h0_sent", "output": "h0_rcvd", "src": 0, "dst": 1, "stage": 1, "microbatch_id": 0},
                    {"type": "MatMul", "a": "h0_rcvd", "b": "w1", "output": "y"},
                ],
                "bwd_ops": [],
                "sharding": {"x": "Replicate", "w0": "Replicate", "w1": "Replicate"},
            })
        elif has_allreduce and has_matmul:
            # Row Parallel Linear pattern
            response = json.dumps({
                "fwd_ops": [
                    {"type": "MatMul", "a": "x", "b": "w", "output": "y_partial"},
                    {"type": "AllReduce", "x": "y_partial", "output": "y", "op_type": "sum"},
                ],
                "bwd_ops": [],
                "sharding": {"x": "Shard(1)", "w": "Shard(0)", "y": "Replicate"},
                "mesh": {"shape": [2], "dim_names": ["tp"]},
            })
        else:
            # Generic: try to extract matmul
            response = json.dumps({
                "fwd_ops": [
                    {"type": "MatMul", "a": "x", "b": "w", "output": "y"},
                ],
                "bwd_ops": [],
                "sharding": {"x": "Replicate", "w": "Replicate"},
            })

        self.call_history.append((prompt, response))
        return response

    def reset(self):
        self.call_count = 0
        self.call_history.clear()


# ── LLM verification loop ────────────────────────────────────────────────────

@dataclass
class LLMVerifyResult:
    """Result of the LLM + Verifier loop."""
    success: bool
    iterations: int
    final_program: Optional[Program] = None
    final_bwd_program: Optional[Program] = None
    verification_results: List = field(default_factory=list)
    llm_call_history: List[Tuple[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"LLM-Verify Loop: {'SUCCESS' if self.success else 'FAILED'}",
            f"  Iterations: {self.iterations}",
            f"  LLM calls: {len(self.llm_call_history)}",
        ]
        if self.final_program:
            lines.append(f"  Final program: {len(self.final_program)} ops")
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        return "\n".join(lines)


class LLMVerificationLoop:
    """The LLM + Verifier feedback loop.

    Flow:
      1. LLM extracts IR from PyTorch code
      2. Verifier checks the IR
      3. If errors found, feed back to LLM for refinement
      4. Repeat until success or max iterations
    """

    def __init__(
        self,
        llm=None,  # LLM interface (Anthropic, OpenAI, or MockLLM)
        max_iterations: int = 5,
    ):
        self.llm = llm or MockLLM()
        self.max_iterations = max_iterations
        self.prompt_builder = PromptBuilder()
        self.verifier = DistributedVerifier()

    def verify_code(
        self,
        code: str,
        mesh: Optional[DeviceMesh] = None,
        tensor_states: Optional[Dict[str, TensorState]] = None,
    ) -> LLMVerifyResult:
        """Run the full LLM + Verification loop on PyTorch code.

        Args:
            code: PyTorch distributed code as a string
            mesh: Device mesh (required for multi-device execution)
            tensor_states: Initial tensor states with sharding info

        Returns:
            LLMVerifyResult with the final verified program
        """
        if mesh is None:
            mesh = DeviceMesh(shape=(2,), dim_names=("tp",))

        call_history = []
        errors_list = []

        # Iteration 1: Initial extraction
        prompt = self.prompt_builder.build_extraction_prompt(code)
        response = self.llm.generate(prompt)
        call_history.append(("extraction", response))

        for iteration in range(self.max_iterations):
            # Parse LLM response
            try:
                ir_response = LLMIRResponse.from_json(response)
                fwd_program = ir_response.to_program("llm_fwd")
            except Exception as e:
                errors_list.append(f"Parse error: {e}")
                if iteration < self.max_iterations - 1:
                    feedback = self.prompt_builder.build_feedback_prompt(
                        code, response, errors_list
                    )
                    response = self.llm.generate(feedback)
                    call_history.append(("feedback", response))
                    continue
                else:
                    return LLMVerifyResult(
                        success=False,
                        iterations=iteration + 1,
                        llm_call_history=call_history,
                        errors=errors_list,
                    )

            # Execute and verify
            try:
                executor = MultiDeviceExecutor(mesh, strict=True)
                if tensor_states:
                    for name, ts in tensor_states.items():
                        executor.register_tensor(ts)
                else:
                    # Create default tensor states from sharding info
                    self._register_default_tensors(executor, ir_response, mesh)

                state = executor.run_program(fwd_program)

                # Reject empty programs
                if not fwd_program.ops or not state:
                    verif_errors = [
                        "Extracted program is empty or produced no output tensors"
                    ]
                    errors_list = verif_errors
                    if iteration < self.max_iterations - 1:
                        feedback = self.prompt_builder.build_feedback_prompt(
                            code, response, verif_errors,
                        )
                        response = self.llm.generate(feedback)
                        call_history.append(("feedback", response))
                    continue

                # Analyze
                analyzer = PlacementAnalyzer()
                analysis = analyzer.analyze(fwd_program, state)

                if analysis.is_correct:
                    # PlacementAnalyzer checks structure (missing/redundant
                    # collectives) but not preconditions. Run the full
                    # communication legality check to catch illegal ops
                    # (e.g. AllGather on Replicate).
                    from verifier.solver import DistributedVerifier
                    dv = DistributedVerifier()
                    legality = dv.verify_communication_legality(
                        fwd_program, tensor_states=state,
                    )
                    if not legality.passed:
                        verif_errors = [legality.details]
                        errors_list = verif_errors
                        if iteration < self.max_iterations - 1:
                            feedback = self.prompt_builder.build_feedback_prompt(
                                code, response, verif_errors,
                            )
                            response = self.llm.generate(feedback)
                            call_history.append(("feedback", response))
                        continue

                    bwd = ir_response.to_bwd_program() if ir_response.bwd_ops else None
                    return LLMVerifyResult(
                        success=True,
                        iterations=iteration + 1,
                        final_program=fwd_program,
                        final_bwd_program=bwd,
                        llm_call_history=call_history,
                    )

                # Collect errors for feedback
                verif_errors = []
                for _, tensor_name, ctype in analysis.missing_collectives:
                    verif_errors.append(
                        f"Missing {ctype.__name__} for tensor '{tensor_name}'"
                    )
                for idx in analysis.redundant_collectives:
                    verif_errors.append(
                        f"Redundant collective at op index {idx}"
                    )

                if verif_errors:
                    errors_list = verif_errors
                    if iteration < self.max_iterations - 1:
                        feedback = self.prompt_builder.build_feedback_prompt(
                            code,
                            json.dumps(ir_response.fwd_ops, indent=2),
                            verif_errors,
                        )
                        response = self.llm.generate(feedback)
                        call_history.append(("feedback", response))

            except Exception as e:
                errors_list.append(f"Execution error: {e}")
                if iteration < self.max_iterations - 1:
                    feedback = self.prompt_builder.build_feedback_prompt(
                        code, response, errors_list
                    )
                    response = self.llm.generate(feedback)
                    call_history.append(("feedback", response))

        return LLMVerifyResult(
            success=False,
            iterations=self.max_iterations,
            final_program=fwd_program if 'fwd_program' in dir() else None,
            llm_call_history=call_history,
            errors=errors_list,
        )

    def _register_default_tensors(
        self,
        executor,
        ir_response: LLMIRResponse,
        mesh: DeviceMesh,
    ):
        """Register default tensor states from sharding info."""
        for name, shard_str in ir_response.sharding.items():
            # Parse sharding string
            if shard_str.startswith("Shard("):
                dim = int(shard_str.replace("Shard(", "").replace(")", ""))
                placement = Shard(dim=dim)
            else:
                placement = Replicate()

            spec = ShardingSpec(placements=(placement,), mesh=mesh)
            # Use dummy shapes
            ts = TensorState(
                name=name,
                global_shape=(8, 16),
                local_shape=compute_local_shape((8, 16), spec),
                sharding=spec,
                expr=name.lower(),
                requires_grad=True,
            )
            executor.register_tensor(ts)


# ── Convenience ──────────────────────────────────────────────────────────────

def extract_and_verify(
    code: str,
    mesh: Optional[DeviceMesh] = None,
    llm=None,
) -> LLMVerifyResult:
    """Convenience: extract IR from PyTorch code and verify."""
    loop = LLMVerificationLoop(llm=llm)
    return loop.verify_code(code, mesh=mesh)
