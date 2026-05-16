"""Blind verification of real-world distributed training bugs.

For each bug scenario, we:
1. Build the IR program (translating the buggy code pattern — no prior
   knowledge of the bug)
2. Run through MultiDeviceExecutor (data-flow validation)
3. Run full spatial verification via verify_all (postcondition,
   communication legality, placement consistency)
4. Run full temporal verification via verify_temporal (races, missing
   waits, buffer aliasing, concurrent collectives)
5. Check whether any general-purpose check caught the issue

We do NOT use bug-specific assertions. The verifier either catches the
bug through its general checks, or it doesn't — and we document the gap.

Sources:
- Megatron-LM #107: AllReduce ordering deadlock with TP+DP
- DeepSpeed #5545: overlap_comm data race (buffer reuse before reduction)
- DeepSpeed #5248: Sequence parallel gradient scaling
- PyTorch FSDP #64803: ReduceScatter/AllGather stream overlap deadlock
- DeepSpeed #5334: ZeRO-3 gradient partitioning on None grad
- Megatron TP: Row-parallel linear missing AllReduce
"""

import warnings as _warnings

import pytest

from verifier.state import (
    TensorState,
    ShardingSpec,
    DeviceMesh,
    Shard,
    Replicate,
    Partial,
    compute_local_shape,
)
from verifier.ir import (
    MatMul, Add, AllReduce, AllGather, ReduceScatter,
    AllReduceAsync, Wait, OverlapRegion,
    LossScale,
    Program,
)
from verifier.ir import ZeROGatherParam, ZeROScatterGrad
from verifier.executor import MultiDeviceExecutor
from verifier.solver import DistributedVerifier
from verifier.temporal import verify_temporal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mesh(n, name="tp"):
    return DeviceMesh(shape=(n,), dim_names=(name,))


def _mesh2d(d0, d1, names=("tp", "dp")):
    return DeviceMesh(shape=(d0, d1), dim_names=names)


def _make(name, shape, spec, **kw):
    local = compute_local_shape(shape, spec)
    return TensorState(name=name, global_shape=shape, local_shape=local,
                       sharding=spec, **kw)


def _rep(mesh):
    return ShardingSpec(placements=tuple(Replicate() for _ in mesh.shape),
                        mesh=mesh)


def _shard(mesh, dim=0, mesh_dim=0):
    p = [Replicate() for _ in mesh.shape]
    p[mesh_dim] = Shard(dim=dim)
    return ShardingSpec(placements=tuple(p), mesh=mesh)


def _partial(mesh, mesh_dim=0):
    p = [Replicate() for _ in mesh.shape]
    p[mesh_dim] = Partial()
    return ShardingSpec(placements=tuple(p), mesh=mesh)


# ── Unified blind verification pipeline ─────────────────────────────────────


class BlindResult:
    """Aggregated result from the full general-purpose verification pipeline."""

    def __init__(self):
        self.executor_warnings = []
        self.executor_errors = []
        self.spatial_results = []
        self.temporal_result = None

    @property
    def has_issues(self):
        return (
            len(self.executor_warnings) > 0
            or len(self.executor_errors) > 0
            or any(not r.passed for r in self.spatial_results)
            or (self.temporal_result is not None
                and not self.temporal_result.is_safe)
        )

    @property
    def spatial_passed(self):
        return all(r.passed for r in self.spatial_results)

    @property
    def temporal_safe(self):
        return (self.temporal_result is None
                or self.temporal_result.is_safe)


def blind_verify(prog, exe, output_names=None):
    """Run the full verification pipeline, agnostic to any specific bug.

    1. Execute program, collecting warnings/errors
    2. Spatial verification (verify_all)
    3. Temporal verification (verify_temporal)
    """
    result = BlindResult()

    # Phase 1: Execute, capturing warnings and errors
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        try:
            exe.run_program(prog)
        except (ValueError, TypeError) as e:
            result.executor_errors.append(str(e))
        result.executor_warnings = [
            x for x in w if issubclass(x.category, UserWarning)
        ]

    # Phase 2: Spatial verification
    verifier = DistributedVerifier()
    final = dict(exe.devices[0].tensors)
    result.spatial_results = verifier.verify_all(
        prog, final,
        multi_device_states=exe.final_state(),
        output_names=output_names,
    )

    # Phase 3: Temporal verification
    result.temporal_result = verify_temporal(prog)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Bug 1: Megatron-LM #107 — AllReduce ordering deadlock with TP+DP
#
# Two AllReduces on different process groups (TP vs DP) launched
# concurrently on different CUDA streams. We model this with
# OverlapRegion and let the verifier decide if it's safe.
# ══════════════════════════════════════════════════════════════════════════════


class TestMegatronAllReduceOrdering:
    """Megatron-LM #107: TP and DP AllReduce ordering conflict."""

    def test_buggy_concurrent_allreduce(self):
        """Two AllReduces in OverlapRegion — the pipeline should flag this."""
        mesh = _mesh2d(2, 2)
        tp_partial = ShardingSpec(placements=(Partial(), Replicate()),
                                  mesh=mesh)
        dp_partial = ShardingSpec(placements=(Replicate(), Partial()),
                                  mesh=mesh)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(_make("y_tp", (8, 16), tp_partial, expr="y_tp"))
        exe.register_tensor(
            _make("grad_dp", (8, 16), dp_partial, expr="grad_dp"))

        prog = Program(name="concurrent_ar")
        prog.add(OverlapRegion(
            compute_ops=[AllReduce(x="y_tp", output="y_tp_reduced")],
            comm_ops=[AllReduce(x="grad_dp", output="grad_dp_reduced")],
        ))

        result = blind_verify(prog, exe)

        # CAUGHT: temporal concurrent-collectives check flags two
        # collectives in the same OverlapRegion.
        assert result.has_issues
        assert not result.temporal_safe
        assert result.temporal_result.num_concurrent_collectives >= 1

    def test_correct_sequential_allreduce(self):
        """Sequential AllReduces — pipeline should pass."""
        mesh = _mesh2d(2, 2)
        tp_partial = ShardingSpec(placements=(Partial(), Replicate()),
                                  mesh=mesh)
        dp_partial = ShardingSpec(placements=(Replicate(), Partial()),
                                  mesh=mesh)

        exe = MultiDeviceExecutor(mesh=mesh)
        exe.register_tensor(_make("y_tp", (8, 16), tp_partial, expr="y_tp"))
        exe.register_tensor(
            _make("grad_dp", (8, 16), dp_partial, expr="grad_dp"))

        prog = Program(name="sequential_ar")
        prog.add(AllReduce(x="y_tp", output="y_tp_reduced"))
        prog.add(AllReduce(x="grad_dp", output="grad_dp_reduced"))

        result = blind_verify(prog, exe)

        assert result.temporal_result.num_concurrent_collectives == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bug 2: DeepSpeed #5545 — overlap_comm data race
#
# AllReduceAsync on buf0, then subsequent op reads the async output
# without Wait. We model the full data-flow and let detect_missing_waits
# find it.
# ══════════════════════════════════════════════════════════════════════════════


class TestDeepSpeedOverlapCommRace:
    """DeepSpeed #5545: data race in overlap_comm double-buffer scheme."""

    def test_buggy_read_without_wait(self):
        """Async AllReduce output consumed without Wait — pipeline flags it."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        buf0 = _make("buf0", (1024,), _partial(mesh), expr="grad_batch1")
        new_grad = _make("new_grad", (1024,), _rep(mesh), expr="new_grad")
        exe.register_tensor(buf0)
        exe.register_tensor(new_grad)

        prog = Program(name="race_overlap")
        prog.add(AllReduceAsync(x="buf0", output="buf0_reducing",
                                handle="h0"))
        # Bug: use the async result for accumulation without Wait
        prog.add(Add(a="buf0_reducing", b="new_grad",
                      output="buf0_next"))

        result = blind_verify(prog, exe)

        # CAUGHT: temporal detect_missing_waits finds that
        # "buf0_reducing" (async output) is read by Add before Wait.
        assert result.has_issues
        assert not result.temporal_safe
        assert result.temporal_result.num_missing_waits >= 1

    def test_correct_with_wait(self):
        """With Wait before reading — pipeline should pass."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        buf0 = _make("buf0", (1024,), _partial(mesh), expr="grad_batch1")
        new_grad = _make("new_grad", (1024,), _rep(mesh), expr="new_grad")
        exe.register_tensor(buf0)
        exe.register_tensor(new_grad)

        prog = Program(name="safe_overlap")
        prog.add(AllReduceAsync(x="buf0", output="buf0_reducing",
                                handle="h0"))
        prog.add(Wait(handle="h0", tensor="buf0_reducing",
                       output="buf0_done"))
        prog.add(Add(a="buf0_done", b="new_grad",
                      output="buf0_next"))

        result = blind_verify(prog, exe)

        assert result.temporal_result.num_missing_waits == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bug 3: DeepSpeed #5248 — Sequence Parallel gradient scaling
#
# Gradient is partial (sum over SP ranks). AllReduce without dividing
# by SP world size first → gradient over-scaled by SP_size.
#
# LIMITATION: Our general pipeline does NOT catch this. The placement
# is correct (Partial → Replicate via AllReduce), the shapes are valid.
# The bug is mathematical (missing division), which is beyond our
# type-system-based spatial checks.
# ══════════════════════════════════════════════════════════════════════════════


class TestDeepSpeedGradientScaling:
    """DeepSpeed #5248: missing gradient scaling with sequence parallelism.

    KNOWN GAP: Our verifier's general checks cannot detect this bug.
    The AllReduce is applied to a Partial tensor (valid precondition),
    produces Replicate output (valid postcondition), with correct shapes.
    The over-scaling is a semantic/mathematical error invisible to
    placement-level analysis.

    To detect this, we would need either:
    - A "gradient normalization" check (application-specific)
    - Tracking reduction semantics (SUM vs MEAN) in placement types
    """

    def test_buggy_no_scaling_not_caught(self):
        """Bug: AllReduce without prior scaling — pipeline does NOT catch it."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        grad = _make("grad", (4096, 4096), _partial(mesh), expr="sum_grad")
        exe.register_tensor(grad)

        prog = Program(name="buggy_no_scale")
        prog.add(AllReduce(x="grad", output="grad_synced"))

        result = blind_verify(prog, exe)

        # NOT CAUGHT: all general checks pass.
        # The AllReduce input is Partial (valid), output is Replicate (valid).
        assert result.spatial_passed
        assert result.temporal_safe

    def test_correct_with_scaling(self):
        """Correct: divide by SP_size before AllReduce — also passes."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        grad = _make("grad", (4096, 4096), _partial(mesh), expr="sum_grad")
        exe.register_tensor(grad)

        prog = Program(name="correct_scaled")
        prog.add(LossScale(x="grad", output="grad_scaled",
                            scale=4.0, direction="unscale"))
        prog.add(AllReduce(x="grad_scaled", output="grad_synced"))

        result = blind_verify(prog, exe)

        assert result.spatial_passed
        assert result.temporal_safe


# ══════════════════════════════════════════════════════════════════════════════
# Bug 4: PyTorch FSDP #64803 — ReduceScatter/AllGather overlap deadlock
#
# ReduceScatter and AllGather on the same stream (sequential) instead
# of overlapping on separate streams. Functionally correct but kills
# throughput and can cause cascading delays in multi-layer FSDP.
#
# LIMITATION: Our verifier does not check for missed parallelism.
# Sequential execution is functionally correct — the bug is about
# performance, not correctness.
# ══════════════════════════════════════════════════════════════════════════════


class TestFSDPOverlapDeadlock:
    """PyTorch FSDP #64803: ReduceScatter/AllGather overlap failure.

    KNOWN GAP: Our verifier treats sequential execution as correct
    (it is — functionally). Detecting missed overlap opportunities
    requires performance modeling, which is beyond correctness checks.
    """

    def test_buggy_sequential_not_caught(self):
        """Bug: sequential on same stream — pipeline does NOT catch it."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        grad = _make("grad_cur", (4096, 4096), _rep(mesh), expr="grad")
        param = _make("param_next", (4096, 4096), _shard(mesh, dim=0),
                       expr="param")
        exe.register_tensor(grad)
        exe.register_tensor(param)

        prog = Program(name="fsdp_serial")
        prog.add(ReduceScatter(x="grad_cur", output="grad_shard",
                                scatter_dim=0))
        prog.add(AllGather(x="param_next", output="param_full",
                            gather_dim=0))

        # grad_shard is intentionally sharded (consumed by optimizer),
        # param_full is the real output that should be Replicate
        result = blind_verify(prog, exe, output_names=["param_full"])

        # NOT CAUGHT: both ops execute correctly in sequence.
        assert result.spatial_passed
        assert result.temporal_safe

    def test_correct_overlap(self):
        """Correct: OverlapRegion for separate streams — also passes
        (but gets a concurrent-collectives warning since both are collectives)."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        grad = _make("grad_cur", (4096, 4096), _rep(mesh), expr="grad")
        param = _make("param_next", (4096, 4096), _shard(mesh, dim=0),
                       expr="param")
        exe.register_tensor(grad)
        exe.register_tensor(param)

        prog = Program(name="fsdp_overlap")
        prog.add(OverlapRegion(
            compute_ops=[ReduceScatter(x="grad_cur", output="grad_shard",
                                        scatter_dim=0)],
            comm_ops=[AllGather(x="param_next", output="param_full",
                                gather_dim=0)],
        ))

        result = blind_verify(prog, exe)

        # OverlapRegion with two collectives triggers the concurrent-
        # collectives warning. In FSDP this is intentional and safe,
        # but the verifier conservatively flags it — the user must
        # confirm the communicators are independent.
        assert result.temporal_result.num_concurrent_collectives >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Bug 5: DeepSpeed #5334 — ZeRO-3 gradient on missing param
#
# ZeRO-3 calls ZeROScatterGrad on a frozen param that has no gradient.
# We model this as scatter on a tensor that was never registered.
# ══════════════════════════════════════════════════════════════════════════════


class TestZeRO3GradPartitioning:
    """DeepSpeed #5334: ZeRO-3 gradient partitioning on missing grad."""

    def test_buggy_scatter_missing_tensor(self):
        """Scatter on non-existent tensor — pipeline flags it."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        # Don't register any tensor — simulates frozen param with no grad
        prog = Program(name="zero3_missing_grad")
        prog.add(ZeROScatterGrad(x="grad_frozen",
                                  output="grad_scattered",
                                  scatter_dim=0))

        result = blind_verify(prog, exe)

        # CAUGHT: executor's _exec_per_device warns about missing input.
        assert result.has_issues
        assert len(result.executor_warnings) > 0
        assert any("not found on device" in str(w.message)
                    for w in result.executor_warnings)

    def test_correct_gather_compute_scatter(self):
        """Correct ZeRO-3 flow — pipeline should pass."""
        mesh = _mesh(4, name="dp")
        exe = MultiDeviceExecutor(mesh=mesh)

        w = _make("w", (4096, 4096), _shard(mesh, dim=0), expr="w_shard")
        x = _make("x", (8, 4096), _rep(mesh), expr="x")
        exe.register_tensor(w)
        exe.register_tensor(x)

        prog = Program(name="zero3_correct")
        prog.add(ZeROGatherParam(x="w", output="w_full", gather_dim=0))
        prog.add(MatMul(a="x", b="w_full", output="y"))

        result = blind_verify(prog, exe)

        assert len(result.executor_warnings) == 0
        assert len(result.executor_errors) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Bug 6: Megatron TP — Missing AllReduce (verifier's core use case)
#
# Row-parallel linear: x[S(1)] @ w[S(0)] → y[Partial].
# Without AllReduce, output is Partial — wrong for downstream consumers.
# ══════════════════════════════════════════════════════════════════════════════


class TestMegatronMissingAllReduce:
    """Classic Megatron bug: row-parallel linear missing AllReduce."""

    def test_buggy_missing_allreduce(self):
        """MatMul without AllReduce — pipeline should flag Partial output."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make("x", (8, 16), _shard(mesh, dim=1), expr="x")
        w = _make("w", (16, 32), _shard(mesh, dim=0), expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="row_parallel_bug")
        prog.add(MatMul(a="x", b="w", output="y"))

        result = blind_verify(prog, exe)

        # CAUGHT: verify_postcondition auto-detects "y" as output,
        # checks expected_partial=False, finds y.partial=True → FAIL.
        assert result.has_issues
        assert not result.spatial_passed

    def test_correct_with_allreduce(self):
        """With AllReduce — pipeline should pass."""
        mesh = _mesh(2)
        exe = MultiDeviceExecutor(mesh=mesh)

        x = _make("x", (8, 16), _shard(mesh, dim=1), expr="x")
        w = _make("w", (16, 32), _shard(mesh, dim=0), expr="w")
        exe.register_tensor(x)
        exe.register_tensor(w)

        prog = Program(name="row_parallel_correct")
        prog.add(MatMul(a="x", b="w", output="y_partial"))
        prog.add(AllReduce(x="y_partial", output="y"))

        result = blind_verify(prog, exe)

        assert result.spatial_passed


# ══════════════════════════════════════════════════════════════════════════════
# Summary: Detection capability matrix
#
# | Bug                          | Caught? | By which general check         |
# |------------------------------|---------|--------------------------------|
# | Megatron AllReduce deadlock   |    Y    | temporal: concurrent_collectives |
# | DeepSpeed overlap_comm race  |    Y    | temporal: missing_waits        |
# | DeepSpeed gradient scaling   |    N    | (mathematical, beyond types)   |
# | FSDP overlap deadlock        |    N    | (performance, not correctness) |
# | ZeRO-3 None grad             |    Y    | executor: missing input warn   |
# | Megatron missing AllReduce   |    Y    | spatial: postcondition         |
# ══════════════════════════════════════════════════════════════════════════════
