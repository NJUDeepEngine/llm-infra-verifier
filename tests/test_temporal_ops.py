"""Temporal verifier tests: multi-input ops, orphaned handles, async pipelines."""

import pytest
from verifier import *
from verifier.temporal import (
    verify_temporal,
    TemporalGraph,
    RaceDetector,
    RaceType,
)
from verifier.ir import (
    AllReduceAsync, Wait, WaitAll,
    FlashAttention, Embedding, CrossEntropyLoss,
    SendAsync, RecvAsync, OverlapRegion,
    COMM_STREAM, DEFAULT_STREAM,
    Handle, Stream,
)


# ── FlashAttention with async inputs ──────────────────────────────────────────


class TestFlashAttentionTemporal:
    """Temporal correctness of FlashAttention with async inputs."""

    def test_all_waits_before_flash_attention_safe(self):
        """3 AllReduceAsync → 3 Waits → FlashAttention: safe."""
        prog = Program("fa_safe", ops=[
            AllReduceAsync(x="q_p", output="q", handle="hq", stream=COMM_STREAM),
            AllReduceAsync(x="k_p", output="k", handle="hk", stream=COMM_STREAM),
            AllReduceAsync(x="v_p", output="v", handle="hv", stream=COMM_STREAM),
            Wait(handle="hq", tensor="q", output="q_safe"),
            Wait(handle="hk", tensor="k", output="k_safe"),
            Wait(handle="hv", tensor="v", output="v_safe"),
            FlashAttention(q="q_safe", k="k_safe", v="v_safe", output="attn"),
        ])
        result = verify_temporal(prog)
        assert result.is_safe
        assert result.num_missing_waits == 0
        assert result.num_orphaned_handles == 0

    def test_no_waits_before_flash_attention_unsafe(self):
        """3 AllReduceAsync → FlashAttention (no Waits): MISSING_WAIT on q/k/v."""
        prog = Program("fa_no_wait", ops=[
            AllReduceAsync(x="q_p", output="q", handle="hq", stream=COMM_STREAM),
            AllReduceAsync(x="k_p", output="k", handle="hk", stream=COMM_STREAM),
            AllReduceAsync(x="v_p", output="v", handle="hv", stream=COMM_STREAM),
            FlashAttention(q="q", k="k", v="v", output="attn"),
        ])
        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_missing_waits >= 3
        assert result.num_orphaned_handles == 3

    def test_partial_waits_before_flash_attention(self):
        """2 Waits but missing Wait for v: partial MISSING_WAIT."""
        prog = Program("fa_partial_wait", ops=[
            AllReduceAsync(x="q_p", output="q", handle="hq", stream=COMM_STREAM),
            AllReduceAsync(x="k_p", output="k", handle="hk", stream=COMM_STREAM),
            AllReduceAsync(x="v_p", output="v", handle="hv", stream=COMM_STREAM),
            Wait(handle="hq", tensor="q", output="q_safe"),
            Wait(handle="hk", tensor="k", output="k_safe"),
            # Missing: Wait(handle="hv", ...)
            FlashAttention(q="q_safe", k="k_safe", v="v", output="attn"),
        ])
        result = verify_temporal(prog)
        assert not result.is_safe
        # Missing wait for v
        missing_wait_reports = [r for r in result.reports
                                if r.race_type == RaceType.MISSING_WAIT]
        assert len(missing_wait_reports) >= 1
        assert any("v" in r.tensor_name for r in missing_wait_reports)


# ── Embedding with async weight ───────────────────────────────────────────────


class TestEmbeddingTemporal:
    """Temporal correctness of Embedding with async weight gather."""

    def test_wait_before_embedding_safe(self):
        """AllReduceAsync(weight) → Wait → Embedding: safe."""
        prog = Program("emb_safe", ops=[
            AllReduceAsync(x="W_p", output="W", handle="hw", stream=COMM_STREAM),
            Wait(handle="hw", tensor="W", output="W_safe"),
            Embedding(indices="ids", weight="W_safe", output="emb"),
        ])
        result = verify_temporal(prog)
        assert result.is_safe

    def test_no_wait_before_embedding_unsafe(self):
        """AllReduceAsync(weight) → Embedding (no Wait): MISSING_WAIT."""
        prog = Program("emb_no_wait", ops=[
            AllReduceAsync(x="W_p", output="W", handle="hw", stream=COMM_STREAM),
            Embedding(indices="ids", weight="W", output="emb"),
        ])
        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_missing_waits >= 1
        missing = [r for r in result.reports if r.race_type == RaceType.MISSING_WAIT]
        assert any("W" in r.tensor_name for r in missing)


# ── CrossEntropyLoss in async pipeline ────────────────────────────────────────


class TestCrossEntropyTemporal:
    """Temporal correctness of CrossEntropyLoss with async inputs."""

    def test_wait_before_cross_entropy_safe(self):
        """Async logits → Wait → CrossEntropyLoss: safe."""
        prog = Program("ce_safe", ops=[
            AllReduceAsync(x="logits_p", output="logits", handle="hl",
                           stream=COMM_STREAM),
            Wait(handle="hl", tensor="logits", output="logits_safe"),
            CrossEntropyLoss(logits="logits_safe", targets="labels", output="loss"),
        ])
        result = verify_temporal(prog)
        assert result.is_safe

    def test_no_wait_before_cross_entropy_unsafe(self):
        """Async logits → CrossEntropyLoss (no Wait): MISSING_WAIT."""
        prog = Program("ce_no_wait", ops=[
            AllReduceAsync(x="logits_p", output="logits", handle="hl",
                           stream=COMM_STREAM),
            CrossEntropyLoss(logits="logits", targets="labels", output="loss"),
        ])
        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_missing_waits >= 1


# ── Orphaned handle detection ─────────────────────────────────────────────────


class TestOrphanedHandles:
    """Detect async ops whose handles are never waited on."""

    def test_orphaned_handle_detected(self):
        """AllReduceAsync with no Wait → ORPHANED_HANDLE."""
        prog = Program("orphan", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduceAsync(x="y_p", output="y", handle="h1", stream=COMM_STREAM),
            MatMul(a="x2", b="w2", output="z"),
        ])
        result = verify_temporal(prog)
        assert not result.is_safe
        assert result.num_orphaned_handles == 1
        orphan_reports = [r for r in result.reports
                          if r.race_type == RaceType.ORPHANED_HANDLE]
        assert len(orphan_reports) == 1
        assert "h1" in orphan_reports[0].description

    def test_waited_handle_not_orphaned(self):
        """AllReduceAsync with Wait → no orphan."""
        prog = Program("no_orphan", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduceAsync(x="y_p", output="y", handle="h1", stream=COMM_STREAM),
            Wait(handle="h1", tensor="y", output="y_safe"),
            MatMul(a="y_safe", b="w2", output="z"),
        ])
        result = verify_temporal(prog)
        assert result.num_orphaned_handles == 0

    def test_waitall_covers_handles(self):
        """WaitAll covers multiple handles — no orphans."""
        prog = Program("waitall_ok", ops=[
            AllReduceAsync(x="a_p", output="a", handle="h1", stream=COMM_STREAM),
            AllReduceAsync(x="b_p", output="b", handle="h2", stream=COMM_STREAM),
            WaitAll(handles=["h1", "h2"], tensors=["a", "b"], outputs=["a_s", "b_s"]),
        ])
        result = verify_temporal(prog)
        assert result.num_orphaned_handles == 0


# ── Multi-op overlap scenarios ────────────────────────────────────────────────


class TestOverlapScenarios:
    """Complex overlap scenarios with multi-input ops."""

    def test_overlap_region_with_compute(self):
        """OverlapRegion correctly tracks reads of sub-ops."""
        prog = Program("overlap_compute", ops=[
            MatMul(a="x", b="w", output="y_p"),
            AllReduceAsync(x="y_p", output="y", handle="h1", stream=COMM_STREAM),
            MatMul(a="x2", b="w2", output="z_independent"),
            Wait(handle="h1", tensor="y", output="y_safe"),
            MatMul(a="y_safe", b="w3", output="final"),
        ])
        result = verify_temporal(prog)
        assert result.is_safe

    def test_send_recv_async_with_layernorm(self):
        """Pipeline: SendAsync/RecvAsync + LayerNorm ordering."""
        prog = Program("pp_async", ops=[
            MatMul(a="x", b="w", output="h"),
            LayerNorm(x="h", output="h_norm", norm_dim=-1),
            SendAsync(x="h_norm", output="h_sent", src=0, dst=1,
                      handle="hs", stage=0, microbatch_id=0, stream=COMM_STREAM),
            Wait(handle="hs", tensor="h_sent", output="h_done"),
        ])
        result = verify_temporal(prog)
        assert result.is_safe
        assert result.num_orphaned_handles == 0
