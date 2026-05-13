"""Large-scale distributed training verification tests.

Simulates production-scale LLaMA-70B configurations:
  1. TP=8 single Transformer layer with real hidden dimensions (8192, 28672)
  2. SP (Sequence Parallelism) with AllGather/ReduceScatter pattern
  3. TP=8 × PP=4 two-dimensional mesh with cross-stage Send/Recv
  4. TP=8 × PP=4 × DP=4 three-dimensional mesh (128 GPUs)
  5. Shape arithmetic verification for all weight/activation configs
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from verifier.state import (
    LocalSPMDType,
    Shard,
    Replicate,
    Partial,
    DeviceMesh,
    ShardingSpec,
    compute_local_shape,
    compute_tensor_slices,
    TensorState,
)
from verifier.ir import (
    MatMul,
    Multiply,
    SiLU,
    FlashAttention,
    AllReduce,
    AllGather,
    ReduceScatter,
    Send,
    Recv,
    Program,
)
from verifier.executor import MultiDeviceExecutor


# ═══════════════════════════════════════════════════════════════════════════════
# LLaMA-70B model configuration
# ═══════════════════════════════════════════════════════════════════════════════

LLAMA_70B = dict(
    hidden=8192,
    intermediate=28672,
    heads=64,
    head_dim=128,
    seq_len=4096,
    vocab_size=32000,
)

TP_SIZE = 8
PP_SIZE = 4
DP_SIZE = 4


def _mesh_1d(tp=TP_SIZE):
    return DeviceMesh(shape=(tp,), dim_names=("tp",))


def _mesh_2d(tp=TP_SIZE, pp=PP_SIZE):
    return DeviceMesh(shape=(tp, pp), dim_names=("tp", "pp"))


def _mesh_3d(tp=TP_SIZE, pp=PP_SIZE, dp=DP_SIZE):
    return DeviceMesh(shape=(tp, pp, dp), dim_names=("tp", "pp", "dp"))


def _spec(placements, mesh):
    return ShardingSpec(placements=tuple(placements), mesh=mesh)


def _tensor(name, global_shape, placements, mesh, expr=None):
    spec = _spec(placements, mesh)
    ls = compute_local_shape(global_shape, spec)
    return TensorState(
        name=name, global_shape=global_shape, local_shape=ls,
        sharding=spec, expr=expr or name,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TP=8 single Transformer layer (Attention + SwiGLU MLP)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLaMA70B_TP8_Layer:

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.tp = TP_SIZE
        self.mesh = _mesh_1d(self.tp)

    def _col_weight(self, name, in_dim, out_dim):
        return _tensor(name, (in_dim, out_dim), [Shard(dim=1)], self.mesh)

    def _row_weight(self, name, in_dim, out_dim):
        return _tensor(name, (in_dim, out_dim), [Shard(dim=0)], self.mesh)

    def test_attention_block(self):
        """QKV column-parallel -> FlashAttention -> O row-parallel -> AllReduce."""
        H, S = self.H, self.S

        x = _tensor("x", (S, H), [Replicate()], self.mesh)
        w_q = self._col_weight("w_q", H, H)
        w_k = self._col_weight("w_k", H, H)
        w_v = self._col_weight("w_v", H, H)
        w_o = self._row_weight("w_o", H, H)

        prog = Program(ops=[
            MatMul(a="x", b="w_q", output="q"),
            MatMul(a="x", b="w_k", output="k"),
            MatMul(a="x", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            AllReduce(x="attn_p", output="y_attn"),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in [x, w_q, w_k, w_v, w_o]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        q = result["q"]
        assert isinstance(q.sharding.placements[0], Shard)
        assert q.local_shape[1] == H // self.tp

        assert result["attn_p"].partial

        y = result["y_attn"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.local_shape == (S, H)

    def test_swiglu_mlp_block(self):
        """Gate+Up column-parallel -> SiLU -> Multiply -> Down row-parallel -> AllReduce."""
        H, I, S = self.H, self.I, self.S

        x = _tensor("x", (S, H), [Replicate()], self.mesh)
        w_gate = self._col_weight("w_gate", H, I)
        w_up = self._col_weight("w_up", H, I)
        w_down = self._row_weight("w_down", I, H)

        prog = Program(ops=[
            MatMul(a="x", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="x", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y_mlp"),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in [x, w_gate, w_up, w_down]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        assert result["gate"].local_shape == (S, I // self.tp)
        assert result["up"].local_shape == (S, I // self.tp)

        h = result["h"]
        assert isinstance(h.sharding.placements[0], Shard)
        assert h.local_shape == (S, I // self.tp)

        assert result["mlp_p"].partial

        y = result["y_mlp"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.local_shape == (S, H)

    def test_full_transformer_layer(self):
        """Full attention + SwiGLU MLP with 70B dimensions, SPMD cross-validated."""
        H, I, S = self.H, self.I, self.S

        tensors = [
            _tensor("x", (S, H), [Replicate()], self.mesh),
            self._col_weight("w_q", H, H),
            self._col_weight("w_k", H, H),
            self._col_weight("w_v", H, H),
            self._row_weight("w_o", H, H),
            self._col_weight("w_gate", H, I),
            self._col_weight("w_up", H, I),
            self._row_weight("w_down", I, H),
        ]

        prog = Program(ops=[
            MatMul(a="x", b="w_q", output="q"),
            MatMul(a="x", b="w_k", output="k"),
            MatMul(a="x", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            AllReduce(x="attn_p", output="y_attn"),
            MatMul(a="y_attn", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="y_attn", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in tensors:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        y = result["y"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.global_shape == (S, H)
        assert y.local_shape == (S, H)

        final = executor.final_state()
        assert len(final) == self.tp
        for did in range(self.tp):
            assert "y" in final[did]
            assert final[did]["y"].local_shape == (S, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Sequence Parallelism (SP) with TP=8
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLaMA70B_SP:

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.tp = TP_SIZE
        self.mesh = _mesh_1d(self.tp)

    def test_sp_attention_block(self):
        """SP: S(0) -> AllGather -> Attention -> ReduceScatter -> S(0)."""
        H, S, tp = self.H, self.S, self.tp

        x_sp = _tensor("x_sp", (S, H), [Shard(dim=0)], self.mesh)
        w_q = _tensor("w_q", (H, H), [Shard(dim=1)], self.mesh)
        w_k = _tensor("w_k", (H, H), [Shard(dim=1)], self.mesh)
        w_v = _tensor("w_v", (H, H), [Shard(dim=1)], self.mesh)
        w_o = _tensor("w_o", (H, H), [Shard(dim=0)], self.mesh)

        prog = Program(ops=[
            AllGather(x="x_sp", output="x_full", gather_dim=0),
            MatMul(a="x_full", b="w_q", output="q"),
            MatMul(a="x_full", b="w_k", output="k"),
            MatMul(a="x_full", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            ReduceScatter(x="attn_p", output="y_sp", scatter_dim=0),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in [x_sp, w_q, w_k, w_v, w_o]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        assert x_sp.local_shape == (S // tp, H)

        x_full = result["x_full"]
        assert x_full.local_shape == (S, H)
        assert isinstance(x_full.sharding.placements[0], Replicate)

        assert result["attn_p"].partial

        y = result["y_sp"]
        assert isinstance(y.sharding.placements[0], Shard)
        assert y.sharding.placements[0].dim == 0
        assert y.local_shape == (S // tp, H)

    def test_sp_mlp_block(self):
        """SP MLP: S(0) -> AllGather -> SwiGLU -> ReduceScatter -> S(0)."""
        H, I, S, tp = self.H, self.I, self.S, self.tp

        x_sp = _tensor("x_sp", (S, H), [Shard(dim=0)], self.mesh)
        w_gate = _tensor("w_gate", (H, I), [Shard(dim=1)], self.mesh)
        w_up = _tensor("w_up", (H, I), [Shard(dim=1)], self.mesh)
        w_down = _tensor("w_down", (I, H), [Shard(dim=0)], self.mesh)

        prog = Program(ops=[
            AllGather(x="x_sp", output="x_full", gather_dim=0),
            MatMul(a="x_full", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="x_full", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            ReduceScatter(x="mlp_p", output="y_sp", scatter_dim=0),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in [x_sp, w_gate, w_up, w_down]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        assert x_sp.local_shape[0] == S // tp
        assert result["gate"].local_shape == (S, I // tp)

        y = result["y_sp"]
        assert isinstance(y.sharding.placements[0], Shard)
        assert y.local_shape == (S // tp, H)

    def test_sp_full_layer(self):
        """Full SP Transformer layer: attn + MLP, start and end in S(0)."""
        H, I, S, tp = self.H, self.I, self.S, self.tp

        tensors = [
            _tensor("x_sp", (S, H), [Shard(dim=0)], self.mesh),
            _tensor("w_q", (H, H), [Shard(dim=1)], self.mesh),
            _tensor("w_k", (H, H), [Shard(dim=1)], self.mesh),
            _tensor("w_v", (H, H), [Shard(dim=1)], self.mesh),
            _tensor("w_o", (H, H), [Shard(dim=0)], self.mesh),
            _tensor("w_gate", (H, I), [Shard(dim=1)], self.mesh),
            _tensor("w_up", (H, I), [Shard(dim=1)], self.mesh),
            _tensor("w_down", (I, H), [Shard(dim=0)], self.mesh),
        ]

        prog = Program(ops=[
            # Attention with SP
            AllGather(x="x_sp", output="x_attn", gather_dim=0),
            MatMul(a="x_attn", b="w_q", output="q"),
            MatMul(a="x_attn", b="w_k", output="k"),
            MatMul(a="x_attn", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            ReduceScatter(x="attn_p", output="y_attn_sp", scatter_dim=0),
            # MLP with SP
            AllGather(x="y_attn_sp", output="x_mlp", gather_dim=0),
            MatMul(a="x_mlp", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="x_mlp", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            ReduceScatter(x="mlp_p", output="y_sp", scatter_dim=0),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        for t in tensors:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        y = result["y_sp"]
        assert isinstance(y.sharding.placements[0], Shard)
        assert y.sharding.placements[0].dim == 0
        assert y.local_shape == (S // tp, H)
        assert y.global_shape == (S, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TP=8 × PP=4 two-dimensional mesh
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLaMA70B_TP_PP:

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.tp = TP_SIZE
        self.pp = PP_SIZE
        self.mesh = _mesh_2d(self.tp, self.pp)

    def _stage_devices(self, stage):
        return self.mesh.devices_in_group(mesh_dim=1, index=stage)

    def _col_weight_2d(self, name, in_dim, out_dim):
        return _tensor(name, (in_dim, out_dim),
                       [Shard(dim=1), Replicate()], self.mesh)

    def _row_weight_2d(self, name, in_dim, out_dim):
        return _tensor(name, (in_dim, out_dim),
                       [Shard(dim=0), Replicate()], self.mesh)

    def test_2d_mesh_topology(self):
        """Verify 2D mesh coordinate mapping and device groups."""
        assert self.mesh.num_devices == self.tp * self.pp

        for stage in range(self.pp):
            group = self._stage_devices(stage)
            assert len(group) == self.tp

        # No overlap between stages
        all_devs = set()
        for stage in range(self.pp):
            group = set(self._stage_devices(stage))
            assert all_devs.isdisjoint(group)
            all_devs |= group
        assert len(all_devs) == self.mesh.num_devices

    def test_2d_placement_shapes(self):
        """Verify 2D placements produce correct local shapes."""
        H, I = self.H, LLAMA_70B["intermediate"]

        w = _tensor("w", (H, H), [Shard(dim=1), Replicate()], self.mesh)
        assert w.local_shape == (H, H // self.tp)

        w_r = _tensor("w_r", (H, H), [Shard(dim=0), Replicate()], self.mesh)
        assert w_r.local_shape == (H // self.tp, H)

        w_gate = _tensor("wg", (H, I), [Shard(dim=1), Replicate()], self.mesh)
        assert w_gate.local_shape == (H, I // self.tp)

        x = _tensor("x", (self.S, H), [Replicate(), Replicate()], self.mesh)
        assert x.local_shape == (self.S, H)

    def test_tp_pp_two_stages(self):
        """Two PP stages with TP within each, connected by Send/Recv."""
        H, S = self.H, self.S

        stage0_devs = self._stage_devices(0)
        stage1_devs = self._stage_devices(1)

        x = _tensor("x", (S, H), [Replicate(), Replicate()], self.mesh)
        w0 = self._col_weight_2d("w0", H, H)
        w0_d = self._row_weight_2d("w0_d", H, H)
        w1 = self._col_weight_2d("w1", H, H)
        w1_d = self._row_weight_2d("w1_d", H, H)

        ops = [
            MatMul(a="x", b="w0", output="h0"),
            MatMul(a="h0", b="w0_d", output="h0_p"),
            AllReduce(x="h0_p", output="y0"),
        ]

        for tp_rank in range(self.tp):
            src = self.mesh.coord_to_device(tp_rank, 0)
            dst = self.mesh.coord_to_device(tp_rank, 1)
            ops.append(Send(x="y0", output=f"sent_{tp_rank}",
                            src=src, dst=dst, stage=0, microbatch_id=0))
            ops.append(Recv(x=f"sent_{tp_rank}", output="y0_recv",
                            src=src, dst=dst, stage=1, microbatch_id=0))

        ops.extend([
            MatMul(a="y0_recv", b="w1", output="h1"),
            MatMul(a="h1", b="w1_d", output="h1_p"),
            AllReduce(x="h1_p", output="y1"),
        ])

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)

        for t in [x, w0, w0_d]:
            executor.register_tensor(t, device_ids=stage0_devs)
        for t in [w1, w1_d]:
            executor.register_tensor(t, device_ids=stage1_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = executor.run_program(prog)

        final = executor.final_state()
        for did in stage1_devs:
            assert "y1" in final[did]
            y1 = final[did]["y1"]
            assert all(isinstance(p, Replicate) for p in y1.sharding.placements)
            assert y1.local_shape == (S, H)

    def test_tp_pp_four_stages(self):
        """Full 4-stage PP pipeline with TP within each stage."""
        H, S = self.H, self.S

        ops = []
        for stage in range(self.pp):
            stage_devs = self._stage_devices(stage)
            in_name = "x" if stage == 0 else f"y{stage - 1}_recv"

            w = self._col_weight_2d(f"w{stage}", H, H)
            w_d = self._row_weight_2d(f"w{stage}_d", H, H)

            ops.extend([
                MatMul(a=in_name, b=f"w{stage}", output=f"h{stage}"),
                MatMul(a=f"h{stage}", b=f"w{stage}_d", output=f"h{stage}_p"),
                AllReduce(x=f"h{stage}_p", output=f"y{stage}"),
            ])

            if stage < self.pp - 1:
                next_devs = self._stage_devices(stage + 1)
                for tp_rank in range(self.tp):
                    src = self.mesh.coord_to_device(tp_rank, stage)
                    dst = self.mesh.coord_to_device(tp_rank, stage + 1)
                    ops.append(Send(
                        x=f"y{stage}", output=f"sent_s{stage}_{tp_rank}",
                        src=src, dst=dst, stage=stage, microbatch_id=0,
                    ))
                    ops.append(Recv(
                        x=f"sent_s{stage}_{tp_rank}",
                        output=f"y{stage}_recv",
                        src=src, dst=dst, stage=stage + 1, microbatch_id=0,
                    ))

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)

        x = _tensor("x", (S, H), [Replicate(), Replicate()], self.mesh)
        executor.register_tensor(x, device_ids=self._stage_devices(0))

        for stage in range(self.pp):
            stage_devs = self._stage_devices(stage)
            w = self._col_weight_2d(f"w{stage}", H, H)
            w_d = self._row_weight_2d(f"w{stage}_d", H, H)
            executor.register_tensor(w, device_ids=stage_devs)
            executor.register_tensor(w_d, device_ids=stage_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        last_stage_devs = self._stage_devices(self.pp - 1)
        final = executor.final_state()
        for did in last_stage_devs:
            y_final = final[did][f"y{self.pp - 1}"]
            assert all(isinstance(p, Replicate) for p in y_final.sharding.placements)
            assert y_final.local_shape == (S, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TP=8 × PP=4 × DP=4 three-dimensional mesh (128 GPUs)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLaMA70B_128GPU:

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.tp = TP_SIZE
        self.pp = PP_SIZE
        self.dp = DP_SIZE
        self.mesh = _mesh_3d(self.tp, self.pp, self.dp)

    def test_128_device_mesh(self):
        """Verify 128-device 3D mesh creation and coordinate mapping."""
        assert self.mesh.num_devices == 128
        assert self.mesh.ndim == 3
        assert self.mesh.shape == (8, 4, 4)

        all_ids = set()
        for tp in range(self.tp):
            for pp in range(self.pp):
                for dp in range(self.dp):
                    did = self.mesh.coord_to_device(tp, pp, dp)
                    assert 0 <= did < 128
                    all_ids.add(did)
        assert len(all_ids) == 128

        for did in range(128):
            coords = self.mesh.device_to_coord(did)
            assert len(coords) == 3
            assert self.mesh.coord_to_device(*coords) == did

    def test_3d_device_groups(self):
        """Verify device group isolation across all 3 dimensions."""
        # TP groups: 8 devices each, 4*4=16 groups
        for pp in range(self.pp):
            for dp in range(self.dp):
                tp_devs = []
                for tp in range(self.tp):
                    tp_devs.append(self.mesh.coord_to_device(tp, pp, dp))
                assert len(set(tp_devs)) == self.tp

        # PP groups: 4 devices each, 8*4=32 groups
        for tp in range(self.tp):
            for dp in range(self.dp):
                pp_devs = []
                for pp in range(self.pp):
                    pp_devs.append(self.mesh.coord_to_device(tp, pp, dp))
                assert len(set(pp_devs)) == self.pp

    def test_3d_placement_shapes(self):
        """Verify 3D placements produce correct local shapes."""
        H, I, S = self.H, self.I, self.S

        w = _tensor("w", (H, H),
                     [Shard(dim=1), Replicate(), Replicate()], self.mesh)
        assert w.local_shape == (H, H // self.tp)

        w_r = _tensor("w_r", (I, H),
                       [Shard(dim=0), Replicate(), Replicate()], self.mesh)
        assert w_r.local_shape == (I // self.tp, H)

        x = _tensor("x", (S, H),
                     [Replicate(), Replicate(), Replicate()], self.mesh)
        assert x.local_shape == (S, H)

        x_sp = _tensor("x_sp", (S, H),
                        [Shard(dim=0), Replicate(), Replicate()], self.mesh)
        assert x_sp.local_shape == (S // self.tp, H)

    def test_3d_tensor_slices(self):
        """Verify compute_tensor_slices for all 128 devices."""
        H = self.H
        spec = _spec([Shard(dim=1), Replicate(), Replicate()], self.mesh)
        slices = compute_tensor_slices((H, H), spec)

        assert len(slices) == 128

        for pp in range(self.pp):
            for dp in range(self.dp):
                offsets_in_group = set()
                for tp in range(self.tp):
                    did = self.mesh.coord_to_device(tp, pp, dp)
                    s = slices[did]
                    assert s.local_shape == (H, H // self.tp)
                    offsets_in_group.add(s.offsets)
                assert len(offsets_in_group) == self.tp

    def test_128gpu_tp_layer(self):
        """Run TP layer on 128-device executor with spmd_checking."""
        H, S = self.H, self.S

        target_devs = [
            self.mesh.coord_to_device(tp, 0, 0) for tp in range(self.tp)
        ]

        x = _tensor("x", (S, H),
                     [Replicate(), Replicate(), Replicate()], self.mesh)
        w = _tensor("w", (H, H),
                    [Shard(dim=1), Replicate(), Replicate()], self.mesh)
        w_d = _tensor("w_d", (H, H),
                      [Shard(dim=0), Replicate(), Replicate()], self.mesh)

        prog = Program(ops=[
            MatMul(a="x", b="w", output="h"),
            MatMul(a="h", b="w_d", output="h_p"),
            AllReduce(x="h_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)
        executor.register_tensor(x, device_ids=target_devs)
        executor.register_tensor(w, device_ids=target_devs)
        executor.register_tensor(w_d, device_ids=target_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for did in target_devs:
            assert "y" in final[did]
            y = final[did]["y"]
            assert y.local_shape == (S, H)
            assert all(isinstance(p, Replicate) for p in y.sharding.placements)

    def test_128gpu_pp_send_recv(self):
        """PP Send/Recv between stages on 3D mesh."""
        H, S = self.H, self.S

        stage0_devs = [
            self.mesh.coord_to_device(tp, 0, 0) for tp in range(self.tp)
        ]
        stage1_devs = [
            self.mesh.coord_to_device(tp, 1, 0) for tp in range(self.tp)
        ]

        x = _tensor("x", (S, H),
                     [Replicate(), Replicate(), Replicate()], self.mesh)

        ops = []
        for i, (src, dst) in enumerate(zip(stage0_devs, stage1_devs)):
            ops.append(Send(x="x", output=f"sent_{i}",
                            src=src, dst=dst, stage=0, microbatch_id=0))
            ops.append(Recv(x=f"sent_{i}", output="x_recv",
                            src=src, dst=dst, stage=1, microbatch_id=0))

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=self.mesh)
        executor.register_tensor(x, device_ids=stage0_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for did in stage1_devs:
            assert "x_recv" in final[did]
            assert final[did]["x_recv"].global_shape == (S, H)

    def test_128gpu_full_tp_pp_layer(self):
        """Full TP+PP: TP layer on stage 0, Send/Recv, TP layer on stage 1."""
        H, S = self.H, self.S

        s0_devs = [self.mesh.coord_to_device(tp, 0, 0) for tp in range(self.tp)]
        s1_devs = [self.mesh.coord_to_device(tp, 1, 0) for tp in range(self.tp)]

        x = _tensor("x", (S, H),
                     [Replicate(), Replicate(), Replicate()], self.mesh)
        w0 = _tensor("w0", (H, H),
                      [Shard(dim=1), Replicate(), Replicate()], self.mesh)
        w0_d = _tensor("w0_d", (H, H),
                        [Shard(dim=0), Replicate(), Replicate()], self.mesh)
        w1 = _tensor("w1", (H, H),
                      [Shard(dim=1), Replicate(), Replicate()], self.mesh)
        w1_d = _tensor("w1_d", (H, H),
                        [Shard(dim=0), Replicate(), Replicate()], self.mesh)

        ops = [
            MatMul(a="x", b="w0", output="h0"),
            MatMul(a="h0", b="w0_d", output="h0_p"),
            AllReduce(x="h0_p", output="y0"),
        ]
        for i, (src, dst) in enumerate(zip(s0_devs, s1_devs)):
            ops.append(Send(x="y0", output=f"sent_{i}",
                            src=src, dst=dst, stage=0, microbatch_id=0))
            ops.append(Recv(x=f"sent_{i}", output="y0_recv",
                            src=src, dst=dst, stage=1, microbatch_id=0))
        ops.extend([
            MatMul(a="y0_recv", b="w1", output="h1"),
            MatMul(a="h1", b="w1_d", output="h1_p"),
            AllReduce(x="h1_p", output="y1"),
        ])

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=self.mesh, spmd_checking=True)

        for t in [x, w0, w0_d]:
            executor.register_tensor(t, device_ids=s0_devs)
        for t in [w1, w1_d]:
            executor.register_tensor(t, device_ids=s1_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for did in s1_devs:
            y1 = final[did]["y1"]
            assert y1.local_shape == (S, H)
            assert all(isinstance(p, Replicate) for p in y1.sharding.placements)
            assert y1.local_type == LocalSPMDType.REPLICATE


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Shape arithmetic verification
# ═══════════════════════════════════════════════════════════════════════════════

class TestShapeArithmetic:

    def test_tp8_all_weight_shapes(self):
        """Verify all 70B weight local shapes with TP=8."""
        H, I = LLAMA_70B["hidden"], LLAMA_70B["intermediate"]
        mesh = _mesh_1d(8)

        cases = [
            ("w_q", (H, H), Shard(dim=1), (H, H // 8)),
            ("w_k", (H, H), Shard(dim=1), (H, H // 8)),
            ("w_v", (H, H), Shard(dim=1), (H, H // 8)),
            ("w_o", (H, H), Shard(dim=0), (H // 8, H)),
            ("w_gate", (H, I), Shard(dim=1), (H, I // 8)),
            ("w_up", (H, I), Shard(dim=1), (H, I // 8)),
            ("w_down", (I, H), Shard(dim=0), (I // 8, H)),
        ]

        for name, shape, placement, expected in cases:
            spec = _spec([placement], mesh)
            local = compute_local_shape(shape, spec)
            assert local == expected, f"{name}: {local} != {expected}"

    def test_intermediate_divisibility(self):
        """28672 must be divisible by TP=8."""
        I = LLAMA_70B["intermediate"]
        assert I % TP_SIZE == 0
        assert I // TP_SIZE == 3584

    def test_3d_mesh_slice_coverage(self):
        """Verify 128 slices collectively cover the full tensor."""
        H = LLAMA_70B["hidden"]
        mesh = _mesh_3d()
        spec = _spec([Shard(dim=1), Replicate(), Replicate()], mesh)
        slices = compute_tensor_slices((H, H), spec)

        col_ranges = set()
        for s in slices.values():
            col_ranges.add((s.offsets[1], s.offsets[1] + s.local_shape[1]))

        assert len(col_ranges) == TP_SIZE
        sorted_ranges = sorted(col_ranges)
        assert sorted_ranges[0][0] == 0
        assert sorted_ranges[-1][1] == H
        for i in range(len(sorted_ranges) - 1):
            assert sorted_ranges[i][1] == sorted_ranges[i + 1][0]

    def test_sp_shape_reduction(self):
        """SP reduces seq_len by TP factor."""
        S, H = LLAMA_70B["seq_len"], LLAMA_70B["hidden"]
        mesh = _mesh_1d(8)

        sp_spec = _spec([Shard(dim=0)], mesh)
        sp_local = compute_local_shape((S, H), sp_spec)
        assert sp_local == (S // 8, H)
        assert sp_local[0] == 512

    def test_2d_activation_shape_independence(self):
        """PP dimension doesn't affect TP-only sharded shapes."""
        H = LLAMA_70B["hidden"]
        mesh_1d = _mesh_1d(8)
        mesh_2d = _mesh_2d(8, 4)

        local_1d = compute_local_shape((H, H), _spec([Shard(dim=1)], mesh_1d))
        local_2d = compute_local_shape(
            (H, H), _spec([Shard(dim=1), Replicate()], mesh_2d),
        )
        assert local_1d == local_2d

    def test_3d_activation_shape_independence(self):
        """DP and PP dimensions don't affect TP-only sharded shapes."""
        H = LLAMA_70B["hidden"]
        mesh_1d = _mesh_1d(8)
        mesh_3d = _mesh_3d(8, 4, 4)

        local_1d = compute_local_shape((H, H), _spec([Shard(dim=1)], mesh_1d))
        local_3d = compute_local_shape(
            (H, H),
            _spec([Shard(dim=1), Replicate(), Replicate()], mesh_3d),
        )
        assert local_1d == local_3d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
