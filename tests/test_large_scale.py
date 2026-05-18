"""Large-scale distributed training verification tests.

Simulates production-scale LLaMA-70B configurations:
  1. TP=8 single Transformer layer (8192, 28672)
  2. SP (Sequence Parallelism) with AllGather/ReduceScatter
  3. TP=8 × PP=4 two-dimensional mesh with cross-stage Send/Recv
  4. TP=8 × PP=4 × DP=4 three-dimensional mesh (128 GPUs)
  5-10. DP4 × TP8 × PP4 × CP2 four-dimensional mesh (256 GPUs):
     MatMul properties, collective contracts, Megatron invariants,
     math constraints, gradient correctness, system integrity.
  11. Multi-dimensional cooperation: DP+TP+PP+CP active simultaneously.
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
    Add,
    Multiply,
    SiLU,
    GELU,
    ReLU,
    Dropout,
    LayerNorm,
    RMSNorm,
    Softmax,
    FlashAttention,
    Embedding,
    CrossEntropyLoss,
    AllReduce,
    AllGather,
    ReduceScatter,
    Send,
    Recv,
    RingAttention,
    AllReduceAsync,
    Wait,
    WaitAll,
    OverlapRegion,
    TopKGate,
    MoEDispatch,
    MoECombine,
    ExpertCompute,
    Program,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine


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
CP_SIZE = 2


def _mesh_1d(tp=TP_SIZE):
    return DeviceMesh(shape=(tp,), dim_names=("tp",))


def _mesh_2d(tp=TP_SIZE, pp=PP_SIZE):
    return DeviceMesh(shape=(tp, pp), dim_names=("tp", "pp"))


def _mesh_3d(tp=TP_SIZE, pp=PP_SIZE, dp=DP_SIZE):
    return DeviceMesh(shape=(tp, pp, dp), dim_names=("tp", "pp", "dp"))


def _mesh_4d(dp=DP_SIZE, tp=TP_SIZE, pp=PP_SIZE, cp=CP_SIZE):
    return DeviceMesh(shape=(dp, tp, pp, cp), dim_names=("dp", "tp", "pp", "cp"))


DP_DIM, TP_DIM, PP_DIM, CP_DIM = 0, 1, 2, 3


def P(dp=Replicate(), tp=Replicate(), pp=Replicate(), cp=Replicate()):
    """Build a 4-element placement tuple for the 4D mesh (dp, tp, pp, cp)."""
    return (dp, tp, pp, cp)


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
# 5. Distributed MatMul Correctness Properties (256 GPU)
#
# Requirements from distributed linear algebra:
#   - Column partition of B: Y_i = A @ B_i is a column block of Y
#   - Row partition (split-k): Y_i = A_i @ B_i is a partial sum, Y = Σ Y_i
#   - Orthogonal mesh dimensions operate independently on different tensor axes
# ═══════════════════════════════════════════════════════════════════════════════

class TestDistributedMatMulProperties:
    """Verify matrix multiplication correctness on 4D mesh (DP4×TP8×PP4×CP2).

    Requirements from distributed linear algebra, not from reading the code.
    """

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def test_column_partition_produces_column_sharded_output(self):
        """Column-parallel: Y = X @ W with W column-partitioned → Y column-sharded.

        Math: X replicated, W ∈ R^{k×n} split along columns across p devices.
        Y_i = X @ W_i ∈ R^{m×n/p} is a contiguous column block of Y.
        """
        x = _tensor("x", (self.S, self.H), P(), self.mesh)
        w = _tensor("w", (self.H, self.I), P(tp=Shard(1)), self.mesh)

        ctx = {"x": x, "w": w}
        y = MatMul(a="x", b="w", output="y").apply(ctx)

        assert y.sharding.placements[TP_DIM] == Shard(1)
        assert y.local_shape == (self.S, self.I // TP_SIZE)
        assert y.global_shape == (self.S, self.I)
        for dim in [DP_DIM, PP_DIM, CP_DIM]:
            assert isinstance(y.sharding.placements[dim], Replicate)

    def test_split_k_matmul_produces_partial_sum(self):
        """Split-k: contracting dim partitioned → partial sum needing AllReduce.

        Math: Y = A @ B = Σ_i(A_i @ B_i) where A column-partitioned and
        B row-partitioned on same device group. Each Y_i is a partial product.
        """
        a = _tensor("a", (self.S, self.H), P(tp=Shard(1)), self.mesh)
        b = _tensor("b", (self.H, self.H), P(tp=Shard(0)), self.mesh)

        ctx = {"a": a, "b": b}
        y = MatMul(a="a", b="b", output="y").apply(ctx)

        assert isinstance(y.sharding.placements[TP_DIM], Partial)
        assert y.local_shape == (self.S, self.H)

    def test_dp_and_tp_shards_are_orthogonal(self):
        """DP (batch split) and TP (model split) are independent.

        Multi-dim parallelism: DP and TP operate on different tensor axes
        and different device groups. Both shards appear independently.
        """
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), self.mesh)
        w = _tensor("w", (self.H, self.I), P(tp=Shard(1)), self.mesh)

        ctx = {"x": x, "w": w}
        y = MatMul(a="x", b="w", output="y").apply(ctx)

        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert y.sharding.placements[TP_DIM] == Shard(1)
        assert y.local_shape == (self.S // DP_SIZE, self.I // TP_SIZE)

    def test_batch_shard_propagates_through_matmul(self):
        """Batch shard on non-contracting dim propagates through MatMul.

        Block row multiplication: Y = X @ W where X row-partitioned, W replicated.
        Y_i = X_i @ W is a row block of Y.
        """
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), self.mesh)
        w = _tensor("w", (self.H, self.H), P(), self.mesh)

        ctx = {"x": x, "w": w}
        y = MatMul(a="x", b="w", output="y").apply(ctx)

        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert y.local_shape == (self.S // DP_SIZE, self.H)

    def test_replicated_matmul_stays_replicated(self):
        """MatMul of two replicated tensors produces replicated result.

        Trivial case: every device computes identical Y = X @ W.
        """
        x = _tensor("x", (self.S, self.H), P(), self.mesh)
        w = _tensor("w", (self.H, self.I), P(), self.mesh)

        ctx = {"x": x, "w": w}
        y = MatMul(a="x", b="w", output="y").apply(ctx)

        for dim in range(4):
            assert isinstance(y.sharding.placements[dim], Replicate)
        assert y.local_shape == (self.S, self.I)

    def test_column_row_allreduce_restores_replicate(self):
        """Col-parallel → row-parallel → AllReduce must restore full replication.

        Megatron TP invariant: the canonical pattern (1) col-parallel → col-sharded,
        (2) row-parallel on col-sharded → partial sum, (3) AllReduce → replicated.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(), mesh)
        w_col = _tensor("w_col", (H, H), P(tp=Shard(1)), mesh)
        w_row = _tensor("w_row", (H, H), P(tp=Shard(0)), mesh)

        ctx = {"x": x, "w_col": w_col, "w_row": w_row}
        h = MatMul(a="x", b="w_col", output="h").apply(ctx)
        h_p = MatMul(a="h", b="w_row", output="h_p").apply(ctx)
        y = AllReduce(x="h_p", output="y").apply(ctx)

        for dim in range(4):
            assert isinstance(y.sharding.placements[dim], Replicate)
        assert y.local_shape == (S, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Collective Communication Contracts
#
# Requirements from NCCL/MPI collective semantics:
#   - AllReduce: sum partial replicas → complete tensor; must not touch Shard
#   - AllGather: concatenate sharded pieces along gather_dim; others untouched
#   - Element-wise: compatible placements merge; incompatible → error
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollectiveContracts256GPU:
    """Verify collective communication contracts on 4D mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def test_allreduce_resolves_all_partials_preserves_shard(self):
        """AllReduce: sum partial replicas, leave Shard placements unchanged.

        AllReduce semantics: resolves ALL Partial→Replicate. Any Shard
        placement (physically distributed data) must NOT be modified.
        """
        mesh = self.mesh
        spec = ShardingSpec(
            placements=(Shard(0), Partial(), Replicate(), Partial()),
            mesh=mesh,
        )
        x = TensorState(
            name="x", global_shape=(self.S, self.H),
            local_shape=compute_local_shape((self.S, self.H), spec),
            sharding=spec, expr="x",
        )
        ctx = {"x": x}
        y = AllReduce(x="x", output="y").apply(ctx)

        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert isinstance(y.sharding.placements[CP_DIM], Replicate)
        assert y.local_shape == (self.S // DP_SIZE, self.H)

    def test_allgather_resolves_only_matching_shard(self):
        """AllGather(dim=0): concatenate Shard(0), leave Shard(1) untouched.

        AllGather concatenates along a specific tensor dim. Only Shard
        placements matching that dim are resolved. Others must remain.
        """
        mesh = self.mesh
        x = _tensor("x", (self.S, self.H),
                     P(dp=Shard(0), tp=Shard(1), cp=Shard(0)), mesh)

        ctx = {"x": x}
        y = AllGather(x="x", output="y", gather_dim=0).apply(ctx)

        assert isinstance(y.sharding.placements[DP_DIM], Replicate)
        assert isinstance(y.sharding.placements[CP_DIM], Replicate)
        assert y.sharding.placements[TP_DIM] == Shard(1)
        assert y.local_shape == (self.S, self.H // TP_SIZE)

    def test_allreduce_rejects_non_partial_input(self):
        """AllReduce on non-Partial input is a semantic error.

        AllReduce only makes sense on partial sums. Calling it on a
        replicated or sharded tensor indicates a pipeline error.
        """
        mesh = self.mesh
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="PARTIAL"):
            AllReduce(x="x", output="y").apply(ctx)

    def test_partial_not_cleared_by_unary_ops(self):
        """Unary ops (GELU, Dropout) must preserve Partial placement.

        Element-wise ops are placement-transparent: if x is a partial sum,
        f(x) is also partial. Accidentally clearing Partial would silently
        produce incorrect results.
        """
        mesh = self.mesh
        x = _tensor("x", (self.S, self.H),
                     P(dp=Shard(0), tp=Partial()), mesh)
        ctx = {"x": x}

        y1 = GELU(x="x", output="y1").apply(ctx)
        assert isinstance(y1.sharding.placements[TP_DIM], Partial)
        assert y1.sharding.placements[DP_DIM] == Shard(0)

        y2 = Dropout(x="y1", output="y2", p=0.1).apply(ctx)
        assert isinstance(y2.sharding.placements[TP_DIM], Partial)
        assert y2.sharding.placements[DP_DIM] == Shard(0)

    def test_element_wise_shard_absorbs_replicate(self):
        """Element-wise: Shard + Replicate → Shard.

        Replicated data is available in full on every device, so the
        output follows the sharded input's layout.
        """
        mesh = self.mesh
        a = _tensor("a", (self.S, self.H), P(dp=Shard(0)), mesh)
        b = _tensor("b", (self.S, self.H), P(), mesh)
        ctx = {"a": a, "b": b}

        result = Add(a="a", b="b", output="c").apply(ctx)
        assert result.sharding.placements[DP_DIM] == Shard(0)
        assert result.local_shape == (self.S // DP_SIZE, self.H)

    def test_element_wise_rejects_incompatible_shards(self):
        """Element-wise: Shard(0) + Shard(1) on same mesh dim → error.

        Data layouts are physically incompatible when two inputs are
        sharded on different tensor dims on the same mesh dim.
        """
        mesh = self.mesh
        a = _tensor("a", (self.S, self.H), P(dp=Shard(0)), mesh)
        b = _tensor("b", (self.S, self.H), P(dp=Shard(1)), mesh)
        ctx = {"a": a, "b": b}

        with pytest.raises(ValueError, match="incompatible placements"):
            Add(a="a", b="b", output="c").apply(ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Megatron-LM Parallelism Invariants
#
# Requirements from Megatron-LM (Shoeybi et al. 2019, Narayanan et al. 2021):
#   - TP layer is transparent: col→row→AllReduce → replicated output
#   - DP batch split is invisible to TP operations
#   - PP Send/Recv preserves tensor metadata between stages
# ═══════════════════════════════════════════════════════════════════════════════

class TestMegatronInvariants256GPU:
    """Verify Megatron-LM parallelism invariants on 256-GPU 4D mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.V = LLAMA_70B["vocab_size"]
        self.mesh = _mesh_4d()

    def test_tp_attention_block_produces_replicated_output(self):
        """Megatron attention: QKV col → FlashAttn → O row → AllReduce → replicated.

        Invariant (Megatron paper, Figure 3): complete attention block with
        tensor parallelism must produce a fully replicated output. Downstream
        ops (residual add, LayerNorm) expect replicated input.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        tensors = [
            _tensor("x", (S, H), P(), mesh),
            _tensor("w_q", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_k", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_v", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_o", (H, H), P(tp=Shard(0)), mesh),
        ]

        prog = Program(ops=[
            MatMul(a="x", b="w_q", output="q"),
            MatMul(a="x", b="w_k", output="k"),
            MatMul(a="x", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            AllReduce(x="attn_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        for t in tensors:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        y = result["y"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.local_shape == (S, H)
        assert y.global_shape == (S, H)

    def test_swiglu_mlp_produces_replicated_output(self):
        """SwiGLU MLP: gate+up col → SiLU → multiply → down row → AllReduce.

        Invariant (LLaMA + Megatron TP for GLU variants): the SwiGLU MLP
        block must produce a fully replicated output, same as attention.
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(), mesh)
        w_gate = _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh)
        w_up = _tensor("w_up", (H, I), P(tp=Shard(1)), mesh)
        w_down = _tensor("w_down", (I, H), P(tp=Shard(0)), mesh)

        prog = Program(ops=[
            MatMul(a="x", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="x", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        for t in [x, w_gate, w_up, w_down]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        y = result["y"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.local_shape == (S, H)

    def test_dp_shard_survives_full_tp_pipeline(self):
        """DP batch shard is preserved through the entire TP pipeline.

        Invariant (Megatron 3D parallelism): DP and TP operate on orthogonal
        dims. A DP batch shard must survive column-parallel, row-parallel,
        element-wise ops, and AllReduce — none of these touch the batch dim.

        spmd_checking disabled: single-enum SPMD type cannot represent the
        mixed state (Shard on DP + Replicate on TP) after AllReduce.
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w_col = _tensor("w_col", (H, I), P(tp=Shard(1)), mesh)
        w_row = _tensor("w_row", (I, H), P(tp=Shard(0)), mesh)

        prog = Program(ops=[
            MatMul(a="x", b="w_col", output="h"),
            SiLU(x="h", output="h_act"),
            MatMul(a="h_act", b="w_row", output="h_p"),
            AllReduce(x="h_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w_col, w_row]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        for name in ["h", "h_act", "h_p", "y"]:
            assert result[name].sharding.placements[DP_DIM] == Shard(0), \
                f"DP shard lost at '{name}'"

        y = result["y"]
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert y.local_shape == (S // DP_SIZE, H)

        slices = compute_tensor_slices((S, H), y.sharding)
        for dp in range(DP_SIZE):
            did = mesh.coord_to_device(dp, 0, 0, 0)
            assert slices[did].offsets[0] == dp * (S // DP_SIZE)

    def test_pp_send_recv_preserves_placement(self):
        """PP transfer preserves tensor metadata unchanged.

        Pipeline parallelism semantics: Send/Recv is a point-to-point copy.
        It must not alter shape, sharding, or any other metadata.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        src = mesh.coord_to_device(0, 0, 0, 0)
        dst = mesh.coord_to_device(0, 0, 1, 0)

        x = _tensor("x", (S, H), P(tp=Shard(1)), mesh)

        prog = Program(ops=[
            Send(x="x", output="sent_0", src=src, dst=dst,
                 stage=0, microbatch_id=0),
            Recv(x="sent_0", output="x_recv", src=src, dst=dst,
                 stage=1, microbatch_id=0),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        executor.register_tensor(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        recv_t = executor.final_state()[dst]["x_recv"]
        assert recv_t.global_shape == x.global_shape
        assert recv_t.sharding.placements[TP_DIM] == Shard(1)
        assert recv_t.local_shape == x.local_shape

    def test_full_megatron_forward_pass(self):
        """Complete Megatron forward: Embed → RMSNorm → Attn+Residual → MLP+Residual → Loss.

        End-to-end invariant (Megatron paper): every intermediate that feeds
        into a norm or residual must be fully replicated. The pipeline:
          1. Vocab-parallel Embed → AllReduce → replicated embeddings
          2. RMSNorm (requires replicated hidden dim)
          3. TP Attention: col-parallel QKV → FlashAttn → row-parallel O → AllReduce
          4. Residual Add (replicated + replicated = replicated)
          5. RMSNorm → TP SwiGLU MLP → AllReduce
          6. Residual Add
          7. Final RMSNorm → output projection → CrossEntropyLoss
        """
        H, I, S, V = self.H, self.I, self.S, self.V
        mesh = self.mesh

        tensors = [
            _tensor("ids", (S,), P(), mesh),
            _tensor("W_emb", (V, H), P(tp=Shard(0)), mesh),
            _tensor("w_q", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_k", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_v", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_o", (H, H), P(tp=Shard(0)), mesh),
            _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh),
            _tensor("w_up", (H, I), P(tp=Shard(1)), mesh),
            _tensor("w_down", (I, H), P(tp=Shard(0)), mesh),
            _tensor("w_out", (H, V), P(tp=Shard(1)), mesh),
            _tensor("targets", (S,), P(), mesh),
        ]

        prog = Program(ops=[
            # 1. Embedding
            Embedding(indices="ids", weight="W_emb", output="emb_p"),
            AllReduce(x="emb_p", output="emb"),
            # 2. Pre-attention norm
            RMSNorm(x="emb", output="x_norm", norm_dim=-1),
            # 3. TP Attention
            MatMul(a="x_norm", b="w_q", output="q"),
            MatMul(a="x_norm", b="w_k", output="k"),
            MatMul(a="x_norm", b="w_v", output="v"),
            FlashAttention(q="q", k="k", v="v", output="attn"),
            MatMul(a="attn", b="w_o", output="attn_p"),
            AllReduce(x="attn_p", output="y_attn"),
            # 4. Residual add
            Add(a="emb", b="y_attn", output="resid1"),
            # 5. Pre-MLP norm + TP SwiGLU MLP
            RMSNorm(x="resid1", output="mlp_norm", norm_dim=-1),
            MatMul(a="mlp_norm", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="mlp_norm", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y_mlp"),
            # 6. Residual add
            Add(a="resid1", b="y_mlp", output="resid2"),
            # 7. Final norm + output projection + loss
            RMSNorm(x="resid2", output="final_norm", norm_dim=-1),
            MatMul(a="final_norm", b="w_out", output="logits"),
            CrossEntropyLoss(logits="logits", targets="targets",
                             output="loss", vocab_dim=-1),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        for t in tensors:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        # Embedding partial → resolved by AllReduce
        assert isinstance(result["emb_p"].sharding.placements[TP_DIM], Partial)
        assert result["emb"].local_type == LocalSPMDType.REPLICATE

        # Attention partial → resolved, residual stays replicated
        assert isinstance(result["attn_p"].sharding.placements[TP_DIM], Partial)
        assert result["y_attn"].local_type == LocalSPMDType.REPLICATE
        assert result["resid1"].local_type == LocalSPMDType.REPLICATE

        # MLP partial → resolved, residual stays replicated
        assert isinstance(result["mlp_p"].sharding.placements[TP_DIM], Partial)
        assert result["y_mlp"].local_type == LocalSPMDType.REPLICATE
        assert result["resid2"].local_type == LocalSPMDType.REPLICATE

        # Output: logits are TP-sharded (col-parallel), loss is partial
        assert result["logits"].sharding.placements[TP_DIM] == Shard(1)
        assert result["logits"].local_shape == (S, V // TP_SIZE)
        loss = result["loss"]
        assert loss.global_shape == (1,)
        assert isinstance(loss.sharding.placements[TP_DIM], Partial)

    def test_tp_pp_two_stage_pipeline(self):
        """TP within each PP stage + Send/Recv between stages.

        Megatron 2D parallelism: each stage runs a complete TP layer.
        The stage output (after AllReduce) is replicated and can be
        safely transferred to the next stage.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        s0_devs = [mesh.coord_to_device(0, tp, 0, 0) for tp in range(TP_SIZE)]
        s1_devs = [mesh.coord_to_device(0, tp, 1, 0) for tp in range(TP_SIZE)]

        x = _tensor("x", (S, H), P(), mesh)
        w0_c = _tensor("w0_c", (H, H), P(tp=Shard(1)), mesh)
        w0_r = _tensor("w0_r", (H, H), P(tp=Shard(0)), mesh)
        w1_c = _tensor("w1_c", (H, H), P(tp=Shard(1)), mesh)
        w1_r = _tensor("w1_r", (H, H), P(tp=Shard(0)), mesh)

        ops = [
            MatMul(a="x", b="w0_c", output="h0"),
            MatMul(a="h0", b="w0_r", output="h0_p"),
            AllReduce(x="h0_p", output="y0"),
        ]
        for i, (s, d) in enumerate(zip(s0_devs, s1_devs)):
            ops.append(Send(x="y0", output=f"sent_{i}",
                            src=s, dst=d, stage=0, microbatch_id=0))
            ops.append(Recv(x=f"sent_{i}", output="y0_recv",
                            src=s, dst=d, stage=1, microbatch_id=0))
        ops.extend([
            MatMul(a="y0_recv", b="w1_c", output="h1"),
            MatMul(a="h1", b="w1_r", output="h1_p"),
            AllReduce(x="h1_p", output="y1"),
        ])

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        for t in [x, w0_c, w0_r]:
            executor.register_tensor(t, device_ids=s0_devs)
        for t in [w1_c, w1_r]:
            executor.register_tensor(t, device_ids=s1_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for did in s1_devs:
            y1 = final[did]["y1"]
            assert y1.local_shape == (S, H)
            assert all(isinstance(p, Replicate) for p in y1.sharding.placements)

    def test_cross_dp_structural_consistency(self):
        """All DP ranks have identical tensor structure, different data.

        Data parallelism: devices with same TP rank but different DP ranks
        must have identical metadata (shape, placements) after any TP op.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w = _tensor("w", (H, H), P(tp=Shard(1)), mesh)

        prog = Program(ops=[MatMul(a="x", b="w", output="h")])
        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w]:
            executor.register_tensor(t)
        executor.run_program(prog)

        final = executor.final_state()
        for tp in range(TP_SIZE):
            ref_did = mesh.coord_to_device(0, tp, 0, 0)
            ref_h = final[ref_did]["h"]
            for dp in range(1, DP_SIZE):
                did = mesh.coord_to_device(dp, tp, 0, 0)
                h = final[did]["h"]
                assert h.local_shape == ref_h.local_shape
                assert h.sharding.placements == ref_h.sharding.placements


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Mathematical Correctness Constraints
#
# Requirements from mathematical definitions:
#   - LayerNorm/RMSNorm: mean and variance require the FULL reduction dimension
#   - Softmax: exp(x_i) / Σexp(x_j) requires the FULL denominator
#   - Embedding: vocab-parallel lookup produces partial matches (need AllReduce)
#   - CrossEntropy: log-sum-exp over vocab requires full softmax range
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathConstraints256GPU:
    """Verify mathematical correctness constraints on 4D mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.V = LLAMA_70B["vocab_size"]
        self.mesh = _mesh_4d()

    def test_layernorm_rejects_sharded_norm_dim(self):
        """LayerNorm MUST reject input sharded on normalization dimension.

        Math: μ = Σx/n, σ² = Σ(x-μ)²/n. Partial statistics are WRONG.
        """
        x = _tensor("x", (self.S, self.H), P(tp=Shard(1)), self.mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on norm_dim"):
            LayerNorm(x="x", output="y", norm_dim=-1).apply(ctx)

    def test_rmsnorm_rejects_sharded_norm_dim(self):
        """RMSNorm MUST reject sharded norm dim, even on non-obvious mesh dim (CP).

        Math: RMS = sqrt(Σx²/n) needs the full dim.
        """
        x = _tensor("x", (self.S, self.H), P(cp=Shard(1)), self.mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on norm_dim"):
            RMSNorm(x="x", output="y", norm_dim=-1).apply(ctx)

    def test_normalization_allows_batch_dim_shard(self):
        """Normalization is safe when only the batch dim is sharded.

        LayerNorm normalizes per-sample. Splitting batch across DP+CP
        doesn't affect per-sample statistics.
        """
        x = _tensor("x", (self.S, self.H),
                     P(dp=Shard(0), cp=Shard(0)), self.mesh)
        ctx = {"x": x}

        y_ln = LayerNorm(x="x", output="y_ln", norm_dim=-1).apply(ctx)
        assert y_ln.sharding.placements[DP_DIM] == Shard(0)
        assert y_ln.sharding.placements[CP_DIM] == Shard(0)
        assert y_ln.local_shape == (self.S // (DP_SIZE * CP_SIZE), self.H)

        y_rms = RMSNorm(x="x", output="y_rms", norm_dim=-1).apply(ctx)
        assert y_rms.local_shape == y_ln.local_shape

    def test_softmax_rejects_sharded_reduction_dim(self):
        """Softmax MUST reject sharded reduction dim.

        Math: softmax(x_i) = exp(x_i)/Σexp(x_j). Partial denominator
        gives wrong probabilities.
        """
        x = _tensor("x", (self.S, self.H), P(tp=Shard(1)), self.mesh)
        ctx = {"x": x}
        with pytest.raises(ValueError, match="sharded on dim"):
            Softmax(x="x", output="y", dim=-1).apply(ctx)

    def test_softmax_allows_batch_dim_shard(self):
        """Softmax with batch-dim shard: each device normalizes its own samples."""
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), self.mesh)
        ctx = {"x": x}
        y = Softmax(x="x", output="y", dim=-1).apply(ctx)
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert y.local_shape == (self.S // DP_SIZE, self.H)

    def test_vocab_parallel_embedding_produces_partial(self):
        """Vocab-parallel embedding: each device matches only its vocab slice.

        Megatron vocab-parallel: embedding table split by vocab rows.
        For index j, only the device holding row j contributes; others
        contribute zero. Result is a partial sum needing AllReduce.
        """
        ids = _tensor("ids", (self.S,), P(), self.mesh)
        W = _tensor("W", (self.V, self.H), P(tp=Shard(0)), self.mesh)
        ctx = {"ids": ids, "W": W}

        result = Embedding(indices="ids", weight="W", output="emb").apply(ctx)

        assert result.global_shape == (self.S, self.H)
        assert isinstance(result.sharding.placements[TP_DIM], Partial)
        for dim in [DP_DIM, PP_DIM, CP_DIM]:
            assert isinstance(result.sharding.placements[dim], Replicate)

    def test_hidden_parallel_embedding_produces_sharded(self):
        """Hidden-parallel embedding: output column-partitioned on hidden dim.

        Each device holds W[:, H/p*i:H/p*(i+1)]. Lookup produces a
        row that is column-partitioned.
        """
        ids = _tensor("ids", (self.S,), P(), self.mesh)
        W = _tensor("W", (self.V, self.H), P(tp=Shard(1)), self.mesh)
        ctx = {"ids": ids, "W": W}

        result = Embedding(indices="ids", weight="W", output="emb").apply(ctx)

        assert result.sharding.placements[TP_DIM] == Shard(1)
        assert result.local_shape == (self.S, self.H // TP_SIZE)

    def test_cross_entropy_vocab_shard_produces_partial(self):
        """CE with vocab-sharded logits → Partial.

        Math: CE = -log(softmax(logits)[target]). Softmax denominator
        Σexp(x_j) requires full vocab. Partial log-sum-exp → partial loss.
        """
        logits = _tensor("logits", (self.S, self.V),
                         P(tp=Shard(1)), self.mesh)
        targets = _tensor("targets", (self.S,), P(), self.mesh)
        ctx = {"logits": logits, "targets": targets}

        result = CrossEntropyLoss(
            logits="logits", targets="targets", output="loss", vocab_dim=-1,
        ).apply(ctx)

        assert result.global_shape == (1,)
        assert isinstance(result.sharding.placements[TP_DIM], Partial)

    def test_cross_entropy_batch_shard_becomes_scalar_replicate(self):
        """CE: batch-sharded logits → scalar output with Replicate.

        Scalar loss has no batch dim to shard, so batch Shard → Replicate.
        """
        logits = _tensor("logits", (self.S, self.V),
                         P(dp=Shard(0)), self.mesh)
        targets = _tensor("targets", (self.S,), P(dp=Shard(0)), self.mesh)
        ctx = {"logits": logits, "targets": targets}

        result = CrossEntropyLoss(
            logits="logits", targets="targets", output="loss", vocab_dim=-1,
        ).apply(ctx)

        assert result.global_shape == (1,)
        assert isinstance(result.sharding.placements[DP_DIM], Replicate)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Gradient Correctness Properties
#
# Requirements from automatic differentiation (AD) theory:
#   - grad(f(x)) has same shape as x (chain rule)
#   - grad must respect input's distribution (same device layout)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradientProperties256GPU:
    """Verify gradient correctness: shape and distribution preservation."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.V = LLAMA_70B["vocab_size"]
        self.mesh = _mesh_4d()

    def _check_grad(self, op, ctx, input_name, input_tensor):
        """Verify grad has same shape and distribution as input."""
        op.apply(ctx)
        grads = op.vjp(ctx, ctx[op.output_name])
        grad = grads[input_name]
        assert grad.global_shape == input_tensor.global_shape
        assert grad.local_shape == input_tensor.local_shape
        assert grad.sharding.placements == input_tensor.sharding.placements

    def test_unary_ops_grad_preserves_dual_shard(self):
        """∂L/∂x for element-wise f(x) has same distribution as x.

        Chain rule: ∂L/∂x = ∂L/∂f(x)·f'(x). Both terms share x's layout.
        """
        x = _tensor("x", (self.S, self.I),
                     P(dp=Shard(0), tp=Shard(1)), self.mesh)

        for op_cls in [GELU, ReLU]:
            self._check_grad(op_cls(x="x", output="y"), {"x": x}, "x", x)

        self._check_grad(
            Dropout(x="x", output="y", p=0.1), {"x": x}, "x", x,
        )

    def test_normalization_grad_preserves_distribution(self):
        """LayerNorm/Softmax grad preserves input distribution.

        AD for normalization: Jacobian is a function of x. Gradient is
        computed locally per sample and has the same distribution as x.
        """
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), self.mesh)

        for op_cls, kw in [(LayerNorm, {"norm_dim": -1}), (Softmax, {"dim": -1})]:
            self._check_grad(op_cls(x="x", output="y", **kw), {"x": x}, "x", x)

    def test_embedding_grad_preserves_weight_distribution(self):
        """Embedding grad: ∂L/∂W has same partition as W.

        Scatter-add gradient: each device updates its own vocab slice.
        """
        ids = _tensor("ids", (self.S,), P(), self.mesh)
        W = _tensor("W", (self.V, self.H), P(tp=Shard(0)), self.mesh)
        ctx = {"ids": ids, "W": W}

        self._check_grad(
            Embedding(indices="ids", weight="W", output="emb"), ctx, "W", W,
        )

    def test_cross_entropy_grad_preserves_logits_distribution(self):
        """CE grad: ∂L/∂logits has same partition as logits.

        softmax(logits) - one_hot(target) is computed per vocab slice.
        """
        logits = _tensor("logits", (self.S, self.V),
                         P(dp=Shard(0), tp=Shard(1)), self.mesh)
        targets = _tensor("targets", (self.S,), P(dp=Shard(0)), self.mesh)
        ctx = {"logits": logits, "targets": targets}

        self._check_grad(
            CrossEntropyLoss(logits="logits", targets="targets", output="loss"),
            ctx, "logits", logits,
        )

    def test_matmul_grad_preserves_both_input_distributions(self):
        """MatMul grad: ∂L/∂A and ∂L/∂B each match their forward input.

        Chain rule: ∂L/∂A = ∂L/∂Y @ B^T (same shape as A),
                    ∂L/∂B = A^T @ ∂L/∂Y (same shape as B).
        """
        a = _tensor("a", (self.S, self.H), P(dp=Shard(0)), self.mesh)
        b = _tensor("b", (self.H, self.I), P(tp=Shard(1)), self.mesh)
        ctx = {"a": a, "b": b}

        op = MatMul(a="a", b="b", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, ctx["y"])

        assert grads["a"].global_shape == a.global_shape
        assert grads["a"].local_shape == a.local_shape
        assert grads["a"].sharding.placements == a.sharding.placements
        assert grads["b"].global_shape == b.global_shape
        assert grads["b"].local_shape == b.local_shape
        assert grads["b"].sharding.placements == b.sharding.placements

    def test_allreduce_vjp_is_self_dual(self):
        """AllReduce backward is AllReduce (self-dual).

        AD theory: AllReduce(sum) forward resolves Partial→Replicate.
        Backward: gradient of Replicate→Replicate is identity in placement;
        the VJP produces a grad with the INPUT's placement (Partial).
        """
        x = _tensor("x", (self.S, self.H),
                     P(dp=Shard(0), tp=Partial()), self.mesh)
        ctx = {"x": x}
        op = AllReduce(x="x", output="y")
        op.apply(ctx)
        grads = op.vjp(ctx, ctx["y"])
        grad_x = grads["x"]
        assert grad_x.global_shape == x.global_shape
        assert grad_x.local_shape == x.local_shape
        assert grad_x.sharding.placements == x.sharding.placements

    def test_allgather_reducescatter_are_duals(self):
        """AllGather and ReduceScatter are VJP duals.

        AD: AllGather(Shard→Replicate) backward = ReduceScatter-like.
        The grad must have the same shape and placement as the input
        (Shard), not the output (Replicate).
        """
        x_ag = _tensor("x_ag", (self.S, self.H),
                        P(dp=Shard(0), tp=Shard(0)), self.mesh)
        ctx = {"x_ag": x_ag}
        op_ag = AllGather(x="x_ag", output="y_ag", gather_dim=0)
        op_ag.apply(ctx)
        grad_ag = op_ag.vjp(ctx, ctx["y_ag"])["x_ag"]
        assert grad_ag.global_shape == x_ag.global_shape
        assert grad_ag.local_shape == x_ag.local_shape
        assert grad_ag.sharding.placements == x_ag.sharding.placements

        x_rs = _tensor("x_rs", (self.S, self.H),
                        P(tp=Partial()), self.mesh)
        ctx2 = {"x_rs": x_rs}
        op_rs = ReduceScatter(x="x_rs", output="y_rs", scatter_dim=0)
        op_rs.apply(ctx2)
        grad_rs = op_rs.vjp(ctx2, ctx2["y_rs"])["x_rs"]
        assert grad_rs.global_shape == x_rs.global_shape
        assert grad_rs.local_shape == (self.S, self.H)
        assert grad_rs.sharding.placements[TP_DIM] == Replicate()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. System Integrity Properties
#
# Requirements for correct multi-device execution:
#   - Device mesh is a proper Cartesian product (coordinate bijection)
#   - Communication groups are disjoint
#   - Sharded slices cover the full tensor without overlap
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemIntegrity256GPU:
    """Verify system-level correctness for 256-GPU execution."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.V = LLAMA_70B["vocab_size"]
        self.mesh = _mesh_4d()

    def test_256_device_coordinate_bijection(self):
        """4D mesh (4×8×4×2) maps 256 coordinates ↔ 256 device IDs bijectively."""
        mesh = self.mesh
        assert mesh.num_devices == 256

        all_ids = set()
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                for pp in range(PP_SIZE):
                    for cp in range(CP_SIZE):
                        all_ids.add(mesh.coord_to_device(dp, tp, pp, cp))
        assert len(all_ids) == 256

        for did in range(256):
            coords = mesh.device_to_coord(did)
            assert mesh.coord_to_device(*coords) == did

    def test_tp_groups_are_disjoint(self):
        """TP groups across different (dp,pp,cp) must be disjoint.

        Overlapping NCCL communicator groups cause incorrect AllReduce.
        """
        mesh = self.mesh
        seen = set()
        for dp in range(DP_SIZE):
            for pp in range(PP_SIZE):
                for cp in range(CP_SIZE):
                    group = set()
                    for tp in range(TP_SIZE):
                        group.add(mesh.coord_to_device(dp, tp, pp, cp))
                    assert len(group) == TP_SIZE
                    assert group.isdisjoint(seen)
                    seen |= group
        assert len(seen) == 256

    def test_dual_shard_slices_cover_full_tensor(self):
        """DP+TP dual-shard slices: non-overlapping, contiguous, full coverage.

        Data integrity: every tensor element on exactly one device.
        """
        S, H = self.S, self.H
        spec = _spec(P(dp=Shard(0), tp=Shard(1)), self.mesh)
        slices = compute_tensor_slices((S, H), spec)
        assert len(slices) == 256

        unique_ranges = set()
        for s in slices.values():
            unique_ranges.add(s.ranges)
        assert len(unique_ranges) == DP_SIZE * TP_SIZE

        row_starts = sorted(set(r[0][0] for r in unique_ranges))
        assert row_starts == [i * (S // DP_SIZE) for i in range(DP_SIZE)]
        assert max(r[0][1] for r in unique_ranges) == S

        col_starts = sorted(set(r[1][0] for r in unique_ranges))
        assert col_starts == [i * (H // TP_SIZE) for i in range(TP_SIZE)]
        assert max(r[1][1] for r in unique_ranges) == H

    def test_all_devices_consistent_after_computation(self):
        """Devices at same (dp,tp) with different (pp,cp) have identical metadata.

        Deterministic execution: same ops on same-shaped data → same metadata.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w = _tensor("w", (H, H), P(tp=Shard(1)), mesh)

        prog = Program(ops=[MatMul(a="x", b="w", output="h")])
        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w]:
            executor.register_tensor(t)
        executor.run_program(prog)

        final = executor.final_state()
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                ref_did = mesh.coord_to_device(dp, tp, 0, 0)
                ref_h = final[ref_did]["h"]
                for pp in range(PP_SIZE):
                    for cp in range(CP_SIZE):
                        did = mesh.coord_to_device(dp, tp, pp, cp)
                        h = final[did]["h"]
                        assert h.local_shape == ref_h.local_shape
                        assert h.sharding.placements == ref_h.sharding.placements

    def test_executor_pipeline_norm_activation_dropout(self):
        """RMSNorm → GELU → Dropout chain preserves distribution on executor.

        Element-wise and per-sample ops must preserve DP batch shard end-to-end.
        """
        mesh = self.mesh
        x = _tensor("x", (self.S, self.H), P(dp=Shard(0)), mesh)

        prog = Program(ops=[
            RMSNorm(x="x", output="normed", norm_dim=-1),
            GELU(x="normed", output="act"),
            Dropout(x="act", output="y", p=0.1),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        executor.register_tensor(x)
        result = executor.run_program(prog)

        y = result["y"]
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert y.local_shape == (self.S // DP_SIZE, self.H)

    def test_executor_embedding_allreduce_layernorm(self):
        """Vocab-parallel Embed → AllReduce → LayerNorm pipeline.

        Embedding produces partial (partial index match). AllReduce resolves
        to complete embedding. LayerNorm normalizes the replicated result.
        """
        mesh = self.mesh
        ids = _tensor("ids", (self.S,), P(), mesh)
        W = _tensor("W", (self.V, self.H), P(tp=Shard(0)), mesh)

        prog = Program(ops=[
            Embedding(indices="ids", weight="W", output="emb_p"),
            AllReduce(x="emb_p", output="emb"),
            LayerNorm(x="emb", output="y", norm_dim=-1),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=True)
        executor.register_tensor(ids)
        executor.register_tensor(W)
        result = executor.run_program(prog)

        assert isinstance(result["emb_p"].sharding.placements[TP_DIM], Partial)
        y = result["y"]
        assert y.local_type == LocalSPMDType.REPLICATE
        assert y.local_shape == (self.S, self.H)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Multi-Dimensional Cooperation
#
# The core value of a 4D mesh: DP, TP, PP, CP operate simultaneously,
# each on a different aspect of the computation. These tests verify that
# dimensions interact correctly — not just that each works in isolation.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiDimCooperation256GPU:
    """Verify multi-dimensional parallelism cooperation on 4D mesh.

    Every test has at least 2 non-Replicate mesh dimensions active.
    These are the scenarios that actually occur in production training.
    """

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.V = LLAMA_70B["vocab_size"]
        self.mesh = _mesh_4d()

    def test_dp_tp_matmul_allreduce(self):
        """DP micro-batch through TP col→row→AllReduce: most common real pattern.

        DP shards the batch dim across DP_SIZE=4 devices.
        TP shards weights across TP_SIZE=8 devices.
        AllReduce resolves TP Partial but must NOT touch DP Shard.

        Expected: y has Shard(0) on DP, Replicate on TP.
        local_shape = (S/4, H) — batch shrunk by DP, hidden restored by AllReduce.
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w_col = _tensor("w_col", (H, I), P(tp=Shard(1)), mesh)
        w_row = _tensor("w_row", (I, H), P(tp=Shard(0)), mesh)

        prog = Program(ops=[
            MatMul(a="x", b="w_col", output="h"),
            MatMul(a="h", b="w_row", output="h_p"),
            AllReduce(x="h_p", output="y"),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w_col, w_row]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        h = result["h"]
        assert h.sharding.placements[DP_DIM] == Shard(0)
        assert h.sharding.placements[TP_DIM] == Shard(1)
        assert h.local_shape == (S // DP_SIZE, I // TP_SIZE)

        h_p = result["h_p"]
        assert h_p.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(h_p.sharding.placements[TP_DIM], Partial)

        y = result["y"]
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert y.local_shape == (S // DP_SIZE, H)

    def test_dp_tp_full_transformer_layer(self):
        """DP+TP full Transformer layer: batch-sharded input through TP attention+MLP.

        Real Megatron 3D: each DP rank processes its micro-batch independently
        through the TP layer. DP Shard must survive the entire layer (MatMul,
        FlashAttention, SiLU, Multiply, AllReduce).
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        tensors = [
            _tensor("x", (S, H), P(dp=Shard(0)), mesh),
            _tensor("w_q", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_k", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_v", (H, H), P(tp=Shard(1)), mesh),
            _tensor("w_o", (H, H), P(tp=Shard(0)), mesh),
            _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh),
            _tensor("w_up", (H, I), P(tp=Shard(1)), mesh),
            _tensor("w_down", (I, H), P(tp=Shard(0)), mesh),
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

        executor = MultiDeviceExecutor(mesh=mesh)
        for t in tensors:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        for name in ["q", "attn", "attn_p", "gate", "gate_act", "up",
                      "h", "mlp_p", "y_attn", "y"]:
            assert result[name].sharding.placements[DP_DIM] == Shard(0), \
                f"DP Shard lost at '{name}'"

        y = result["y"]
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert y.local_shape == (S // DP_SIZE, H)

    def test_dp_tp_pp_three_dim_pipeline(self):
        """DP+TP+PP: micro-batch sharded by DP, TP within each PP stage, Send/Recv between stages.

        Three active dimensions. DP Shard must survive cross-stage transfer.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        s0_devs = [mesh.coord_to_device(dp, tp, 0, 0)
                    for dp in range(DP_SIZE) for tp in range(TP_SIZE)]
        s1_devs = [mesh.coord_to_device(dp, tp, 1, 0)
                    for dp in range(DP_SIZE) for tp in range(TP_SIZE)]

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w0_c = _tensor("w0_c", (H, H), P(tp=Shard(1)), mesh)
        w0_r = _tensor("w0_r", (H, H), P(tp=Shard(0)), mesh)
        w1_c = _tensor("w1_c", (H, H), P(tp=Shard(1)), mesh)
        w1_r = _tensor("w1_r", (H, H), P(tp=Shard(0)), mesh)

        ops = [
            MatMul(a="x", b="w0_c", output="h0"),
            MatMul(a="h0", b="w0_r", output="h0_p"),
            AllReduce(x="h0_p", output="y0"),
        ]
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                src = mesh.coord_to_device(dp, tp, 0, 0)
                dst = mesh.coord_to_device(dp, tp, 1, 0)
                ops.append(Send(x="y0", output=f"sent_{dp}_{tp}",
                                src=src, dst=dst, stage=0, microbatch_id=dp))
                ops.append(Recv(x=f"sent_{dp}_{tp}", output="y0_recv",
                                src=src, dst=dst, stage=1, microbatch_id=dp))
        ops.extend([
            MatMul(a="y0_recv", b="w1_c", output="h1"),
            MatMul(a="h1", b="w1_r", output="h1_p"),
            AllReduce(x="h1_p", output="y1"),
        ])

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w0_c, w0_r]:
            executor.register_tensor(t, device_ids=s0_devs)
        for t in [w1_c, w1_r]:
            executor.register_tensor(t, device_ids=s1_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                did = mesh.coord_to_device(dp, tp, 1, 0)
                y1 = final[did]["y1"]
                assert y1.sharding.placements[DP_DIM] == Shard(0), \
                    f"DP Shard lost at device ({dp},{tp},1,0)"
                assert isinstance(y1.sharding.placements[TP_DIM], Replicate)
                assert y1.local_shape == (S // DP_SIZE, H)

    def test_cp_tp_ring_attention(self):
        """CP+TP: context parallelism (ring attention) with tensor parallelism.

        CP shards sequence across CP_SIZE=2, TP shards heads across TP_SIZE=8.
        RingAttention combines attention across CP ring; output should resolve
        CP sharding while preserving TP head sharding.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        q = _tensor("q", (S, H), P(cp=Shard(0), tp=Shard(1)), mesh)
        k = _tensor("k", (S, H), P(cp=Shard(0), tp=Shard(1)), mesh)
        v = _tensor("v", (S, H), P(cp=Shard(0), tp=Shard(1)), mesh)

        assert q.local_shape == (S // CP_SIZE, H // TP_SIZE)

        ctx = {"q": q, "k": k, "v": v}
        result = RingAttention(
            q="q", k="k", v="v", output="attn",
            ring_size=CP_SIZE,
        ).apply(ctx)

        assert result.sharding.placements[CP_DIM] == Shard(0)
        assert result.sharding.placements[TP_DIM] == Shard(1)
        assert result.local_shape == (S // CP_SIZE, H // TP_SIZE)

    def test_dp_cp_tp_cooperative_attention(self):
        """DP+CP+TP: three dimensions active in a single attention block.

        DP splits micro-batch, CP splits sequence, TP splits heads.
        After col-parallel Q/K/V projection, we have 3 active shards.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0), cp=Shard(0)), mesh)
        w_q = _tensor("w_q", (H, H), P(tp=Shard(1)), mesh)

        assert x.local_shape == (S // (DP_SIZE * CP_SIZE), H)

        ctx = {"x": x, "w_q": w_q}
        q = MatMul(a="x", b="w_q", output="q").apply(ctx)

        assert q.sharding.placements[DP_DIM] == Shard(0)
        assert q.sharding.placements[CP_DIM] == Shard(0)
        assert q.sharding.placements[TP_DIM] == Shard(1)
        assert q.local_shape == (S // (DP_SIZE * CP_SIZE), H // TP_SIZE)

    def test_all_four_dims_active(self):
        """All 4 dims active: DP batch + CP seq + TP model + PP stage transfer.

        Most realistic scenario: every mesh dimension does real work.
        """
        H, S = self.H, self.S
        mesh = self.mesh

        s0_devs = [mesh.coord_to_device(dp, tp, 0, cp)
                    for dp in range(DP_SIZE)
                    for tp in range(TP_SIZE)
                    for cp in range(CP_SIZE)]
        s1_devs = [mesh.coord_to_device(dp, tp, 1, cp)
                    for dp in range(DP_SIZE)
                    for tp in range(TP_SIZE)
                    for cp in range(CP_SIZE)]

        x = _tensor("x", (S, H), P(dp=Shard(0), cp=Shard(0)), mesh)
        w_col = _tensor("w_col", (H, H), P(tp=Shard(1)), mesh)
        w_row = _tensor("w_row", (H, H), P(tp=Shard(0)), mesh)

        assert x.local_shape == (S // (DP_SIZE * CP_SIZE), H)

        ops = [
            MatMul(a="x", b="w_col", output="h"),
            MatMul(a="h", b="w_row", output="h_p"),
            AllReduce(x="h_p", output="y0"),
        ]
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                for cp in range(CP_SIZE):
                    src = mesh.coord_to_device(dp, tp, 0, cp)
                    dst = mesh.coord_to_device(dp, tp, 1, cp)
                    tag = f"{dp}_{tp}_{cp}"
                    ops.append(Send(x="y0", output=f"sent_{tag}",
                                    src=src, dst=dst, stage=0, microbatch_id=dp))
                    ops.append(Recv(x=f"sent_{tag}", output="y0_recv",
                                    src=src, dst=dst, stage=1, microbatch_id=dp))

        prog = Program(ops=ops)
        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w_col, w_row]:
            executor.register_tensor(t, device_ids=s0_devs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executor.run_program(prog)

        final = executor.final_state()
        for dp in range(DP_SIZE):
            for tp in range(TP_SIZE):
                for cp in range(CP_SIZE):
                    did = mesh.coord_to_device(dp, tp, 0, cp)
                    y0 = final[did]["y0"]
                    assert y0.sharding.placements[DP_DIM] == Shard(0)
                    assert y0.sharding.placements[CP_DIM] == Shard(0)
                    assert isinstance(y0.sharding.placements[TP_DIM], Replicate)
                    assert y0.local_shape == (S // (DP_SIZE * CP_SIZE), H)

                    dst_did = mesh.coord_to_device(dp, tp, 1, cp)
                    recv = final[dst_did]["y0_recv"]
                    assert recv.sharding.placements[DP_DIM] == Shard(0)
                    assert recv.sharding.placements[CP_DIM] == Shard(0)

    def test_dp_tp_matmul_to_loss(self):
        """DP+TP end-to-end: batch-sharded activation through TP to CrossEntropyLoss.

        Production pattern: DP micro-batch + TP output projection + loss.
        Verifies both DP and TP survive through MatMul and into CE loss.
        """
        H, S, V = self.H, self.S, self.V
        mesh = self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0)), mesh)
        w_out = _tensor("w_out", (H, V), P(tp=Shard(1)), mesh)
        targets = _tensor("targets", (S,), P(dp=Shard(0)), mesh)

        prog = Program(ops=[
            RMSNorm(x="x", output="x_norm", norm_dim=-1),
            MatMul(a="x_norm", b="w_out", output="logits"),
            CrossEntropyLoss(logits="logits", targets="targets",
                             output="loss", vocab_dim=-1),
        ])

        executor = MultiDeviceExecutor(mesh=mesh)
        for t in [x, w_out, targets]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        logits = result["logits"]
        assert logits.sharding.placements[DP_DIM] == Shard(0)
        assert logits.sharding.placements[TP_DIM] == Shard(1)
        assert logits.local_shape == (S // DP_SIZE, V // TP_SIZE)

        loss = result["loss"]
        assert loss.global_shape == (1,)
        assert isinstance(loss.sharding.placements[TP_DIM], Partial)

    def test_sp_with_dp_allgather_compute_reducescatter(self):
        """SP+DP: sequence parallelism with data parallelism on 4D mesh.

        SP pattern: AllGather(hidden_shard)→col-parallel→row-parallel→ReduceScatter.
        DP shards batch (dim 0), SP shards hidden (dim 1) across TP ranks.

        mesh_dim=TP_DIM targets the TP mesh dimension specifically,
        leaving DP Shard untouched.
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        x_sp = _tensor("x_sp", (S, H), P(dp=Shard(0), tp=Shard(1)), mesh)
        assert x_sp.local_shape == (S // DP_SIZE, H // TP_SIZE)

        ctx = {"x_sp": x_sp}
        x_full = AllGather(x="x_sp", output="x_full", gather_dim=1, mesh_dim=TP_DIM).apply(ctx)
        assert x_full.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(x_full.sharding.placements[TP_DIM], Replicate)
        assert x_full.local_shape == (S // DP_SIZE, H)

        w_col = _tensor("w_col", (H, I), P(tp=Shard(1)), mesh)
        ctx["w_col"] = w_col
        h = MatMul(a="x_full", b="w_col", output="h").apply(ctx)
        assert h.sharding.placements[DP_DIM] == Shard(0)
        assert h.sharding.placements[TP_DIM] == Shard(1)

        w_row = _tensor("w_row", (I, H), P(tp=Shard(0)), mesh)
        ctx["w_row"] = w_row
        h_p = MatMul(a="h", b="w_row", output="h_p").apply(ctx)
        assert isinstance(h_p.sharding.placements[TP_DIM], Partial)
        assert h_p.sharding.placements[DP_DIM] == Shard(0)

        y_sp = ReduceScatter(x="h_p", output="y_sp", scatter_dim=1, mesh_dim=TP_DIM).apply(ctx)
        assert y_sp.sharding.placements[DP_DIM] == Shard(0)
        assert y_sp.sharding.placements[TP_DIM] == Shard(1)
        assert y_sp.local_shape == x_sp.local_shape

    def test_sp_full_mlp_on_4d_executor(self):
        """SP+DP MLP end-to-end on 4D executor.

        Full SP pipeline with DP active: AllGather→SwiGLU→ReduceScatter.
        DP shards batch (dim 0), SP shards hidden (dim 1) across TP ranks.
        mesh_dim=TP_DIM ensures collectives target only the TP dimension.

        spmd_checking=False: single-enum SPMD type can't represent
        "Shard on dp, Replicate on tp" after AllGather gathers only tp.
        """
        H, I, S = self.H, self.I, self.S
        mesh = self.mesh

        x_sp = _tensor("x_sp", (S, H), P(dp=Shard(0), tp=Shard(1)), mesh)
        w_gate = _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh)
        w_up = _tensor("w_up", (H, I), P(tp=Shard(1)), mesh)
        w_down = _tensor("w_down", (I, H), P(tp=Shard(0)), mesh)

        prog = Program(ops=[
            AllGather(x="x_sp", output="x_full", gather_dim=1, mesh_dim=TP_DIM),
            MatMul(a="x_full", b="w_gate", output="gate"),
            SiLU(x="gate", output="gate_act"),
            MatMul(a="x_full", b="w_up", output="up"),
            Multiply(a="gate_act", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            ReduceScatter(x="mlp_p", output="y_sp", scatter_dim=1, mesh_dim=TP_DIM),
        ])

        executor = MultiDeviceExecutor(mesh=mesh, spmd_checking=False)
        for t in [x_sp, w_gate, w_up, w_down]:
            executor.register_tensor(t)
        result = executor.run_program(prog)

        y = result["y_sp"]
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert y.sharding.placements[TP_DIM] == Shard(1)
        assert y.local_shape == (S // DP_SIZE, H // TP_SIZE)
        assert y.global_shape == (S, H)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. End-to-End Backward Pass (AutogradEngine)
#
# Requirements from autograd theory:
#   - Every forward collective has a backward dual (AllReduce↔AllReduce,
#     AllGather↔ReduceScatter)
#   - Gradient shapes match forward input shapes
#   - Gradient placements match forward input placements
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutogradBackward256GPU:
    """Verify end-to-end backward pass generation on 256-GPU mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.I = LLAMA_70B["intermediate"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def _run_forward_and_record(self, tensors, ops):
        engine = AutogradEngine()
        ctx = {t.name: t for t in tensors}
        for op in ops:
            op.apply(ctx)
            engine.record(op, ctx)
        return engine, ctx

    def test_tp_linear_backward_generates_dual_allreduce(self):
        """Col→Row→AllReduce generates self-dual AllReduce in backward.

        Forward collectives: [AllReduce]
        Expected backward: [AllReduce] (AllReduce is self-dual)
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(), mesh, expr="x")
        w_col = _tensor("w_col", (H, H), P(tp=Shard(1)), mesh, expr="w_col")
        w_row = _tensor("w_row", (H, H), P(tp=Shard(0)), mesh, expr="w_row")

        fwd_ops = [
            MatMul(a="x", b="w_col", output="h"),
            MatMul(a="h", b="w_row", output="h_p"),
            AllReduce(x="h_p", output="y"),
        ]
        fwd_program = Program("fwd", ops=fwd_ops)
        engine, ctx = self._run_forward_and_record([x, w_col, w_row], fwd_ops)

        bwd_program = engine.generate_backward("y")
        check = engine.verify_gradient_correctness(fwd_program, bwd_program)

        assert check.passed, f"Gradient check failed: {check.errors}"
        assert len(check.collective_pairs) == 1
        fwd_op, bwd_op = check.collective_pairs[0]
        assert isinstance(fwd_op, AllReduce)
        assert isinstance(bwd_op, AllReduce)

    def test_sp_backward_swaps_allgather_and_reducescatter(self):
        """SP AllGather→compute→ReduceScatter duals swap in backward.

        Forward: [AllGather, ReduceScatter]
        Expected: AllGather↔ReduceScatter duality (dual types swap).
        mesh_dim=TP_DIM ensures collectives target only the TP dimension.
        """
        H, I, S, mesh = self.H, self.I, self.S, self.mesh

        x_sp = _tensor("x_sp", (S, H), P(dp=Shard(0), tp=Shard(1)), mesh, expr="x_sp")
        w_col = _tensor("w_col", (H, I), P(tp=Shard(1)), mesh, expr="w_col")
        w_row = _tensor("w_row", (I, H), P(tp=Shard(0)), mesh, expr="w_row")

        fwd_ops = [
            AllGather(x="x_sp", output="x_full", gather_dim=1, mesh_dim=TP_DIM),
            MatMul(a="x_full", b="w_col", output="h"),
            MatMul(a="h", b="w_row", output="h_p"),
            ReduceScatter(x="h_p", output="y_sp", scatter_dim=1, mesh_dim=TP_DIM),
        ]
        fwd_program = Program("sp_fwd", ops=fwd_ops)
        engine, ctx = self._run_forward_and_record([x_sp, w_col, w_row], fwd_ops)

        bwd_program = engine.generate_backward("y_sp")
        check = engine.verify_gradient_correctness(fwd_program, bwd_program)

        assert len(check.collective_pairs) == 2
        fwd_types = {type(f).__name__ for f, _ in check.collective_pairs}
        bwd_types = {type(b).__name__ for _, b in check.collective_pairs}
        assert "AllGather" in fwd_types
        assert "ReduceScatter" in fwd_types
        assert "AllGather" in bwd_types
        assert "ReduceScatter" in bwd_types

        for name in ["x_sp", "w_col", "w_row"]:
            grad = engine.get_gradient(name)
            assert grad is not None, f"No gradient for {name}"
            assert grad.global_shape == ctx[name].global_shape

    def test_megatron_mlp_backward_grad_shapes_match_inputs(self):
        """All gradient shapes match their corresponding forward input shapes.

        MLP: x→MatMul(w_gate)→SiLU→MatMul(w_up)→Multiply→MatMul(w_down)→AllReduce.
        Every weight gradient must have the correct global_shape.
        """
        H, I, S, mesh = self.H, self.I, self.S, self.mesh

        x = _tensor("x", (S, H), P(), mesh, expr="x")
        w_gate = _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh, expr="w_gate")
        w_up = _tensor("w_up", (H, I), P(tp=Shard(1)), mesh, expr="w_up")
        w_down = _tensor("w_down", (I, H), P(tp=Shard(0)), mesh, expr="w_down")

        fwd_ops = [
            MatMul(a="x", b="w_gate", output="gate_raw"),
            SiLU(x="gate_raw", output="gate"),
            MatMul(a="x", b="w_up", output="up"),
            Multiply(a="gate", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y"),
        ]

        engine, ctx = self._run_forward_and_record(
            [x, w_gate, w_up, w_down], fwd_ops
        )
        engine.generate_backward("y")

        for name, shape in [("w_gate", (H, I)), ("w_up", (H, I)), ("w_down", (I, H))]:
            grad = engine.get_gradient(name)
            assert grad is not None, f"No gradient for {name}"
            assert grad.global_shape == shape

        grad_x = engine.get_gradient("x")
        assert grad_x is not None
        assert grad_x.global_shape == (S, H)

    def test_full_layer_backward_verify_passes(self):
        """verify_gradient_correctness passes for a full Transformer layer.

        Attention (col→row→AllReduce) + MLP (col→row→AllReduce) = 2 AllReduce.
        Backward must produce 2 dual AllReduce ops, all placements consistent.
        """
        H, I, S, mesh = self.H, self.I, self.S, self.mesh

        tensors = [
            _tensor("x", (S, H), P(), mesh, expr="x"),
            _tensor("w_q", (H, H), P(tp=Shard(1)), mesh, expr="w_q"),
            _tensor("w_o", (H, H), P(tp=Shard(0)), mesh, expr="w_o"),
            _tensor("w_gate", (H, I), P(tp=Shard(1)), mesh, expr="w_gate"),
            _tensor("w_up", (H, I), P(tp=Shard(1)), mesh, expr="w_up"),
            _tensor("w_down", (I, H), P(tp=Shard(0)), mesh, expr="w_down"),
        ]

        fwd_ops = [
            MatMul(a="x", b="w_q", output="q"),
            MatMul(a="q", b="w_o", output="attn_p"),
            AllReduce(x="attn_p", output="y_attn"),
            MatMul(a="y_attn", b="w_gate", output="gate_raw"),
            SiLU(x="gate_raw", output="gate"),
            MatMul(a="y_attn", b="w_up", output="up"),
            Multiply(a="gate", b="up", output="h"),
            MatMul(a="h", b="w_down", output="mlp_p"),
            AllReduce(x="mlp_p", output="y"),
        ]
        fwd_program = Program("fwd", ops=fwd_ops)
        engine, ctx = self._run_forward_and_record(tensors, fwd_ops)

        bwd_program = engine.generate_backward("y")
        check = engine.verify_gradient_correctness(fwd_program, bwd_program)

        assert check.passed, f"Errors: {check.errors}"
        assert check.fwd_ops == 9
        assert check.bwd_ops >= 2
        assert all(isinstance(f, AllReduce) for f, _ in check.collective_pairs)

    def test_backward_collective_count_matches_forward(self):
        """Backward has same number of collective duals as forward collectives.

        Two col→row→AllReduce blocks = 2 forward AllReduce ops.
        """
        H, S, mesh = self.H, self.S, self.mesh

        tensors = [
            _tensor("x", (S, H), P(), mesh, expr="x"),
            _tensor("w1_c", (H, H), P(tp=Shard(1)), mesh, expr="w1c"),
            _tensor("w1_r", (H, H), P(tp=Shard(0)), mesh, expr="w1r"),
            _tensor("w2_c", (H, H), P(tp=Shard(1)), mesh, expr="w2c"),
            _tensor("w2_r", (H, H), P(tp=Shard(0)), mesh, expr="w2r"),
        ]

        fwd_ops = [
            MatMul(a="x", b="w1_c", output="h1"),
            MatMul(a="h1", b="w1_r", output="h1_p"),
            AllReduce(x="h1_p", output="y1"),
            MatMul(a="y1", b="w2_c", output="h2"),
            MatMul(a="h2", b="w2_r", output="h2_p"),
            AllReduce(x="h2_p", output="y2"),
        ]
        fwd_program = Program("fwd", ops=fwd_ops)
        engine, ctx = self._run_forward_and_record(tensors, fwd_ops)

        bwd_program = engine.generate_backward("y2")
        check = engine.verify_gradient_correctness(fwd_program, bwd_program)

        assert check.passed, f"Errors: {check.errors}"
        assert check.num_collectives_fwd == 2
        assert check.num_collectives_bwd == 2
        assert len(check.errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Async Communication Overlap
#
# Requirements from CUDA stream semantics:
#   - AllReduceAsync launches on comm stream, returns immediately
#   - Wait synchronizes: output is safe to read only after Wait
#   - OverlapRegion: compute and communication run concurrently
#   - is_async_in_flight tracks handle lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncOverlap256GPU:
    """Verify async communication overlap on 256-GPU mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def test_async_allreduce_resolves_partial_on_4d_mesh(self):
        """AllReduceAsync resolves Partial→Replicate with mixed placements.

        Input: DP Shard(0) + TP Partial (realistic post-row-parallel state).
        After AllReduceAsync+Wait: TP Replicate, DP Shard(0) preserved.
        Uses mesh_dim=TP_DIM to target only the TP dimension.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0), tp=Partial()), mesh)
        ctx = {"x": x}

        ar = AllReduceAsync(x="x", output="x_async", handle="h0", mesh_dim=TP_DIM)
        x_async = ar.apply(ctx)

        assert x_async.is_async_in_flight
        assert x_async._async_handle == "h0"
        assert isinstance(x_async.sharding.placements[TP_DIM], Replicate)
        assert x_async.sharding.placements[DP_DIM] == Shard(0)
        assert x_async.local_shape == (S // DP_SIZE, H)

        w = Wait(handle="h0", tensor="x_async", output="x_done")
        x_done = w.apply(ctx)

        assert not x_done.is_async_in_flight
        assert isinstance(x_done.sharding.placements[TP_DIM], Replicate)
        assert x_done.sharding.placements[DP_DIM] == Shard(0)
        assert x_done.global_shape == (S, H)
        assert x_done.local_shape == (S // DP_SIZE, H)

    def test_overlap_matmul_and_allreduce(self):
        """OverlapRegion: compute next MatMul while AllReduce runs on comm stream.

        Megatron overlap pattern: layer N's AllReduce overlaps with
        layer N+1's column-parallel MatMul.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x_next = _tensor("x_next", (S, H), P(), mesh)
        w_next = _tensor("w_next", (H, H), P(tp=Shard(1)), mesh)
        prev_partial = _tensor("prev_partial", (S, H), P(tp=Partial()), mesh)

        ctx = {"x_next": x_next, "w_next": w_next, "prev_partial": prev_partial}

        overlap = OverlapRegion(
            compute_ops=[MatMul(a="x_next", b="w_next", output="h_next")],
            comm_ops=[AllReduce(x="prev_partial", output="prev_done")],
        )
        overlap.apply(ctx)

        h_next = ctx["h_next"]
        assert h_next.sharding.placements[TP_DIM] == Shard(1)
        assert h_next.local_shape == (S, H // TP_SIZE)

        prev_done = ctx["prev_done"]
        assert isinstance(prev_done.sharding.placements[TP_DIM], Replicate)
        assert prev_done.local_shape == (S, H)

    def test_wait_all_resolves_multiple_handles(self):
        """WaitAll synchronizes multiple async AllReduce ops simultaneously.

        Two independent AllReduceAsync ops (attention + MLP partials),
        WaitAll resolves both in one sync point.
        """
        H, S, mesh = self.H, self.S, self.mesh

        attn_p = _tensor("attn_p", (S, H), P(tp=Partial()), mesh)
        mlp_p = _tensor("mlp_p", (S, H), P(tp=Partial()), mesh)
        ctx = {"attn_p": attn_p, "mlp_p": mlp_p}

        AllReduceAsync(x="attn_p", output="attn_async", handle="h_attn").apply(ctx)
        AllReduceAsync(x="mlp_p", output="mlp_async", handle="h_mlp").apply(ctx)

        assert ctx["attn_async"].is_async_in_flight
        assert ctx["mlp_async"].is_async_in_flight

        WaitAll(
            handles=("h_attn", "h_mlp"),
            tensors=("attn_async", "mlp_async"),
            outputs=("attn_done", "mlp_done"),
        ).apply(ctx)

        for name in ("attn_done", "mlp_done"):
            t = ctx[name]
            assert not t.is_async_in_flight
            assert isinstance(t.sharding.placements[TP_DIM], Replicate)
            assert t.local_shape == (S, H)

    def test_async_handle_lifecycle(self):
        """Async handle is active after AllReduceAsync, cleared after Wait.

        This is the fundamental safety property: reading the output buffer
        before Wait is a data race; the handle tracks this.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(tp=Partial()), mesh)
        ctx = {"x": x}

        assert not x.is_async_in_flight

        AllReduceAsync(x="x", output="x_async", handle="h0").apply(ctx)
        assert ctx["x_async"].is_async_in_flight
        assert ctx["x_async"]._async_handle == "h0"

        Wait(handle="h0", tensor="x_async", output="x_ready").apply(ctx)
        assert not ctx["x_ready"].is_async_in_flight
        assert ctx["x_ready"]._async_handle is None


# ═══════════════════════════════════════════════════════════════════════════════
# 14. MoE Expert Routing
#
# Requirements from Mixture-of-Experts semantics:
#   - TopKGate selects experts per token, preserves batch sharding
#   - MoEDispatch (AllToAll): routes tokens to expert devices,
#     transforms Shard(split_dim) → Shard(concat_dim)
#   - MoECombine (AllToAll): reverses dispatch transformation
#   - Dispatch→Combine roundtrip restores original placement
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoERouting256GPU:
    """Verify MoE expert routing on 256-GPU mesh."""

    NUM_EXPERTS = 8

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def test_moe_dispatch_transforms_shard_on_4d_mesh(self):
        """MoEDispatch transforms Shard(split_dim)→Shard(concat_dim).

        Tokens sharded on dim 0 (batch) across TP dim get re-sharded
        to dim 1 (hidden) after AllToAll dispatch to expert devices.
        """
        H, S, mesh = self.H, self.S, self.mesh

        tokens = _tensor("tokens", (S, H), P(tp=Shard(0)), mesh)
        assert tokens.local_shape == (S // TP_SIZE, H)

        ctx = {"tokens": tokens}
        dispatched = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=self.NUM_EXPERTS, split_dim=0, concat_dim=1,
        ).apply(ctx)

        assert dispatched.sharding.placements[TP_DIM] == Shard(1)
        assert dispatched.local_shape == (S, H // TP_SIZE)
        for dim in [DP_DIM, PP_DIM, CP_DIM]:
            assert isinstance(dispatched.sharding.placements[dim], Replicate)

    def test_moe_dispatch_combine_roundtrip(self):
        """Dispatch→ExpertCompute→Combine restores original placement.

        MoEDispatch: Shard(0)→Shard(1) on TP dim.
        MoECombine (reverse): Shard(1)→Shard(0) on TP dim.
        Round-trip must match original placement and local_shape.
        """
        H, S, mesh = self.H, self.S, self.mesh

        tokens = _tensor("tokens", (S, H), P(tp=Shard(0)), mesh)
        original_placements = tokens.sharding.placements
        original_local = tokens.local_shape

        ctx = {"tokens": tokens}
        MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=self.NUM_EXPERTS, split_dim=0, concat_dim=1,
        ).apply(ctx)

        ExpertCompute(
            x="dispatched", output="expert_out",
            expert_id=0, num_experts=self.NUM_EXPERTS,
        ).apply(ctx)

        combined = MoECombine(
            x="expert_out", output="combined",
            num_experts=self.NUM_EXPERTS, split_dim=1, concat_dim=0,
        ).apply(ctx)

        assert combined.sharding.placements == original_placements
        assert combined.local_shape == original_local

    def test_moe_full_pipeline_gate_dispatch_expert_combine(self):
        """Full MoE layer: TopKGate→Dispatch→expert FFN→Combine.

        End-to-end MoE routing with expert-local MatMul computation.
        """
        H, S, mesh = self.H, self.S, self.mesh

        tokens = _tensor("tokens", (S, H), P(tp=Shard(0)), mesh)
        gate_w = _tensor("gate_w", (H, self.NUM_EXPERTS), P(tp=Shard(0)), mesh)
        expert_w = _tensor("expert_w", (H, H), P(tp=Shard(1)), mesh)

        ctx = {"tokens": tokens, "gate_w": gate_w, "expert_w": expert_w}

        TopKGate(
            x="tokens", gate_weight="gate_w",
            output="gate_scores", indices_output="gate_indices",
            num_experts=self.NUM_EXPERTS, top_k=2,
        ).apply(ctx)
        assert ctx["gate_indices"].global_shape == (S, 2)

        MoEDispatch(
            x="gate_scores", output="dispatched",
            num_experts=self.NUM_EXPERTS, split_dim=0, concat_dim=1,
        ).apply(ctx)
        assert ctx["dispatched"].sharding.placements[TP_DIM] == Shard(1)

        ExpertCompute(
            x="dispatched", output="expert_input",
            expert_id=0, num_experts=self.NUM_EXPERTS,
        ).apply(ctx)

        MatMul(a="expert_input", b="expert_w", output="expert_out").apply(ctx)

        MoECombine(
            x="expert_out", output="combined",
            num_experts=self.NUM_EXPERTS, split_dim=1, concat_dim=0,
        ).apply(ctx)

        combined = ctx["combined"]
        assert combined.sharding.placements[TP_DIM] == Shard(0)
        assert combined.local_shape == (S // TP_SIZE, H)

    def test_moe_dispatch_combine_are_vjp_duals(self):
        """MoEDispatch and MoECombine are VJP duals of each other.

        Dispatch VJP expr mentions MoECombine; Combine VJP mentions MoEDispatch.
        Gradient shapes and placements match forward inputs.
        """
        H, S, mesh = self.H, self.S, self.mesh

        tokens = _tensor("tokens", (S, H), P(tp=Shard(0)), mesh, expr="tokens")
        ctx = {"tokens": tokens}

        dispatch = MoEDispatch(
            x="tokens", output="dispatched",
            num_experts=self.NUM_EXPERTS, split_dim=0, concat_dim=1,
        )
        dispatch.apply(ctx)
        grad_dispatch = dispatch.vjp(ctx, ctx["dispatched"])["tokens"]
        assert grad_dispatch.global_shape == tokens.global_shape
        assert grad_dispatch.local_shape == tokens.local_shape
        assert grad_dispatch.sharding.placements == tokens.sharding.placements
        assert "MoECombine" in grad_dispatch.expr

        expert_out = _tensor("expert_out", (S, H), P(tp=Shard(1)), mesh, expr="expert_out")
        ctx2 = {"expert_out": expert_out}

        combine = MoECombine(
            x="expert_out", output="combined",
            num_experts=self.NUM_EXPERTS, split_dim=1, concat_dim=0,
        )
        combine.apply(ctx2)
        grad_combine = combine.vjp(ctx2, ctx2["combined"])["expert_out"]
        assert grad_combine.global_shape == expert_out.global_shape
        assert grad_combine.local_shape == expert_out.local_shape
        assert "MoEDispatch" in grad_combine.expr


# ═══════════════════════════════════════════════════════════════════════════════
# 15. mesh_dim Targeting for Collectives
#
# Verifies that mesh_dim restricts collective placement transforms to a
# single mesh dimension, fixing the multi-dim bug where AllGather/ReduceScatter
# would modify ALL matching placements instead of just the targeted one.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeshDimCollective256GPU:
    """Verify mesh_dim targeting on 256-GPU 4D mesh."""

    def setup_method(self):
        self.H = LLAMA_70B["hidden"]
        self.S = LLAMA_70B["seq_len"]
        self.mesh = _mesh_4d()

    def test_allgather_mesh_dim_targets_only_specified_dim(self):
        """AllGather with mesh_dim gathers only the targeted mesh dimension.

        Both dp and tp shard tensor dim 0, but mesh_dim=TP_DIM means
        only the TP placement is gathered — DP Shard(0) survives.
        Without mesh_dim, both would be gathered (the old bug).
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0), tp=Shard(0)), mesh)
        assert x.local_shape == (S // DP_SIZE // TP_SIZE, H)

        ctx = {"x": x}
        y = AllGather(x="x", output="y", gather_dim=0, mesh_dim=TP_DIM).apply(ctx)
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert y.local_shape == (S // DP_SIZE, H)

    def test_allgather_without_mesh_dim_gathers_all_matching(self):
        """AllGather without mesh_dim gathers ALL mesh dims with matching Shard.

        Legacy behavior: both dp=Shard(0) and tp=Shard(0) become Replicate.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0), tp=Shard(0)), mesh)
        ctx = {"x": x}
        y = AllGather(x="x", output="y", gather_dim=0).apply(ctx)
        assert isinstance(y.sharding.placements[DP_DIM], Replicate)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)
        assert y.local_shape == (S, H)

    def test_reducescatter_mesh_dim_targets_exact_dim(self):
        """ReduceScatter with mesh_dim targets exactly the specified mesh dim.

        Without mesh_dim, first-match picks dp=Replicate (wrong).
        With mesh_dim=TP_DIM, it correctly targets tp=Partial.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Replicate(), tp=Partial()), mesh)
        ctx = {"x": x}

        y = ReduceScatter(x="x", output="y", scatter_dim=0, mesh_dim=TP_DIM).apply(ctx)
        assert isinstance(y.sharding.placements[DP_DIM], Replicate)
        assert y.sharding.placements[TP_DIM] == Shard(0)
        assert y.local_shape == (S // TP_SIZE, H)

    def test_reducescatter_without_mesh_dim_uses_first_match(self):
        """ReduceScatter without mesh_dim picks the FIRST Replicate/Partial.

        Legacy first-match: dp=Replicate comes first, gets transformed.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Replicate(), tp=Partial()), mesh)
        ctx = {"x": x}

        y = ReduceScatter(x="x", output="y", scatter_dim=0).apply(ctx)
        assert y.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(y.sharding.placements[TP_DIM], Partial)

    def test_allreduce_mesh_dim_resolves_only_targeted_partial(self):
        """AllReduce with mesh_dim resolves only the targeted Partial.

        Two Partial dims (dp and tp): mesh_dim=TP_DIM resolves only tp.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Partial(), tp=Partial()), mesh)
        ctx = {"x": x}

        y = AllReduce(x="x", output="y", mesh_dim=TP_DIM).apply(ctx)
        assert isinstance(y.sharding.placements[DP_DIM], Partial)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)

    def test_allreduce_without_mesh_dim_resolves_all_partials(self):
        """AllReduce without mesh_dim resolves ALL Partial dims."""
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Partial(), tp=Partial()), mesh)
        ctx = {"x": x}

        y = AllReduce(x="x", output="y").apply(ctx)
        assert isinstance(y.sharding.placements[DP_DIM], Replicate)
        assert isinstance(y.sharding.placements[TP_DIM], Replicate)

    def test_async_allreduce_mesh_dim_with_shard_and_partial(self):
        """AllReduceAsync with mesh_dim preserves Shard on non-targeted dims.

        Previously bugged: local_shape was hardcoded to global_shape.
        Now correctly computes local_shape from output sharding spec.
        """
        H, S, mesh = self.H, self.S, self.mesh

        x = _tensor("x", (S, H), P(dp=Shard(0), tp=Partial()), mesh)
        ctx = {"x": x}

        ar = AllReduceAsync(x="x", output="x_async", handle="h0", mesh_dim=TP_DIM)
        x_async = ar.apply(ctx)

        assert x_async.sharding.placements[DP_DIM] == Shard(0)
        assert isinstance(x_async.sharding.placements[TP_DIM], Replicate)
        assert x_async.local_shape == (S // DP_SIZE, H)
        assert x_async.is_async_in_flight

    def test_mesh_dim_propagated_in_clone_with_names(self):
        """mesh_dim is preserved through clone_with_names."""
        ag = AllGather(x="a", output="b", gather_dim=0, mesh_dim=TP_DIM)
        ag2 = ag.clone_with_names({"a": "c"}, "d")
        assert ag2.mesh_dim == TP_DIM
        assert ag2.x == "c"
        assert ag2.output == "d"

        rs = ReduceScatter(x="a", output="b", scatter_dim=1, mesh_dim=DP_DIM)
        rs2 = rs.clone_with_names({"a": "c"}, "d")
        assert rs2.mesh_dim == DP_DIM

        ar = AllReduce(x="a", output="b", mesh_dim=CP_DIM)
        ar2 = ar.clone_with_names({"a": "c"}, "d")
        assert ar2.mesh_dim == CP_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
