"""
Example: Context Parallelism — Ring Attention with Flash Attention.

Models full ring attention where:
  - Q is replicated on each device (full sequence context)
  - K, V are sharded on seq_len dimension
  - Devices communicate in a ring: send K_i, V_i to next device
  - Each device computes partial attention with local + received K, V
  - Final output requires AllReduce across CP ranks

Ring Pattern (4 devices):
  Device 0: Q, K0(seq[0:4]), V0(seq[0:4]) → attn0_partial0
            Recv K1, V1 from device 1 → attn0_partial1
            Recv K2, V2 from device 2 → attn0_partial2
            Recv K3, V3 from device 3 → attn0_partial3
            AllReduce → O0

Simplified for 2-device prototype:
  Device 0: Q, K0(S/2), V0(S/2) → Send K0,V0 → Recv K1,V1
  Device 1: Q, K1(S/2), V1(S/2) → Send K1,V1 → Recv K0,V0
  Each computes: O_partial = FA(Q, K_local, V_local) + FA(Q, K_remote, V_remote)
  AllReduce → O_replicated
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    FlashAttention,
    Add,
    AllReduce,
    Send,
    Recv,
    ir_to_str,
)
from verifier.executor import MultiDeviceExecutor
from verifier.autograd import AutogradEngine
from verifier.solver import DistributedVerifier


def build_cp_ring_attention(num_devices: int = 2):
    """Build a ring attention program for context parallelism.

    Args:
        num_devices: Number of CP ranks (default: 2)

    Each device holds:
      - Q: full (replicated)
      - K: shard on seq_len (dim=1)
      - V: shard on seq_len (dim=1)

    Ring communication:
      1. Compute local FA: O_local = FA(Q, K_i, V_i)
      2. Send K_i, V_i to next rank
      3. Recv K_j, V_j from previous rank
      4. Compute remote FA: O_remote = FA(Q, K_j, V_j)
      5. O = O_local + O_remote (partial sum)
      6. AllReduce → O_final
    """
    B, S, H, D = 2, 8, 4, 16  # batch, seq_len, heads, head_dim
    S_per_device = S // num_devices

    mesh = DeviceMesh(shape=(num_devices,), dim_names=("cp",))

    # ── Tensors ──
    q = TensorState(
        name="q",
        global_shape=(B, S, H, D),
        local_shape=(B, S, H, D),
        sharding=ShardingSpec(placements=(Replicate(),), mesh=mesh),
        expr="q",
        requires_grad=True,
    )
    k = TensorState(
        name="k",
        global_shape=(B, S, H, D),
        local_shape=(B, S_per_device, H, D),
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="k",
        requires_grad=True,
    )
    v = TensorState(
        name="v",
        global_shape=(B, S, H, D),
        local_shape=(B, S_per_device, H, D),
        sharding=ShardingSpec(placements=(Shard(dim=1),), mesh=mesh),
        expr="v",
        requires_grad=True,
    )

    # ── Forward program (2-device ring) ──
    fwd = Program(name="cp_ring_attn_fwd")

    # Step 1: Local FA on each device
    fwd.add(FlashAttention(q="q", k="k", v="v", output="o_local"))

    # Step 2: Ring rotation — Send local K, V to next device
    fwd.add(Send(x="k", output="k_sent", src=0, dst=1, stage=0, microbatch_id=0))
    fwd.add(Send(x="v", output="v_sent", src=0, dst=1, stage=0, microbatch_id=0))
    fwd.add(Send(x="k", output="k_sent_1", src=1, dst=0, stage=0, microbatch_id=0))
    fwd.add(Send(x="v", output="v_sent_1", src=1, dst=0, stage=0, microbatch_id=0))

    # Step 3: Recv remote K, V
    fwd.add(Recv(x="k_sent_1", output="k_remote_0", src=1, dst=0, stage=0, microbatch_id=0))
    fwd.add(Recv(x="v_sent_1", output="v_remote_0", src=1, dst=0, stage=0, microbatch_id=0))
    fwd.add(Recv(x="k_sent", output="k_remote_1", src=0, dst=1, stage=0, microbatch_id=0))
    fwd.add(Recv(x="v_sent", output="v_remote_1", src=0, dst=1, stage=0, microbatch_id=0))

    # Step 4: Remote FA (using received K, V)
    # Device 0 uses received K, V from device 1
    fwd.add(FlashAttention(
        q="q", k="k_remote_0", v="v_remote_0", output="o_remote_0"
    ))
    # Device 1 uses received K, V from device 0
    fwd.add(FlashAttention(
        q="q", k="k_remote_1", v="v_remote_1", output="o_remote_1"
    ))

    # Step 5: Accumulate partial outputs
    fwd.add(Add(a="o_local", b="o_remote_0", output="o_partial_0"))
    fwd.add(Add(a="o_local", b="o_remote_1", output="o_partial_1"))

    # Step 6: AllReduce across CP ranks
    # Both devices have partial results that need reduction
    fwd.add(AllReduce(x="o_partial_0", output="o_0", op_type="sum"))
    fwd.add(AllReduce(x="o_partial_1", output="o_1", op_type="sum"))

    tensors = {"q": q, "k": k, "v": v}

    return fwd, tensors, mesh


def demo_cp_ring_attn():
    print("=" * 60)
    print("  DTENSOR-VERIFIER: CP Ring Attention Demo")
    print("=" * 60)

    fwd, tensors, mesh = build_cp_ring_attention(num_devices=2)

    print("\nForward program (2-device ring attention):")
    print(ir_to_str(fwd))

    n_coll = len(fwd.collectives)
    n_p2p = len(fwd.p2p_ops)
    n_comp = len(fwd.compute_ops)
    print(f"\n  Ops: {len(fwd.ops)} total")
    print(f"    Compute:     {n_comp} (FlashAttention + Add)")
    print(f"    Collectives: {n_coll} (AllReduce)")
    print(f"    P2P:         {n_p2p} (Send/Recv)")

    # ── Execute ──
    executor = MultiDeviceExecutor(mesh)
    executor.register_tensor(tensors["q"])
    executor.register_tensor(tensors["k"])
    executor.register_tensor(tensors["v"])
    result = executor.run_program(fwd)

    print("\nDevice 0 final tensors:")
    for name, t in executor.devices[0].tensors.items():
        partial_str = " [PARTIAL]" if t.partial else ""
        cp_str = f" cp_rank={t.cp_rank}" if t.cp_rank is not None else ""
        print(f"  {name}: shape={t.global_shape}{partial_str}{cp_str}")

    print("\nDevice 1 final tensors:")
    for name, t in executor.devices[1].tensors.items():
        partial_str = " [PARTIAL]" if t.partial else ""
        print(f"  {name}: shape={t.global_shape}{partial_str}")

    # ── Verify ──
    verifier = DistributedVerifier()

    # Check postcondition: output tensors should NOT be partial
    for dev_id in range(mesh.num_devices):
        o_name = f"o_{dev_id}"
        o_tensor = executor.get_tensor(o_name, device_id=dev_id)
        if o_tensor:
            verifier.verify_postcondition(o_tensor, expected_partial=False)

    verifier.verify_communication_legality(
        fwd, tensor_states=result,
        multi_device_states=executor.final_state(),
    )

    print("\n" + verifier.summary())

    # Verify key properties
    o0 = executor.get_tensor("o_0", device_id=0)
    o1 = executor.get_tensor("o_1", device_id=1)

    checks = []
    if o0:
        checks.append(("o_0.partial == False", not o0.partial))
    if o1:
        checks.append(("o_1.partial == False", not o1.partial))

    # Check Send/Recv pairing
    sends = [op for op in fwd.ops if isinstance(op, Send)]
    recvs = [op for op in fwd.ops if isinstance(op, Recv)]
    checks.append(("Send/Recv balanced", len(sends) == len(recvs)))

    print("\nKey checks:")
    all_ok = True
    for desc, ok in checks:
        status = "PASSED" if ok else "FAILED"
        if not ok:
            all_ok = False
        print(f"  {desc}: {status}")

    print(f"\n  Overall CP verification: {'PASSED' if all_ok else 'FAILED'}")

    return all_ok


if __name__ == "__main__":
    demo_cp_ring_attn()
