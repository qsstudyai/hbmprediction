"""M1：attention + pure-EP routed experts 的 MoE decoder 图。"""

from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .dense import build_dense_decoder, hyper_connection_ops


def build_moe_decoder(dims: DimTable) -> LayerSpec:
    if dims.n_experts <= 0 or dims.topk <= 0 or dims.moe_F <= 0:
        raise ValueError("MoE 图要求 n_experts、topk、moe_F 均为正数")

    attention = tuple(build_dense_decoder(dims).ops[:6])
    hidden = TensorRef("h1", ("S", "B", "H"), shard={0: "sp"})
    logits = TensorRef("logits", ("S", "B", "n_experts"))
    dispatched = TensorRef("dispatched", ("T_routed", "H"), shard={0: "ep"})
    w1 = TensorRef(
        "expert_w1",
        ("n_experts", "H", "2*moe_F"),
        shard={0: "ep"},
        is_weight=True,
    )
    gate = TensorRef("expert_gate", ("T_routed", "2*moe_F"), shard={0: "ep"})
    act = TensorRef("expert_act", ("T_routed", "moe_F"), shard={0: "ep"})
    w2 = TensorRef(
        "expert_w2",
        ("n_experts", "moe_F", "H"),
        shard={0: "ep"},
        is_weight=True,
    )
    expert_out = TensorRef("expert_out", ("T_routed", "H"), shard={0: "ep"})
    combined = TensorRef("combined", ("S", "B", "H"), shard={0: "sp"})

    routed_ffn = (
        OpSpec("router", OpType.MOE_ROUTER, (hidden,), logits, saves=(logits,)),
        OpSpec(
            "dispatch",
            OpType.DISPATCH,
            (hidden,),
            dispatched,
            saves=(dispatched,),
        ),
        OpSpec(
            "expert_fc1",
            OpType.MOE_GEMM,
            (dispatched, w1),
            gate,
            params=(w1,),
            saves=(dispatched,),
        ),
        OpSpec(
            "expert_swiglu",
            OpType.ELEMENTWISE,
            (gate,),
            act,
            saves=(gate,),
        ),
        OpSpec(
            "expert_fc2",
            OpType.MOE_GEMM,
            (act, w2),
            expert_out,
            params=(w2,),
            saves=(act,),
        ),
        OpSpec(
            "combine",
            OpType.COMBINE,
            (expert_out,),
            combined,
            saves=(combined,),
        ),
    )
    return LayerSpec(attention + routed_ffn + hyper_connection_ops(dims, "combined"))
