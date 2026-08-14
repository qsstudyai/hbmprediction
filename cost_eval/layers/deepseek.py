"""MindFormers PyNative DeepSeek-V3 MLA, shared-expert and MTP graphs."""

from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .dense import hyper_connection_ops


def _norm_weight(name: str, dim="H") -> TensorRef:
    return TensorRef(name, (dim,), is_weight=True, dtype_bytes=4)


def build_mla_attention(dims: DimTable) -> tuple[OpSpec, ...]:
    if not all((dims.q_lora_rank, dims.kv_lora_rank,
                dims.qk_nope_head_dim, dims.qk_rope_head_dim,
                dims.v_head_dim)):
        raise ValueError("MLA requires q/kv ranks and qk/v head dimensions")
    x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
    ln1 = TensorRef("ln1", ("S", "B", "H"), {0: "sp"})
    ln1_w = _norm_weight("ln1_weight")
    combo_dim = "q_lora_rank+kv_lora_rank+qk_rope_head_dim"
    combo_w = TensorRef("mla_qkv_down_w", ("H", combo_dim), is_weight=True)
    combo = TensorRef("mla_qkv_combo", ("S", "B", combo_dim), {0: "sp"})
    qa = TensorRef("mla_q_a", ("S", "B", "q_lora_rank"), {0: "sp"})
    qan = TensorRef("mla_q_a_norm", ("S", "B", "q_lora_rank"), {0: "sp"})
    qn_w = _norm_weight("mla_q_norm_weight", "q_lora_rank")
    ckv = TensorRef("mla_compressed_kv", ("S", "B", "kv_lora_rank"), {0: "sp"})
    ckvn = TensorRef("mla_compressed_kv_norm", ("S", "B", "kv_lora_rank"), {0: "sp"})
    kn_w = _norm_weight("mla_kv_norm_weight", "kv_lora_rank")
    kpe_local = TensorRef("mla_k_pe", ("S", "B", "qk_rope_head_dim"), {0: "sp"})

    qdim = "n_heads*(qk_nope_head_dim+qk_rope_head_dim)"
    q_up_w = TensorRef("mla_q_up_w", ("q_lora_rank", qdim), {1: "tp"}, True)
    qa_full = TensorRef("mla_q_a_norm", ("S", "B", "q_lora_rank"))
    query = TensorRef("mla_query", ("S", "B", "n_heads", "qk_nope_head_dim+qk_rope_head_dim"), {2: "tp"})
    query_rope = TensorRef("mla_query_rope", ("S", "B", "n_heads", "qk_nope_head_dim+qk_rope_head_dim"), {2: "tp"})

    kvdim = "n_heads*(qk_nope_head_dim+v_head_dim)"
    kv_up_w = TensorRef("mla_kv_up_w", ("kv_lora_rank", kvdim), {1: "tp"}, True)
    ckv_full = TensorRef("mla_compressed_kv_norm", ("S", "B", "kv_lora_rank"))
    kv = TensorRef("mla_kv", ("S", "B", "n_heads", "qk_nope_head_dim+v_head_dim"), {2: "tp"})
    key = TensorRef("mla_key", ("S", "B", "n_heads", "qk_nope_head_dim+qk_rope_head_dim"), {2: "tp"})
    value = TensorRef("mla_value", ("S", "B", "n_heads", "v_head_dim"), {2: "tp"})
    kpe_full = TensorRef("mla_k_pe", ("S", "B", "qk_rope_head_dim"))
    attn = TensorRef("mla_attn", ("S", "B", "n_heads", "v_head_dim"), {2: "tp"})
    lse = TensorRef("mla_lse", ("S", "B", "n_heads"), {2: "tp"})
    out_w = TensorRef("mla_o_w", ("n_heads*v_head_dim", "H"), {0: "tp"}, True)
    partial = TensorRef("mla_o", ("S", "B", "H"), partial="tp")
    sharded = TensorRef("mla_o", ("S", "B", "H"), {0: "sp"})
    h1 = TensorRef("h1", ("S", "B", "H"), {0: "sp"})

    return (
        OpSpec("ln1", OpType.NORM, (x, ln1_w), ln1, (ln1_w,), (x,), module_paths=("input_layernorm",)),
        OpSpec("mla_qkv_down", OpType.MATMUL, (ln1, combo_w), combo, (combo_w,), (ln1,),
               module_paths=("self_attention.linear_qkv",)),
        OpSpec("mla_q_extract", OpType.ELEMENTWISE, (combo,), qa),
        OpSpec("mla_q_norm", OpType.NORM, (qa, qn_w), qan, (qn_w,), (qa,),
               module_paths=("self_attention.q_layernorm",)),
        OpSpec("mla_kv_extract", OpType.ELEMENTWISE, (combo,), ckv),
        OpSpec("mla_kv_norm", OpType.NORM, (ckv, kn_w), ckvn, (kn_w,), (ckv,),
               module_paths=("self_attention.k_layernorm",)),
        OpSpec("mla_kpe_extract", OpType.ELEMENTWISE, (combo,), kpe_local),
        OpSpec("mla_q_up", OpType.MATMUL, (qa_full, q_up_w), query, (q_up_w,), (qa_full,),
               module_paths=("self_attention.linear_qb",)),
        OpSpec("mla_rope_q", OpType.ROPE, (query,), query_rope, saves=(query,),
               module_paths=("self_attention.apply_rotary_emb_q",)),
        OpSpec("mla_kv_up", OpType.MATMUL, (ckv_full, kv_up_w), kv, (kv_up_w,), (ckv_full,),
               module_paths=("self_attention.linear_kvb",)),
        OpSpec("mla_kv_split", OpType.ELEMENTWISE, (kv,), value, saves=(kv,)),
        OpSpec("mla_rope_k", OpType.ROPE, (kpe_full, kv), key, saves=(kpe_full,),
               module_paths=("self_attention.apply_rotary_emb_k",)),
        OpSpec("mla_flash", OpType.FLASH_ATTN, (query_rope, key, value), attn,
               saves=(query_rope, key, value, attn, lse),
               workspace="S*B*n_heads*v_head_dim*dtype_bytes",
               attrs={"n_heads": dims.n_heads, "head_dim": dims.v_head_dim},
               module_paths=("self_attention.core_attention",)),
        OpSpec("mla_o_proj", OpType.MATMUL, (attn, out_w), partial, (out_w,), (attn,),
               module_paths=("self_attention.linear_proj",)),
        OpSpec("add1", OpType.ELEMENTWISE, (sharded, x), h1),
    )


def _moe_tail(dims: DimTable) -> tuple[OpSpec, ...]:
    h1 = TensorRef("h1", ("S", "B", "H"), {0: "sp"})
    ln2_w = _norm_weight("ln2_weight")
    ln2 = TensorRef("ln2", ("S", "B", "H"), {0: "sp"})
    router_w = TensorRef("router_w", ("H", "n_experts"), is_weight=True, dtype_bytes=4)
    logits = TensorRef("logits", ("S", "B", "n_experts"), dtype_bytes=4)
    dispatched = TensorRef("dispatched", ("T_routed", "H"), {0: "ep"})
    w1 = TensorRef("expert_w1", ("n_experts", "H", "2*moe_F"), {0: "ep"}, True)
    gate = TensorRef("expert_gate", ("T_routed", "2*moe_F"), {0: "ep"})
    act = TensorRef("expert_act", ("T_routed", "moe_F"), {0: "ep"})
    w2 = TensorRef("expert_w2", ("n_experts", "moe_F", "H"), {0: "ep"}, True)
    expert_out = TensorRef("expert_out", ("T_routed", "H"), {0: "ep"})
    routed = TensorRef("routed_combined", ("S", "B", "H"), {0: "sp"})
    ops = [
        OpSpec("ln2", OpType.NORM, (h1, ln2_w), ln2, (ln2_w,), (h1,), module_paths=("pre_mlp_layernorm",)),
        OpSpec("router", OpType.MOE_ROUTER, (ln2, router_w), logits, (router_w,), (logits,), module_paths=("mlp.router",)),
        OpSpec("dispatch", OpType.DISPATCH, (ln2,), dispatched, saves=(dispatched,), module_paths=("mlp.experts.dispatch",)),
        OpSpec("expert_fc1", OpType.MOE_GEMM, (dispatched, w1), gate, (w1,), (dispatched,), module_paths=("mlp.experts.weight1",)),
        OpSpec("expert_swiglu", OpType.ELEMENTWISE, (gate,), act, saves=(gate,)),
        OpSpec("expert_fc2", OpType.MOE_GEMM, (act, w2), expert_out, (w2,), (act,), module_paths=("mlp.experts.weight2",)),
        OpSpec("combine", OpType.COMBINE, (expert_out,), routed, saves=(routed,), module_paths=("mlp.experts.combine",)),
    ]
    final_input = routed
    if dims.n_shared and dims.shared_F:
        sw1 = TensorRef("shared_fc1_w", ("H", "2*shared_F"), is_weight=True)
        sgate = TensorRef("shared_gate_act", ("S", "B", "2*shared_F"), {0: "sp"})
        sact = TensorRef("shared_act", ("S", "B", "shared_F"), {0: "sp"})
        sw2 = TensorRef("shared_fc2_w", ("shared_F", "H"), is_weight=True)
        sout = TensorRef("shared_out", ("S", "B", "H"), {0: "sp"})
        ops.extend([
            OpSpec("shared_fc1", OpType.MATMUL, (ln2, sw1), sgate, (sw1,), (ln2,), module_paths=("mlp.shared_experts.linear_fc1",)),
            OpSpec("shared_swiglu", OpType.ELEMENTWISE, (sgate,), sact, saves=(sgate,)),
            OpSpec("shared_fc2", OpType.MATMUL, (sact, sw2), sout, (sw2,), (sact,), module_paths=("mlp.shared_experts.linear_fc2",)),
        ])
        if dims.use_shared_expert_gating:
            sgw = TensorRef("shared_expert_gate_w", ("H", 1), is_weight=True, dtype_bytes=4)
            sg = TensorRef("shared_expert_score", ("S", "B", 1), {0: "sp"}, dtype_bytes=4)
            gated = TensorRef("shared_gated", ("S", "B", "H"), {0: "sp"})
            ops.extend([
                OpSpec("shared_score", OpType.MATMUL, (ln2, sgw), sg, (sgw,), (ln2,), module_paths=("mlp.shared_experts.shared_experts_gate",)),
                OpSpec("shared_scale", OpType.ELEMENTWISE, (sout, sg), gated, saves=(sout, sg)),
            ])
            sout = gated
        merged = TensorRef("moe_merged", ("S", "B", "H"), {0: "sp"})
        ops.append(OpSpec("shared_add", OpType.ELEMENTWISE, (routed, sout), merged, saves=(routed, sout)))
        final_input = merged
    h2 = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    ops.append(OpSpec("add2", OpType.ELEMENTWISE, (final_input, h1), h2))
    return tuple(ops)


def _dense_tail() -> tuple[OpSpec, ...]:
    h1 = TensorRef("h1", ("S", "B", "H"), {0: "sp"})
    ln2_w = _norm_weight("ln2_weight")
    ln2 = TensorRef("ln2", ("S", "B", "H"), {0: "sp"})
    fc1_w = TensorRef("fc1_w", ("H", "2*F"), {1: "tp"}, True)
    gate = TensorRef("gate", ("S", "B", "2*F"), {2: "tp"})
    act = TensorRef("act", ("S", "B", "F"), {2: "tp"})
    fc2_w = TensorRef("fc2_w", ("F", "H"), {0: "tp"}, True)
    partial = TensorRef("o2", ("S", "B", "H"), partial="tp")
    sharded = TensorRef("o2", ("S", "B", "H"), {0: "sp"})
    h2 = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    return (
        OpSpec("ln2", OpType.NORM, (h1, ln2_w), ln2, (ln2_w,), (h1,),
               module_paths=("pre_mlp_layernorm",)),
        OpSpec("fc1", OpType.MATMUL, (ln2, fc1_w), gate, (fc1_w,), (ln2,),
               module_paths=("mlp.linear_fc1",)),
        OpSpec("swiglu", OpType.ELEMENTWISE, (gate,), act, saves=(gate,)),
        OpSpec("fc2", OpType.MATMUL, (act, fc2_w), partial, (fc2_w,), (act,),
               module_paths=("mlp.linear_fc2",)),
        OpSpec("add2", OpType.ELEMENTWISE, (sharded, h1), h2),
    )


def build_mla_dense_decoder(dims: DimTable, include_hc: bool = True) -> LayerSpec:
    ops = build_mla_attention(dims) + _dense_tail()
    return LayerSpec(ops + (hyper_connection_ops(dims, "h2") if include_hc else ()))


def build_mla_moe_decoder(dims: DimTable, include_hc: bool = True) -> LayerSpec:
    ops = build_mla_attention(dims) + _moe_tail(dims)
    return LayerSpec(ops + (hyper_connection_ops(dims, "h2") if include_hc else ()))


def build_mtp_layer(dims: DimTable) -> LayerSpec:
    embedding = TensorRef("mtp_embedding", ("S", "B", "H"), {0: "sp"}, storage_id="token_embedding_activation")
    prev = TensorRef("mtp_prev_hidden", ("S", "B", "H"), {0: "sp"})
    ew = _norm_weight("mtp_enorm_weight")
    hw = _norm_weight("mtp_hnorm_weight")
    enorm = TensorRef("mtp_enorm", ("S", "B", "H"), {0: "sp"})
    hnorm = TensorRef("mtp_hnorm", ("S", "B", "H"), {0: "sp"})
    joined = TensorRef("mtp_joined", ("S", "B", "2*H"), {0: "sp"})
    proj_w = TensorRef("mtp_eh_proj_w", ("2*H", "H"), is_weight=True)
    x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
    outer = (
        OpSpec("mtp_enorm", OpType.NORM, (embedding, ew), enorm, (ew,), (embedding,), module_paths=("enorm",)),
        OpSpec("mtp_hnorm", OpType.NORM, (prev, hw), hnorm, (hw,), (prev,), module_paths=("hnorm",)),
        OpSpec("mtp_concat", OpType.ELEMENTWISE, (enorm, hnorm), joined, saves=(enorm, hnorm)),
        OpSpec("mtp_eh_proj", OpType.MATMUL, (joined, proj_w), x, (proj_w,), (joined,), module_paths=("eh_proj",)),
    )
    # The MTP block owns its post-MLP HC path; do not duplicate the decoder
    # block's HC ops inside the nested MTP layer.
    inner = build_mla_moe_decoder(dims, include_hc=False).ops
    final_w = _norm_weight("mtp_final_norm_weight")
    inner_out = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    final = TensorRef("mtp_final", ("S", "B", "H"), {0: "sp"})
    return LayerSpec(outer + tuple(inner) + (
        OpSpec("mtp_final_norm", OpType.NORM, (inner_out, final_w), final, (final_w,), (inner_out,), module_paths=("final_layernorm",)),
    ) + hyper_connection_ops(dims, "mtp_final"))
