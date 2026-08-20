"""MindFormers DeepSeek-V4 legacy-MLA and hybrid-CSA memory graphs."""

from __future__ import annotations

from dataclasses import replace

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .deepseek import _moe_tail, build_mla_attention


def _norm_weight(name: str, dim: str = "H") -> TensorRef:
    return TensorRef(name, (dim,), is_weight=True, dtype_bytes=4)


def _with_hash_router(layer: LayerSpec, enabled: bool) -> LayerSpec:
    if not enabled:
        return layer
    table = TensorRef(
        "router_tid2eid",
        ("vocab", "topk"),
        is_weight=True,
        dtype_bytes=4,
        trainable=False,
        swappable=False,
        recomputable=False,
        fsdp_replicated=True,
    )
    ops = []
    found = False
    for op in layer.ops:
        if op.name == "router":
            found = True
            op = replace(
                op,
                params=tuple(op.params) + (table,),
                attrs={**dict(op.attrs), "hash_router": True},
                module_paths=tuple(op.module_paths) + ("mlp.router.tid2eid",),
            )
        ops.append(op)
    if not found:
        raise ValueError("DeepSeek-V4 hash routing requires a MoE router")
    return LayerSpec(tuple(ops))


def build_v4_legacy_mla_moe_decoder(
    dims: DimTable, hash_router: bool = False
) -> LayerSpec:
    """Archived V4 configs explicitly selecting the legacy MLA runtime."""

    attention = build_mla_attention(dims)
    tail = _moe_tail(dims)
    ops = _with_mhc(dims, attention, tail) if dims.enable_hyper_connections else attention + tail
    return _with_hash_router(LayerSpec(ops), hash_router)


def _mhc_pre(
    dims: DimTable, prefix: str, stream_name: str, output_name: str
) -> tuple[OpSpec, TensorRef, TensorRef, TensorRef]:
    """One source-exact FP32 mHC pre/Sinkhorn contract."""

    if dims.hc_mult < 1:
        raise ValueError("DeepSeek-V4 mHC requires hc_mult >= 1")
    mapping_dim = "2*hc_mult+hc_mult*hc_mult"
    streams = TensorRef(
        stream_name, ("S", "B", "hc_mult*H"), {0: "sp"}
    )
    mapping = TensorRef(
        f"{prefix}_mapping_w",
        ("hc_mult*H", mapping_dim),
        is_weight=True,
        dtype_bytes=4,
    )
    alpha_pre = TensorRef(
        f"{prefix}_alpha_pre", (1,), is_weight=True, dtype_bytes=4,
        fsdp_replicated=True,
    )
    alpha_post = TensorRef(
        f"{prefix}_alpha_post", (1,), is_weight=True, dtype_bytes=4,
        fsdp_replicated=True,
    )
    alpha_res = TensorRef(
        f"{prefix}_alpha_res", (1,), is_weight=True, dtype_bytes=4,
        fsdp_replicated=True,
    )
    bias = TensorRef(
        f"{prefix}_bias", (mapping_dim,), is_weight=True, dtype_bytes=4,
        fsdp_replicated=True,
    )
    rms_weight = TensorRef(
        f"{prefix}_rms_weight",
        ("hc_mult*H",),
        is_weight=True,
        dtype_bytes=4,
        trainable=False,
    )
    aggregated = TensorRef(output_name, ("S", "B", "H"), {0: "sp"})
    h_res = TensorRef(
        f"{prefix}_h_res", ("S", "B", "hc_mult", "hc_mult"),
        {0: "sp"}, dtype_bytes=4,
    )
    h_post = TensorRef(
        f"{prefix}_h_post", ("S", "B", "hc_mult", 1),
        {0: "sp"}, dtype_bytes=4,
    )
    op = OpSpec(
        f"{prefix}_pre",
        OpType.MATMUL,
        (streams, mapping),
        aggregated,
        params=(mapping, alpha_pre, alpha_post, alpha_res, bias, rms_weight),
        saves=(streams, h_res, h_post),
        workspace="S*B*(hc_mult*hc_mult+2*hc_mult)*4",
        attrs={"hc_mult": dims.hc_mult, "compute_dtype_bytes": 4},
        module_paths=(prefix,),
    )
    return op, streams, h_res, h_post


def _mhc_post(
    prefix: str,
    streams: TensorRef,
    h_res: TensorRef,
    h_post: TensorRef,
    sublayer_output: TensorRef,
    output_name: str,
) -> OpSpec:
    output = TensorRef(
        output_name, ("S", "B", "hc_mult*H"), {0: "sp"}
    )
    return OpSpec(
        f"{prefix}_post",
        OpType.ELEMENTWISE,
        (h_res, h_post, streams, sublayer_output),
        output,
        saves=(h_res, h_post, streams, sublayer_output),
        workspace="S*B*hc_mult*H*4",
        attrs={"hyper_connection": True, "compute_dtype_bytes": 4},
        module_paths=(f"{prefix}.output_cell",),
    )


def _with_mhc(
    dims: DimTable,
    attention: tuple[OpSpec, ...],
    tail: tuple[OpSpec, ...],
) -> tuple[OpSpec, ...]:
    """Wrap attention and FFN independently in the V4 two-module mHC path."""

    if not attention or attention[-1].name != "add1":
        raise ValueError("V4 attention graph lacks the residual boundary required by mHC")
    if not tail or tail[-1].name != "add2":
        raise ValueError("V4 MoE graph lacks the residual boundary required by mHC")
    attn_pre, input_streams, attn_h_res, attn_h_post = _mhc_pre(
        dims, "attn_hc", "v4_input_streams", "x"
    )
    # The normal residual add is replaced by the mHC output-cell update.
    attn_output = attention[-1].inputs[0]
    attn_post = _mhc_post(
        "attn_hc", input_streams, attn_h_res, attn_h_post,
        attn_output, "h1_streams",
    )
    ffn_pre, ffn_streams, ffn_h_res, ffn_h_post = _mhc_pre(
        dims, "ffn_hc", "h1_streams", "h1"
    )
    ffn_output = tail[-1].inputs[0]
    ffn_post = _mhc_post(
        "ffn_hc", ffn_streams, ffn_h_res, ffn_h_post,
        ffn_output, "h2",
    )
    return (
        (attn_pre,)
        + tuple(attention[:-1])
        + (attn_post, ffn_pre)
        + tuple(tail[:-1])
        + (ffn_post,)
    )


def _compressor_op(
    dims: DimTable, ratio: int, head_dim: str, prefix: str
) -> tuple[OpSpec, TensorRef]:
    coff = 2 if ratio == 4 else 1
    projected_dim = f"{coff}*{head_dim}"
    source = TensorRef("ln1", ("S", "B", "H"), {0: "cp"})
    wkv = TensorRef(
        f"{prefix}_wkv", ("H", projected_dim), is_weight=True
    )
    wgate = TensorRef(
        f"{prefix}_wgate", ("H", projected_dim), is_weight=True
    )
    ape = TensorRef(
        f"{prefix}_ape", (ratio, projected_dim), is_weight=True
    )
    norm = _norm_weight(f"{prefix}_norm_weight", head_dim)
    output = TensorRef(
        f"{prefix}_kv", (f"S//{ratio}", "B", head_dim), {0: "cp"}
    )
    return (
        OpSpec(
            prefix,
            OpType.COMPRESS,
            (source,),
            output,
            params=(wkv, wgate, ape, norm),
            saves=(source, output),
            attrs={"compress_ratio": ratio, "coff": coff},
            module_paths=(f"self_attention.core_attention.{prefix}",),
        ),
        output,
    )


def build_v4_hybrid_attention(
    dims: DimTable, compress_ratio: int, dense_mode: bool = False
) -> tuple[OpSpec, ...]:
    """Build the exact parameter-bearing branches of DSv4 hybrid attention."""

    if compress_ratio not in {0, 4, 128}:
        raise ValueError("DeepSeek-V4 hybrid attention supports ratios 0, 4 and 128")
    x_sp = TensorRef("x", ("S", "B", "H"), {0: "sp"})
    x_full = TensorRef("x", ("S", "B", "H"), {0: "cp"})
    ln1_w = _norm_weight("ln1_weight")
    ln1_sp = TensorRef("ln1", ("S", "B", "H"), {0: "sp"})
    ln1_full = TensorRef("ln1", ("S", "B", "H"), {0: "cp"})
    q_down_w = TensorRef(
        "v4_q_down_w", ("H", "q_lora_rank"), is_weight=True
    )
    q_compressed = TensorRef("v4_q_compressed", ("S", "B", "q_lora_rank"), {0: "cp"})
    q_norm_w = _norm_weight("v4_q_norm_weight", "q_lora_rank")
    q_norm = TensorRef("v4_q_norm", ("S", "B", "q_lora_rank"), {0: "cp"})
    q_up_w = TensorRef(
        "v4_q_up_w",
        ("q_lora_rank", "n_heads*v_head_dim"),
        {1: "tp"},
        True,
    )
    q_sharded = TensorRef(
        "v4_query", ("S", "B", "n_heads", "v_head_dim"), {0: "cp", 2: "tp"}
    )
    q_rms_gamma = TensorRef(
        "v4_q_rms_gamma",
        ("v_head_dim",),
        is_weight=True,
        dtype_bytes=4,
        trainable=False,
    )
    q_rms = TensorRef(
        "v4_query", ("S", "B", "n_heads", "v_head_dim"), {0: "cp", 2: "tp"}
    )
    q_full = TensorRef("v4_query", ("S", "B", "n_heads", "v_head_dim"), {0: "cp"})
    kv_w = TensorRef("v4_kv_w", ("H", "v_head_dim"), is_weight=True)
    kv = TensorRef("v4_kv", ("S", "B", "v_head_dim"), {0: "cp"})
    kv_norm_w = _norm_weight("v4_kv_norm_weight", "v_head_dim")
    kv_norm = TensorRef("v4_kv_norm", ("S", "B", "v_head_dim"), {0: "cp"})

    ops: list[OpSpec] = [
        OpSpec(
            "ln1", OpType.NORM, (x_sp, ln1_w), ln1_sp,
            params=(ln1_w,), saves=(x_sp,), module_paths=("input_layernorm",),
        ),
        OpSpec(
            "v4_q_down", OpType.MATMUL, (ln1_full, q_down_w), q_compressed,
            params=(q_down_w,), saves=(ln1_full,),
            module_paths=("self_attention.linear_q_down_proj",),
        ),
        OpSpec(
            "v4_q_norm", OpType.NORM, (q_compressed, q_norm_w), q_norm,
            params=(q_norm_w,), saves=(q_compressed,),
            module_paths=("self_attention.q_layernorm",),
        ),
        OpSpec(
            "v4_q_up", OpType.MATMUL, (q_norm, q_up_w), q_sharded,
            params=(q_up_w,), saves=(q_norm,),
            module_paths=("self_attention.linear_q_up_proj",),
        ),
        OpSpec(
            "v4_q_head_rms", OpType.NORM, (q_sharded, q_rms_gamma), q_rms,
            params=(q_rms_gamma,), saves=(q_sharded,),
            module_paths=("self_attention.q_rms_gamma",),
        ),
        OpSpec(
            "v4_kv_proj", OpType.MATMUL, (ln1_full, kv_w), kv,
            params=(kv_w,), saves=(ln1_full,),
            module_paths=("self_attention.linear_kv_proj",),
        ),
        OpSpec(
            "v4_kv_norm", OpType.NORM, (kv, kv_norm_w), kv_norm,
            params=(kv_norm_w,), saves=(kv,),
            module_paths=("self_attention.kv_layernorm",),
        ),
    ]

    compressed_kv = None
    if compress_ratio:
        compressor, compressed_kv = _compressor_op(
            dims, compress_ratio, "v_head_dim", "compressor"
        )
        ops.append(compressor)

    index_ids = None
    if compress_ratio == 4 and not dense_mode:
        index_coff = 2
        index_q_w = TensorRef(
            "indexer_q_w",
            ("q_lora_rank", "index_n_heads*index_head_dim"),
            is_weight=True,
        )
        index_wkv = TensorRef(
            "indexer_wkv", ("H", f"{index_coff}*index_head_dim"), is_weight=True
        )
        index_wgate = TensorRef(
            "indexer_wgate", ("H", f"{index_coff}*index_head_dim"), is_weight=True
        )
        index_ape = TensorRef(
            "indexer_ape", (4, f"{index_coff}*index_head_dim"), is_weight=True
        )
        index_norm = _norm_weight("indexer_norm_weight", "index_head_dim")
        index_score_w = TensorRef(
            "indexer_score_w", ("H", "index_n_heads"), is_weight=True
        )
        index_ids = TensorRef(
            "indexer_topk", ("S", "B", "index_topk"), {0: "cp"}, dtype_bytes=4
        )
        index_q = TensorRef(
            "indexer_query", ("S", "B", "index_n_heads", "index_head_dim"), {0: "cp"}
        )
        index_k = TensorRef(
            "indexer_key", ("S//4", "B", "index_head_dim"), {0: "cp"}
        )
        index_scores = TensorRef(
            "indexer_weights", ("S", "B", "index_n_heads"), {0: "cp"}, dtype_bytes=4
        )
        ops.append(OpSpec(
            "indexer", OpType.INDEXER, (ln1_full, q_norm), index_ids,
            params=(index_q_w, index_wkv, index_wgate, index_ape, index_norm, index_score_w),
            saves=(index_q, index_k, index_scores, index_ids),
            attrs={"compress_ratio": 4, "topk": dims.index_topk},
            module_paths=("self_attention.core_attention.indexer",),
        ))

    sink = TensorRef(
        "attention_sink", ("n_heads",), is_weight=True, dtype_bytes=4
    )
    attn = TensorRef("v4_attn", ("S", "B", "n_heads", "v_head_dim"), {0: "cp"})
    lse = TensorRef("v4_sparse_lse", ("S", "B", "n_heads"), {0: "cp"}, dtype_bytes=4)
    sparse_inputs = [q_full, kv_norm, x_full, q_norm]
    if compressed_kv is not None:
        sparse_inputs.append(compressed_kv)
    if index_ids is not None:
        sparse_inputs.append(index_ids)
    ops.append(OpSpec(
        "v4_sparse_attention", OpType.SPARSE_ATTN, tuple(sparse_inputs), attn,
        params=(sink,), saves=(q_full, kv_norm, attn, lse),
        workspace="S*B*n_heads*v_head_dim*dtype_bytes",
        attrs={
            "compress_ratio": compress_ratio,
            "sliding_window": dims.sliding_window,
            "dense_mode": dense_mode,
        },
        module_paths=("self_attention.core_attention",),
    ))

    group_w = TensorRef(
        "v4_o_group_w",
        ("o_groups*o_lora_rank", "n_heads*v_head_dim//o_groups"),
        is_weight=True,
    )
    grouped = TensorRef(
        "v4_o_grouped", ("S", "B", "o_groups*o_lora_rank"), {0: "cp"}
    )
    out_w = TensorRef(
        "v4_o_w", ("o_groups*o_lora_rank", "H"), is_weight=True
    )
    projected = TensorRef("v4_o", ("S", "B", "H"), {0: "cp"})
    h1 = TensorRef("h1", ("S", "B", "H"), {0: "cp"})
    ops.extend((
        OpSpec(
            "v4_o_group", OpType.MATMUL, (attn, group_w), grouped,
            params=(group_w,), saves=(attn,),
            module_paths=("self_attention.linear_o_group_proj",),
        ),
        OpSpec(
            "v4_o_proj", OpType.MATMUL, (grouped, out_w), projected,
            params=(out_w,), saves=(grouped,),
            module_paths=("self_attention.linear_proj",),
        ),
        OpSpec("add1", OpType.ELEMENTWISE, (projected, x_full), h1),
    ))
    return tuple(ops)


def build_v4_hybrid_moe_decoder(
    dims: DimTable,
    compress_ratio: int,
    hash_router: bool = False,
    dense_mode: bool = False,
    include_hc: bool = True,
) -> LayerSpec:
    attention = build_v4_hybrid_attention(dims, compress_ratio, dense_mode)
    tail = _moe_tail(dims)
    # ``include_hc`` is retained for API compatibility. mHC is intrinsic to
    # the transformer layer when enabled; MTP therefore also owns both cells.
    _ = include_hc
    ops = _with_mhc(dims, attention, tail) if dims.enable_hyper_connections else attention + tail
    return _with_hash_router(LayerSpec(ops), hash_router)


def _mtp_outer(dims: DimTable) -> tuple[OpSpec, ...]:
    embedding = TensorRef(
        "mtp_embedding", ("S", "B", "H"), {0: "sp"},
        storage_id="token_embedding_activation",
    )
    prev = TensorRef("mtp_prev_hidden", ("S", "B", "H"), {0: "sp"})
    ew = _norm_weight("mtp_enorm_weight")
    hw = _norm_weight("mtp_hnorm_weight")
    enorm = TensorRef("mtp_enorm", ("S", "B", "H"), {0: "sp"})
    hnorm = TensorRef("mtp_hnorm", ("S", "B", "H"), {0: "sp"})
    joined = TensorRef("mtp_joined", ("S", "B", "2*H"), {0: "sp"})
    proj_w = TensorRef("mtp_eh_proj_w", ("2*H", "H"), is_weight=True)
    x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
    return (
        OpSpec("mtp_enorm", OpType.NORM, (embedding, ew), enorm, (ew,), (embedding,), module_paths=("enorm",)),
        OpSpec("mtp_hnorm", OpType.NORM, (prev, hw), hnorm, (hw,), (prev,), module_paths=("hnorm",)),
        OpSpec("mtp_concat", OpType.ELEMENTWISE, (enorm, hnorm), joined, saves=(enorm, hnorm)),
        OpSpec("mtp_eh_proj", OpType.MATMUL, (joined, proj_w), x, (proj_w,), (joined,), module_paths=("eh_proj",)),
    )


def build_v4_hybrid_mtp_layer(
    dims: DimTable, compress_ratio: int = 0, dense_mode: bool = False
) -> LayerSpec:
    """V4 MTP owns one hybrid MoE layer and never enables hash routing."""

    inner = build_v4_hybrid_moe_decoder(
        dims, compress_ratio, hash_router=False, dense_mode=dense_mode,
        include_hc=False,
    ).ops
    final_w = _norm_weight("mtp_final_norm_weight")
    inner_out = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    final = TensorRef("mtp_final", ("S", "B", "H"), {0: "sp"})
    expand_collapse: tuple[OpSpec, ...] = ()
    collapse: tuple[OpSpec, ...] = ()
    if dims.enable_hyper_connections:
        outer_x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
        streams = TensorRef(
            "v4_input_streams", ("S", "B", "hc_mult*H"), {0: "sp"}
        )
        stream_output = TensorRef(
            "h2", ("S", "B", "hc_mult*H"), {0: "sp"}
        )
        collapsed = TensorRef("h2_collapsed", ("S", "B", "H"), {0: "sp"})
        expand_collapse = (OpSpec(
            "mtp_hc_expand", OpType.ELEMENTWISE, (outer_x,), streams,
            saves=(outer_x,), attrs={"hc_mult": dims.hc_mult},
        ),)
        collapse = (OpSpec(
            "mtp_hc_collapse", OpType.ELEMENTWISE, (stream_output,), collapsed,
            saves=(stream_output,), attrs={"hc_mult": dims.hc_mult, "mode": "mean"},
        ),)
        inner_out = collapsed
    final_ops = collapse + (
        OpSpec(
            "mtp_final_norm", OpType.NORM, (inner_out, final_w), final,
            (final_w,), (inner_out,), module_paths=("final_layernorm",),
        ),
    )
    return LayerSpec(_mtp_outer(dims) + expand_collapse + tuple(inner) + final_ops)


def build_v4_legacy_mtp_layer(dims: DimTable) -> LayerSpec:
    """Legacy MLA V4 MTP with V4 hash/mHC rules."""

    inner = build_v4_legacy_mla_moe_decoder(dims, hash_router=False).ops
    final_w = _norm_weight("mtp_final_norm_weight")
    inner_out = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    final = TensorRef("mtp_final", ("S", "B", "H"), {0: "sp"})
    expand: tuple[OpSpec, ...] = ()
    collapse: tuple[OpSpec, ...] = ()
    if dims.enable_hyper_connections:
        outer_x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
        streams = TensorRef(
            "v4_input_streams", ("S", "B", "hc_mult*H"), {0: "sp"}
        )
        stream_output = TensorRef(
            "h2", ("S", "B", "hc_mult*H"), {0: "sp"}
        )
        collapsed = TensorRef("h2_collapsed", ("S", "B", "H"), {0: "sp"})
        expand = (OpSpec(
            "mtp_hc_expand", OpType.ELEMENTWISE, (outer_x,), streams,
            saves=(outer_x,), attrs={"hc_mult": dims.hc_mult},
        ),)
        collapse = (OpSpec(
            "mtp_hc_collapse", OpType.ELEMENTWISE, (stream_output,), collapsed,
            saves=(stream_output,), attrs={"hc_mult": dims.hc_mult, "mode": "mean"},
        ),)
        inner_out = collapsed
    final_op = OpSpec(
        "mtp_final_norm", OpType.NORM, (inner_out, final_w), final,
        (final_w,), (inner_out,), module_paths=("final_layernorm",),
    )
    return LayerSpec(_mtp_outer(dims) + expand + tuple(inner) + collapse + (final_op,))
