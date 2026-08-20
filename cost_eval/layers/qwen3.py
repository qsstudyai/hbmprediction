"""Qwen3-specific GQA/RMSNorm/RoPE saved-tensor contract."""

from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .dense import NHD, QKV


def _norm_weight(name: str) -> TensorRef:
    return TensorRef(name, ("H",), is_weight=True, dtype_bytes=4)


def build_qwen3_decoder(dims: DimTable) -> LayerSpec:
    x = TensorRef("x", ("S", "B", "H"), {0: "sp"})
    ln1 = TensorRef("ln1", ("S", "B", "H"), {0: "sp"})
    ln1_full = TensorRef("ln1", ("S", "B", "H"), {0: "cp"})
    ln1_w = _norm_weight("input_norm_weight")
    qkv_w = TensorRef("qkv_w", ("H", QKV), {1: "tp"}, True)
    qkv = TensorRef("qkv", ("S", "B", QKV), {0: "cp", 2: "tp"})
    qk_normed = TensorRef(
        "qkv_normed", ("S", "B", QKV), {0: "cp", 2: "tp"}
    )
    q_norm_w = TensorRef(
        "q_norm_weight", ("head_dim",), is_weight=True, dtype_bytes=4
    )
    k_norm_w = TensorRef(
        "k_norm_weight", ("head_dim",), is_weight=True, dtype_bytes=4
    )
    position_ids = TensorRef(
        "position_ids", ("S", "B"), {0: "cp"}, dtype_bytes=4,
        trainable=False, swappable=False, recomputable=False,
    )
    rope_cos = TensorRef(
        "rope_cos", ("S", "head_dim"), {0: "cp"}, dtype_bytes=4,
        trainable=False, swappable=False,
    )
    rope_sin = TensorRef(
        "rope_sin", ("S", "head_dim"), {0: "cp"}, dtype_bytes=4,
        trainable=False, swappable=False,
    )
    qkv_rope = TensorRef(
        "qkv_rope", ("S", "B", QKV), {0: "cp", 2: "tp"}
    )
    attn = TensorRef("attn", ("S", "B", NHD), {0: "cp", 2: "tp"})
    lse = TensorRef(
        "lse", ("S", "B", "n_heads"), {0: "cp", 2: "tp"},
        dtype_bytes=4,
    )
    o_w = TensorRef("o_w", (NHD, "H"), {0: "tp"}, True)
    o_partial = TensorRef("o", ("S", "B", "H"), {0: "cp"}, partial="tp")
    o_sharded = TensorRef("o", ("S", "B", "H"), {0: "sp"})
    h1 = TensorRef("h1", ("S", "B", "H"), {0: "sp"})
    ln2 = TensorRef("ln2", ("S", "B", "H"), {0: "sp"})
    ln2_full = TensorRef("ln2", ("S", "B", "H"), {0: "cp"})
    ln2_w = _norm_weight("post_attention_norm_weight")
    fc1_w = TensorRef("fc1_w", ("H", "2*F"), {1: "tp"}, True)
    gate = TensorRef("gate", ("S", "B", "2*F"), {0: "cp", 2: "tp"})
    act = TensorRef("act", ("S", "B", "F"), {0: "cp", 2: "tp"})
    fc2_w = TensorRef("fc2_w", ("F", "H"), {0: "tp"}, True)
    o2_partial = TensorRef("o2", ("S", "B", "H"), {0: "cp"}, partial="tp")
    o2_sharded = TensorRef("o2", ("S", "B", "H"), {0: "sp"})
    h2 = TensorRef("h2", ("S", "B", "H"), {0: "sp"})
    return LayerSpec((
        OpSpec("ln1", OpType.NORM, (x, ln1_w), ln1, (ln1_w,), (x,)),
        OpSpec("qkv", OpType.MATMUL, (ln1_full, qkv_w), qkv, (qkv_w,), (ln1_full,)),
        OpSpec(
            "qk_norm", OpType.NORM, (qkv, q_norm_w, k_norm_w), qk_normed,
            (q_norm_w, k_norm_w), (qkv,),
        ),
        OpSpec(
            "rope", OpType.ROPE, (qk_normed, position_ids), qkv_rope,
            saves=(qk_normed, position_ids, rope_cos, rope_sin),
        ),
        OpSpec(
            "flash", OpType.FLASH_ATTN, (qkv_rope,), attn,
            saves=(qkv_rope, attn, lse),
            workspace="S*B*n_heads*head_dim*dtype_bytes",
            attrs={"n_heads": dims.n_heads, "head_dim": dims.head_dim},
        ),
        OpSpec("o_proj", OpType.MATMUL, (attn, o_w), o_partial, (o_w,), (attn,)),
        OpSpec("add1", OpType.ELEMENTWISE, (o_sharded, x), h1),
        OpSpec("ln2", OpType.NORM, (h1, ln2_w), ln2, (ln2_w,), (h1,)),
        OpSpec("fc1", OpType.MATMUL, (ln2_full, fc1_w), gate, (fc1_w,), (ln2_full,)),
        OpSpec("swiglu", OpType.ELEMENTWISE, (gate,), act, saves=(gate,)),
        OpSpec("fc2", OpType.MATMUL, (act, fc2_w), o2_partial, (fc2_w,), (act,)),
        OpSpec("add2", OpType.ELEMENTWISE, (o2_sharded, h1), h2),
    ))
