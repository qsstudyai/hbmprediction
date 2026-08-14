"""M1：GQA + FlashAttention + SwiGLU dense decoder 图。"""

from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef

QKV = "(n_heads+2*n_kv)*head_dim"
NHD = "n_heads*head_dim"


def hyper_connection_ops(dims: DimTable, source_name: str) -> tuple[OpSpec, ...]:
    """Add the projection/mixing cost contract for Hyper-Connections."""

    if not getattr(dims, "enable_hyper_connections", False):
        return ()
    mult = max(1, int(getattr(dims, "hc_mult", 1)))
    source = TensorRef(source_name, ("S", "B", "H"), shard={0: "sp"})
    weight = TensorRef(
        "hc_projection_w", ("H", "H*hc_mult"), shard={1: "tp"}, is_weight=True
    )
    projected = TensorRef("hc_projected", ("S", "B", "H*hc_mult"), shard={2: "tp"})
    mixed = TensorRef("hc_mixed", ("S", "B", "H"), shard={0: "sp"})
    return (
        OpSpec(
            "hc_project",
            OpType.MATMUL,
            (source, weight),
            projected,
            params=(weight,),
            saves=(source,),
            module_paths=("hyper_connections.projection",),
        ),
        OpSpec(
            "hc_mix",
            OpType.ELEMENTWISE,
            (source, projected),
            mixed,
            saves=(source, projected),
            attrs={"hc_mult": mult},
            module_paths=("hyper_connections.mix",),
        ),
    )


def build_dense_decoder(dims: DimTable) -> LayerSpec:
    x = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})
    ln1 = TensorRef("ln1", ("S", "B", "H"), shard={0: "sp"})
    qkv_w = TensorRef(
        "qkv_w", ("H", QKV), shard={1: "tp"}, is_weight=True
    )
    qkv = TensorRef("qkv", ("S", "B", QKV), shard={2: "tp"})
    attn = TensorRef("attn", ("S", "B", NHD), shard={2: "tp"})
    lse = TensorRef("lse", ("S", "B", "n_heads"), shard={2: "tp"})
    o_w = TensorRef("o_w", (NHD, "H"), shard={0: "tp"}, is_weight=True)
    o_partial = TensorRef("o", ("S", "B", "H"), partial="tp")
    o_sharded = TensorRef("o", ("S", "B", "H"), shard={0: "sp"})
    h1 = TensorRef("h1", ("S", "B", "H"), shard={0: "sp"})
    ln2 = TensorRef("ln2", ("S", "B", "H"), shard={0: "sp"})
    fc1_w = TensorRef(
        "fc1_w", ("H", "2*F"), shard={1: "tp"}, is_weight=True
    )
    gate = TensorRef("gate", ("S", "B", "2*F"), shard={2: "tp"})
    act = TensorRef("act", ("S", "B", "F"), shard={2: "tp"})
    fc2_w = TensorRef(
        "fc2_w", ("F", "H"), shard={0: "tp"}, is_weight=True
    )
    o2_partial = TensorRef("o2", ("S", "B", "H"), partial="tp")
    o2_sharded = TensorRef("o2", ("S", "B", "H"), shard={0: "sp"})
    h2 = TensorRef("h2", ("S", "B", "H"), shard={0: "sp"})

    return LayerSpec(
        (
            OpSpec("ln1", OpType.NORM, (x,), ln1, saves=(x,)),
            OpSpec(
                "qkv",
                OpType.MATMUL,
                (ln1, qkv_w),
                qkv,
                params=(qkv_w,),
                saves=(ln1,),
            ),
            OpSpec("rope", OpType.ROPE, (qkv,), qkv),
            OpSpec(
                "flash",
                OpType.FLASH_ATTN,
                (qkv,),
                attn,
                saves=(qkv, attn, lse),
                workspace="S*B*n_heads*head_dim*dtype_bytes",
                attrs={"n_heads": dims.n_heads, "head_dim": dims.head_dim},
            ),
            OpSpec(
                "o_proj",
                OpType.MATMUL,
                (attn, o_w),
                o_partial,
                params=(o_w,),
                saves=(attn,),
            ),
            OpSpec("add1", OpType.ELEMENTWISE, (o_sharded, x), h1),
            OpSpec("ln2", OpType.NORM, (h1,), ln2, saves=(h1,)),
            OpSpec(
                "fc1",
                OpType.MATMUL,
                (ln2, fc1_w),
                gate,
                params=(fc1_w,),
                saves=(ln2,),
            ),
            OpSpec(
                "swiglu",
                OpType.ELEMENTWISE,
                (gate,),
                act,
                saves=(gate,),
            ),
            OpSpec(
                "fc2",
                OpType.MATMUL,
                (act, fc2_w),
                o2_partial,
                params=(fc2_w,),
                saves=(act,),
            ),
            OpSpec("add2", OpType.ELEMENTWISE, (o2_sharded, h1), h2),
            *hyper_connection_ops(dims, "h2"),
        )
    )
