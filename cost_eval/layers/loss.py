"""Explicit training loss/logits memory contracts."""

from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef


SUPPORTED_LOSS_VARIANTS = frozenset({
    "fused_cross_entropy", "fallback_cross_entropy"
})


def build_loss_ops(
    dims: DimTable, variant: str, loss_parallel: bool
) -> tuple[OpSpec, ...]:
    if variant not in SUPPORTED_LOSS_VARIANTS:
        raise ValueError(f"unsupported loss implementation: {variant}")
    logits_shard = {0: "sp"}
    if loss_parallel:
        logits_shard[2] = "tp"
    logits = TensorRef("logits", ("S", "B", "vocab"), logits_shard)
    labels = TensorRef(
        "labels", ("S", "B"), {0: "sp"}, dtype_bytes=4,
        trainable=False, swappable=False, recomputable=False,
    )
    mask = TensorRef(
        "loss_mask", ("S", "B"), {0: "sp"}, dtype_bytes=4,
        trainable=False, swappable=False, recomputable=False,
    )
    stats = TensorRef(
        "loss_stats", ("S", "B", 2), {0: "sp"}, dtype_bytes=4
    )
    loss = TensorRef("loss", (1,), dtype_bytes=4, recomputable=False)

    if variant == "fused_cross_entropy":
        return (OpSpec(
            "loss_fused_cross_entropy", OpType.ELEMENTWISE,
            (logits, labels, mask), loss,
            saves=(logits, labels, mask, stats), workspace="S*B*2*4",
            attrs={
                "loss_variant": variant,
                "backward_workspace_largest_input_multiplier": 1,
            },
            module_paths=("loss_func.fused_cross_entropy",),
        ),)

    logits_fp32 = TensorRef(
        "loss_logits_fp32", ("S", "B", "vocab"), logits_shard,
        dtype_bytes=4,
    )
    probabilities = TensorRef(
        "loss_probabilities", ("S", "B", "vocab"), logits_shard,
        dtype_bytes=4,
    )
    return (
        OpSpec(
            "loss_cast_fp32", OpType.ELEMENTWISE, (logits,), logits_fp32,
            module_paths=("loss_func.cast",),
        ),
        OpSpec(
            "loss_fwd_softmax", OpType.ELEMENTWISE,
            (logits_fp32,), probabilities,
            saves=(probabilities, stats), workspace="S*B*2*4",
            module_paths=("loss_func.log_softmax",),
        ),
        OpSpec(
            "loss_reduce", OpType.ELEMENTWISE,
            (probabilities, labels, mask), loss,
            saves=(probabilities, labels, mask),
            attrs={
                "loss_variant": variant,
                # Fallback backward materializes both the incoming FP32
                # probability gradient and FP32 dlogits before the cast sent
                # to lm_head.
                "backward_workspace_largest_input_multiplier": 2,
            },
            module_paths=("loss_func.nll_loss",),
        ),
    )
