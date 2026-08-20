"""内置 Transformer 层图。"""

from .dense import build_dense_decoder
from .moe import build_moe_decoder
from .deepseek import build_mla_dense_decoder, build_mla_moe_decoder, build_mtp_layer
from .deepseek_v4 import (
    build_v4_hybrid_moe_decoder,
    build_v4_hybrid_mtp_layer,
    build_v4_legacy_mla_moe_decoder,
    build_v4_legacy_mtp_layer,
)
from .loss import SUPPORTED_LOSS_VARIANTS, build_loss_ops
from .qwen3 import build_qwen3_decoder

__all__ = [
    "build_dense_decoder", "build_moe_decoder", "build_mla_dense_decoder",
    "build_mla_moe_decoder", "build_mtp_layer",
    "build_v4_hybrid_moe_decoder", "build_v4_hybrid_mtp_layer",
    "build_v4_legacy_mla_moe_decoder", "build_v4_legacy_mtp_layer",
    "SUPPORTED_LOSS_VARIANTS", "build_loss_ops",
    "build_qwen3_decoder",
]
