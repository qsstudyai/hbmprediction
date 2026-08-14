"""内置 Transformer 层图。"""

from .dense import build_dense_decoder
from .moe import build_moe_decoder
from .deepseek import build_mla_dense_decoder, build_mla_moe_decoder, build_mtp_layer

__all__ = ["build_dense_decoder", "build_moe_decoder", "build_mla_dense_decoder", "build_mla_moe_decoder", "build_mtp_layer"]
