"""Auditable MindFormers DeepSeek-V4 source-to-evaluator contract.

The contract is pinned to a concrete neighbouring MindFormers checkout.  It
keeps source provenance and validation rules separate from graph construction,
so a model cannot silently fall back to the DeepSeek-V3 MLA formulas.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEEPSEEK_V4_CONTRACT_VERSION = "mindformers-75e57dc-dsv4-v1"
DEEPSEEK_V4_SOURCE_COMMIT = "75e57dc680c0a349d3ad9ec2641cdcce4f98f986"

DEEPSEEK_V4_PYNATIVE = {
    "version": DEEPSEEK_V4_CONTRACT_VERSION,
    "source_commit": DEEPSEEK_V4_SOURCE_COMMIT,
    "files": {
        "model": {
            "path": "mindformers/models/deepseek4/modeling_deepseek_v4_pynative.py",
            "sha256": "84dd3fb2eed7ff5daf39fc96287ab37543e16e68461f0cd8710e3959392bf476",
        },
        "configuration": {
            "path": "mindformers/models/deepseek4/configuration_deepseek_v4.py",
            "sha256": "85301ae9931ff9be4a02530dacd050ac1d742a6989f01937d504b581a376cd2a",
        },
        "converter": {
            "path": "mindformers/models/deepseek4/config_converter_deepseek_v4.py",
            "sha256": "e3ddfa4ca5b6aa109e78a4cb93e9f4f429ff9d2fc0959c43dbbd9c8f44c4dbfd",
        },
        "checkpoint_mapping": {
            "path": "mindformers/models/deepseek4/utils.py",
            "sha256": "e24857abaa3b311f347b1bd11fc0d955e609b363b36e4f16dc1c03706bac7967",
        },
        "legacy_mla": {
            "path": "mindformers/pynative/transformers/multi_latent_attention.py",
            "sha256": "439a823fd87f5aa809558b7b5ae6053e4183b9ebd834a988e7ba7b2d73d911a2",
        },
        "hybrid_attention": {
            "path": "mindformers/pynative/transformers/experimental_attention_variant/deepseek_v4_hybrid_attention.py",
            "sha256": "8a7f6a362f34b4a28e32216c73de632034ec832e68c596300bf352734d7e22f6",
        },
        "compressor": {
            "path": "mindformers/pynative/transformers/experimental_attention_variant/compressor.py",
            "sha256": "bed1e7a131cb5f0e78ab08a4667aeb3d5afcd460810d7ba2f7c21ff8893a58e3",
        },
        "indexer": {
            "path": "mindformers/pynative/transformers/experimental_attention_variant/indexer.py",
            "sha256": "fbcb3531edfc308934fcc94eaf0fe4df739981d36a853332b35b855cf3483540",
        },
        "compressed_sparse_attention": {
            "path": "mindformers/pynative/transformers/experimental_attention_variant/csa.py",
            "sha256": "2eb3fe3baafc02958473a0af0d4e90dce9e0c5129328c4a6c9bf3ec06cd37a13",
        },
        "moe_router": {
            "path": "mindformers/pynative/transformers/moe/router.py",
            "sha256": "6e41ea5c05afd2bb638ae227e7ad70301ffc1d0867e7e6cb368d50f39eea879c",
        },
        "moe_layer": {
            "path": "mindformers/pynative/transformers/moe/moe_layer.py",
            "sha256": "2b899a3d22e1b85c01352ab1bf747ec261b587b0bdf542aa51820f067a7be957",
        },
        "moe_experts": {
            "path": "mindformers/pynative/transformers/moe/experts.py",
            "sha256": "a4063de51ae93fc2044a79b4e9e35f4e746665463b256fd6ea76968cd43d2274",
        },
        "shared_experts": {
            "path": "mindformers/pynative/transformers/moe/shared_experts.py",
            "sha256": "9d0d1b6d31c51ba382a01bf0e173bd0c6b6a0cb9393e92d40cb18b6c0e20afe7",
        },
        "variant_specs": {
            "path": "mindformers/pynative/base_models/gpt/experimental_attention_variant_module_specs.py",
            "sha256": "c8fe4d549841cccbeffd882972e8d42b8f2411615ad3d99184a3486593b7e1fd",
        },
        "gpt_layer_specs": {
            "path": "mindformers/pynative/base_models/gpt/gpt_layer_specs.py",
            "sha256": "30e246b3d45d542d2cad706062b5bfb3a00a30703ef618c5a81298ad0fde531a",
        },
        "parallelize": {
            "path": "mindformers/pynative/base_models/gpt/parallelize.py",
            "sha256": "8d67004010759fdc60c2df326173f727c520d835466438914997055f04f2c8b7",
        },
        "hyper_connection": {
            "path": "mindformers/pynative/transformers/hyper_connection.py",
            "sha256": "4902279c3b6a21f6d60570ea767b409184b4416bf4c11084da6d62bee0a3f19f",
        },
        "mtp": {
            "path": "mindformers/pynative/transformers/multi_token_prediction.py",
            "sha256": "befab26d1442473faea599128b1e82f0a3f61615a22cad96080033f9b2ce7dd9",
        },
    },
    "variants": {
        "mla": "legacy MLA used by the archived DeepSeek-V4 real-NPU configs",
        "dsv4_hybrid": "V4 sliding-window/compressed sparse attention",
    },
    "hybrid_parameters": {
        "q": "H*q_lora_rank + q_lora_rank*n_heads*v_head_dim",
        "kv": "H*v_head_dim",
        "grouped_output": (
            "(o_groups*o_lora_rank)*(n_heads*v_head_dim//o_groups) + "
            "(o_groups*o_lora_rank)*H"
        ),
        "compressor": (
            "2*H*(coff*head_dim) + compress_ratio*(coff*head_dim) + head_dim; "
            "coff=2 for ratio 4, otherwise 1"
        ),
        "indexer_ratio4": (
            "q_lora_rank*index_n_heads*index_head_dim + "
            "2*H*(2*index_head_dim) + 4*(2*index_head_dim) + "
            "index_head_dim + H*index_n_heads"
        ),
        "attention_sink": "n_heads fp32 trainable values",
        "q_rms_gamma": "v_head_dim fp32 non-trainable values",
    },
    "hash_router": {
        "scope": "leading num_hash_layers decoder layers; never MTP",
        "table": "vocab*topk int32, non-trainable and FSDP-replicated",
        "learned_scores": "router weight remains trainable",
    },
    "hyper_connections": {
        "per_layer": "independent attention and FFN modules",
        "mapping": "fp32 (hc_mult*H) * (2*hc_mult + hc_mult^2)",
        "parameters": "three fp32 scalar alphas, fp32 bias, fixed fp32 RMS gamma",
        "saved": "packed streams, h_res[S,B,n,n], h_post[S,B,n,1]",
        "mtp": "expand H->nH, run both modules, collapse by mean when hc_head is disabled",
    },
}


def deepseek_v4_attention_variant(model: Mapping[str, Any]) -> str:
    """Return the normalized V4 attention variant."""

    return str(model.get("experimental_attention_variant") or "dsv4_hybrid").lower()


def deepseek_v4_compress_ratios(
    model: Mapping[str, Any], n_layers: int, n_mtp_layers: int
) -> tuple[int, ...]:
    """Resolve the per-decoder/MTP ratios used by the V4 runtime."""

    raw = model.get("compress_ratios")
    if raw is None:
        return (128,) * n_layers + (0,) * n_mtp_layers
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("DeepSeek-V4 compress_ratios must be a sequence")
    ratios = tuple(int(value) for value in raw)
    required = n_layers + n_mtp_layers
    if len(ratios) < required:
        raise ValueError(
            "DeepSeek-V4 compress_ratios must cover all decoder and MTP layers"
        )
    return ratios[:required]


def validate_deepseek_v4_contract(
    model: Mapping[str, Any], dims: Any, ratios: Sequence[int]
) -> None:
    """Reject V4 configurations whose memory graph is not contract-covered."""

    if not dims.n_experts:
        raise ValueError("DeepSeek-V4 contract supports the runtime's MoE model only")
    if not bool(model.get("multi_latent_attention", True)):
        raise ValueError("DeepSeek-V4 requires multi_latent_attention=True")
    variant = deepseek_v4_attention_variant(model)
    if variant not in DEEPSEEK_V4_PYNATIVE["variants"]:
        raise ValueError(f"unsupported DeepSeek-V4 attention variant: {variant!r}")
    if dims.num_hash_layers > dims.n_layers:
        raise ValueError("DeepSeek-V4 num_hash_layers cannot exceed decoder layers")
    if bool(model.get("enable_hc_head", False)):
        raise ValueError(
            "DeepSeek-V4 learnable HC head is not covered; disable enable_hc_head"
        )
    invalid_ratios = sorted(set(ratios) - {0, 4, 128})
    if invalid_ratios:
        raise ValueError(
            f"unsupported DeepSeek-V4 compression ratios: {invalid_ratios}"
        )
    if variant == "mla":
        if not all((
            dims.q_lora_rank,
            dims.kv_lora_rank,
            dims.qk_nope_head_dim,
            dims.qk_rope_head_dim,
            dims.v_head_dim,
        )):
            raise ValueError("DeepSeek-V4 legacy MLA dimensions are incomplete")
        return
    required = {
        "q_lora_rank": dims.q_lora_rank,
        "v_head_dim": dims.v_head_dim,
        "qk_rope_head_dim": dims.qk_rope_head_dim,
        "o_lora_rank": dims.o_lora_rank,
        "o_groups": dims.o_groups,
        "sliding_window": dims.sliding_window,
    }
    missing = [name for name, value in required.items() if value <= 0]
    if missing:
        raise ValueError("DeepSeek-V4 hybrid dimensions are incomplete: " + ", ".join(missing))
    if dims.qk_rope_head_dim > dims.v_head_dim:
        raise ValueError("DeepSeek-V4 qk_rope_head_dim cannot exceed v_head_dim")
    if dims.qk_nope_head_dim + dims.qk_rope_head_dim != dims.v_head_dim:
        raise ValueError(
            "DeepSeek-V4 hybrid requires qk_nope_head_dim + "
            "qk_rope_head_dim == v_head_dim"
        )
    if (dims.n_heads * dims.v_head_dim) % dims.o_groups:
        raise ValueError("DeepSeek-V4 grouped output width must be divisible by o_groups")
    dense_mode = bool(model.get("csa_dense_mode", False))
    if not bool(model.get("apply_dsa_kernel_fusion", True)):
        raise ValueError(
            "DeepSeek-V4 unfused hybrid attention has a different memory contract"
        )
    if dims.sliding_window != 128:
        raise ValueError(
            "DeepSeek-V4 fused hybrid attention requires sliding_window=128"
        )
    if 4 in ratios and not dense_mode:
        index_required = {
            "index_n_heads": dims.index_n_heads,
            "index_head_dim": dims.index_head_dim,
            "index_topk": dims.index_topk,
        }
        missing = [name for name, value in index_required.items() if value <= 0]
        if missing:
            raise ValueError(
                "DeepSeek-V4 ratio-4 indexer dimensions are incomplete: "
                + ", ".join(missing)
            )
