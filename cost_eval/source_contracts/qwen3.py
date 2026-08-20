"""Versioned Qwen3 PyNative source contract and probe state."""

QWEN3_CONTRACT_VERSION = "mindformers-75e57dc-qwen3-v1"
QWEN3_SOURCE_COMMIT = "75e57dc680c0a349d3ad9ec2641cdcce4f98f986"

QWEN3_PYNATIVE = {
    "version": QWEN3_CONTRACT_VERSION,
    "source_commit": QWEN3_SOURCE_COMMIT,
    "decoder": {
        "family": "GQA + Q/K RMSNorm + RoPE + FlashAttention + SwiGLU",
        "norm_storage_dtype": "fp32",
        "activation_storage_dtype": "model compute dtype",
        "lse_storage_dtype": "fp32",
    },
    "probe_status": "pending",
}
