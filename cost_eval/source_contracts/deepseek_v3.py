"""MindFormers PyNative DeepSeek-V3 source-to-evaluator contract.

This is intentionally data rather than executable graph construction: tests and
reviews can audit which runtime module justifies each evaluator formula without
importing MindSpore.
"""

DEEPSEEK_V3_PYNATIVE = {
    "mla": {
        "source": "mindformers/pynative/transformers/multi_latent_attention.py",
        "down_projection": "H * (q_lora_rank + kv_lora_rank + qk_rope_head_dim)",
        "q_up_projection": "q_lora_rank * n_heads * (qk_nope_head_dim + qk_rope_head_dim)",
        "kv_up_projection": "kv_lora_rank * n_heads * (qk_nope_head_dim + v_head_dim)",
        "output_projection": "n_heads * v_head_dim * H",
        "tp": "down=SequenceParallel, q/kv_up=Colwise, output=Rowwise",
    },
    "shared_expert": {
        "source": "mindformers/pynative/transformers/moe/shared_experts.py",
        "parameters": "H * 2*shared_F + shared_F * H + optional(H * 1)",
        "tp": "replicated parameters with sequence-parallel activations",
        "fsdp": "dense FSDP; fp32 gate wrapped separately",
    },
    "mtp": {
        "source": "mindformers/pynative/transformers/multi_token_prediction.py",
        "outer": "enorm(H), hnorm(H), concat(2H), eh_proj(2H->H), final_norm(H)",
        "inner": "one full transformer layer",
        "placement": "global IDs after decoder layers, resident on final PP stage",
        "sharing": "token embedding and output head are shared with the base model",
    },
    "parallelize": {
        "source": "mindformers/pynative/base_models/gpt/parallelize.py",
        "edge_fsdp": "embedding, final norm and output head participate in FSDP",
        "loss_parallel": "logits retain TP vocabulary sharding",
        "edge_graph": "embedding -> decoder layers -> final norm -> output head/loss",
        "edge_prefetch": "embedding and tail modules participate in the FSDP prefetch chain",
    },
    "recompute_comm": {
        "source": "mindformers/pynative/distributed/activation_checkpoint.py",
        "runtime": "checkpoint_wrapper(operator, output_recompute=True)",
        "memory": "drop selected communication output in forward; reissue it in backward",
    },
    "context_parallel": {
        "source": "mindformers/pynative/distributed/style.py",
        "hybrid": "inner Ulysses all-to-all plus outer ring over cp/ulysses_degree",
    },
}
