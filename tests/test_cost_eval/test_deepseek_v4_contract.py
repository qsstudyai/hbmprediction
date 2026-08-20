import unittest

from cost_eval.adapters import MindFormersAdapter
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.source_contracts.deepseek_v4 import (
    DEEPSEEK_V4_CONTRACT_VERSION,
    DEEPSEEK_V4_PYNATIVE,
)


def v4_mapping(variant="dsv4_hybrid"):
    model = {
        "model_type": "deepseek_v4",
        "architectures": "DeepseekV4ForCausalLM",
        "vocab_size": 128,
        "seq_length": 128,
        "hidden_size": 64,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "num_attention_heads": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "multi_latent_attention": True,
        "q_lora_rank": 48,
        "qk_rope_head_dim": 8,
        "num_hash_layers": 1,
        "experimental_attention_variant": variant,
        "params_dtype": "bfloat16",
    }
    if variant == "mla":
        model.update({
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "v_head_dim": 16,
        })
    else:
        model.update({
            "head_dim": 32,
            "o_lora_rank": 16,
            "o_groups": 4,
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 8,
            "sliding_window": 128,
            "compress_ratios": [4, 128, 0],
        })
    return {
        "training": {"local_batch_size": 1, "global_batch_size": 1},
        "parallelism": {},
        "model": model,
    }


class DeepSeekV4ContractTests(unittest.TestCase):
    def test_legacy_mla_has_distinct_v4_identity_hash_scope_and_mtp_rule(self):
        inputs = MindFormersAdapter.from_mapping(v4_mapping("mla"), world_size=1)
        spec = inputs.model_spec
        self.assertEqual(spec.capabilities["model_family"], "deepseek_v4")
        self.assertEqual(
            spec.capabilities["source_contract_version"],
            DEEPSEEK_V4_CONTRACT_VERSION,
        )
        self.assertEqual(spec.capabilities["attention_variant"], "mla")
        self.assertEqual(spec.layer_pattern, (
            "deepseek_v4_mla_moe_hash",
            "deepseek_v4_mla_moe",
            "deepseek_v4_mtp_mla",
        ))
        hash_params = {
            param.name: param
            for op in spec.get_layer(spec.layer_pattern[0]).ops
            for param in op.params
        }
        normal_params = {
            param.name
            for op in spec.get_layer(spec.layer_pattern[1]).ops
            for param in op.params
        }
        mtp_params = {
            param.name
            for op in spec.get_layer(spec.layer_pattern[2]).ops
            for param in op.params
        }
        self.assertEqual(hash_params["router_tid2eid"].shape, ("vocab", "topk"))
        self.assertFalse(hash_params["router_tid2eid"].trainable)
        self.assertTrue(hash_params["router_tid2eid"].fsdp_replicated)
        self.assertNotIn("router_tid2eid", normal_params)
        self.assertNotIn("router_tid2eid", mtp_params)

    def test_hybrid_contract_covers_ratio_specific_parameters(self):
        inputs = MindFormersAdapter.from_mapping(v4_mapping(), world_size=1)
        spec = inputs.model_spec
        self.assertEqual(spec.layer_pattern, (
            "deepseek_v4_hybrid_r4_moe_hash",
            "deepseek_v4_hybrid_r128_moe",
            "deepseek_v4_mtp_hybrid_r0",
        ))
        ratio4 = spec.get_layer(spec.layer_pattern[0])
        ratio128 = spec.get_layer(spec.layer_pattern[1])
        ops4 = {op.name: op for op in ratio4.ops}
        ops128 = {op.name: op for op in ratio128.ops}
        self.assertIn("compressor", ops4)
        self.assertIn("indexer", ops4)
        self.assertIn("compressor", ops128)
        self.assertNotIn("indexer", ops128)
        group = next(
            param for param in ops4["v4_o_group"].params
            if param.name == "v4_o_group_w"
        )
        self.assertEqual(
            group.shape,
            ("o_groups*o_lora_rank", "n_heads*v_head_dim//o_groups"),
        )
        q_gamma = ops4["v4_q_head_rms"].params[0]
        sink = ops4["v4_sparse_attention"].params[0]
        self.assertEqual((q_gamma.dtype_bytes, q_gamma.trainable), (4, False))
        self.assertEqual((sink.dtype_bytes, sink.trainable), (4, True))
        mhc = ops4["attn_hc_pre"]
        mhc_params = {param.name: param for param in mhc.params}
        self.assertEqual(
            mhc_params["attn_hc_mapping_w"].shape,
            ("hc_mult*H", "2*hc_mult+hc_mult*hc_mult"),
        )
        self.assertEqual(mhc_params["attn_hc_mapping_w"].dtype_bytes, 4)
        self.assertTrue(mhc_params["attn_hc_alpha_pre"].fsdp_replicated)
        self.assertTrue(mhc_params["attn_hc_bias"].fsdp_replicated)
        self.assertIn("ffn_hc_pre", ops4)
        graph = ShapeEval().resolve(
            spec, ParallelModel(inputs.parallel_config, spec.dims.n_layers)
        )
        self.assertEqual(graph.stages[0][-1].layer_type, "deepseek_v4_mtp_hybrid_r0")
        v4_report = inputs.evaluator().evaluate()
        self.assertGreater(v4_report.per_stage[0].peak_bytes, 0)
        self.assertEqual(v4_report.oom_status, "unsupported")
        self.assertNotEqual(v4_report.oom_status, "definitely_safe")

        sharded = v4_mapping()
        sharded["parallelism"]["data_parallel_shard"] = 2
        sharded["training"]["global_batch_size"] = 2
        sharded_inputs = MindFormersAdapter.from_mapping(sharded, world_size=2)
        self.assertGreater(
            sharded_inputs.evaluator().evaluate().per_stage[0].peak_bytes, 0
        )

        tp = v4_mapping()
        tp["parallelism"].update({
            "tensor_parallel": 2,
            "sequence_parallel": True,
        })
        tp_inputs = MindFormersAdapter.from_mapping(tp, world_size=2)
        tp_graph = ShapeEval().resolve(
            tp_inputs.model_spec,
            ParallelModel(tp_inputs.parallel_config, tp_inputs.model_spec.dims.n_layers),
        )
        sparse = next(
            op for op in tp_graph.stages[0][0].ops
            if op.name == "v4_sparse_attention"
        )
        self.assertTrue(any(item.ctype == "all_gather" for item in sparse.collectives))

    def test_contract_rejects_uncovered_v4_shapes(self):
        bad = v4_mapping()
        bad["model"]["experimental_attention_variant"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported DeepSeek-V4"):
            MindFormersAdapter.from_mapping(bad, world_size=1)

        bad = v4_mapping()
        bad["model"]["compress_ratios"] = [16, 128, 0]
        with self.assertRaisesRegex(ValueError, "compression ratios"):
            MindFormersAdapter.from_mapping(bad, world_size=1)

        bad = v4_mapping()
        bad["model"]["n_routed_experts"] = 0
        with self.assertRaisesRegex(ValueError, "MoE"):
            MindFormersAdapter.from_mapping(bad, world_size=1)

        bad = v4_mapping()
        bad["model"]["enable_hc_head"] = True
        with self.assertRaisesRegex(ValueError, "HC head"):
            MindFormersAdapter.from_mapping(bad, world_size=1)

    def test_explicit_pipeline_mapping_excludes_mtp_ids_and_mtp_is_final_stage(self):
        data = v4_mapping("mla")
        data["model"]["num_hidden_layers"] = 4
        data["model"]["num_hash_layers"] = 1
        data["parallelism"] = {
            "pipeline_parallel": 2,
            "pipeline_parallel_layers_per_stage": ["0-1", "2-3"],
        }
        data["training"]["global_batch_size"] = 2
        inputs = MindFormersAdapter.from_mapping(data, world_size=2)
        self.assertEqual(
            inputs.parallel_config.layers_per_stage,
            ((0, 1), (2, 3)),
        )
        report = inputs.evaluator().evaluate()
        self.assertEqual([stage.stage for stage in report.per_stage], [0, 1])

    def test_source_contract_is_content_addressed(self):
        files = DEEPSEEK_V4_PYNATIVE["files"]
        self.assertIn("hybrid_attention", files)
        self.assertIn("compressor", files)
        self.assertIn("indexer", files)
        self.assertIn("compressed_sparse_attention", files)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in files.values()))


if __name__ == "__main__":
    unittest.main()
