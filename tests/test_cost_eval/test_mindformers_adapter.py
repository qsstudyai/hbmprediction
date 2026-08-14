import unittest
from pathlib import Path

from cost_eval.adapters import MindFormersAdapter


class MindFormersAdapterTests(unittest.TestCase):
    def test_parallel_dims_from_config_matches_mindformers_hsdp_rule(self):
        data = {
            "training": {"local_batch_size": 2, "global_batch_size": 16},
            "parallelism": {
                "data_parallel_shard": 2,
                "tensor_parallel": 2,
                "context_parallel": 1,
                "pipeline_parallel": 1,
                "expert_parallel": 1,
                "sequence_parallel": True,
            },
            "optimizer": {"type": "AdamW"},
            "context": {"max_device_memory": "32GB"},
            "model": {
                "model_type": "qwen-like",
                "vocab_size": 32000,
                "seq_length": 2048,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "params_dtype": "bfloat16",
            },
        }
        inputs = MindFormersAdapter.from_mapping(data, world_size=16)
        parallel = inputs.parallel_config
        self.assertEqual((parallel.dp_replicate, parallel.dp_shard), (4, 2))
        self.assertEqual(parallel.world_size, 16)
        self.assertEqual(parallel.fsdp_degree, 2)
        self.assertEqual(inputs.model_spec.dims.n_kv, 4)
        self.assertEqual(parallel.num_microbatches, 1)

    def test_recompute_and_swap_fields_follow_pynative_schema(self):
        data = {
            "training": {"local_batch_size": 1, "global_batch_size": 1},
            "parallelism": {},
            "model": {
                "vocab_size": 1000,
                "seq_length": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
            },
            "recompute": {
                "mode": "select",
                "select_module": {"mlp": ["0-1"]},
                "exclude_op": ["matmul"],
            },
            "recompute_comm": {"enable": True, "select_module": {"attention": ["0"]}},
            "swap": {
                "enable": True,
                "default_prefetch": 1,
                "layer_swap": [{"layers": ["2"]}],
                "op_swap": [{"op_name": "attention", "layers": ["0"]}],
            },
        }
        inputs = MindFormersAdapter.from_mapping(data, world_size=1)
        self.assertEqual(inputs.recompute.mode, "select")
        self.assertEqual(inputs.recompute.select_modules["mlp"], frozenset({0, 1}))
        self.assertEqual(inputs.recompute.exclude_ops, frozenset({"matmul"}))
        self.assertTrue(inputs.recompute.recompute_comm)
        self.assertEqual(
            inputs.recompute.comm_select_modules["attention"], frozenset({0})
        )
        self.assertEqual(inputs.swap.layers, frozenset({2}))
        self.assertEqual(inputs.swap.op_layers["attention"], frozenset({0}))
        self.assertEqual(inputs.swap.prefetch_depth, 1)

    def test_mindformers_tp_requires_sequence_parallel(self):
        data = {
            "parallelism": {"tensor_parallel": 2, "sequence_parallel": False},
            "model": {
                "vocab_size": 1000,
                "seq_length": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            },
        }
        with self.assertRaisesRegex(ValueError, "sequence_parallel"):
            MindFormersAdapter.from_mapping(data, world_size=2)

    def test_real_deepseek_pynative_yaml_runs_end_to_end(self):
        relative = Path(
            "tests/st/test_multi_cards_cases/test_pynative/test_models/"
            "test_deepseek3/pynative_ds3.yaml"
        )
        project_root = Path(__file__).resolve().parents[2]
        candidates = (
            project_root.parent / "mindformers" / relative,
            project_root.parent / "mindformers" / "mindformers" / relative,
        )
        config = next((item for item in candidates if item.exists()), candidates[0])
        if not config.exists():
            self.skipTest("local MindFormers checkout is not available")
        inputs = MindFormersAdapter.from_yaml(config, world_size=1)
        self.assertEqual(inputs.model_spec.dims.H, 1792)
        self.assertEqual(inputs.model_spec.layer_pattern[0], "mla_dense")
        self.assertEqual(inputs.model_spec.layer_pattern.count("mla_moe"), 11)
        self.assertEqual(inputs.model_spec.layer_pattern[-1], "mtp")
        self.assertEqual(inputs.model_spec.dims.total_layers, 13)
        self.assertEqual(inputs.parallel_config.dp_shard, 1)
        self.assertEqual(inputs.recompute.layers, frozenset(range(4)))
        report = inputs.evaluator().evaluate()
        self.assertGreater(report.per_stage[0].peak_bytes, 0)
        self.assertFalse(any("MLA" in item or "MTP" in item or "shared" in item
                             for item in report.warnings))

    def test_deepseek4_moe_width_can_fill_base_ffn_alias(self):
        data = {
            "training": {"local_batch_size": 1, "global_batch_size": 1},
            "parallelism": {"expert_parallel": 1},
            "model": {
                "model_type": "deepseek_v4",
                "vocab_size": 1024,
                "seq_length": 128,
                "hidden_size": 128,
                "moe_intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
            },
        }
        inputs = MindFormersAdapter.from_mapping(data, world_size=1)
        self.assertEqual(inputs.model_spec.dims.F, 256)
        self.assertEqual(inputs.model_spec.dims.moe_F, 256)

    def test_explicit_interleaved_pipeline_ranges_are_mapped_by_physical_stage(self):
        data = {
            "training": {"local_batch_size": 1, "global_batch_size": 2},
            "parallelism": {
                "tensor_parallel": 1,
                "pipeline_parallel": 2,
                "pipeline_parallel_interleave_num": 2,
                "pipeline_parallel_layers_per_stage": ["0-4,9-12", "5-8,13-17"],
            },
            "model": {
                "vocab_size": 1024,
                "seq_length": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 18,
                "num_attention_heads": 4,
            },
        }
        inputs = MindFormersAdapter.from_mapping(data, world_size=2)
        report = inputs.evaluator().evaluate()
        self.assertEqual(inputs.parallel_config.layers_per_stage, (
            (0, 1, 2, 3, 4, 9, 10, 11, 12),
            (5, 6, 7, 8, 13, 14, 15, 16, 17),
        ))
        self.assertEqual([stage.stage for stage in report.per_stage], [0, 1])
