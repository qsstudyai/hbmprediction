from cost_eval.adapters import MindFormersAdapter
from cost_eval.shape_eval import ShapeEval
from cost_eval.parallel_model import ParallelModel


def _config():
    return {
        "model": {
            "model_type": "qwen3",
            "hidden_size": 64,
            "ffn_hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_layers": 2,
            "seq_length": 64,
            "vocab_size": 128,
        },
        "training": {"local_batch_size": 1, "global_batch_size": 2},
        "parallelism": {"tensor_parallel": 2, "sequence_parallel": True},
        "context": {"max_device_memory": "60GB"},
    }


def test_adapter_selects_qwen3_family_contract():
    inputs = MindFormersAdapter.from_mapping(_config(), world_size=2)
    assert inputs.model_spec.capabilities["model_family"] == "qwen3"
    assert inputs.model_spec.capabilities["source_commit"]
    assert inputs.model_spec.layer_specs["dense"].ops[2].name == "qk_norm"


def test_qwen3_norm_parameters_and_saved_dtypes_are_explicit():
    inputs = MindFormersAdapter.from_mapping(_config(), world_size=2)
    graph = ShapeEval().resolve(
        inputs.model_spec,
        ParallelModel.build(inputs.parallel_config, inputs.model_spec.dims),
    )
    ops = {op.name: op for op in graph.stages[0][0].ops}
    assert all(t.dtype_bytes == 4 for t in ops["qk_norm"].params)
    assert next(t for t in ops["flash"].saves if t.name == "lse").dtype_bytes == 4
    assert {t.name for t in ops["rope"].saves} >= {
        "position_ids", "rope_cos", "rope_sin"
    }
