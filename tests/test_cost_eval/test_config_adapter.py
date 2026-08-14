import json

import pytest

from cost_eval.config_adapter import ConfigAdapter


def base_config():
    return {
        "name": "adapter-dense",
        "model": {
            "type": "dense",
            "H": 8,
            "F": 16,
            "n_heads": 2,
            "n_kv": 2,
            "head_dim": 4,
            "S": 4,
            "B": 1,
            "vocab": 10,
            "n_layers": 2,
        },
        "parallel": {
            "tensor_parallel": 2,
            "data_parallel_shard": 2,
            "num_microbatches": 2,
        },
        "optimizer": {"type": "AdamW"},
        "hardware": {
            "max_device_memory": "1 GiB",
            "framework_reserve": "4 MiB",
        },
        "recompute": {
            "mode": "select",
            "select_ops": {0: ["fc1", "swiglu"]},
        },
        "swap": {
            "enable": True,
            "swap_ops": {1: ["flash"]},
        },
    }


def test_dict_adapter_builds_runnable_evaluator():
    inputs = ConfigAdapter.from_dict(base_config())
    assert inputs.parallel.tp == 2
    assert inputs.hardware.max_device_memory == 2**30
    assert inputs.recompute.recomputed_ops(0, ("fc1", "swiglu")) == {
        "fc1",
        "swiglu",
    }
    report = inputs.evaluator().evaluate()
    assert report.per_stage[0].peak_bytes > 0


def test_json_adapter(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_config()), encoding="utf-8")
    inputs = ConfigAdapter.load(path)
    assert inputs.model_spec.name == "adapter-dense"


def test_yaml_subset_adapter(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
name: yaml-dense
model:
  type: dense
  H: 8
  F: 16
  n_heads: 2
  n_kv: 2
  head_dim: 4
  S: 4
  B: 1
  vocab: 10
  n_layers: 2
parallel:
  tensor_parallel: 2
  data_parallel_shard: 2
  num_microbatches: 2
optimizer:
  type: AdamW
hardware:
  max_device_memory: 1 GiB
  framework_reserve: 4 MiB
recompute:
  mode: select
  select_ops:
    0: [fc1, swiglu]
swap:
  enable: true
  swap_ops:
    1: [flash]
""".strip(),
        encoding="utf-8",
    )
    inputs = ConfigAdapter.load(path)
    assert inputs.model_spec.name == "yaml-dense"
    assert inputs.swap.swapped_ops(1, ("flash", "fc1")) == {"flash"}
    assert not inputs.evaluator().evaluate().oom


def test_adapter_rejects_unknown_fields():
    config = base_config()
    config["parallel"]["mystery_axis"] = 2
    with pytest.raises(ValueError, match="未知并行配置项"):
        ConfigAdapter.from_dict(config)


def test_single_op_name_is_not_split_into_characters():
    config = base_config()
    config["recompute"]["select_ops"] = {0: "fc1"}
    inputs = ConfigAdapter.from_dict(config)
    assert inputs.recompute.recomputed_ops(0, ("fc1", "flash")) == {"fc1"}
