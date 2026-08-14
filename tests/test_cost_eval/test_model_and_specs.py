import pytest

import cost_eval
from cost_eval.model_spec import (
    DimTable,
    LayerSpec,
    ModelSpec,
    OpSpec,
    OpType,
    TensorRef,
)
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)


def dims(n_layers=2):
    return DimTable(
        H=8,
        F=16,
        n_heads=2,
        n_kv=2,
        head_dim=4,
        S=4,
        B=1,
        vocab=10,
        n_layers=n_layers,
    )


def test_public_package_imports():
    assert cost_eval.Evaluator is not None
    assert OptimizerSpec.adamw().state_bytes_per_param == 16
    assert OptimizerSpec.adamw(fp32_grad=True).state_bytes_per_param == 18


def test_memory_contract_and_model_lookup():
    weight = TensorRef(
        "weight", ("H", "F"), shard={1: "tp"}, is_weight=True
    )
    x = TensorRef("x", ("S", "B", "H"))
    op = OpSpec(
        "fc",
        OpType.MATMUL,
        (x, weight),
        TensorRef("y", ("S", "B", "F"), shard={2: "tp"}),
        params=(weight,),
        saves=(x,),
    )
    layer = LayerSpec((op,))
    spec = ModelSpec("toy", dims(), ("dense", "dense"), {"dense": layer})
    assert spec.get_layer("dense") is layer
    assert op.params[0].is_weight
    assert op.saves[0].name == "x"


def test_model_rejects_inconsistent_layer_pattern():
    with pytest.raises(ValueError, match="layer_pattern"):
        ModelSpec("bad", dims(), ("dense",), {"dense": LayerSpec(())})


def test_parallel_degrees_and_mindformers_stage_assignment():
    pc = ParallelConfig(
        tp=4, dp_shard=2, ep=2, pp=2, sequence_parallel=True
    )
    pm = ParallelModel(pc, n_layers=6)
    assert pm.degree("sp") == 4
    assert pm.fsdp_degree() == 2
    assert pm.efsdp_degree() == 4
    assert pm.stage_layers(0) == [0, 1, 2]
    assert pm.stage_layers(1) == [3, 4, 5]
    uneven = ParallelModel(pc, n_layers=5)
    assert uneven.stage_layers(0) == [0, 1, 2]
    assert uneven.stage_layers(1) == [3, 4]


def test_ep_one_experts_use_dense_fsdp_mesh():
    pc = ParallelConfig(
        tp=2, cp=2, ep=1, dp_shard=2, sequence_parallel=True
    )
    pm = ParallelModel(pc, n_layers=2)
    # MindFormers does not create an eFSDP wrapper for ep=1; the expert
    # weights are part of the layer's ordinary fsdp wrapper.
    assert pm.fsdp_degree() == 4
    assert pm.efsdp_degree() == 4
    assert pc.expert_fsdp_degree == 4


def test_parallel_rejects_invalid_expert_region_and_world_size():
    with pytest.raises(ValueError, match="必须整除"):
        ParallelModel(ParallelConfig(tp=2, ep=4), 2)
    with pytest.raises(ValueError, match="world_size"):
        ParallelModel(ParallelConfig(tp=2), 2, world_size=4)


def test_config_selection_semantics_and_validation():
    assert RecomputeSpec("FULL").is_full(999)
    assert RecomputeSpec("full", {1}).is_full(1)
    assert not RecomputeSpec("full", {1}).is_full(0)
    assert SwapSpec(True).swaps(3)
    assert not SwapSpec(True, swap_layers={1}).swaps(0)
    with pytest.raises(ValueError):
        HardwareSpec(0)
    with pytest.raises(ValueError):
        ParallelConfig(tp=0)


def test_fine_grained_recompute_and_swap_selection():
    available = ("flash", "fc1", "swiglu", "fc2")
    selected = RecomputeSpec(
        "select",
        select_ops={-1: {"fc1", "swiglu"}},
        exclude_ops={0: {"swiglu"}},
    )
    assert selected.recomputed_ops(0, available) == {"fc1"}
    assert selected.recomputed_ops(1, available) == {"fc1", "swiglu"}

    swapped = SwapSpec(
        True,
        swap_ops={0: {"flash"}, 1: {"fc1", "swiglu"}},
    )
    assert swapped.swapped_ops(0, available) == {"flash"}
    assert swapped.swapped_ops(1, available) == {"fc1", "swiglu"}
    assert swapped.swapped_ops(2, available) == frozenset()

    with pytest.raises(ValueError, match="未知算子"):
        RecomputeSpec(
            "select", select_ops={0: {"missing"}}
        ).recomputed_ops(0, available)
