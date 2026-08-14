import pytest

from cost_eval.model_spec import (
    DimTable,
    LayerSpec,
    ModelSpec,
    OpSpec,
    OpType,
    TensorRef,
)
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import (
    Placement,
    ShapeEval,
    detect_reshard,
    eval_expr,
    resolve_tensor,
)
from cost_eval.specs import ParallelConfig

DIMS = DimTable(
    H=8,
    F=16,
    n_heads=2,
    n_kv=2,
    head_dim=4,
    S=4,
    B=1,
    vocab=10,
    n_layers=2,
)


def pm(**kwargs):
    return ParallelModel(ParallelConfig(**kwargs), n_layers=2)


def test_restricted_expression_evaluation():
    assert eval_expr("H", DIMS) == 8
    assert eval_expr("(n_heads+2*n_kv)*head_dim", DIMS) == 24
    assert eval_expr("2*F//4", DIMS) == 8
    with pytest.raises(ValueError):
        eval_expr("__import__('os').system('echo unsafe')", DIMS)
    with pytest.raises(ValueError):
        eval_expr("H/2", DIMS)


def test_resolve_tensor_applies_tp_and_marks_expert():
    resolved = resolve_tensor(
        TensorRef("y", ("S", "B", "2*F"), shard={2: "tp"}),
        DIMS,
        pm(tp=8),
    )
    assert resolved.local_numel == 4 * 1 * 4

    expert_dims = DimTable(
        **{
            **DIMS.__dict__,
            "n_experts": 8,
        }
    )
    expert = resolve_tensor(
        TensorRef(
            "w",
            ("n_experts", "H"),
            shard={0: "ep"},
            is_weight=True,
        ),
        expert_dims,
        pm(tp=2, dp_shard=2, ep=4),
    )
    assert expert.is_expert
    assert expert.local_numel == 2 * 8


def test_resolve_tensor_rejects_indivisible_dimension():
    with pytest.raises(ValueError, match="不被"):
        resolve_tensor(
            TensorRef("y", ("H",), shard={0: "tp"}),
            DIMS,
            pm(tp=3),
        )


def test_placement_algebra_detects_collectives():
    reduction = detect_reshard(
        Placement(partial="tp"), Placement(), 128, 2
    )
    assert reduction.ctype == "all_reduce"
    assert reduction.group_axis == "tp"
    assert reduction.volume_bytes == 256

    all_to_all = detect_reshard(
        Placement({0: "ep"}), Placement({1: "ep"}), 64, 2
    )
    assert all_to_all.ctype == "all_to_all"
    placement = Placement({2: "tp"})
    assert detect_reshard(placement, placement, 64, 2) is None


def test_shape_eval_groups_layers_and_derives_reduce_scatter():
    x = TensorRef("x", ("S", "B", "H"))
    weight = TensorRef(
        "w", ("H", "2*F"), shard={1: "tp"}, is_weight=True
    )
    partial = TensorRef("partial", ("S", "B", "2*F"), partial="tp")
    sharded = TensorRef("partial", ("S", "B", "2*F"), shard={2: "tp"})
    output = TensorRef("out", ("S", "B", "2*F"), shard={2: "tp"})
    layer = LayerSpec(
        (
            OpSpec(
                "fc",
                OpType.MATMUL,
                (x, weight),
                partial,
                params=(weight,),
                saves=(x, x),
            ),
            OpSpec("consume", OpType.ELEMENTWISE, (sharded,), output),
        )
    )
    spec = ModelSpec(
        "toy", DIMS, ("dense", "dense"), {"dense": layer}
    )
    graph = ShapeEval().resolve(spec, pm(tp=2, pp=2))
    assert set(graph.stages) == {0, 1}
    first = graph.stages[0][0].ops[0]
    assert first.params[0].local_numel == 8 * 16
    assert len(first.saves) == 1
    comm = graph.stages[0][0].ops[1].collectives[0]
    assert comm.ctype == "reduce_scatter"
