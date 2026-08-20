import pytest

from cost_eval.model_spec import DimTable, TensorRef
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import Placement, detect_reshards, eval_expr, resolve_tensor
from cost_eval.specs import ParallelConfig


DIMS = DimTable(
    H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
    S=16, B=1, vocab=32, n_layers=1,
)


def test_sp_cp_to_cp_only_removes_tp_but_keeps_cp():
    pm = ParallelModel.build(
        ParallelConfig(tp=2, cp=4, sequence_parallel=True), DIMS
    )
    source = resolve_tensor(TensorRef("x", ("S", "B", "H"), {0: "sp"}), DIMS, pm)
    dest = resolve_tensor(TensorRef("x", ("S", "B", "H"), {0: "cp"}), DIMS, pm)
    actions = detect_reshards(source, dest)

    assert source.placement.shards == ((0, "cp"), (0, "tp"))
    assert dest.placement.shards == ((0, "cp"),)
    assert [(item.kind, item.group_axis) for item in actions] == [("all_gather", "tp")]
    assert dest.local_numel == source.local_numel * 2


def test_partial_tp_to_sp_cp_is_one_reduce_scatter():
    source = Placement(((0, "cp"),), partial="tp")
    dest = Placement(((0, "cp"), (0, "tp")))
    actions = detect_reshards(source, dest, 64, 2)
    assert [(item.kind, item.group_axis) for item in actions] == [("reduce_scatter", "tp")]


def test_multi_axis_removal_has_stable_order():
    source = Placement(((0, "tp"), (1, "ep")))
    actions = detect_reshards(source, Placement(), 64, 2)
    assert [(item.kind, item.group_axis) for item in actions] == [
        ("all_gather", "ep"),
        ("all_gather", "tp"),
    ]


def test_floor_division_requires_exact_shape_contract():
    with pytest.raises(ValueError, match="整除"):
        eval_expr("S//3", DIMS)
