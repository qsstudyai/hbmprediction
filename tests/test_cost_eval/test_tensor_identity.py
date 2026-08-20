from cost_eval.mem_timeline import _unique_tensors
from cost_eval.model_spec import DimTable, TensorRef
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import resolve_tensor
from cost_eval.specs import ParallelConfig


DIMS = DimTable(
    H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
    S=16, B=1, vocab=32, n_layers=1,
)


def _pm():
    return ParallelModel.build(
        ParallelConfig(tp=2, cp=2, sequence_parallel=True), DIMS
    )


def test_same_logical_name_with_different_placement_is_not_aliased():
    sp = resolve_tensor(TensorRef("x", ("S", "B", "H"), {0: "sp"}), DIMS, _pm())
    cp = resolve_tensor(TensorRef("x", ("S", "B", "H"), {0: "cp"}), DIMS, _pm())
    replicated = resolve_tensor(TensorRef("x", ("S", "B", "H")), DIMS, _pm())

    assert len({sp.value_id, cp.value_id, replicated.value_id}) == 3
    assert sum(t.local_bytes for t in _unique_tensors((sp, cp, replicated))) == (
        sp.local_bytes + cp.local_bytes + replicated.local_bytes
    )


def test_only_explicit_storage_id_aliases_values():
    left = resolve_tensor(
        TensorRef("embedding", ("vocab", "H"), is_weight=True, storage_id="tokens"),
        DIMS,
        _pm(),
    )
    right = resolve_tensor(
        TensorRef("lm_head", ("vocab", "H"), is_weight=True, storage_id="tokens"),
        DIMS,
        _pm(),
    )
    assert left.value_id != right.value_id
    assert left.storage_key == right.storage_key == "tokens"
    assert len(tuple(_unique_tensors((left, right)))) == 1
