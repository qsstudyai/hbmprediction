import pytest

from cost_eval.layers import build_dense_decoder, build_loss_ops
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import ParallelConfig


def _dims(seq=16, vocab=32):
    return DimTable(
        H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
        S=seq, B=1, vocab=vocab, n_layers=1,
    )


def _graph(seq=16, vocab=32, tp=1, loss_parallel=False, variant="fallback_cross_entropy"):
    dims = _dims(seq, vocab)
    spec = ModelSpec(
        "loss-test", dims, ("dense",),
        {"dense": build_dense_decoder(dims)},
        {"loss_variant": variant},
    )
    pc = ParallelConfig(
        tp=tp, sequence_parallel=tp > 1, enable_loss_parallel=loss_parallel
    )
    return ShapeEval().resolve(spec, ParallelModel.build(pc, dims))


def test_fallback_contract_has_fp32_probabilities_and_dlogits_workspace():
    graph = _graph()
    ops = {op.name: op for op in graph.edge_ops[0]}
    probs = ops["loss_fwd_softmax"].output
    assert probs.dtype_bytes == 4
    assert probs.local_shape == (16, 1, 32)
    assert ops["loss_reduce"].backward_workspace_bytes == 2 * probs.local_bytes


def test_loss_buffers_scale_with_sequence_and_vocab():
    base = _graph().edge_ops[0][-2].output.local_bytes
    seq2 = _graph(seq=32).edge_ops[0][-2].output.local_bytes
    vocab2 = _graph(vocab=64).edge_ops[0][-2].output.local_bytes
    assert seq2 == 2 * base
    assert vocab2 == 2 * base


def test_loss_parallel_shards_local_vocab_and_avoids_logits_all_gather():
    plain = _graph(tp=2, loss_parallel=False)
    parallel = _graph(tp=2, loss_parallel=True)
    plain_ops = {op.name: op for op in plain.edge_ops[0]}
    parallel_ops = {op.name: op for op in parallel.edge_ops[0]}
    assert parallel_ops["loss_fwd_softmax"].output.local_bytes * 2 == plain_ops["loss_fwd_softmax"].output.local_bytes
    assert any(item.kind == "all_gather" for item in plain_ops["lm_head"].collectives)
    assert not parallel_ops["lm_head"].collectives


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="unsupported loss"):
        build_loss_ops(_dims(), "mystery", False)
