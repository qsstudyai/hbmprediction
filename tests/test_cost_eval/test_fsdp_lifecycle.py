from cost_eval.layers import build_dense_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import OptimizerSpec, ParallelConfig
from cost_eval.static_mem import StaticMem


def test_flatten_group_padding_allows_individually_indivisible_parameters():
    dims = DimTable(
        H=6, F=9, n_heads=2, n_kv=2, head_dim=3,
        S=4, B=1, vocab=10, n_layers=1,
    )
    spec = ModelSpec(
        "dense", dims, ("dense",), {"dense": build_dense_decoder(dims)}
    )
    pc = ParallelConfig(
        dp_shard=4, fsdp_flatten_alignment_bytes=16,
    )
    pm = ParallelModel.build(pc, dims)
    graph = ShapeEval().resolve(spec, pm)
    memory = StaticMem().compute(graph, OptimizerSpec.adamw(), pm)[0]
    assert memory.persistent_bytes > 0
    assert memory.breakdown.total == memory.persistent_bytes
