from cost_eval.layers import build_dense_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import OptimizerSpec, ParallelConfig
from cost_eval.static_mem import StaticMem


def _static(pc):
    dims = DimTable(
        H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
        S=16, B=1, vocab=32, n_layers=1,
    )
    spec = ModelSpec(
        "dense", dims, ("dense",), {"dense": build_dense_decoder(dims)}
    )
    pm = ParallelModel.build(pc, dims)
    graph = ShapeEval().resolve(spec, pm)
    return StaticMem().compute(graph, OptimizerSpec.adamw(), pm)[0]


def test_optimizer_only_offload_keeps_parameters_and_gradients_on_device():
    normal = _static(ParallelConfig())
    offloaded = _static(ParallelConfig(optimizer_offload=True))
    assert offloaded.breakdown.parameter == normal.breakdown.parameter
    assert offloaded.breakdown.gradient == normal.breakdown.gradient
    assert offloaded.breakdown.master_weight == 0
    assert offloaded.breakdown.optimizer_state == 0
    assert offloaded.host_breakdown.master_weight == normal.breakdown.master_weight


def test_legacy_cpu_offload_maps_to_all_independent_policies():
    offloaded = _static(ParallelConfig(cpu_offload=True))
    assert offloaded.persistent_bytes == 0
    assert offloaded.host_pinned_bytes > 0
