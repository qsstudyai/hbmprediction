from math import prod

from cost_eval.layers import build_dense_decoder, build_moe_decoder
from cost_eval.model_spec import DimTable, ModelSpec, OpType
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, eval_expr
from cost_eval.specs import OptimizerSpec, ParallelConfig
from cost_eval.static_mem import StaticMem

DENSE_DIMS = DimTable(
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


def parameter_numel(layer, dims):
    return sum(
        prod(eval_expr(dim, dims) for dim in weight.shape)
        for op in layer.ops
        for weight in op.params
    )


def test_dense_graph_parameter_formula_and_flash_contract():
    layer = build_dense_decoder(DENSE_DIMS)
    qkv = (
        DENSE_DIMS.H
        * (DENSE_DIMS.n_heads + 2 * DENSE_DIMS.n_kv)
        * DENSE_DIMS.head_dim
    )
    output = (
        DENSE_DIMS.n_heads * DENSE_DIMS.head_dim * DENSE_DIMS.H
    )
    mlp = 3 * DENSE_DIMS.H * DENSE_DIMS.F
    assert parameter_numel(layer, DENSE_DIMS) == qkv + output + mlp
    flash = next(op for op in layer.ops if op.type == OpType.FLASH_ATTN)
    assert {tensor.name for tensor in flash.saves} == {"qkv", "attn", "lse"}


def test_dense_uses_all_reduce_without_sp_and_reduce_scatter_with_sp():
    layer = build_dense_decoder(DENSE_DIMS)
    spec = ModelSpec(
        "dense", DENSE_DIMS, ("dense", "dense"), {"dense": layer}
    )
    no_sp = ShapeEval().resolve(
        spec, ParallelModel(ParallelConfig(tp=2), 2)
    )
    with_sp = ShapeEval().resolve(
        spec,
        ParallelModel(
            ParallelConfig(tp=2, sequence_parallel=True), 2
        ),
    )
    no_sp_add = next(
        op for op in no_sp.stages[0][0].ops if op.name == "add1"
    )
    with_sp_add = next(
        op for op in with_sp.stages[0][0].ops if op.name == "add1"
    )
    assert no_sp_add.collectives[0].ctype == "all_reduce"
    assert with_sp_add.collectives[0].ctype == "reduce_scatter"


def test_moe_experts_are_sharded_exactly_once_by_ep():
    dims = DimTable(
        H=8,
        F=16,
        n_heads=2,
        n_kv=2,
        head_dim=4,
        S=4,
        B=1,
        vocab=10,
        n_layers=2,
        n_experts=8,
        topk=2,
        moe_F=16,
    )
    layer = build_moe_decoder(dims)
    expert_ops = [op for op in layer.ops if op.type == OpType.MOE_GEMM]
    assert len(expert_ops) == 2
    assert all(
        list(weight.shard.values()) == ["ep"]
        for op in expert_ops
        for weight in op.params
    )

    spec = ModelSpec("moe", dims, ("moe", "moe"), {"moe": layer})
    pm = ParallelModel(
        ParallelConfig(tp=2, dp_shard=2, ep=4), dims.n_layers
    )
    graph = ShapeEval().resolve(spec, pm)
    resolved_w1 = next(
        weight
        for op in graph.stages[0][0].ops
        for weight in op.params
        if weight.name == "expert_w1"
    )
    assert resolved_w1.local_numel == 2 * 8 * 32


def test_static_memory_single_device_and_sharding_conservation():
    layer = build_dense_decoder(DENSE_DIMS)
    spec = ModelSpec(
        "dense",
        DENSE_DIMS,
        ("dense", "dense"),
        {"dense": layer},
    )
    global_params = 2 * parameter_numel(layer, DENSE_DIMS)

    single_pm = ParallelModel(ParallelConfig(), 2)
    single_graph = ShapeEval().resolve(spec, single_pm)
    single = StaticMem().compute(
        single_graph, OptimizerSpec.adamw(), single_pm, False
    )
    assert single[0] == global_params * 16

    sharded_pm = ParallelModel(ParallelConfig(tp=2, dp_shard=2), 2)
    sharded_graph = ShapeEval().resolve(spec, sharded_pm)
    sharded = StaticMem().compute(
        sharded_graph, OptimizerSpec.adamw(), sharded_pm, False
    )
    assert (sharded[0] // 16) * 4 == global_params
    offloaded = StaticMem().compute(
        sharded_graph, OptimizerSpec.adamw(), sharded_pm, True
    )
    assert offloaded[0] == 0
