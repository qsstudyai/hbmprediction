from cost_eval.layers.deepseek import build_mla_moe_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import ParallelConfig


def _dims():
    return DimTable(
        H=64, F=128, n_heads=8, n_kv=1, head_dim=8,
        S=64, B=1, vocab=128, n_layers=1,
        n_experts=8, topk=2, moe_F=64,
        q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8,
    )


def test_router_and_dispatch_metadata_have_explicit_shape_dtype_and_placement():
    dims = _dims()
    spec = ModelSpec(
        "deepseek", dims, ("mla_moe",),
        {"mla_moe": build_mla_moe_decoder(dims)},
    )
    pc = ParallelConfig(tp=2, dp_shard=2, ep=2, sequence_parallel=True)
    graph = ShapeEval().resolve(spec, ParallelModel.build(pc, dims))
    ops = {op.name: op for op in graph.stages[0][0].ops}
    router_saved = {tensor.name: tensor for tensor in ops["router"].saves}
    assert router_saved["logits"].dtype_bytes == 4
    assert router_saved["logits"].placement.shards == ((0, "tp"),)
    assert router_saved["router_topk_indices"].dtype_bytes == 4
    assert router_saved["router_token_counts"].local_shape == (4,)
    assert {tensor.name for tensor in ops["dispatch"].saves} >= {
        "router_permute_map", "router_inverse_map"
    }


def test_grouped_gemm_workspace_scales_with_local_routed_output():
    dims = _dims()
    spec = ModelSpec(
        "deepseek", dims, ("mla_moe",),
        {"mla_moe": build_mla_moe_decoder(dims)},
    )
    pc = ParallelConfig(dp_shard=2, ep=2)
    graph = ShapeEval().resolve(spec, ParallelModel.build(pc, dims))
    op = next(op for op in graph.stages[0][0].ops if op.name == "expert_fc1")
    assert op.workspace_bytes == op.output.local_bytes
