import unittest
from math import prod

from cost_eval import (
    DimTable, Evaluator, HardwareSpec, ModelSpec, OptimizerSpec,
    ParallelConfig, RecomputeSpec, SwapSpec,
)
from cost_eval.layers import (
    build_mla_dense_decoder,
    build_mla_moe_decoder,
    build_mtp_layer,
)
from cost_eval.mem_timeline import _save_plan
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem


def deepseek_dims(**updates):
    values = dict(
        H=128, F=256, n_heads=8, n_kv=8, head_dim=16,
        S=128, B=1, vocab=1024, n_layers=2,
        n_experts=8, topk=2, moe_F=64, n_shared=1, shared_F=96,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    )
    values.update(updates)
    return DimTable(**values)


class DeepSeekP05Tests(unittest.TestCase):
    def test_mla_parameter_formulas_and_tp_placements(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        graph = ShapeEval().resolve(
            spec, ParallelModel.build(ParallelConfig(tp=2, sequence_parallel=True), dims)
        )
        params = {
            tensor.name: tensor
            for op in graph.stages[0][0].ops for tensor in op.params
        }
        self.assertEqual(
            prod(params["mla_qkv_down_w"].global_shape),
            dims.H * (dims.q_lora_rank + dims.kv_lora_rank + dims.qk_rope_head_dim),
        )
        self.assertEqual(params["mla_qkv_down_w"].placement.shards, ())
        self.assertEqual(params["mla_q_up_w"].placement.shards, ((1, "tp"),))
        self.assertEqual(params["mla_kv_up_w"].placement.shards, ((1, "tp"),))
        self.assertEqual(params["mla_o_w"].placement.shards, ((0, "tp"),))

    def test_shared_expert_is_tp_replicated_and_dense_fsdp(self):
        dims = deepseek_dims(n_layers=1, use_shared_expert_gating=True)
        spec = ModelSpec("moe", dims, ("mla_moe",), {
            "mla_moe": build_mla_moe_decoder(dims),
        })
        graph = ShapeEval().resolve(
            spec, ParallelModel.build(
                ParallelConfig(tp=2, ep=2, dp_shard=2, sequence_parallel=True), dims
            )
        )
        params = {
            tensor.name: tensor
            for op in graph.stages[0][0].ops for tensor in op.params
        }
        for name in ("shared_fc1_w", "shared_fc2_w", "shared_expert_gate_w"):
            self.assertEqual(params[name].placement.shards, ())
            self.assertFalse(params[name].is_expert)
        self.assertEqual(params["shared_expert_gate_w"].dtype_bytes, 4)

    def test_mtp_uses_global_layer_id_and_final_pipeline_stage(self):
        dims = deepseek_dims(n_mtp_layers=1)
        spec = ModelSpec(
            "mtp", dims, ("mla_dense", "mla_moe", "mtp"),
            {
                "mla_dense": build_mla_dense_decoder(dims),
                "mla_moe": build_mla_moe_decoder(dims),
                "mtp": build_mtp_layer(dims),
            },
        )
        graph = ShapeEval().resolve(spec, ParallelModel.build(ParallelConfig(pp=2), dims))
        self.assertEqual([layer.layer_id for layer in graph.stages[1]], [1, 2])
        self.assertEqual(graph.stages[1][-1].layer_type, "mtp")

    def test_loss_parallel_retains_vocab_shard(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        outputs = []
        for enabled in (False, True):
            pc = ParallelConfig(tp=2, sequence_parallel=True, enable_loss_parallel=enabled)
            graph = ShapeEval().resolve(spec, ParallelModel.build(pc, dims))
            outputs.append(graph.stage_output_bytes[0])
        self.assertEqual(outputs[0], outputs[1] * 2)

    def test_hybrid_cp_emits_inner_and_outer_collectives(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        pc = ParallelConfig(cp=4, context_parallel_method="hybrid", ulysses_degree_in_cp=2)
        graph = ShapeEval().resolve(spec, ParallelModel.build(pc, dims))
        flash = next(op for op in graph.stages[0][0].ops if op.name == "mla_flash")
        self.assertEqual({item.group_axis for item in flash.collectives}, {"ulysses_cp", "ring_cp"})

    def test_deredundancy_dispatcher_partitions_routed_tokens_over_tp(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("moe", dims, ("mla_moe",), {
            "mla_moe": build_mla_moe_decoder(dims),
        })
        sizes = []
        for dispatcher in ("alltoall", "alltoall_deredundency"):
            pc = ParallelConfig(
                tp=2, sequence_parallel=True, moe_token_dispatcher=dispatcher
            )
            layer = ShapeEval().resolve(spec, ParallelModel.build(pc, dims)).stages[0][0]
            dispatched = next(op.output for op in layer.ops if op.name == "dispatch")
            sizes.append(dispatched.local_numel)
        self.assertEqual(sizes[0], sizes[1] * 2)

    def test_deredundancy_reduces_dispatch_temporary_buffers(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("moe", dims, ("mla_moe",), {
            "mla_moe": build_mla_moe_decoder(dims),
        })
        workspaces = []
        for dispatcher in ("alltoall", "alltoall_deredundency"):
            pc = ParallelConfig(tp=2, ep=2, sequence_parallel=True, moe_token_dispatcher=dispatcher)
            layer = ShapeEval().resolve(spec, ParallelModel.build(pc, dims)).stages[0][0]
            dispatch = next(op for op in layer.ops if op.name == "dispatch")
            workspaces.append(
                dispatch.workspace_bytes + max(item.volume_bytes for item in dispatch.collectives)
            )
        self.assertEqual(workspaces[0], workspaces[1] * 2)

    def test_edge_modules_contribute_fsdp_gather_peak(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        report = Evaluator(
            spec, ParallelConfig(dp_shard=2), OptimizerSpec.adamw(),
            HardwareSpec(10**12),
        ).evaluate()
        self.assertGreater(report.per_stage[0].bucket_peaks.gather_buf, 0)

    def test_edge_modules_are_resolved_ops_not_static_only_parameters(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        pc = ParallelConfig(dp_shard=2)
        pm = ParallelModel.build(pc, dims)
        graph = ShapeEval().resolve(spec, pm)
        self.assertEqual(
            [op.name for op in graph.edge_ops[0]],
            [
                "embedding", "final_norm", "lm_head",
                "loss_cast_fp32", "loss_fwd_softmax", "loss_reduce",
            ],
        )
        # Static memory is sourced from edge ops; stage_params is compatibility-only.
        graph_without_compat = graph.__class__(
            graph.stages, {}, graph.stage_output_bytes, graph.edge_ops
        )
        stage = StaticMem().compute(graph_without_compat, OptimizerSpec.adamw(), pm)[0]
        self.assertGreater(stage.edge_parameter_bytes, 0)

    def test_recompute_comm_releases_forward_activation_and_reissues_bytes(self):
        dims = deepseek_dims(n_layers=1)
        spec = ModelSpec("mla", dims, ("mla_dense",), {
            "mla_dense": build_mla_dense_decoder(dims),
        })
        pc = ParallelConfig(tp=2, sequence_parallel=True)
        layer = ShapeEval().resolve(spec, ParallelModel.build(pc, dims)).stages[0][0]
        baseline = _save_plan(layer, RecomputeSpec(), SwapSpec())
        selected = _save_plan(
            layer,
            RecomputeSpec(
                recompute_comm=True,
                comm_select_modules={"self_attention.linear_qb": frozenset({0})},
            ),
            SwapSpec(),
        )
        self.assertGreater(selected.recompute_comm_bytes, 0)
        self.assertLess(selected.resident_bytes, baseline.resident_bytes)
        q_up = next(op for op in layer.ops if op.name == "mla_q_up")
        self.assertEqual(q_up.collectives[0].output_tensor, "mla_q_a_norm")


if __name__ == "__main__":
    unittest.main()
