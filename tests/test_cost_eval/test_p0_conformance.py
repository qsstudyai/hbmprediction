import unittest
from math import prod

from cost_eval import (
    DimTable,
    Evaluator,
    HardwareSpec,
    ModelSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
    TensorRef,
)
from cost_eval.layers import build_dense_decoder, build_moe_decoder
from cost_eval.mem_timeline import _save_plan, build_1f1b
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, detect_reshard, resolve_tensor
from cost_eval.static_mem import StaticMem


def dims(**updates):
    values = dict(
        H=128,
        F=256,
        n_heads=8,
        n_kv=2,
        head_dim=16,
        S=128,
        B=1,
        vocab=1000,
        n_layers=4,
    )
    values.update(updates)
    return DimTable(**values)


def dense_spec(n_layers=4):
    table = dims(n_layers=n_layers)
    return ModelSpec("dense", table, ("dense",) * n_layers, {"dense": build_dense_decoder(table)})


class PlacementConformanceTests(unittest.TestCase):
    def test_sp_uses_tp_mesh_and_degenerates_when_disabled(self):
        table = dims()
        ref = TensorRef("x", ("S", "B", "H"), {0: "sp"})
        enabled = ParallelModel.build(ParallelConfig(tp=4, sequence_parallel=True), table)
        disabled = ParallelModel.build(ParallelConfig(tp=4), table)
        self.assertEqual(resolve_tensor(ref, table.as_dict(), enabled).local_shape, (32, 1, 128))
        self.assertEqual(resolve_tensor(ref, table.as_dict(), disabled).local_shape, (128, 1, 128))
        self.assertEqual(resolve_tensor(ref, table.as_dict(), enabled).placement.shards, ((0, "tp"),))

    def test_cp_and_sp_both_reduce_sequence_activation(self):
        table = dims()
        ref = TensorRef("x", ("S", "B", "H"), {0: "sp"})
        parallel = ParallelModel.build(
            ParallelConfig(tp=2, cp=4, sequence_parallel=True), table
        )
        resolved = resolve_tensor(ref, table.as_dict(), parallel)
        self.assertEqual(resolved.local_shape, (16, 1, 128))
        self.assertEqual(resolved.placement.shards, ((0, "cp"), (0, "tp")))

    def test_reshard_table(self):
        table = dims()
        plain = ParallelModel.build(ParallelConfig(tp=2), table)
        sp = ParallelModel.build(ParallelConfig(tp=2, sequence_parallel=True), table)
        source_partial = resolve_tensor(TensorRef("y", ("S", "B", "H"), partial="tp"), table.as_dict(), plain)
        replicated = resolve_tensor(TensorRef("y", ("S", "B", "H")), table.as_dict(), plain)
        sharded_seq = resolve_tensor(TensorRef("y", ("S", "B", "H"), {0: "sp"}), table.as_dict(), sp)
        self.assertEqual(detect_reshard(source_partial, replicated).kind, "all_reduce")
        self.assertEqual(detect_reshard(source_partial, sharded_seq).kind, "reduce_scatter")

        sharded_hidden = resolve_tensor(TensorRef("z", ("S", "B", "H"), {2: "tp"}), table.as_dict(), plain)
        replicated_z = resolve_tensor(TensorRef("z", ("S", "B", "H")), table.as_dict(), plain)
        sharded_sequence_z = resolve_tensor(TensorRef("z", ("S", "B", "H"), {0: "tp"}), table.as_dict(), plain)
        self.assertEqual(detect_reshard(sharded_hidden, replicated_z).kind, "all_gather")
        self.assertEqual(detect_reshard(sharded_hidden, sharded_sequence_z).kind, "all_to_all")
        self.assertIsNone(detect_reshard(replicated_z, sharded_hidden))

    def test_dense_graph_derives_two_tp_reductions(self):
        spec = dense_spec(1)
        pm = ParallelModel.build(ParallelConfig(tp=2, sequence_parallel=True), spec.dims)
        graph = ShapeEval().resolve(spec, pm)
        collectives = [comm.kind for op in graph.stages[0][0].ops for comm in op.collectives]
        self.assertEqual(collectives.count("reduce_scatter"), 2)

    def test_context_parallel_collective_matches_method(self):
        spec = dense_spec(1)
        kinds = {}
        for method in ("colossal", "ulysses"):
            pc = ParallelConfig(cp=2, context_parallel_method=method)
            graph = ShapeEval().resolve(spec, ParallelModel.build(pc, spec.dims))
            kinds[method] = {
                comm.kind
                for op in graph.stages[0][0].ops
                for comm in op.collectives
                if comm.group_axis == "cp"
            }
        self.assertEqual(kinds["colossal"], {"ring_p2p"})
        self.assertEqual(kinds["ulysses"], {"all_to_all"})

    def test_ep_collectives_degenerate_at_degree_one(self):
        table = dims(n_layers=1, n_experts=8, topk=2, moe_F=256)
        spec = ModelSpec("moe", table, ("moe",), {"moe": build_moe_decoder(table)})
        no_ep = ShapeEval().resolve(spec, ParallelModel.build(ParallelConfig(), table))
        with_ep = ShapeEval().resolve(spec, ParallelModel.build(ParallelConfig(ep=2, tp=2), table))
        no_ep_kinds = [
            comm.kind for op in no_ep.stages[0][0].ops for comm in op.collectives
            if comm.group_axis == "ep"
        ]
        with_ep_kinds = [
            comm.kind for op in with_ep.stages[0][0].ops for comm in op.collectives
            if comm.group_axis == "ep"
        ]
        self.assertEqual(no_ep_kinds, [])
        self.assertEqual(with_ep_kinds.count("all_to_all"), 2)


class MemoryConformanceTests(unittest.TestCase):
    def test_expert_weights_are_pure_ep_and_efsdp_conserves_numel(self):
        table = dims(n_layers=1, n_experts=8, topk=2, moe_F=256)
        spec = ModelSpec("moe", table, ("moe",), {"moe": build_moe_decoder(table)})
        pc = ParallelConfig(tp=2, ep=4, dp_shard=4, sequence_parallel=True)
        graph = ShapeEval().resolve(spec, ParallelModel.build(pc, table))
        expert = {
            tensor.tid: tensor
            for op in graph.stages[0][0].ops
            for tensor in op.params
            if tensor.is_expert
        }
        self.assertTrue(expert)
        for tensor in expert.values():
            self.assertEqual(tensor.placement.shards, ((0, "ep"),))
            self.assertEqual(
                tensor.local_numel * pc.ep * pc.expert_fsdp_degree,
                prod(tensor.global_shape) * pc.expert_fsdp_degree,
            )

    def test_static_breakdown_sums_to_persistent(self):
        spec = dense_spec(1)
        pc = ParallelConfig(tp=2, dp_shard=2, sequence_parallel=True)
        graph = ShapeEval().resolve(spec, ParallelModel.build(pc, spec.dims))
        stage = StaticMem().compute(graph, OptimizerSpec.adamw(), pc)[0]
        self.assertEqual(stage.breakdown.total, stage.persistent_bytes)
        self.assertGreater(stage.breakdown.parameter, 0)
        self.assertGreater(stage.breakdown.optimizer_state, 0)
        self.assertGreater(stage.edge_parameter_bytes, 0)

    def test_embedding_output_and_final_norm_are_counted(self):
        table = dims(n_layers=1)
        spec = ModelSpec(
            "full-model",
            table,
            ("dense",),
            {"dense": build_dense_decoder(table)},
        )
        pc = ParallelConfig(tp=2, sequence_parallel=True)
        graph = ShapeEval().resolve(spec, ParallelModel.build(pc, table))
        names = {tensor.tid for tensor in graph.stage_params[0]}
        self.assertEqual(names, {"embedding_weight", "lm_head_weight", "final_norm_weight"})
        stage = StaticMem().compute(graph, OptimizerSpec.adamw(), pc)[0]
        expected_edge_bytes = 2 * table.vocab * table.H * 2 + table.H * 4
        self.assertEqual(stage.edge_parameter_bytes, expected_edge_bytes)

    def test_never_reshard_has_larger_gather_peak_than_always(self):
        spec = dense_spec(4)
        common = dict(tp=2, dp_shard=2, sequence_parallel=True, num_microbatches=2)
        hardware = HardwareSpec(10**12)
        always = Evaluator(
            spec,
            ParallelConfig(**common, reshard_after_forward_policy="always"),
            OptimizerSpec.adamw(),
            hardware,
        ).evaluate()
        never = Evaluator(
            spec,
            ParallelConfig(**common, reshard_after_forward_policy="never"),
            OptimizerSpec.adamw(),
            hardware,
        ).evaluate()
        self.assertGreater(never.per_stage[0].breakdown.gather_buf, always.per_stage[0].breakdown.gather_buf)
        self.assertGreater(never.per_stage[0].peak_bytes, always.per_stage[0].peak_bytes)

    def test_default_reshard_policy_matches_mindformers_pp_rule(self):
        self.assertTrue(ParallelConfig().reshard_after_forward)
        self.assertFalse(ParallelConfig(pp=2).reshard_after_forward)

    def test_cpu_offload_zeroes_persistent_but_keeps_gather(self):
        spec = dense_spec(1)
        pc = ParallelConfig(cpu_offload=True)
        report = Evaluator(
            spec, pc, OptimizerSpec.adamw(), HardwareSpec(10**12)
        ).evaluate()
        stage = report.per_stage[0]
        self.assertEqual(stage.breakdown.persistent, 0)
        self.assertGreater(stage.bucket_peaks.gather_buf, 0)

    def test_fsdp_prefetch_depth_increases_gather_high_water_mark(self):
        spec = dense_spec(4)
        peaks = []
        for depth in (0, 1, 2):
            report = Evaluator(
                spec,
                ParallelConfig(dp_shard=2, prefetch_depth=depth),
                OptimizerSpec.adamw(),
                HardwareSpec(10**12),
            ).evaluate()
            peaks.append(report.per_stage[0].bucket_peaks.gather_buf)
        self.assertLess(peaks[0], peaks[1])
        self.assertLess(peaks[1], peaks[2])

    def test_select_exclude_and_op_swap_partition_saved_tensors(self):
        spec = dense_spec(1)
        graph = ShapeEval().resolve(spec, ParallelModel.build(ParallelConfig(), spec.dims))
        layer = graph.stages[0][0]
        selected = _save_plan(layer, RecomputeSpec("select", select_modules={"mlp": frozenset()}), SwapSpec())
        excluded = _save_plan(
            layer,
            RecomputeSpec("full", exclude_ops=frozenset({"flash_attn"})),
            SwapSpec(),
        )
        swapped = _save_plan(
            layer,
            RecomputeSpec(),
            SwapSpec(enabled=True, op_names=frozenset({"flash_attn"})),
        )
        self.assertGreater(selected.recomputed, 0)
        self.assertGreater(excluded.resident, layer.checkpoint_bytes)
        self.assertGreater(swapped.offloaded, 0)
        for plan in (selected, excluded, swapped):
            self.assertEqual(plan.resident + plan.recomputed + plan.offloaded, layer.activation_bytes)


class ScheduleConformanceTests(unittest.TestCase):
    def test_unsupported_schedule_is_rejected_and_interleave_is_supported(self):
        with self.assertRaisesRegex(ValueError, "1f1b"):
            ParallelConfig(pipeline_schedule="gpipe")
        parallel = ParallelConfig(
            pp=2,
            pipeline_interleave=2,
            num_microbatches=2,
        )
        self.assertEqual(parallel.interleave, 2)

    def test_1f1b_warmup_and_event_balance(self):
        events = build_1f1b(stage=0, pp=4, microbatches=8)
        prefix = []
        for event in events:
            if event.kind != "FWD":
                break
            prefix.append(event)
        self.assertEqual(len(prefix), 4)
        self.assertEqual(sum(event.kind == "FWD" for event in events), 8)
        self.assertEqual(sum(event.kind == "BWD" for event in events), 8)
