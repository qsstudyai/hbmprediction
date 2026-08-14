import unittest

from cost_eval import (
    DimTable,
    Evaluator,
    HardwareSpec,
    ModelSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)
from cost_eval.layers import build_dense_decoder, build_moe_decoder


GIB = 2**30


def dense_spec(n_layers=4):
    dims = DimTable(
        H=1024, F=4096, n_heads=16, n_kv=16, head_dim=64,
        S=1024, B=1, vocab=32000, n_layers=n_layers,
    )
    return ModelSpec("dense", dims, ("dense",) * n_layers, {"dense": build_dense_decoder(dims)})


def evaluate(spec, parallel, memory=80 * GIB, recompute=None, swap=None):
    return Evaluator(
        spec,
        parallel,
        OptimizerSpec.adamw(),
        HardwareSpec(memory, framework_reserve=256 * 2**20),
        recompute,
        swap,
    ).evaluate()


class EvaluatorTests(unittest.TestCase):
    def test_dense_report_has_breakdown_and_oom_decision(self):
        report = evaluate(dense_spec(), ParallelConfig(tp=2, dp_shard=2, num_microbatches=2))
        peak = report.per_stage[0]
        self.assertGreater(peak.breakdown.persistent, 0)
        self.assertNotEqual(peak.peak_event, "initial")
        self.assertEqual(report.oom, peak.peak_bytes > 80 * GIB)
        self.assertIsNotNone(report.summary)
        self.assertEqual(report.summary.num_layers, 4)
        self.assertEqual(report.static_per_stage[0].total, peak.breakdown.persistent)

    def test_more_tensor_parallelism_reduces_peak_memory(self):
        spec = dense_spec()
        tp1 = evaluate(spec, ParallelConfig(tp=1, dp_shard=2)).per_stage[0].peak_bytes
        tp4 = evaluate(spec, ParallelConfig(tp=4, dp_shard=2)).per_stage[0].peak_bytes
        self.assertLess(tp4, tp1)

    def test_full_recompute_reduces_pipeline_activation_peak(self):
        spec = dense_spec(n_layers=8)
        parallel = ParallelConfig(tp=2, pp=2, num_microbatches=6)
        normal = evaluate(spec, parallel)
        recomputed = evaluate(spec, parallel, recompute=RecomputeSpec("full"))
        self.assertLess(recomputed.per_stage[0].peak_bytes, normal.per_stage[0].peak_bytes)

    def test_low_memory_limit_reports_oom(self):
        report = evaluate(dense_spec(), ParallelConfig(tp=1), memory=1)
        self.assertTrue(report.oom)
        self.assertTrue(any(stage.oom for stage in report.per_stage))

    def test_moe_and_swap_paths_run(self):
        dims = DimTable(
            H=512, F=1024, n_heads=8, n_kv=8, head_dim=64,
            S=512, B=1, vocab=32000, n_layers=2,
            n_experts=8, topk=2, moe_F=1024,
        )
        spec = ModelSpec("moe", dims, ("moe", "moe"), {"moe": build_moe_decoder(dims)})
        report = evaluate(
            spec,
            ParallelConfig(tp=2, ep=4, dp_shard=2, num_microbatches=2),
            swap=SwapSpec(enabled=True, prefetch_depth=1),
        )
        self.assertGreater(report.per_stage[0].peak_bytes, 0)
        self.assertFalse(report.oom)
        self.assertGreater(report.per_stage[0].bucket_peaks.swap_buf, 0)
import unittest
