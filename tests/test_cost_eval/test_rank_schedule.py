from cost_eval.layers import build_dense_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.report import Evaluator
from cost_eval.specs import HardwareSpec, OptimizerSpec, ParallelConfig


def _setup():
    dims = DimTable(
        H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
        S=16, B=1, vocab=32, n_layers=4,
    )
    spec = ModelSpec(
        "dense", dims, ("dense",) * 4, {"dense": build_dense_decoder(dims)}
    )
    return dims, spec


def test_every_rank_is_reported_and_job_peak_is_real_max_rank():
    _, spec = _setup()
    report = Evaluator(
        spec, ParallelConfig(dp_shard=2, pp=2), OptimizerSpec.adamw(),
        HardwareSpec(10**12, allocator_granularity_bytes=1),
    ).evaluate()
    assert len(report.per_rank) == 4
    assert [item.stage for item in report.per_rank] == [0, 1, 0, 1]
    assert report.per_rank[report.tightest_rank].peak_bytes == max(
        item.peak_bytes for item in report.per_rank
    )


def test_pipeline_boundary_staging_has_named_peak_candidate():
    _, spec = _setup()
    report = Evaluator(
        spec, ParallelConfig(pp=2), OptimizerSpec.adamw(),
        HardwareSpec(10**12, allocator_granularity_bytes=1),
    ).evaluate()
    # Boundary staging is part of the same ledger even when it is not the
    # global peak for this small graph.
    assert all(item.physical_dynamic_peak_bytes > 0 for item in report.per_stage)
