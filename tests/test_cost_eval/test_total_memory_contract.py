from cost_eval.layers import build_dense_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.report import Evaluator
from cost_eval.specs import HardwareSpec, OptimizerSpec, ParallelConfig


def _report(limit=10**12, **hardware):
    dims = DimTable(
        H=16, F=32, n_heads=4, n_kv=2, head_dim=4,
        S=16, B=1, vocab=32, n_layers=1,
    )
    spec = ModelSpec(
        "dense", dims, ("dense",), {"dense": build_dense_decoder(dims)}
    )
    return Evaluator(
        spec, ParallelConfig(), OptimizerSpec.adamw(),
        HardwareSpec(limit, allocator_granularity_bytes=1, **hardware),
    ).evaluate()


def test_report_v2_keeps_physical_point_and_safe_upper_separate():
    report = _report(
        device_baseline_bytes=10,
        untracked_runtime_point_bytes=20,
        calibrated_upper_margin_bytes=30,
    )
    peak = report.per_stage[0]
    assert report.schema_version == "hbmprediction.report.v2"
    assert peak.total_point_bytes == peak.physical_dynamic_peak_bytes + 30
    assert peak.safe_upper_bytes == peak.total_point_bytes + 30
    assert peak.peak_bytes == peak.total_point_bytes
    assert peak.breakdown.total == peak.total_point_bytes


def test_oom_status_does_not_claim_unknown_profile_is_definitely_safe():
    report = _report()
    assert report.oom_status == "risky"
    assert not report.oom


def test_point_over_usable_memory_is_predicted_oom():
    report = _report(limit=1)
    assert report.oom_status == "predicted_oom"
    assert report.oom
