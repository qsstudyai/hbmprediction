import json

from cost_eval.runtime_profiles import RuntimeProfileRegistry
from cost_eval.specs import HardwareSpec, ParallelConfig


def test_profile_application_is_layout_exact_and_singletons_are_wide(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema": "hbmprediction.runtime-profiles.v1",
        "profiles": [{
            "family": "qwen3",
            "layout": "tp2-cp1-pp1-ep1-dp4x1-i1",
            "sample_count": 1,
            "untracked_runtime_point_bytes": 100,
            "untracked_runtime_p90_bytes": 100,
            "allocator_pool_slack_point_bytes": 20,
            "allocator_pool_slack_p90_bytes": 20,
            "device_baseline_point_bytes": 30,
            "device_baseline_p90_bytes": 30,
            "physical_active_shortfall_p90_bytes": 40,
            "label_sha256": ["a" * 64],
        }],
    }))
    registry = RuntimeProfileRegistry.load(path)
    assert registry.profile_id.startswith("sha256:")
    hardware = registry.apply(
        HardwareSpec(10**12), "qwen3", ParallelConfig(tp=2, dp_shard=4)
    )
    assert hardware.untracked_runtime_point_bytes == 100
    assert hardware.allocator_pool_slack_point_bytes == 20
    assert hardware.device_baseline_bytes == 30
    assert hardware.calibrated_upper_margin_bytes >= 2 * 2**30
    assert hardware.runtime_profile.startswith("sha256:")


def test_missing_layout_is_ood_and_does_not_invent_a_point(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema": "hbmprediction.runtime-profiles.v1", "profiles": []
    }))
    hardware = RuntimeProfileRegistry.load(path).apply(
        HardwareSpec(10**12), "qwen3", ParallelConfig()
    )
    assert hardware.untracked_runtime_point_bytes == 0
    assert hardware.ood_margin_bytes >= 2 * 2**30
