import copy
import json
from pathlib import Path

import pytest

from cost_eval.measurement_contract import (
    HBMComponents,
    HBMMeasurement,
)


ROOT = Path(__file__).resolve().parents[2]


def test_components_use_max_active_pool_and_are_mutually_exclusive():
    components = HBMComponents(
        device_baseline_bytes=100,
        model_active_peak_bytes=500,
        allocator_pool_peak_bytes=600,
        untracked_runtime_bytes=50,
    )
    assert components.dynamic_total_bytes == 650
    assert components.nominal_total_bytes == 750
    components.validate_dynamic_total(651, tolerance_bytes=1)
    with pytest.raises(ValueError, match="不守恒"):
        components.validate_dynamic_total(700, tolerance_bytes=1)


def test_real_case_result_obeys_component_and_stage_contract():
    path = (
        ROOT
        / "validation/real_npu/case_results/c32-qwen3-46l-dp8.json"
    )
    measurement = HBMMeasurement.from_case_result(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert measurement.quality_status == "valid"
    assert len(measurement.per_rank) == 8
    assert measurement.eligibility_issues(expected_rank_count=8) == ()
    assert measurement.worst_dynamic_bytes == 12_503_220_224
    assert measurement.worst_total_bytes == 16_086_204_416
    measurement.validate_stage_aggregation()


def test_negative_or_double_counted_components_are_rejected():
    with pytest.raises(ValueError, match="不能为负数"):
        HBMComponents(0, 1, 1, -1)

    path = (
        ROOT
        / "validation/real_npu/case_results/c32-qwen3-46l-dp8.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    broken = copy.deepcopy(payload)
    broken["measured"]["hbm_components"]["untracked_runtime_bytes"] += 2**30
    with pytest.raises(ValueError, match="不守恒"):
        HBMMeasurement.from_case_result(broken)
