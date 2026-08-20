import hashlib
import json
from pathlib import Path

from cost_eval.measurement_contract import HBMMeasurement


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/hbm_components/representative_cases.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_representative_component_goldens_match_content_addressed_evidence():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema"] == "hbmprediction.component-goldens.v1"
    for name, expected in fixture["cases"].items():
        path = ROOT / f"validation/real_npu/case_results/{name}.json"
        assert _sha256(path) == expected["case_result_sha256"]
        measurement = HBMMeasurement.from_case_result(
            json.loads(path.read_text(encoding="utf-8"))
        )
        measurement.validate_stage_aggregation()
        assert measurement.components.device_baseline_bytes == expected[
            "device_baseline_bytes"
        ]
        assert measurement.components.model_active_peak_bytes == expected[
            "model_active_peak_bytes"
        ]
        assert measurement.components.allocator_pool_peak_bytes == expected[
            "allocator_pool_peak_bytes"
        ]
        assert measurement.components.untracked_runtime_bytes == expected[
            "untracked_runtime_bytes"
        ]
        assert measurement.worst_dynamic_bytes == expected["dynamic_peak_bytes"]
        assert measurement.worst_total_bytes == expected["total_peak_bytes"]
        assert len(measurement.per_rank) == expected["rank_count"]
        assert min(item.sample_count for item in measurement.per_rank) == expected[
            "min_samples"
        ]
