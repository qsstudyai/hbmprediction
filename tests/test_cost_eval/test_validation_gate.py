import pytest

from cost_eval.validation_gate import assess_hbm_gates, render_gate_markdown


def _report(active=(9.0, 30.0), dynamic=(8.0, 24.0), total=(7.0, 20.5)):
    metrics = {
        "active": {"mean_ape_pct": active[0], "max_ape_pct": active[1]},
        "dynamic": {"mean_ape_pct": dynamic[0], "max_ape_pct": dynamic[1]},
        "total": {"mean_ape_pct": total[0], "max_ape_pct": total[1]},
    }
    return {
        "summary": {"overall": metrics},
        "cases": [{
            "label_sha256": "label-a",
            "prediction": {"safe_upper_bytes": 110},
            "measured": {"total_bytes": 100},
        }],
    }


def test_gate_keeps_calibration_and_production_status_separate():
    report = _report()
    result = assess_hbm_gates(report, report)
    assert result["p0_numeric_status"] == "pass"
    assert result["p1_numeric_status"] == "fail"
    assert result["production_status"] == "validation_unavailable"
    assert {item["status"] for item in result["gates"]} >= {
        "pass", "fail", "unavailable"
    }
    assert "unavailable" in render_gate_markdown(result)


def test_gate_rejects_mixed_evidence_sets():
    offline = _report()
    calibrated = _report()
    calibrated["cases"][0]["label_sha256"] = "label-b"
    with pytest.raises(ValueError, match="different labels"):
        assess_hbm_gates(offline, calibrated)
