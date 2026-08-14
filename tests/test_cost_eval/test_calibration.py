import json

import pytest

from cost_eval.calibration import CalibrationRunner, main
from cost_eval.config_adapter import ConfigAdapter


def calibration_config(max_memory="1 GiB"):
    return {
        "name": "calibration-toy",
        "model": {
            "type": "dense",
            "H": 8,
            "F": 16,
            "n_heads": 2,
            "n_kv": 2,
            "head_dim": 4,
            "S": 4,
            "B": 1,
            "vocab": 10,
            "n_layers": 2,
        },
        "parallel": {
            "tensor_parallel": 2,
            "data_parallel_shard": 2,
            "num_microbatches": 2,
        },
        "optimizer": {"type": "AdamW"},
        "hardware": {
            "max_device_memory": max_memory,
            "framework_reserve": "4 MiB",
        },
        "recompute": {"mode": "none"},
        "swap": {"enable": False},
    }


def measured_from_prediction(inputs):
    prediction = inputs.evaluator().evaluate()
    return {
        "per_stage_peak_bytes": {
            str(stage.stage): stage.peak_bytes
            for stage in prediction.per_stage
        },
        "per_stage_peak_event": {
            str(stage.stage): stage.peak_event
            for stage in prediction.per_stage
        },
        "per_stage_breakdown": {
            str(stage.stage): {
                "persistent": stage.breakdown.persistent,
                "act_live": stage.breakdown.act_live,
            }
            for stage in prediction.per_stage
        },
        "idle_bytes": inputs.hardware.framework_reserve,
        "oom": prediction.oom,
    }


def test_exact_case_has_zero_error_and_bucket_diagnostics():
    inputs = ConfigAdapter.from_dict(calibration_config())
    case = CalibrationRunner().run_case(
        "exact", inputs, measured_from_prediction(inputs)
    )
    stage = case.stages[0]
    assert stage.peak.relative_error == 0
    assert stage.event_match is True
    assert case.peak_stage_order_match is True
    assert stage.bucket_errors["persistent"].relative_error == 0
    assert case.framework_reserve.relative_error == 0
    assert case.oom_match


def test_suite_statistics_and_oom_false_negative():
    inputs = ConfigAdapter.from_dict(calibration_config())
    exact_measured = measured_from_prediction(inputs)
    exact = CalibrationRunner().run_case(
        "exact", inputs, exact_measured
    )
    high_measured = measured_from_prediction(inputs)
    high_measured["per_stage_peak_bytes"]["0"] *= 2
    high_measured["oom"] = True
    high = CalibrationRunner().run_case("high", inputs, high_measured)
    report = CalibrationRunner(target_relative_error=0.10).run_cases(
        (exact, high)
    )
    assert report.summary.case_count == 2
    assert report.summary.stage_count == 2
    assert report.summary.p50_relative_error == pytest.approx(0.25)
    assert report.summary.p90_relative_error == pytest.approx(0.45)
    assert report.summary.max_relative_error == pytest.approx(0.5)
    assert report.summary.within_target_rate == pytest.approx(0.5)
    assert report.summary.oom_false_negatives == 1
    assert report.summary.suggested_framework_reserve == 4 * 2**20
    assert report.summary.peak_stage_order_match_rate == 1.0


def test_stage_peak_order_mismatch_is_reported():
    config = calibration_config()
    config["parallel"]["pipeline_parallel"] = 2
    inputs = ConfigAdapter.from_dict(config)
    predicted = inputs.evaluator().evaluate().per_stage
    assert len(predicted) == 2
    predicted_order = [
        item.stage
        for item in sorted(predicted, key=lambda item: (-item.peak_bytes, item.stage))
    ]
    measured = {
        "per_stage_peak_bytes": {
            str(predicted_order[0]): 1,
            str(predicted_order[1]): 2,
        },
        "oom": False,
    }
    case = CalibrationRunner().run_case("order-mismatch", inputs, measured)
    assert case.peak_stage_order_match is False
    report = CalibrationRunner().run_cases((case,))
    assert report.summary.peak_stage_order_match_rate == 0.0


def test_profile_file_resolves_relative_config_and_cli_output(tmp_path):
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(calibration_config()), encoding="utf-8"
    )
    inputs = ConfigAdapter.load(config_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "target_relative_error": 0.10,
                "cases": {
                    "relative-config": {
                        "config": "model.json",
                        "measured": measured_from_prediction(inputs),
                        "metadata": {"device": "synthetic"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = CalibrationRunner().run_profile(profile_path)
    assert report.summary.p90_relative_error == 0
    assert report.cases[0].metadata["device"] == "synthetic"

    output = tmp_path / "report.json"
    assert main([str(profile_path), "-o", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 1


def test_fail_on_threshold_returns_nonzero(tmp_path):
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(calibration_config()), encoding="utf-8"
    )
    inputs = ConfigAdapter.load(config_path)
    measured = measured_from_prediction(inputs)
    measured["per_stage_peak_bytes"]["0"] *= 2
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "cases": {
                    "bad": {
                        "config": "model.json",
                        "measured": measured,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert main([str(profile_path), "--fail-on-threshold"]) == 2


def test_stage_mismatch_is_rejected():
    inputs = ConfigAdapter.from_dict(calibration_config())
    with pytest.raises(ValueError, match="stage 不一致"):
        CalibrationRunner().run_case(
            "bad-stage",
            inputs,
            {"per_stage_peak_bytes": {"1": 123}, "oom": False},
        )


def test_classification_only_oom_case_counts_false_negative():
    inputs = ConfigAdapter.from_dict(calibration_config())
    case = CalibrationRunner().run_case(
        "allocation-failed", inputs, {"oom": True}
    )
    assert case.stages == ()
    report = CalibrationRunner().run_cases((case,))
    assert report.summary.stage_count == 0
    assert report.summary.oom_false_negatives == 1
    assert report.summary.p90_relative_error is None
