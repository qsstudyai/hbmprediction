"""预测显存与真实 profiling 数据的对账、诊断和验收报告。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Optional, Sequence

from .config_adapter import ConfigAdapter, load_data_file, parse_bytes


@dataclass(frozen=True)
class MetricComparison:
    predicted: int
    measured: int
    signed_error: int
    absolute_error: int
    relative_error: Optional[float]

    @classmethod
    def compare(cls, predicted: int, measured: int) -> "MetricComparison":
        signed = predicted - measured
        relative = abs(signed) / measured if measured else (
            0.0 if predicted == 0 else None
        )
        return cls(predicted, measured, signed, abs(signed), relative)


@dataclass(frozen=True)
class StageCalibration:
    stage: int
    peak: MetricComparison
    predicted_event: str
    measured_event: Optional[str]
    event_match: Optional[bool]
    predicted_breakdown: Mapping[str, int]
    bucket_errors: Mapping[str, MetricComparison]


@dataclass(frozen=True)
class CaseCalibration:
    name: str
    stages: tuple[StageCalibration, ...]
    predicted_oom: bool
    measured_oom: bool
    oom_match: bool
    framework_reserve: Optional[MetricComparison]
    metadata: Mapping[str, Any]
    peak_stage_order_match: Optional[bool] = None


@dataclass(frozen=True)
class CalibrationSummary:
    case_count: int
    stage_count: int
    mean_relative_error: Optional[float]
    p50_relative_error: Optional[float]
    p90_relative_error: Optional[float]
    max_relative_error: Optional[float]
    mean_signed_bias: Optional[float]
    within_target_rate: Optional[float]
    target_relative_error: float
    peak_event_match_rate: Optional[float]
    oom_true_positives: int
    oom_true_negatives: int
    oom_false_positives: int
    oom_false_negatives: int
    suggested_framework_reserve: Optional[int]
    peak_stage_order_match_rate: Optional[float] = None


@dataclass(frozen=True)
class CalibrationReport:
    cases: tuple[CaseCalibration, ...]
    summary: CalibrationSummary


def _percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _event_phase(event: Optional[str]) -> Optional[str]:
    if event is None:
        return None
    lowered = event.lower()
    if lowered.startswith("fwd") or lowered == "forward":
        return "forward"
    if lowered.startswith("bwd") or lowered == "backward":
        return "backward"
    return lowered


def _event_matches(predicted: str, measured: Optional[str]) -> Optional[bool]:
    if measured is None:
        return None
    return predicted == measured or _event_phase(predicted) == _event_phase(measured)


def _stage_order_match(predicted_stages, measured_peaks: Mapping[int, int]) -> bool:
    """Check whether measured and predicted stages have the same peak order.

    Ties are resolved by stage id so the result is deterministic.  A one-stage
    pipeline is considered a valid (vacuous) ordering match.
    """
    predicted_order = tuple(
        item.stage
        for item in sorted(
            predicted_stages,
            key=lambda item: (-item.peak_bytes, item.stage),
        )
    )
    measured_order = tuple(
        stage
        for stage, _ in sorted(
            measured_peaks.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return predicted_order == measured_order


def _normalize_stage_mapping(
    value: Mapping[Any, Any], field_name: str
) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} 必须是 stage -> value 映射")
    result = {}
    for stage, item in value.items():
        try:
            stage_id = int(stage)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} 包含非法 stage: {stage!r}") from exc
        result[stage_id] = item
    return result


class CalibrationRunner:
    def __init__(self, target_relative_error: float = 0.10):
        if not 0 < target_relative_error < 1:
            raise ValueError("target_relative_error 必须位于 0 和 1 之间")
        self.target_relative_error = target_relative_error

    def run_case(
        self,
        name: str,
        inputs,
        measured: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CaseCalibration:
        prediction = inputs.evaluator().evaluate()
        predicted_stages = {item.stage: item for item in prediction.per_stage}
        raw_peaks = measured.get("per_stage_peak_bytes")
        measured_oom_value = measured.get("oom")
        if measured_oom_value is not None and not isinstance(
            measured_oom_value, bool
        ):
            raise ValueError(f"case {name} 的 measured.oom 必须是布尔值")
        # New collector profiles distinguish the baseline/runtime floor from
        # the device-wide Ascend allocator reserve.  Keep ``idle_bytes`` as a
        # backwards-compatible fallback for older profiles.
        idle = measured.get(
            "framework_reserve_bytes", measured.get("idle_bytes")
        )
        framework = (
            MetricComparison.compare(
                inputs.hardware.framework_reserve, parse_bytes(idle)
            )
            if idle is not None
            else None
        )
        if not raw_peaks:
            if measured_oom_value is True:
                return CaseCalibration(
                    name,
                    tuple(),
                    prediction.oom,
                    True,
                    prediction.oom,
                    framework,
                    dict(metadata or {}),
                    None,
                )
            raise ValueError(
                f"case {name} 缺少 measured.per_stage_peak_bytes；"
                "仅 measured.oom=true 的分类案例可以省略"
            )
        measured_peaks = {
            stage: parse_bytes(value)
            for stage, value in _normalize_stage_mapping(
                raw_peaks, "per_stage_peak_bytes"
            ).items()
        }
        unknown_stages = set(measured_peaks) - set(predicted_stages)
        missing_stages = set(predicted_stages) - set(measured_peaks)
        if unknown_stages or missing_stages:
            raise ValueError(
                f"case {name} stage 不一致；实测多出={sorted(unknown_stages)}，"
                f"实测缺少={sorted(missing_stages)}"
            )

        raw_events = measured.get(
            "per_stage_peak_event", measured.get("peak_event")
        )
        if isinstance(raw_events, Mapping):
            measured_events = _normalize_stage_mapping(
                raw_events, "per_stage_peak_event"
            )
        else:
            measured_events = {
                stage: raw_events for stage in measured_peaks
            }
        raw_breakdowns = measured.get("per_stage_breakdown", {})
        measured_breakdowns = _normalize_stage_mapping(
            raw_breakdowns, "per_stage_breakdown"
        )

        stages = []
        for stage in sorted(predicted_stages):
            predicted = predicted_stages[stage]
            predicted_breakdown = dict(asdict(predicted.breakdown))
            measured_breakdown = measured_breakdowns.get(stage, {})
            if not isinstance(measured_breakdown, Mapping):
                raise ValueError(
                    f"case {name} stage {stage} breakdown 必须是映射"
                )
            unknown_buckets = set(measured_breakdown) - set(
                predicted_breakdown
            )
            if unknown_buckets:
                raise ValueError(
                    f"case {name} stage {stage} 包含未知内存桶: "
                    f"{', '.join(sorted(unknown_buckets))}"
                )
            bucket_errors = {
                bucket: MetricComparison.compare(
                    predicted_breakdown[bucket], parse_bytes(value)
                )
                for bucket, value in measured_breakdown.items()
            }
            measured_event = measured_events.get(stage)
            if measured_event is not None:
                measured_event = str(measured_event)
            stages.append(
                StageCalibration(
                    stage,
                    MetricComparison.compare(
                        predicted.peak_bytes, measured_peaks[stage]
                    ),
                    predicted.peak_event,
                    measured_event,
                    _event_matches(predicted.peak_event, measured_event),
                    predicted_breakdown,
                    bucket_errors,
                )
            )

        measured_oom = measured_oom_value
        if measured_oom is None:
            measured_oom = any(
                value > inputs.hardware.max_device_memory
                for value in measured_peaks.values()
            )
        stage_order_match = _stage_order_match(
            tuple(predicted_stages.values()), measured_peaks
        )
        return CaseCalibration(
            name,
            tuple(stages),
            prediction.oom,
            measured_oom,
            prediction.oom == measured_oom,
            framework,
            dict(metadata or {}),
            stage_order_match,
        )

    def run_cases(self, cases: Sequence[CaseCalibration]) -> CalibrationReport:
        cases = tuple(cases)
        if not cases:
            raise ValueError("至少需要一个 calibration case")
        comparisons = [stage.peak for case in cases for stage in case.stages]
        relative = [
            item.relative_error
            for item in comparisons
            if item.relative_error is not None
        ]
        event_matches = [
            stage.event_match
            for case in cases
            for stage in case.stages
            if stage.event_match is not None
        ]
        stage_order_matches = [
            case.peak_stage_order_match
            for case in cases
            if case.peak_stage_order_match is not None
        ]
        idle_values = [
            case.framework_reserve.measured
            for case in cases
            if case.framework_reserve is not None
        ]
        true_positive = sum(
            case.predicted_oom and case.measured_oom for case in cases
        )
        true_negative = sum(
            not case.predicted_oom and not case.measured_oom for case in cases
        )
        false_positive = sum(
            case.predicted_oom and not case.measured_oom for case in cases
        )
        false_negative = sum(
            not case.predicted_oom and case.measured_oom for case in cases
        )
        summary = CalibrationSummary(
            len(cases),
            len(comparisons),
            mean(relative) if relative else None,
            _percentile(relative, 0.50),
            _percentile(relative, 0.90),
            max(relative) if relative else None,
            mean(item.signed_error for item in comparisons)
            if comparisons
            else None,
            (
                sum(
                    value <= self.target_relative_error for value in relative
                )
                / len(relative)
                if relative
                else None
            ),
            self.target_relative_error,
            (
                sum(event_matches) / len(event_matches)
                if event_matches
                else None
            ),
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            int(median(idle_values)) if idle_values else None,
            (
                sum(stage_order_matches) / len(stage_order_matches)
                if stage_order_matches
                else None
            ),
        )
        return CalibrationReport(cases, summary)

    def run_profile(self, path) -> CalibrationReport:
        path = Path(path)
        root = load_data_file(path)
        if not isinstance(root, Mapping):
            raise ValueError("profiling 文件根节点必须是映射")
        target = root.get("target_relative_error")
        runner = (
            CalibrationRunner(float(target))
            if target is not None
            else self
        )
        raw_cases = root.get("cases")
        if raw_cases is None:
            raw_cases = {"case": root}
        if isinstance(raw_cases, Mapping):
            items = [
                (str(name), value) for name, value in raw_cases.items()
            ]
        elif isinstance(raw_cases, list):
            items = [
                (str(value.get("name", f"case-{index}")), value)
                for index, value in enumerate(raw_cases)
            ]
        else:
            raise ValueError("cases 必须是映射或列表")

        results = []
        for name, case in items:
            if not isinstance(case, Mapping):
                raise ValueError(f"case {name} 必须是映射")
            config = case.get("config")
            adapter = str(case.get("adapter", "native")).lower()
            framework_reserve = int(case.get("framework_reserve_bytes", 0))
            if isinstance(config, Mapping) and adapter == "mindformers":
                from .adapters import MindFormersAdapter
                inputs = MindFormersAdapter.from_mapping(
                    config,
                    world_size=int(case.get("world_size", 1)),
                    framework_reserve=framework_reserve,
                    source=f"{path}::{name}",
                )
            elif isinstance(config, Mapping):
                inputs = ConfigAdapter.from_dict(config)
            elif isinstance(config, str):
                config_path = path.parent / config
                if adapter == "mindformers":
                    from .adapters import MindFormersAdapter
                    inputs = MindFormersAdapter.from_yaml(
                        config_path,
                        world_size=int(case.get("world_size", 1)),
                        framework_reserve=framework_reserve,
                    )
                else:
                    inputs = ConfigAdapter.load(config_path)
            else:
                raise ValueError(
                    f"case {name} 的 config 必须是配置映射或相对路径"
                )
            measured = case.get("measured")
            if not isinstance(measured, Mapping):
                raise ValueError(f"case {name} 缺少 measured 映射")
            metadata = case.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise ValueError(f"case {name} 的 metadata 必须是映射")
            results.append(
                runner.run_case(
                    name, inputs, measured, metadata
                )
            )
        return runner.run_cases(results)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="对比离线显存预测与真实 profiling 数据"
    )
    parser.add_argument("profile", help="profiling JSON/YAML 文件")
    parser.add_argument("-o", "--output", help="将 JSON 报告写入文件")
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="P90 超标、PP stage 峰值排序不一致或出现 OOM 漏报时返回非零状态",
    )
    args = parser.parse_args(argv)
    report = CalibrationRunner().run_profile(args.profile)
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.fail_on_threshold:
        summary = report.summary
        if (
            summary.oom_false_negatives
            or (
                summary.p90_relative_error is not None
                and summary.p90_relative_error
                > summary.target_relative_error
            )
            or (
                summary.peak_stage_order_match_rate is not None
                and summary.peak_stage_order_match_rate < 1.0
            )
        ):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
