#!/usr/bin/env python3
"""Validate and summarize the imported Qwen3/DeepSeek-V3 HBM evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SCHEMA = "hbmprediction.real-npu-analysis.v1"


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ape(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100.0


def _metric_summary(values: Iterable[float]) -> Mapping[str, float | int | None]:
    numbers = list(values)
    if not numbers:
        return {"count": 0, "mean_ape_pct": None, "max_ape_pct": None}
    return {
        "count": len(numbers),
        "mean_ape_pct": round(mean(numbers), 3),
        "max_ape_pct": round(max(numbers), 3),
    }


def _linear_fit(points: list[tuple[int, int]]) -> Mapping[str, float | int] | None:
    if len(points) < 3 or len({x for x, _ in points}) < 2:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    xbar, ybar = mean(xs), mean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator
    intercept = ybar - slope * xbar
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - ybar) ** 2 for y in ys)
    r_squared = 1.0 if total == 0 else 1.0 - residual / total
    return {
        "case_count": len(points),
        "slope_gib_per_layer": round(slope / 2**30, 4),
        "intercept_gib": round(intercept / 2**30, 4),
        "r_squared": round(r_squared, 4),
    }


def _layout(name: str, family: str) -> str:
    if "tp2-cp2-ep2" in name:
        return f"{family}:tp2-cp2-ep2"
    if "pp2-dp2-interleave" in name:
        return f"{family}:tp2-pp2-dp2-interleave"
    if "tp4-dp2" in name:
        return f"{family}:tp4-dp2"
    if "-dp8" in name:
        return f"{family}:dp8"
    match = re.search(r"-(tp\d+(?:-[a-z]+\d+)*)", name)
    return f"{family}:{match.group(1) if match else 'other'}"


def _layer_count(name: str) -> int | None:
    match = re.search(r"-(\d+)l-", name)
    return int(match.group(1)) if match else None


def analyze(root: Path) -> Mapping[str, Any]:
    root = root.resolve()
    manifest = _read(root / "manifest.json")
    if manifest.get("schema") != "hbmprediction.real-npu-evidence.v1":
        raise ValueError("unsupported real-NPU evidence manifest")

    integrity_errors: list[str] = []
    copied_files = 0
    copied_bytes = 0
    file_records = list(manifest.get("control_files", {}).values())
    for case in manifest["cases"]:
        file_records.extend(
            record for record in case["files"].values() if record is not None
        )
    for record in file_records:
        path = root / record["file"]
        copied_files += 1
        if not path.is_file():
            integrity_errors.append(f"missing: {record['file']}")
            continue
        copied_bytes += path.stat().st_size
        actual = _sha256(path)
        if actual != record["sha256"]:
            integrity_errors.append(f"sha256 mismatch: {record['file']}")

    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        comparison = _read(root / case["files"]["comparison"]["file"])
        rows.append({"manifest": case, "comparison": comparison})

    # Calibration rows and their historical holdout aliases can point to the
    # same HBM run.  Metrics operate on one row per content-addressed label.
    unique_rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for row in rows:
        case = row["manifest"]
        if not case["formal_eligible"]:
            continue
        label = case["source_label_hashes"].get("hbm")
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        unique_rows.append(row)

    family_counts = Counter(row["manifest"]["family"] for row in unique_rows)
    split_counts = Counter(case["split"] for case in manifest["cases"])
    eligible_counts = Counter(
        case["split"] for case in manifest["cases"] if case["formal_eligible"]
    )

    physical: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"total": [], "dynamic": []}
    )
    baselines: list[int] = []
    sample_counts: list[int] = []
    actual_oom = 0
    trends: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {"total": [], "dynamic": []}
    )
    for row in unique_rows:
        case, comparison = row["manifest"], row["comparison"]
        measured = comparison["measured"]
        base = comparison["physical_base"]
        family = case["family"]
        actual_oom += int(bool(measured.get("oom")))
        for key, short in (
            ("peak_hbm_total_bytes", "total"),
            ("peak_hbm_dynamic_bytes", "dynamic"),
        ):
            if measured.get(key) and base.get(key) is not None:
                physical[family][short].append(_ape(base[key], measured[key]))
                physical["overall"][short].append(_ape(base[key], measured[key]))
        quality = case["measurement_quality"]
        baselines.extend(int(value) for value in quality.get("per_rank_baseline_bytes", ()))
        sample_counts.extend(int(value) for value in quality.get("per_rank_sample_counts", ()))
        layers = _layer_count(case["name"])
        if layers is not None:
            layout = _layout(case["name"], family)
            trends[layout]["total"].append((layers, measured["peak_hbm_total_bytes"]))
            trends[layout]["dynamic"].append((layers, measured["peak_hbm_dynamic_bytes"]))

    physical_summary = {
        family: {
            metric: _metric_summary(values)
            for metric, values in metrics.items()
        }
        for family, metrics in sorted(physical.items())
    }

    frozen: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"total": [], "dynamic": []}
    )
    frozen_cases: list[str] = []
    for row in rows:
        case, comparison = row["manifest"], row["comparison"]
        frozen_record = case["files"].get("frozen_prediction")
        if (
            case["split"] != "historical_frozen_holdout"
            or not case["formal_eligible"]
            or frozen_record is None
        ):
            continue
        prediction = _read(root / frozen_record["file"])["prediction"]
        measured = comparison["measured"]
        family = case["family"]
        frozen_cases.append(case["name"])
        for key, short in (
            ("peak_hbm_total_bytes", "total"),
            ("peak_hbm_dynamic_bytes", "dynamic"),
        ):
            frozen[family][short].append(_ape(prediction[key], measured[key]))
            frozen["overall"][short].append(_ape(prediction[key], measured[key]))
    frozen_summary = {
        family: {
            metric: _metric_summary(values)
            for metric, values in metrics.items()
        }
        for family, metrics in sorted(frozen.items())
    }

    trend_summary: dict[str, Any] = {}
    for layout, metrics in sorted(trends.items()):
        total_fit = _linear_fit(metrics["total"])
        dynamic_fit = _linear_fit(metrics["dynamic"])
        if total_fit is not None:
            trend_summary[layout] = {
                "measured_total": total_fit,
                "measured_dynamic": dynamic_fit,
            }

    matrix = _read(root / manifest["control_files"]["v17_matrix"]["file"])
    diagnostic_holdouts = [
        {
            "name": item["name"],
            "family": "deepseek_v3" if item["family"] == "deepseek3" else item["family"],
            "reason": item["reason"],
        }
        for item in matrix.get("diagnostic", ())
        if item.get("split") == "holdout"
    ]
    required_holdouts = int(
        matrix.get("acceptance_policy", {}).get("minimum_holdout_cases", 0)
    )
    required_oom = int(
        matrix.get("acceptance_policy", {}).get("minimum_actual_oom_cases", 0)
    )
    current_holdouts = len(matrix.get("holdout", ()))

    result = {
        "schema": SCHEMA,
        "source_commit": manifest["source"].get("git_commit"),
        "integrity": {
            "status": "valid" if not integrity_errors else "invalid",
            "copied_files_checked": copied_files,
            "copied_bytes_checked": copied_bytes,
            "errors": integrity_errors,
        },
        "coverage": {
            "manifest_cases": len(rows),
            "formal_eligible_rows": sum(
                int(case["formal_eligible"]) for case in manifest["cases"]
            ),
            "unique_formal_hbm_labels": len(unique_rows),
            "unique_labels_by_family": dict(sorted(family_counts.items())),
            "rows_by_split": dict(sorted(split_counts.items())),
            "eligible_rows_by_split": dict(sorted(eligible_counts.items())),
            "actual_oom_labels": actual_oom,
            "baseline_bytes_min": min(baselines) if baselines else None,
            "baseline_bytes_max": max(baselines) if baselines else None,
            "per_rank_sample_count_min": min(sample_counts) if sample_counts else None,
        },
        "physical_base_error": physical_summary,
        "historical_frozen_holdout_error": {
            "case_count": len(frozen_cases),
            "cases": sorted(frozen_cases),
            "metrics": frozen_summary,
            "interpretation": (
                "historical content-addressed predictions; not a valid current-v17 blind holdout"
            ),
        },
        "measured_layer_trends": trend_summary,
        "v17_acceptance": {
            "status": "pass" if (
                current_holdouts >= required_holdouts and actual_oom >= required_oom
            ) else "fail",
            "required_holdout_cases": required_holdouts,
            "eligible_holdout_cases": current_holdouts,
            "required_actual_oom_cases": required_oom,
            "actual_oom_labels": actual_oom,
            "diagnostic_holdouts": diagnostic_holdouts,
        },
        "conclusion": (
            "The evidence supports calibration and historical error analysis for Qwen3 and "
            "DeepSeek-V3, but it does not establish current blind-holdout generalization or "
            "the OOM decision boundary."
        ),
    }
    return result


def _format_pct(metric: Mapping[str, Any]) -> str:
    if not metric or metric.get("mean_ape_pct") is None:
        return "n/a"
    return f"{metric['mean_ape_pct']:.3f}% (max {metric['max_ape_pct']:.3f}%, n={metric['count']})"


def render_markdown(result: Mapping[str, Any]) -> str:
    coverage = result["coverage"]
    physical = result["physical_base_error"]
    frozen = result["historical_frozen_holdout_error"]
    acceptance = result["v17_acceptance"]
    lines = [
        "# Qwen3 / DeepSeek-V3 real-NPU HBM evidence analysis",
        "",
        "## Bottom line",
        "",
        (
            f"All {result['integrity']['copied_files_checked']} copied evidence files pass SHA-256 "
            f"verification. The set contains {coverage['manifest_cases']} rows but only "
            f"{coverage['unique_formal_hbm_labels']} independent, formally eligible HBM labels "
            f"after content-hash deduplication ({coverage['unique_labels_by_family']})."
        ),
        "",
        (
            "The physical model is a useful baseline, not an accurate final predictor: overall "
            f"total-HBM MAPE is {_format_pct(physical['overall']['total'])}, and dynamic-HBM "
            f"MAPE is {_format_pct(physical['overall']['dynamic'])}."
        ),
        "",
        (
            f"The {frozen['case_count']} historical frozen holdouts improve this to total-HBM "
            f"MAPE {_format_pct(frozen['metrics'].get('overall', {}).get('total', {}))} and "
            f"dynamic-HBM MAPE {_format_pct(frozen['metrics'].get('overall', {}).get('dynamic', {}))}. "
            "These are versioned historical predictions, not a current blind test set."
        ),
        "",
        (
            f"The current v17 acceptance gate is **{acceptance['status'].upper()}**: it has "
            f"{acceptance['eligible_holdout_cases']}/{acceptance['required_holdout_cases']} eligible "
            f"holdouts and {acceptance['actual_oom_labels']}/{acceptance['required_actual_oom_cases']} "
            "actual OOM labels. Therefore the evidence does not yet validate generalization or the "
            "OOM classification boundary."
        ),
        "",
        "## Error by family",
        "",
        "| Model | Physical total APE | Physical dynamic APE | Frozen total APE | Frozen dynamic APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("qwen3", "deepseek_v3"):
        p = physical.get(family, {})
        f = frozen["metrics"].get(family, {})
        lines.append(
            f"| {family} | {_format_pct(p.get('total', {}))} | "
            f"{_format_pct(p.get('dynamic', {}))} | {_format_pct(f.get('total', {}))} | "
            f"{_format_pct(f.get('dynamic', {}))} |"
        )
    lines.extend((
        "",
        "## Measured scaling at fixed layouts",
        "",
        "| Layout | Total GiB/layer | Total R² | Dynamic GiB/layer | Dynamic R² | n |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for layout, metrics in result["measured_layer_trends"].items():
        total = metrics["measured_total"]
        dynamic = metrics["measured_dynamic"]
        lines.append(
            f"| {layout} | {total['slope_gib_per_layer']:.4f} | {total['r_squared']:.4f} | "
            f"{dynamic['slope_gib_per_layer']:.4f} | {dynamic['r_squared']:.4f} | "
            f"{total['case_count']} |"
        )
    lines.extend((
        "",
        "The slopes are descriptive within each fixed layout; they are not interchangeable across "
        "TP/CP/PP/EP strategies.",
        "",
        "## Data-quality limits",
        "",
        f"- One calibration row is diagnostic-only because of a contaminated cross-rank baseline.",
        f"- Minimum valid per-rank HBM sample count is {coverage['per_rank_sample_count_min']}.",
        f"- Valid baseline range is {coverage['baseline_bytes_min'] / 2**30:.3f}–{coverage['baseline_bytes_max'] / 2**30:.3f} GiB.",
        "- The four intended v17 holdouts are diagnostic-only because their timing archives lack `case_result.json`.",
        "- Raw profiler traces, checkpoints, logs, and datasets remain in the source repository and are intentionally not copied.",
        "",
        "Reproduce with `python tools/import_parallelsearch_hbm.py` followed by "
        "`python tools/analyze_real_npu.py`.",
        "",
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "validation/real_npu",
    )
    args = parser.parse_args()
    result = analyze(args.root)
    if result["integrity"]["status"] != "valid":
        raise SystemExit("evidence integrity validation failed")
    (args.root / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.root / "ANALYSIS.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(
        f"analyzed {result['coverage']['unique_formal_hbm_labels']} unique HBM labels; "
        f"v17 acceptance={result['v17_acceptance']['status']}"
    )


if __name__ == "__main__":
    main()
