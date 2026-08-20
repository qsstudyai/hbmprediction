#!/usr/bin/env python3
"""对导入的真实 NPU 标签重新运行当前 hbmprediction 物理模型。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from cost_eval.adapters import MindFormersAdapter
from cost_eval.measurement_contract import HBMMeasurement
from cost_eval.runtime_profiles import RuntimeProfileRegistry
from cost_eval.source_fingerprint import python_source_tree_sha256


SCHEMA = "hbmprediction.current-hbm-baseline.v1"


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _comparison(predicted: int, measured: int) -> Mapping[str, Any]:
    signed = predicted - measured
    return {
        "predicted_bytes": predicted,
        "measured_bytes": measured,
        "signed_error_bytes": signed,
        "ape_pct": abs(signed) / measured * 100.0 if measured else None,
        "ratio": predicted / measured if measured else None,
    }


def _metric_summary(rows: list[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    values = [row[key]["ape_pct"] for row in rows if row[key]["ape_pct"] is not None]
    biases = [row[key]["signed_error_bytes"] for row in rows]
    return {
        "count": len(values),
        "mean_ape_pct": mean(values) if values else None,
        "max_ape_pct": max(values) if values else None,
        "mean_signed_error_bytes": mean(biases) if biases else None,
    }


def evaluate(root: Path, runtime_profiles: Path | None = None) -> Mapping[str, Any]:
    root = root.resolve()
    manifest = _read(root / "manifest.json")
    if manifest.get("schema") != "hbmprediction.real-npu-evidence.v1":
        raise ValueError("unsupported real-NPU manifest")
    project = root.parents[1]
    model_hash = python_source_tree_sha256(project / "cost_eval")

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    registry = (
        RuntimeProfileRegistry.load(runtime_profiles)
        if runtime_profiles is not None else None
    )
    for case in manifest.get("cases", ()):
        name = str(case["name"])
        label_hash = case.get("source_label_hashes", {}).get("hbm")
        if not case.get("formal_eligible"):
            diagnostics.append({"name": name, "reason": "manifest diagnostic-only"})
            continue
        if not label_hash:
            diagnostics.append({"name": name, "reason": "missing HBM label hash"})
            continue
        if label_hash in seen_labels:
            diagnostics.append({"name": name, "reason": "duplicate HBM label hash"})
            continue
        seen_labels.add(str(label_hash))
        records = case.get("files", {})
        config_record = records.get("config")
        result_record = records.get("case_result")
        if not config_record or not result_record:
            diagnostics.append({"name": name, "reason": "missing config/case_result"})
            continue
        config_path = root / config_record["file"]
        result_path = root / result_record["file"]
        try:
            measurement = HBMMeasurement.from_case_result(_read(result_path))
            measurement.validate_stage_aggregation()
            rank_count = len(measurement.per_rank)
            issues = measurement.eligibility_issues(expected_rank_count=rank_count)
            if issues:
                raise ValueError("; ".join(issues))
            inputs = MindFormersAdapter.from_yaml(
                config_path, world_size=rank_count, framework_reserve=0
            )
            if registry is not None:
                inputs = replace(
                    inputs,
                    hardware=registry.apply(
                        inputs.hardware, str(case["family"]),
                        inputs.parallel_config,
                    ),
                )
            report = inputs.evaluator().evaluate()
        except Exception as exc:  # keep the audit complete across all cases
            diagnostics.append({
                "name": name,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        predicted_active = max(
            item.physical_dynamic_peak_bytes for item in report.per_stage
        )
        predicted_dynamic = max(
            item.total_point_bytes - item.device_baseline_bytes
            for item in report.per_stage
        )
        predicted_total = max(
            item.total_point_bytes for item in report.per_stage
        )
        legacy_reserve = int(
            _read(result_path).get("framework_reserve_bytes", 0)
        )
        predicted_legacy = predicted_active + legacy_reserve
        measured_active = measurement.components.model_active_peak_bytes
        measured_dynamic = measurement.worst_dynamic_bytes
        measured_total = measurement.worst_total_bytes
        rows.append({
            "name": name,
            "family": case["family"],
            "split": case["split"],
            "label_sha256": label_hash,
            "config_sha256": config_record.get("sha256"),
            "collection_attempt_id": measurement.collection_attempt_id,
            "prediction": {
                "raw_physical_active_bytes": predicted_active,
                "dynamic_point_bytes": predicted_dynamic,
                "total_point_bytes": predicted_total,
                "safe_upper_bytes": max(
                    item.safe_upper_bytes for item in report.per_stage
                ),
                "legacy_reserve_bytes": legacy_reserve,
                "legacy_total_bytes": predicted_legacy,
                "tightest_stage": report.tightest_stage,
                "peak_event": next(
                    item.peak_event for item in report.per_stage
                    if item.stage == report.tightest_stage
                ),
                "per_stage": [asdict(item) for item in report.per_stage],
                "warnings": list(report.warnings),
            },
            "measured": {
                "model_active_bytes": measured_active,
                "allocator_pool_bytes": measurement.components.allocator_pool_peak_bytes,
                "untracked_runtime_bytes": measurement.components.untracked_runtime_bytes,
                "dynamic_bytes": measured_dynamic,
                "total_bytes": measured_total,
            },
            "active": _comparison(predicted_active, measured_active),
            "dynamic": _comparison(predicted_dynamic, measured_dynamic),
            "total": _comparison(predicted_total, measured_total),
            "legacy_total": _comparison(predicted_legacy, measured_total),
        })

    summary: dict[str, Any] = {}
    families = sorted({row["family"] for row in rows})
    for family in [*families, "overall"]:
        selected = rows if family == "overall" else [
            row for row in rows if row["family"] == family
        ]
        summary[family] = {
            metric: _metric_summary(selected, metric)
            for metric in ("active", "dynamic", "total", "legacy_total")
        }
    return {
        "schema": SCHEMA,
        "model_source_sha256": model_hash,
        "evidence_source_commit": manifest.get("source", {}).get("git_commit"),
        "case_count": len(rows),
        "diagnostic_count": len(diagnostics),
        "runtime_profiles": str(runtime_profiles) if runtime_profiles else None,
        "summary": summary,
        "cases": rows,
        "diagnostics": diagnostics,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Current hbmprediction historical-label replay",
        "",
        f"Model source SHA-256: `{result['model_source_sha256']}`",
        "",
        f"Eligible unique labels: {result['case_count']}; diagnostics: {result['diagnostic_count']}.",
        "",
        "| Family | Metric | Mean APE | Max APE | Mean signed error GiB | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, metrics in result["summary"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {family} | {metric} | {values['mean_ape_pct']:.3f}% | "
                f"{values['max_ape_pct']:.3f}% | "
                f"{values['mean_signed_error_bytes'] / 2**30:.3f} | "
                f"{values['count']} |"
            )
    lines.extend((
        "",
        "`active` compares the raw evaluator ledger with profiler model-active bytes. "
        "`dynamic` compares it with max(active, allocator pool) plus untracked runtime. "
        "`legacy_total` adds the archived legacy reserve and compares device total HBM.",
        "",
    ))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "validation/real_npu",
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--runtime-profiles", type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.root, args.runtime_profiles)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown:
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
