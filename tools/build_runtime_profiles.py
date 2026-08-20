#!/usr/bin/env python3
"""Build non-leaky allocator/runtime profiles from calibration labels only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from cost_eval.adapters import MindFormersAdapter
from cost_eval.measurement_contract import HBMMeasurement
from cost_eval.source_fingerprint import python_source_tree_sha256


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo
    return round(ordered[lo] * (1 - fraction) + ordered[hi] * fraction)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="validation/real_npu")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    project = root.resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    groups = defaultdict(list)
    seen_labels = set()
    for case in manifest["cases"]:
        if case.get("split") != "calibration" or not case.get("formal_eligible"):
            continue
        label_hash = case["source_label_hashes"]["hbm"]
        if label_hash in seen_labels:
            continue
        seen_labels.add(label_hash)
        files = case["files"]
        config = root / files["config"]["file"]
        label = root / files["case_result"]["file"]
        if not config.exists() or not label.exists():
            continue
        measurement = HBMMeasurement.from_case_result(json.loads(label.read_text()))
        inputs = MindFormersAdapter.from_yaml(config, world_size=8)
        pc = inputs.parallel_config
        prediction = inputs.evaluator().evaluate()
        predicted_active = max(
            item.physical_dynamic_peak_bytes for item in prediction.per_stage
        )
        layout = (
            f"tp{pc.tp}-cp{pc.cp}-pp{pc.pp}-ep{pc.ep}-"
            f"dp{pc.dp_shard}x{pc.dp_replicate}-i{pc.interleave}"
        )
        groups[(case["family"], layout)].append((
            measurement.components.untracked_runtime_bytes,
            measurement.components.allocator_pool_peak_bytes
            - measurement.components.model_active_peak_bytes,
            measurement.components.device_baseline_bytes,
            max(
                0,
                measurement.components.model_active_peak_bytes
                - predicted_active,
            ),
            label_hash,
        ))
    profiles = []
    for (family, layout), samples in sorted(groups.items()):
        runtime = [max(0, item[0]) for item in samples]
        pool_slack = [max(0, item[1]) for item in samples]
        baseline = [max(0, item[2]) for item in samples]
        active_shortfall = [max(0, item[3]) for item in samples]
        profiles.append({
            "family": family,
            "layout": layout,
            "sample_count": len(samples),
            "untracked_runtime_point_bytes": round(median(runtime)),
            "untracked_runtime_p90_bytes": percentile(runtime, 0.9),
            "allocator_pool_slack_point_bytes": round(median(pool_slack)),
            "allocator_pool_slack_p90_bytes": percentile(pool_slack, 0.9),
            "device_baseline_point_bytes": round(median(baseline)),
            "device_baseline_p90_bytes": percentile(baseline, 0.9),
            "physical_active_shortfall_p90_bytes": percentile(active_shortfall, 0.9),
            "label_sha256": sorted(item[4] for item in samples),
        })
    payload = {
        "schema": "hbmprediction.runtime-profiles.v1",
        "source_manifest": str((root / "manifest.json").resolve()),
        "model_source_sha256": python_source_tree_sha256(project / "cost_eval"),
        "split_policy": "calibration-only; label-hash deduplicated",
        "profiles": profiles,
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
