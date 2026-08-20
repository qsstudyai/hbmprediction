#!/usr/bin/env python3
"""Freeze a source/config/profile-addressed prediction before an NPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from cost_eval.adapters import MindFormersAdapter
from cost_eval.runtime_profiles import RuntimeProfileRegistry


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--runtime-profiles", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = MindFormersAdapter.from_yaml(args.config, args.world_size)
    if args.runtime_profiles:
        registry = RuntimeProfileRegistry.load(args.runtime_profiles)
        family = str(inputs.model_spec.capabilities.get("model_family", inputs.model_spec.name))
        inputs = replace(
            inputs,
            hardware=registry.apply(inputs.hardware, family, inputs.parallel_config),
        )
    report = inputs.evaluator().evaluate()
    source_files = list((Path(__file__).resolve().parents[1] / "cost_eval").rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(path.relative_to(path.parents[1]).as_posix().encode())
        digest.update(path.read_bytes())
    payload = {
        "schema": "hbmprediction.frozen-prediction.v2",
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "model_source_sha256": digest.hexdigest(),
        "runtime_profiles": str(args.runtime_profiles.resolve()) if args.runtime_profiles else None,
        "runtime_profiles_sha256": sha256(args.runtime_profiles) if args.runtime_profiles else None,
        "prediction": asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
