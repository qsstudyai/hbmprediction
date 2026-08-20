#!/usr/bin/env python3
"""Import the auditable HBM evidence layer from the neighbouring repository.

The raw profiler tree is hundreds of GiB.  This importer intentionally copies
the small, durable evidence needed for review and analysis: comparison rows,
frozen predictions, embedded case results, and the exact profile configs.  The
manifest records hashes and the location of the original raw archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "hbmprediction.real-npu-evidence.v1"
FAMILIES = {"qwen3", "deepseek3"}


def _read_json(path: Path) -> Mapping[str, Any]:
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


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _copy(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = _sha256(source)
    copied_hash = _sha256(target)
    if source_hash != copied_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {target}")
    return {
        "source": str(source),
        "file": str(target),
        "sha256": copied_hash,
        "bytes": target.stat().st_size,
    }


def import_evidence(source_root: Path, destination: Path) -> Mapping[str, Any]:
    source_root = source_root.resolve()
    destination = destination.resolve()
    comparisons_root = source_root / "real_tests/p1v3/comparisons"
    frozen_root = source_root / "real_tests/p1v3/frozen_predictions"
    matrix_path = source_root / "real_tests/p1v3/v17_calibration_matrix.json"
    audit_path = source_root / "real_tests/p1v3/v17_calibration_audit.json"

    def copy_evidence(source: Path, relative_target: Path) -> dict[str, Any]:
        copied = _copy(source, destination / relative_target)
        copied["file"] = relative_target.as_posix()
        return copied

    matrix = _read_json(matrix_path)
    split_by_name: dict[str, str] = {}
    config_by_name: dict[str, str] = {}
    for split in ("calibration", "validation", "holdout"):
        for item in matrix.get(split, ()):
            split_by_name[str(item["name"])] = split
            if item.get("config"):
                config_by_name[str(item["name"])] = str(item["config"])
    diagnostic_reasons = {
        str(item["name"]): str(item.get("reason", "diagnostic-only"))
        for item in matrix.get("diagnostic", ())
    }

    cases: list[dict[str, Any]] = []
    for comparison_path in sorted(comparisons_root.glob("*.json")):
        comparison = _read_json(comparison_path)
        family = str(comparison.get("family", ""))
        if family not in FAMILIES:
            continue
        name = str(comparison.get("case") or comparison_path.stem)
        split = split_by_name.get(name)
        if name in diagnostic_reasons:
            split = "diagnostic"
        elif split is None and name.startswith("h"):
            split = "historical_frozen_holdout"
        elif split is None:
            split = "unclassified"

        copied_comparison = copy_evidence(
            comparison_path, Path("comparisons") / comparison_path.name
        )
        source = comparison.get("source", {})
        hbm_archive_value = source.get("hbm_archive")
        if not hbm_archive_value:
            raise ValueError(f"comparison lacks hbm_archive: {comparison_path}")
        hbm_archive = _resolve(source_root, str(hbm_archive_value))
        case_results = sorted(hbm_archive.rglob("case_result.json"))
        if len(case_results) != 1:
            raise ValueError(
                f"expected exactly one case_result.json in {hbm_archive}, "
                f"found {len(case_results)}"
            )
        copied_result = copy_evidence(
            case_results[0], Path("case_results") / f"{name}.json"
        )

        frozen_source = source.get("frozen_prediction") or {}
        frozen_value = frozen_source.get("path")
        # Historical DeepSeek case IDs insert separators between TP/CP/EP in
        # comparison filenames, while the frozen artifacts use ``tp2cp2ep2``.
        # The comparison records the authoritative path, so prefer it over a
        # filename reconstructed from the case ID.
        frozen_path = (
            _resolve(source_root, str(frozen_value))
            if frozen_value
            else frozen_root / f"{name}.json"
        )
        copied_frozen = None
        config_value = config_by_name.get(name)
        if frozen_path.is_file():
            frozen = _read_json(frozen_path)
            copied_frozen = copy_evidence(
                frozen_path, Path("frozen_predictions") / f"{name}.json"
            )
            config_value = str(frozen.get("inputs", {}).get("config") or config_value or "")

        copied_config = None
        if config_value:
            config_path = _resolve(source_root, config_value)
            if config_path.is_file():
                copied_config = copy_evidence(
                    config_path, Path("configs") / f"{name}{config_path.suffix}"
                )

        hbm_quality = (
            comparison.get("measured", {})
            .get("measurement_quality", {})
            .get("hbm", {})
        )
        cases.append(
            {
                "name": name,
                "family": "deepseek_v3" if family == "deepseek3" else family,
                "source_family": family,
                "split": split,
                "formal_eligible": split != "diagnostic"
                and hbm_quality.get("status") == "valid",
                "diagnostic_reason": diagnostic_reasons.get(name),
                "measurement_quality": hbm_quality,
                "original_archives": {
                    "hbm": str(hbm_archive),
                    "timing": source.get("timing_archive"),
                },
                "source_label_hashes": {
                    "hbm": source.get("hbm_label_sha256"),
                    "timing": source.get("timing_label_sha256"),
                },
                "files": {
                    "comparison": copied_comparison,
                    "case_result": copied_result,
                    "frozen_prediction": copied_frozen,
                    "config": copied_config,
                },
            }
        )

    control_files = {
        "v17_matrix": copy_evidence(matrix_path, Path("v17_calibration_matrix.json")),
        "v17_audit": copy_evidence(audit_path, Path("v17_calibration_audit.json")),
    }
    manifest = {
        "schema": SCHEMA,
        "source": {
            "repository": str(source_root),
            "git_commit": _git_commit(source_root),
            "raw_profiler_tree": str(source_root / ".npu-calibration"),
            "raw_profiler_bytes_are_not_copied": True,
        },
        "policy": {
            "families": ["qwen3", "deepseek_v3"],
            "copied": [
                "comparison summaries",
                "frozen predictions when present",
                "collector case_result.json",
                "profile configs when present",
                "v17 matrix and audit",
            ],
            "excluded": ["raw profiler traces", "checkpoints", "logs", "datasets"],
        },
        "control_files": control_files,
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "parallelsearch",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "validation/real_npu",
    )
    args = parser.parse_args()
    manifest = import_evidence(args.source, args.destination)
    print(f"imported {manifest['case_count']} cases into {args.destination.resolve()}")


if __name__ == "__main__":
    main()
