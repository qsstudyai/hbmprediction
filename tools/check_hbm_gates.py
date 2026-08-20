#!/usr/bin/env python3
"""Check P0/P1 HBM gates from offline and calibrated reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cost_eval.validation_gate import assess_hbm_gates, render_gate_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline", type=Path,
        default=Path("validation/real_npu/p0_p1_offline_after.json"),
    )
    parser.add_argument(
        "--calibrated", type=Path,
        default=Path("validation/real_npu/p0_p1_calibrated_after.json"),
    )
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--strict", action="store_true",
        help="return 2 unless every gate passes",
    )
    args = parser.parse_args(argv)
    offline = json.loads(args.offline.read_text(encoding="utf-8"))
    calibrated = json.loads(args.calibrated.read_text(encoding="utf-8"))
    result = assess_hbm_gates(offline, calibrated)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown:
        args.markdown.write_text(render_gate_markdown(result), encoding="utf-8")
    if args.strict and any(
        item["status"] != "pass" for item in result["gates"]
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
