"""命令行入口：``python -m cost_eval CONFIG.yaml``。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .config_adapter import ConfigAdapter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="离线估算并行训练配置的每卡峰值显存和 OOM 状态"
    )
    parser.add_argument("config", help="JSON/YAML 配置文件")
    parser.add_argument(
        "--trace",
        help="把逐事件显存 trace 写入 .json 或 .csv 文件",
    )
    parser.add_argument(
        "--trace-tensors",
        action="store_true",
        help="在 trace 中保存每一步完整的 layer-local live tensor 快照",
    )
    args = parser.parse_args(argv)
    evaluator = ConfigAdapter.load(args.config).evaluator()
    if args.trace:
        report, trace = evaluator.evaluate_with_trace(
            capture_live_tensors=args.trace_tensors
        )
        trace.write(args.trace)
    else:
        report = evaluator.evaluate()
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
