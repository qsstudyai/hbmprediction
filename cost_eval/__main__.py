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
    args = parser.parse_args(argv)
    report = ConfigAdapter.load(args.config).evaluator().evaluate()
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
