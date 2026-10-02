"""命令行入口：python3 -m writingaid [--config ...] [--chunk-size N] [文件 ...]

无文件参数时从标准输入读取；输出 JSON 数组。加 --chunk-size 可强制走增量路径
（用于演示/验证增量一致性，结果与整体处理相同）。
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .engine import WritingAid


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="writingaid", description="写作纠错建议库")
    parser.add_argument("--config", help="规则配置 JSON 路径")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help=">0 时分块走增量路径（默认整体处理）")
    parser.add_argument("files", nargs="*", help="待检查文件；缺省读标准输入")
    args = parser.parse_args(argv)

    aid = WritingAid(load_config(args.config))
    if args.files:
        text = ""
        for path in args.files:
            with open(path, "r", encoding="utf-8") as fh:
                text += fh.read()
    else:
        text = sys.stdin.read()

    suggestions = (
        aid.analyze_chunked(text, args.chunk_size) if args.chunk_size > 0
        else aid.analyze(text)
    )
    json.dump([s.to_dict() for s in suggestions], sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
