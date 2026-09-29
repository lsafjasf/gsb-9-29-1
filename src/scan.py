"""命令行扫描入口。

用法：
  python3 src/scan.py "要检查的文本"
  echo "文本" | python3 src/scan.py
  python3 src/scan.py --config config/mappings.json --json file.txt
退出码：0 无命中；1 有命中；2 参数/配置错误。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_guard import ContentModerator, load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="敏感词变体扫描")
    parser.add_argument("text", nargs="?", help="待扫描文本；省略则从标准输入读取")
    parser.add_argument("--config", default=str(ROOT / "config" / "mappings.json"))
    parser.add_argument("--json", action="store_true", help="输出 JSON 证据")
    args = parser.parse_args()

    try:
        moderator = ContentModerator(load_config(args.config))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"配置加载失败：{exc}", file=sys.stderr)
        return 2

    text = args.text if args.text is not None else sys.stdin.read()
    result = moderator.scan(text)

    if args.json:
        payload = {
            "text": text,
            "normalized": result.normalized.text,
            "matches": [m.to_dict() for m in result.matches],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if not result.matches:
            print("未命中敏感词。")
        for match in result.matches:
            print(
                f"[{match.start},{match.end}) {match.entry_id} "
                f"({match.category}, {match.confidence}) "
                f"原文={text[match.start:match.end]!r} "
                f"还原={match.normalized_text!r}"
            )
            for rule in match.rules[1:]:
                print(f"    证据: {rule}")
    return 1 if result.matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
