# -*- coding: utf-8 -*-
"""演示：对典型绕过样本输出命中证据（JSON）。

用法: python3 run_demo.py [自定义文本]
"""

import json
import sys
from pathlib import Path

from sensitive_filter import Matcher

ROOT = Path(__file__).resolve().parent

SAMPLES = [
    "赌※博",
    "加 我 唯★欣 看 贝者 博",
    "v​x​加​我，​c​4​s​i​n​o​在​线",
    "他很有威信，大家都服他。",   # 白名单抑制示例
    "今天天气不错，适合出门。",     # 无命中示例
]


def main():
    matcher = Matcher.from_files(ROOT / "config" / "rules.json", ROOT / "config" / "maps.json")
    texts = sys.argv[1:] or SAMPLES
    for text in texts:
        res = matcher.scan(text)
        print(f"原文: {text!r}")
        print(f"归一化: {res.normalized!r}")
        if res.hits:
            print("命中证据:")
            print(json.dumps(res.evidence(), ensure_ascii=False, indent=2))
        else:
            print("命中证据: 无")
        for h in res.suppressed:
            print(f"白名单抑制: {h.variant!r} -> {h.word!r} ({h.suppress_reason})")
        print("-" * 60)


if __name__ == "__main__":
    main()
