#!/usr/bin/env python3
"""与人工标注对拍：逐文档输出命中 / 漏检，并汇总 P@k / R@k / F1。

匹配口径：抽取结果与 annotations.json 中的标注做字符串精确匹配
（抽取结果均为原文中的连续片段，大小写已统一为小写）。

用法：python3 evaluate.py [--k 5]
"""

import argparse
import json
from pathlib import Path

from keyword_extractor import KeywordExtractor

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
ANNOTATIONS = ROOT / "annotations.json"


def load_doc(path):
    head, _, body = path.read_text(encoding="utf-8").partition("\n")
    return head.strip(), body


def main():
    parser = argparse.ArgumentParser(description="关键词抽取对拍")
    parser.add_argument("--k", type=int, default=5, help="每篇抽取的关键词个数（默认 5）")
    args = parser.parse_args()

    answers = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    docs = {name: load_doc(CORPUS_DIR / name) for name in answers}
    extractor = KeywordExtractor(corpus=docs.values())

    header = f"文档 ({args.k} 词/篇)"
    print(f"{header:<26}{'P@k':>6}{'R@k':>6}{'F1':>6}")
    print("-" * 44)
    sums = [0.0, 0.0, 0.0]
    total_hits = total_answers = total_pred = 0
    for name in sorted(answers):
        title, text = docs[name]
        pred = extractor.extract(text, title=title, topk=args.k)
        gold = answers[name]
        hits = [w for w in pred if w in gold]
        misses = [w for w in gold if w not in pred]
        p = len(hits) / len(pred) if pred else 0.0
        r = len(hits) / len(gold) if gold else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        sums[0] += p
        sums[1] += r
        sums[2] += f1
        total_hits += len(hits)
        total_answers += len(gold)
        total_pred += len(pred)
        print(f"{name:<26}{p:>6.2f}{r:>6.2f}{f1:>6.2f}")
        print(f"    命中: {'、'.join(hits) or '无'}")
        print(f"    漏检: {'、'.join(misses) or '无'}")
    n = len(answers)
    print("-" * 44)
    print(f"{'宏平均':<26}{sums[0]/n:>6.2f}{sums[1]/n:>6.2f}{sums[2]/n:>6.2f}")
    micro_p = total_hits / total_pred
    micro_r = total_hits / total_answers
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    print(f"{'微平均':<26}{micro_p:>6.2f}{micro_r:>6.2f}{micro_f1:>6.2f}")


if __name__ == "__main__":
    main()
