#!/usr/bin/env python3
"""对拍评估：转换结果与标注语料逐句比对，输出逐句准确率与错误分类。

用法:
    python3 evaluate.py --mapping data/mapping.json --corpus data/corpus.jsonl \
        --report eval_report.json

错误分类:
    unresolved_pending  字被标记为待确认，但标注给出了确定转换（词表覆盖不足）
    under_conversion    字原样保留（非待确认），但标注要求转换（映射缺失）
    over_conversion     字被转换，但标注要求保留原字（误转）
    wrong_conversion    字被转换成了错误的字（消歧选错）
    length_mismatch     输出与标注长度不一致，无法逐字对齐
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from zhconv import load_mapping, make_converters

ERR_UNRESOLVED_PENDING = "unresolved_pending"
ERR_UNDER_CONVERSION = "under_conversion"
ERR_OVER_CONVERSION = "over_conversion"
ERR_WRONG_CONVERSION = "wrong_conversion"
ERR_LENGTH_MISMATCH = "length_mismatch"


def classify_errors(source: str, plain: str, expected: str, pending_indices: set[int]):
    """逐字对齐 source/plain/expected（三者等长时），返回错误列表。"""
    errors = []
    if len(plain) != len(expected) or len(source) != len(expected):
        return [{"index": -1, "type": ERR_LENGTH_MISMATCH,
                 "detail": f"len(source)={len(source)} len(output)={len(plain)} len(expected)={len(expected)}"}]
    for i, (s_char, o_char, e_char) in enumerate(zip(source, plain, expected)):
        if o_char == e_char:
            continue
        if i in pending_indices:
            err_type = ERR_UNRESOLVED_PENDING
        elif o_char == s_char:
            err_type = ERR_UNDER_CONVERSION
        elif e_char == s_char:
            err_type = ERR_OVER_CONVERSION
        else:
            err_type = ERR_WRONG_CONVERSION
        errors.append({"index": i, "type": err_type,
                       "source_char": s_char, "output_char": o_char, "expected_char": e_char})
    return errors


def evaluate(mapping_path: str, corpus_path: str) -> dict:
    mapping = load_mapping(mapping_path)
    converters = make_converters(mapping)

    sentences = []
    error_counter: Counter = Counter()
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exact": 0, "chars": 0, "correct": 0})

    with open(corpus_path, encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]

    for case in cases:
        converter = converters[case["direction"]]
        result = converter.convert(case["source"])
        expected = case["target"]
        pending_indices = {item.index for item in result.pending}
        errors = classify_errors(case["source"], result.plain_text, expected, pending_indices)

        # 语料可声明 expect_pending：校验待确认标记是否符合预期
        expect_pending = case.get("expect_pending")
        if expect_pending is not None:
            got = [item.char for item in result.pending]
            if got != expect_pending:
                errors.append({"index": -1, "type": "pending_mismatch",
                               "detail": f"expect_pending={expect_pending} got={got}"})

        correct_chars = len(expected) - sum(1 for e in errors if e["type"] != "pending_mismatch")
        exact = not errors
        for err in errors:
            error_counter[err["type"]] += 1
        stat = category_stats[case["category"]]
        stat["total"] += 1
        stat["exact"] += int(exact)
        stat["chars"] += len(expected)
        stat["correct"] += max(correct_chars, 0)

        sentences.append({
            "id": case["id"],
            "category": case["category"],
            "direction": case["direction"],
            "source": case["source"],
            "expected": expected,
            "output": result.plain_text,
            "marked_output": result.text,
            "exact_match": exact,
            "char_accuracy": round(correct_chars / len(expected), 4) if expected else 1.0,
            "errors": errors,
        })

    total = len(sentences)
    exact = sum(1 for s in sentences if s["exact_match"])
    total_chars = sum(len(s["expected"]) for s in sentences)
    correct_chars = sum(round(s["char_accuracy"] * len(s["expected"])) for s in sentences)

    return {
        "mapping": str(mapping_path),
        "corpus": str(corpus_path),
        "summary": {
            "sentences": total,
            "exact_match": exact,
            "sentence_accuracy": round(exact / total, 4) if total else 0.0,
            "char_accuracy": round(correct_chars / total_chars, 4) if total_chars else 0.0,
            "errors_by_type": dict(error_counter.most_common()),
            "by_category": {
                cat: {
                    "sentences": st["total"],
                    "exact_match": st["exact"],
                    "sentence_accuracy": round(st["exact"] / st["total"], 4),
                    "char_accuracy": round(st["correct"] / st["chars"], 4) if st["chars"] else 0.0,
                }
                for cat, st in sorted(category_stats.items())
            },
        },
        "sentences": sentences,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="简繁转换对拍评估")
    parser.add_argument("--mapping", default="data/mapping.json")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--report", default="eval_report.json", help="JSON 报告输出路径；'-' 表示不写文件")
    args = parser.parse_args(argv)

    report = evaluate(args.mapping, args.corpus)

    header = f"{'id':<5} {'category':<17} {'dir':<4} {'exact':<6} {'char_acc':<9} errors"
    print(header)
    print("-" * len(header))
    for s in report["sentences"]:
        err_desc = ",".join(sorted({e["type"] for e in s["errors"]})) or "-"
        print(f"{s['id']:<5} {s['category']:<17} {s['direction']:<4} "
              f"{'OK' if s['exact_match'] else 'FAIL':<6} {s['char_accuracy']:<9.4f} {err_desc}")
    summary = report["summary"]
    print("-" * len(header))
    print(f"句子准确率: {summary['exact_match']}/{summary['sentences']} = {summary['sentence_accuracy']:.4f}")
    print(f"字符准确率: {summary['char_accuracy']:.4f}")
    print(f"错误分类: {summary['errors_by_type'] or '无'}")
    for cat, st in summary["by_category"].items():
        print(f"  [{cat}] 句子 {st['exact_match']}/{st['sentences']} 字符 {st['char_accuracy']:.4f}")

    if args.report != "-":
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"报告已写入 {args.report}")

    return 0 if summary["exact_match"] == summary["sentences"] else 1


if __name__ == "__main__":
    sys.exit(main())
