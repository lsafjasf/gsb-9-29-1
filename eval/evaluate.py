"""在带标注样例集上评测精确率 / 召回率 / F1，并检查保护区违规。

判定口径：
  - 一个建议算 TP，当且仅当存在标注错误与其 (start, end, suggestion) 完全一致；
  - 标注未被任何建议覆盖记 FN，多余建议记 FP；
  - 任何与受保护片段重叠的建议记为「保护区违规」，必须为零。

用法：`python3 eval/evaluate.py`，退出码非零表示评测未通过。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writingaid import WritingAid, load_config

HERE = os.path.dirname(os.path.abspath(__file__))


def locate(text: str, span: str, occurrence: int = 1) -> tuple[int, int]:
    pos = -1
    for _ in range(occurrence):
        pos = text.find(span, pos + 1)
        if pos < 0:
            raise ValueError(f"找不到标注片段: {span!r} (第 {occurrence} 次)")
    return pos, pos + len(span)


def main() -> int:
    aid = WritingAid(load_config())
    tp = fp = fn = 0
    violations = []
    fp_rows = []
    fn_rows = []
    with open(os.path.join(HERE, "samples.jsonl"), encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    for sample in samples:
        text = sample["text"]
        expected = []
        for err in sample.get("errors", []):
            start, end = locate(text, err["original"], err.get("occurrence", 1))
            expected.append({"start": start, "end": end, "suggestion": err["suggestion"]})
        protected = [locate(text, p["span"]) for p in sample.get("protected", [])]

        got = aid.analyze(text)
        matched = set()
        for sug in got:
            hit = None
            for i, exp in enumerate(expected):
                if i in matched:
                    continue
                if (sug.start, sug.end, sug.suggestion) == (exp["start"], exp["end"], exp["suggestion"]):
                    hit = i
                    break
            if hit is not None:
                matched.add(hit)
                tp += 1
            else:
                fp += 1
                fp_rows.append((sample["id"], sug.to_dict()))
            for lo, hi in protected:
                if sug.start < hi and sug.end > lo:
                    violations.append((sample["id"], sug.to_dict(), (lo, hi)))
        for i, exp in enumerate(expected):
            if i not in matched:
                fn += 1
                fn_rows.append((sample["id"], exp))

    total = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"样例数: {len(samples)}  标注错误: {sum(len(s['errors']) for s in samples)}")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"精确率 P = {precision:.4f}")
    print(f"召回率 R = {recall:.4f}")
    print(f"F1       = {f1:.4f}")
    print(f"保护区违规 = {len(violations)}")

    if fp_rows:
        print("\n误报明细:")
        for sid, row in fp_rows:
            print(f"  [{sid}] {row}")
    if fn_rows:
        print("\n漏报明细:")
        for sid, row in fn_rows:
            print(f"  [{sid}] {row}")
    if violations:
        print("\n保护区违规明细:")
        for sid, sug, span in violations:
            print(f"  [{sid}] {sug} 与保护区 {span} 重叠")

    ok = not violations and fp == 0 and fn == 0
    print("\n结论:", "通过" if ok else "未通过")

    report = {
        "sample_count": len(samples),
        "annotated_errors": sum(len(s["errors"]) for s in samples),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "protected_violations": len(violations),
    }
    out = os.path.join(HERE, "..", "reports", "metrics.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"指标已写入 {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
