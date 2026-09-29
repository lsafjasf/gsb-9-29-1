"""变体还原与匹配：自测入口。

运行：python3 src/demo.py
输出：
  - 变体检出结果与证据样例
  - 正常语料误伤率（单句 + 拼接全文两种口径）
  - 边界用例结果
  - 超长文本性能（AC 自动机线性扫描）
报告同时写入 reports/。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_guard import ContentModerator, load_config  # noqa: E402

CONFIG = ROOT / "config" / "mappings.json"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def show(text: str) -> str:
    return (
        text.replace("\u200b", "<ZWSP>")
        .replace("\u200c", "<ZWNJ>")
        .replace("\u200d", "<ZWJ>")
        .replace("\ufeff", "<BOM>")
        .replace("\ue000", "<B>")
        .replace(" ", "·")
    )


def build_edge_text(case: dict) -> str:
    if "text_repeat" in case:
        return case["text_repeat"] * int(case.get("repeat", 1))
    if "text_prefix" in case:
        return case["text_prefix"] * int(case.get("prefix_repeat", 1)) + case["text"]
    if "middle" in case:
        return case["text"] + case["middle"] * int(case.get("middle_repeat", 1)) + case["tail"]
    return case["text"]


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    moderator = ContentModerator(load_config(CONFIG))
    failures = 0

    variants = json.loads((DATA / "variants.json").read_text(encoding="utf-8"))["cases"]
    benign = json.loads((DATA / "benign.json").read_text(encoding="utf-8"))["sentences"]
    edges = json.loads((DATA / "edge_cases.json").read_text(encoding="utf-8"))["cases"]

    variant_rows = []
    detected = 0
    for case in variants:
        result = moderator.scan(case["text"])
        ids = [m.entry_id for m in result.matches]
        ok = all(expected in ids for expected in case["expect_ids"])
        detected += 1 if ok else 0
        if not ok:
            failures += 1
        variant_rows.append(
            {
                "name": case["name"],
                "text": case["text"],
                "normalized": result.normalized.text,
                "ok": ok,
                "expect_ids": case["expect_ids"],
                "got_ids": ids,
                "matches": [m.to_dict() for m in result.matches],
            }
        )

    # 误伤：单句口径与整段拼接口径（拼接可暴露跨句边界被打通的问题）
    fp_sentence_hits = []
    for index, sentence in enumerate(benign):
        result = moderator.scan(sentence)
        for match in result.matches:
            fp_sentence_hits.append({"index": index, "sentence": sentence, "match": match.to_dict()})

    joined = "\n".join(benign)
    joined_result = moderator.scan(joined)
    fp_joined_hits = [m.to_dict() for m in joined_result.matches]

    edge_rows = []
    long_rows = []
    for case in edges:
        text = build_edge_text(case)
        length = len(text)
        start = time.perf_counter()
        result = moderator.scan(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ids = [m.entry_id for m in result.matches]
        ok = all(expected in ids for expected in case["expect_ids"])
        if not ok:
            failures += 1
        row = {"name": case["name"], "length": length, "ok": ok,
               "expect_ids": case["expect_ids"], "got_ids": ids}
        edge_rows.append(row)
        if length >= 10000:
            long_rows.append({**row, "elapsed_ms": round(elapsed_ms, 2),
                              "throughput_kchars_per_s": round(length / elapsed_ms, 0) if elapsed_ms else None})

    # 额外超长文本微基准：200k 字正常文本 + 末尾命中
    big = "正常流水账内容，今天处理了很多日常事务。" * 10000 + "结尾发现赌\u200b博"
    start = time.perf_counter()
    big_result = moderator.scan(big)
    big_ms = (time.perf_counter() - start) * 1000
    bench = {
        "chars": len(big),
        "elapsed_ms": round(big_ms, 2),
        "kchars_per_s": round(len(big) / big_ms, 0),
        "hits": [m.to_dict() for m in big_result.matches],
    }

    n = len(benign)
    fp_rate = len(fp_sentence_hits) / n
    # 规则 of three：0 次观测时，95% 置信上界约 3/n
    upper_95 = round(3 / n * 100, 2)

    report = {
        "summary": {
            "variant_cases": len(variants),
            "variant_detected": detected,
            "variant_recall": round(detected / len(variants), 4),
            "benign_sentences": n,
            "false_positives_per_sentence": len(fp_sentence_hits),
            "false_positive_rate": fp_rate,
            "false_positive_rate_pct": round(fp_rate * 100, 2),
            "rule_of_three_upper_bound_pct_95conf": upper_95,
            "false_positives_joined_corpus": len(fp_joined_hits),
            "edge_cases": len(edges),
            "edge_passed": sum(1 for r in edge_rows if r["ok"]),
        },
        "variant_cases": variant_rows,
        "false_positive_sentence_hits": fp_sentence_hits,
        "false_positive_joined_hits": fp_joined_hits,
        "edge_cases": edge_rows,
        "long_text": long_rows,
        "micro_benchmark_200k_plus": bench,
    }
    (REPORTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evidence_samples = [r for r in variant_rows if r["matches"]][:6]
    (REPORTS / "evidence_samples.json").write_text(
        json.dumps(evidence_samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("变体还原与匹配自测报告")
    print("=" * 72)
    s = report["summary"]
    print(f"变体用例：{s['variant_detected']}/{s['variant_cases']} 检出，召回率 {s['variant_recall']*100:.1f}%")
    print(f"误伤评估：{n} 条正常句子，误伤 {s['false_positives_per_sentence']} 条，"
          f"误伤率 {s['false_positive_rate_pct']}%")
    print(f"           0 误伤时 95% 置信上界约 {upper_95}%（rule of three）")
    print(f"           全文拼接口径误伤：{s['false_positives_joined_corpus']} 处")
    print(f"边界用例：{s['edge_passed']}/{s['edge_cases']} 通过")
    print(f"超长文本：{bench['chars']:,} 字用时 {bench['elapsed_ms']} ms"
          f"（{bench['kchars_per_s']} 千字/秒，末尾命中 {len(bench['hits'])} 处）")
    print()
    print("证据样例（前 3 条）")
    print("-" * 72)
    for row in evidence_samples[:3]:
        m = row["matches"][0]
        print(f"用例：{row['name']}")
        print(f"  原文     : {show(row['text'])}")
        print(f"  还原后   : {show(row['normalized'])}")
        print(f"  命中位置 : [{m['start']},{m['end']})  原文片段={show(m['matched_text'])}")
        print(f"  规则     : {m['entry_id']} / {m['category']} / 置信度={m['confidence']}")
        for rule in m["rules"][1:]:
            print(f"             - {rule}")
        print()

    if failures:
        print(f"存在 {failures} 个未通过用例，详见 reports/report.json")
        return 1
    print("全部用例通过。详细报告：reports/report.json，证据：reports/evidence_samples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
