"""与人工标注对拍：输出每篇文档的命中率、漏检与误检，以及整体指标。

用法：
    python3 eval_keywords.py            # 语料模式（idf 跨 8 篇文档计算）
    python3 eval_keywords.py --single   # 单文档模式（每篇独立抽取，idf 退化为 1）

判定口径：top_k 取该文档人工标注的关键词个数；抽取结果与标注均做
小写化、去空白归一后精确匹配。
"""

import os
import sys

from keyword_extract import extract_keywords, load_corpus

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE, "data", "corpus")
ANNOTATIONS = os.path.join(BASE, "data", "annotations.json")


def normalize(term):
    return " ".join(term.lower().split())


def evaluate(texts, gold, doc_ids):
    top_ks = [len(gold[d]) for d in doc_ids]
    extracted = [
        [t for t, _ in kws]
        for kws in (extract_keywords(texts, top_k=max(top_ks)))
    ]
    rows = []
    sum_p = sum_r = 0.0
    for doc_id, kws, top_k in zip(doc_ids, extracted, top_ks):
        pred = [normalize(t) for t in kws[:top_k]]
        truth = [normalize(t) for t in gold[doc_id]]
        hits = [t for t in pred if t in truth]
        misses = [t for t in truth if t not in pred]
        false_alarms = [t for t in pred if t not in truth]
        precision = len(hits) / len(pred) if pred else 0.0
        recall = len(hits) / len(truth) if truth else 0.0
        sum_p += precision
        sum_r += recall
        rows.append((doc_id, pred, hits, misses, false_alarms, precision, recall))
    return rows, sum_p / len(rows), sum_r / len(rows)


def main():
    single = "--single" in sys.argv
    doc_ids, texts, gold = load_corpus(CORPUS_DIR, ANNOTATIONS)

    if single:
        print("== 单文档模式（每篇独立构成语料，idf = 1）==")
        all_rows = []
        for doc_id, text in zip(doc_ids, texts):
            rows, _, _ = evaluate([text], {doc_id: gold[doc_id]}, [doc_id])
            all_rows.extend(rows)
        rows = all_rows
        macro_p = sum(r[5] for r in rows) / len(rows)
        macro_r = sum(r[6] for r in rows) / len(rows)
    else:
        print("== 语料模式（idf 跨全部文档计算）==")
        rows, macro_p, macro_r = evaluate(texts, gold, doc_ids)

    for doc_id, pred, hits, misses, false_alarms, p, r in rows:
        print(f"\n[{doc_id}]  准确率 P={p:.2f}  召回 R={r:.2f}")
        print(f"  抽取: {', '.join(pred)}")
        print(f"  命中: {', '.join(hits) if hits else '(无)'}")
        print(f"  漏检: {', '.join(misses) if misses else '(无)'}")
        print(f"  误检: {', '.join(false_alarms) if false_alarms else '(无)'}")

    print("\n==================================================")
    print(f"宏平均  准确率 P={macro_p:.3f}  召回 R={macro_r:.3f}")
    total_miss = sum(len(r[3]) for r in rows)
    total_gold = sum(len(gold[d]) for d in (r[0] for r in rows))
    print(f"漏检总数 {total_miss} / 标注总数 {total_gold}")


if __name__ == "__main__":
    main()
