"""参照实现：与 ranking_metrics.py 独立编写，用于对拍。

刻意采用不同的代码路径，但遵循同一份确定性规范：
  - 并列：先按 doc_id 升序稳定排序，再按 -score 稳定排序（等价于 (-score, doc_id)）。
  - 缺失标注：defaultdict(float)，缺省 0.0。
  - 零相关查询：全部指标 0.0。
  - AP 分母为相关文档总数 R。
浮点累加顺序与库保持一致（位置升序、查询顺序），因此对拍要求逐位相等。
"""

from __future__ import annotations

import math
from collections import defaultdict


def _ordered(ranking, k):
    by_id = sorted(ranking, key=lambda pair: pair[0])
    by_score = sorted(by_id, key=lambda pair: -pair[1])
    return by_score[: max(k, 0)]


def _label_map(labels):
    table = defaultdict(float)
    for doc_id, rel in labels.items():
        table[doc_id] = float(rel)
    return table


def _dcg(gains):
    # 显式 += 累加：与库保持相同的浮点运算顺序（CPython 3.12 起 sum()
    # 对浮点使用 Neumaier 补偿求和，会与朴素累加差 1 ULP）。
    total = 0.0
    for rank, gain in enumerate(gains):
        total += (2.0 ** gain - 1.0) / math.log2(rank + 2.0)
    return total


def dcg_at(ranking, labels, k):
    table = _label_map(labels)
    return _dcg([table[doc_id] for doc_id, _ in _ordered(ranking, k)])


def idcg_at(labels, k):
    return _dcg(sorted(map(float, labels.values()), reverse=True)[: max(k, 0)])


def ndcg_at(ranking, labels, k):
    ideal = idcg_at(labels, k)
    return dcg_at(ranking, labels, k) / ideal if ideal != 0.0 else 0.0


def _ap_numerator_at(ranking, labels, k):
    table = _label_map(labels)
    hits = 0
    acc = 0.0
    for rank, (doc_id, _score) in enumerate(_ordered(ranking, k), start=1):
        if table[doc_id] > 0.0:
            hits += 1
            acc += hits / float(rank)
    return acc


def average_precision_at(ranking, labels, k):
    relevant = sum(1 for rel in labels.values() if float(rel) > 0.0)
    if relevant == 0:
        return 0.0
    return _ap_numerator_at(ranking, labels, k) / relevant


def reciprocal_rank_at(ranking, labels, k):
    table = _label_map(labels)
    for rank, (doc_id, _score) in enumerate(_ordered(ranking, k), start=1):
        if table[doc_id] > 0.0:
            return 1.0 / float(rank)
    return 0.0


def hit_rate_at(ranking, labels, k):
    table = _label_map(labels)
    return 1.0 if any(table[doc_id] > 0.0 for doc_id, _ in _ordered(ranking, k)) else 0.0


def evaluate_query(ranking, labels, k):
    return {
        "ndcg": ndcg_at(ranking, labels, k),
        "ap": average_precision_at(ranking, labels, k),
        "mrr": reciprocal_rank_at(ranking, labels, k),
        "hit": hit_rate_at(ranking, labels, k),
        "dcg": dcg_at(ranking, labels, k),
        "idcg": idcg_at(labels, k),
        "ap_numerator": _ap_numerator_at(ranking, labels, k),
        "relevant": sum(1 for rel in labels.values() if float(rel) > 0.0),
    }


def evaluate(dataset, k):
    per_query = [evaluate_query(item["ranking"], item["labels"], k) for item in dataset]
    count = len(per_query)
    macro = {}
    for metric in ("ndcg", "ap", "mrr", "hit"):
        macro[metric] = sum(q[metric] for q in per_query) / count if count else 0.0
    total_rel = sum(q["relevant"] for q in per_query)
    total_ideal = sum(q["idcg"] for q in per_query)
    micro = {
        "ndcg": (sum(q["dcg"] for q in per_query) / total_ideal) if total_ideal else 0.0,
        "ap": (sum(q["ap_numerator"] for q in per_query) / total_rel) if total_rel else 0.0,
        "mrr": macro["mrr"],
        "hit": macro["hit"],
    }
    return {"per_query": per_query, "macro": macro, "micro": micro}
