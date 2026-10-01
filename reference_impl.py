"""reference_impl.py

Deliberately naive, independent reference implementation used ONLY for
differential testing. It re-sorts and re-scans from scratch for every cutoff
and shares no code with ranking_metrics, so an agreement across thousands of
random cases is real evidence rather than a tautology.

Conventions are intentionally the same documented ones:
  score desc, then doc id asc; unlabeled -> rel 0; AP denominator is R.
"""

import math


def _ranked_ids(scored_docs):
    by_id = sorted(scored_docs, key=lambda item: item[0])
    by_score = sorted(by_id, key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in by_score]


def _gain(rel, gain):
    r = float(rel)
    if gain == "linear":
        return r
    if gain == "exp":
        return 2.0 ** r - 1.0
    raise ValueError(gain)


def dcg_at_k(scored_docs, relevance, k, gain="linear"):
    total = 0.0
    for rank, doc_id in enumerate(_ranked_ids(scored_docs)[:k], start=1):
        g = _gain(relevance.get(doc_id, 0.0), gain)
        total += g / math.log2(rank + 1)
    return total


def idcg_at_k(relevance, k, gain="linear"):
    total = 0.0
    gains = sorted((_gain(v, gain) for v in relevance.values()), reverse=True)
    for rank, g in enumerate(gains[:k], start=1):
        total += g / math.log2(rank + 1)
    return total


def ndcg_at_k(scored_docs, relevance, k, gain="linear", zero_rel_ndcg=0.0):
    ideal = idcg_at_k(relevance, k, gain)
    if ideal == 0.0:
        return float(zero_rel_ndcg)
    return dcg_at_k(scored_docs, relevance, k, gain) / ideal


def ap_num_at_k(scored_docs, relevance, k):
    """Unnormalized AP numerator: sum of precision at each hit rank <= k."""
    total = 0.0
    hits = 0
    for rank, doc_id in enumerate(_ranked_ids(scored_docs)[:k], start=1):
        if relevance.get(doc_id, 0.0) > 0:
            hits += 1
            total += hits / rank
    return total


def ap_at_k(scored_docs, relevance, k):
    total_r = sum(1 for v in relevance.values() if v > 0)
    if total_r == 0:
        return 0.0
    return ap_num_at_k(scored_docs, relevance, k) / total_r


def rr_at_k(scored_docs, relevance, k):
    for rank, doc_id in enumerate(_ranked_ids(scored_docs)[:k], start=1):
        if relevance.get(doc_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def hit_at_k(scored_docs, relevance, k):
    for doc_id in _ranked_ids(scored_docs)[:k]:
        if relevance.get(doc_id, 0.0) > 0:
            return 1.0
    return 0.0


METRIC_FNS = {
    "dcg": dcg_at_k,
    "ndcg": ndcg_at_k,
    "ap": ap_at_k,
    "rr": rr_at_k,
    "hit": hit_at_k,
}


def evaluate(queries, ks, gain="linear", zero_rel_ndcg=0.0):
    """Macro/micro evaluation mirroring ranking_metrics.evaluate."""
    ks = sorted(set(ks))
    per_query = {}
    rows = []
    for qid, scored_docs, relevance in queries:
        row = {"scored": scored_docs, "rel": relevance}
        values = {}
        for metric, fn in METRIC_FNS.items():
            values[metric] = {}
            for k in ks:
                if metric == "ndcg":
                    values[metric][k] = fn(
                        scored_docs, relevance, k, gain, zero_rel_ndcg
                    )
                elif metric == "dcg":
                    values[metric][k] = fn(
                        scored_docs, relevance, k, gain
                    )
                else:
                    values[metric][k] = fn(scored_docs, relevance, k)
        per_query[qid] = values
        rows.append(row)

    n = len(rows)
    macro = {m: {} for m in METRIC_FNS}
    micro = {m: {} for m in METRIC_FNS}
    for k in ks:
        for metric in METRIC_FNS:
            macro[metric][k] = (
                sum(per_query[qid][metric][k] for qid in per_query) / n
                if n else 0.0
            )
            if metric == "ndcg":
                total_dcg = sum(
                    dcg_at_k(r["scored"], r["rel"], k, gain) for r in rows
                )
                total_idcg = sum(
                    idcg_at_k(r["rel"], k, gain) for r in rows
                )
                micro[metric][k] = (
                    total_dcg / total_idcg if total_idcg > 0 else 0.0
                )
            elif metric == "ap":
                total_num = sum(
                    ap_num_at_k(r["scored"], r["rel"], k) for r in rows
                )
                total_r = sum(
                    sum(1 for v in r["rel"].values() if v > 0) for r in rows
                )
                micro[metric][k] = (
                    total_num / total_r if total_r > 0 else 0.0
                )
            else:
                micro[metric][k] = (
                    sum(per_query[qid][metric][k] for qid in per_query) / n
                    if n else 0.0
                )
    return {"per_query": per_query, "macro": macro, "micro": micro,
            "n_queries": n}
