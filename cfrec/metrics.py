"""Offline ranking / coverage metrics (standard library only).

Conventions:
    test_matrix : {user_id: {item_id: rating}} ; only positive items
                  (rating >= pos_threshold) are treated as relevant targets.
    recommendations : {user_id: [(item_id, score), ...]} or None if the model
                  cannot serve that user.
"""

import math


def rmse(pairs):
    if not pairs:
        return 0.0
    return math.sqrt(sum((pred - true) ** 2 for pred, true in pairs) / len(pairs))


def precision_at_k(recommended, relevant, k):
    if not recommended:
        return 0.0
    top = [item_id for item_id, _ in recommended[:k]]
    if not top:
        return 0.0
    return len(set(top) & relevant) / float(k)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    top = [item_id for item_id, _ in recommended[:k]]
    return len(set(top) & relevant) / float(len(relevant))


def ndcg_at_k(recommended, relevant, k):
    top = [item_id for item_id, _ in recommended[:k]]
    dcg = 0.0
    for pos, item_id in enumerate(top):
        if item_id in relevant:
            dcg += 1.0 / math.log2(pos + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def aggregate_ranking(recs_by_user, test_matrix, k, pos_threshold=4.0):
    out = {"precision": [], "recall": [], "ndcg": [], "served_users": 0,
           "evaluated_users": len(test_matrix)}
    for user_id, items in test_matrix.items():
        recs = recs_by_user.get(user_id)
        if recs:
            out["served_users"] += 1
        relevant = {i for i, r in items.items() if r >= pos_threshold}
        if not relevant:
            continue
        out["precision"].append(precision_at_k(recs or [], relevant, k))
        out["recall"].append(recall_at_k(recs or [], relevant, k))
        out["ndcg"].append(ndcg_at_k(recs or [], relevant, k))
    n = len(out["precision"]) or 1
    return {
        "precision@%d" % k: sum(out["precision"]) / n,
        "recall@%d" % k: sum(out["recall"]) / n,
        "ndcg@%d" % k: sum(out["ndcg"]) / n,
        "user_coverage": out["served_users"] / float(out["evaluated_users"] or 1),
    }


def catalog_coverage(recs_by_user, catalog_size):
    if catalog_size == 0:
        return 0.0
    seen = set()
    for recs in recs_by_user.values():
        if recs:
            seen.update(item_id for item_id, _ in recs)
    return len(seen) / float(catalog_size)


def intra_list_diversity(recs_by_user, item_sim):
    """Average 1 - pairwise cosine similarity within each list (ILD)."""
    values = []
    for recs in recs_by_user.values():
        items = [item_id for item_id, _ in recs]
        if len(items) < 2:
            continue
        total = 0.0
        count = 0
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                total += 1.0 - item_sim.get(items[a], {}).get(items[b], 0.0)
                count += 1
        if count:
            values.append(total / count)
    return sum(values) / len(values) if values else 0.0


def category_entropy(recs_by_user, catalog):
    """Shannon entropy of the category distribution of recommended items."""
    counts = {}
    total = 0
    for recs in recs_by_user.values():
        for item_id, _ in recs:
            item = catalog.get(item_id)
            if item is None:
                continue
            counts[item.category] = counts.get(item.category, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / float(total)
        entropy -= p * math.log(p)
    return entropy


def novelty(recs_by_user, popularity):
    """Mean self-information -log2(population item probability) of recommendations.

    Lower popularity -> higher novelty. Popular items dominate -> low novelty.
    """
    total_views = sum(popularity.values()) or 1
    values = []
    for recs in recs_by_user.values():
        for item_id, _ in recs:
            p = max(popularity.get(item_id, 0), 1) / float(total_views)
            values.append(-math.log2(p))
    return sum(values) / len(values) if values else 0.0


def long_tail_share(recs_by_user, popularity, tail_fraction=0.5):
    """Fraction of recommendations in the least-populated half of the catalog."""
    if not popularity:
        return 0.0
    ordered = sorted(popularity, key=lambda i: (popularity[i], i))
    tail = set(ordered[:int(len(ordered) * tail_fraction)])
    hits = 0
    total = 0
    for recs in recs_by_user.values():
        for item_id, _ in recs:
            total += 1
            if item_id in tail:
                hits += 1
    return hits / float(total) if total else 0.0
