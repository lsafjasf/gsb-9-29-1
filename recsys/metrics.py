"""Offline evaluation metrics.

Accuracy (top-N recommendation, positive = test rating >= relevance_threshold):

* precision@N, recall@N
* ndcg@N, relevance graded as (rating - threshold), clipped at 0

Beyond-accuracy (the difference vs. a pure-popularity fallback):

* catalog_coverage : fraction of catalog items ever recommended
* category_coverage: fraction of categories ever recommended
* diversity        : mean pairwise content dissimilarity inside lists
                     (1 - Jaccard(category+tag) over all list pairs)
* intra_list_similarity : mean fraction of list pairs sharing a category
                     (low = varied; popularity lists concentrate -> high)
* novelty         : mean -log2 popularity probability of recommended items
                     (high novelty = long-tail exposure)
* gini            : Gini coefficient of the recommendation-frequency
                     distribution (higher = more concentrated/unfair)

Cold-start views:
metrics are also computed on the cold-user subset, and ``cold_item_exposure``
measures what fraction of recommendations point at cold items plus the
recall over test interactions on cold items.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Dataset
from .recommenders import BaseRecommender

Pair = Tuple[int, float]


def precision_recall_ndcg(
    recommended: Sequence[int],
    relevant: Dict[int, float],
    threshold: float = 4.0,
) -> Tuple[float, float, float]:
    n = len(recommended)
    if n == 0:
        return 0.0, 0.0, 0.0
    hits = sum(1 for item in recommended if relevant.get(item, 0.0) >= threshold)
    precision = hits / n
    num_relevant = sum(1 for rating in relevant.values() if rating >= threshold)
    recall = hits / num_relevant if num_relevant else 0.0

    dcg = 0.0
    for position, item in enumerate(recommended, start=1):
        rating = relevant.get(item)
        if rating is not None and rating >= threshold:
            graded = rating - threshold  # 0.0 .. 1.0
            dcg += (2.0 ** graded - 1.0) / math.log2(position + 1)
    ideal = sorted(
        (r - threshold for r in relevant.values() if r >= threshold),
        reverse=True,
    )
    idcg = sum(
        (2.0 ** graded - 1.0) / math.log2(position + 1)
        for position, graded in enumerate(ideal[:n], start=1)
    )
    ndcg = dcg / idcg if idcg > 0.0 else 0.0
    return precision, recall, ndcg


def gini_coefficient(freq: Dict[int, int], num_items: int) -> float:
    values = sorted(freq.get(item, 0) for item in range(num_items))
    total = sum(values)
    if total == 0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2.0 * weighted) / (num_items * total) - (num_items + 1.0) / num_items


def evaluate(
    recommender: BaseRecommender,
    dataset: Dataset,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
    users: Optional[Iterable[int]] = None,
    seed: int = 7,
) -> Dict[str, float]:
    test_matrix = dataset.test_matrix()
    eval_users = sorted(set(users) if users is not None else test_matrix.keys())

    def feature_set(item: int) -> frozenset:
        return frozenset(
            [dataset.item_categories[item], *dataset.item_tags[item]]
        )

    item_features = {item: feature_set(item) for item in range(dataset.num_items)}

    def dissimilarity(a: int, b: int) -> float:
        fa = item_features[a]
        fb = item_features[b]
        if not fa and not fb:
            return 0.0
        return 1.0 - len(fa & fb) / len(fa | fb)

    num_raters = defaultdict(int)
    for ratings in dataset.train_matrix().values():
        for item in ratings:
            num_raters[item] += 1
    num_users = max(len(dataset.train_matrix()), 1)

    precision_sum = recall_sum = ndcg_sum = 0.0
    diversity_sum = novelty_sum = 0.0
    ils_sum = 0.0
    evaluated = 0
    rec_frequency: Dict[int, int] = defaultdict(int)
    recommended_items: set = set()
    recommended_categories: set = set()
    cold_item_recs = 0
    cold_item_hits = 0
    cold_item_relevant = 0

    for user in eval_users:
        relevant = test_matrix.get(user, {})
        if not any(r >= relevance_threshold for r in relevant.values()):
            # Nothing positive to gain recall on; still count coverage below
            # only when the user has test items at all.
            if not relevant:
                continue
        rng = random.Random(seed + user)
        recommendations = recommender.recommend(user, top_n, rng=rng)
        items = [item for item, _score in recommendations]
        if not items:
            continue
        evaluated += 1

        precision, recall, ndcg = precision_recall_ndcg(
            items, relevant, relevance_threshold
        )
        precision_sum += precision
        recall_sum += recall
        ndcg_sum += ndcg

        if len(items) > 1:
            pair_count = 0
            pair_sum = 0.0
            for idx, item_a in enumerate(items):
                for item_b in items[idx + 1:]:
                    pair_sum += dissimilarity(item_a, item_b)
                    pair_count += 1
            diversity_sum += pair_sum / pair_count
            same_category = sum(
                1 for p, item_a in enumerate(items)
                for item_b in items[p + 1:]
                if dataset.item_categories[item_a]
                == dataset.item_categories[item_b]
            )
            ils_sum += same_category / pair_count

        for item in items:
            novelty_sum += -math.log2((num_raters.get(item, 0) + 1) / num_users)
            rec_frequency[item] += 1
            recommended_items.add(item)
            recommended_categories.add(dataset.item_categories[item])
            if item in dataset.cold_items:
                cold_item_recs += 1
        for item, rating in relevant.items():
            if item in dataset.cold_items and rating >= relevance_threshold:
                cold_item_relevant += 1
                if item in items:
                    cold_item_hits += 1

    total_recs = sum(rec_frequency.values())
    return {
        "users_evaluated": float(evaluated),
        "precision@%d" % top_n: precision_sum / evaluated if evaluated else 0.0,
        "recall@%d" % top_n: recall_sum / evaluated if evaluated else 0.0,
        "ndcg@%d" % top_n: ndcg_sum / evaluated if evaluated else 0.0,
        "catalog_coverage": len(recommended_items) / dataset.num_items,
        "category_coverage": len(recommended_categories) / len(dataset.categories),
        "diversity": diversity_sum / evaluated if evaluated else 0.0,
        "intra_list_similarity": ils_sum / evaluated if evaluated else 0.0,
        "novelty": novelty_sum / total_recs if total_recs else 0.0,
        "gini": gini_coefficient(rec_frequency, dataset.num_items),
        "cold_item_exposure": cold_item_recs / total_recs if total_recs else 0.0,
        "cold_item_recall": (
            cold_item_hits / cold_item_relevant if cold_item_relevant else 0.0
        ),
    }
