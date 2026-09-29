"""Similarity functions and neighbor index construction.

Missing-value handling
----------------------
All sims here are *co-rated* sims: only items (or users) observed on both
sides participate in the formula. Missing ratings are never imputed as
zero. Predictions additionally center with per-user / per-item means, so
an unseen cell does not drag a score toward 0 ("the user hates it").

* User similarity: **Pearson correlation** on co-rated items, shrunk by
  overlap size so that a single shared item cannot imply perfect
  correlation. A co-rating count below ``min_overlap`` yields similarity 0.
* Item similarity: **adjusted cosine** -- each rating is mean-centered by
  the *user* who gave it (removing systematic easy/hard raters) -- with
  the same overlap shrinkage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


def user_means(matrix: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    return {u: sum(r.values()) / len(r) for u, r in matrix.items() if r}


def item_means(matrix: Dict[int, Dict[int, float]]) -> Dict[int, float]:
    totals: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    for ratings in matrix.values():
        for item, rating in ratings.items():
            totals[item] += rating
            counts[item] += 1
    return {item: totals[item] / counts[item] for item in totals}


def _shrunk(numerator: float, norm_a: float, norm_b: float,
            overlap: int, shrink: float) -> float:
    if overlap <= 0 or norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    raw = numerator / (norm_a * norm_b)
    # Shrinkage: tiny overlap regresses similarity toward 0.
    return raw * overlap / (overlap + shrink)


def pearson_users(
    ratings_a: Dict[int, float],
    ratings_b: Dict[int, float],
    shrink: float = 3.0,
) -> float:
    """Pearson corr. of two users over co-rated items, with shrinkage."""

    common = ratings_a.keys() & ratings_b.keys()
    if not common:
        return 0.0
    mean_a = sum(ratings_a.values()) / len(ratings_a)
    mean_b = sum(ratings_b.values()) / len(ratings_b)
    numerator = norm_a = norm_b = 0.0
    for item in common:
        da = ratings_a[item] - mean_a
        db = ratings_b[item] - mean_b
        numerator += da * db
        norm_a += da * da
        norm_b += db * db
    return _shrunk(numerator, math.sqrt(norm_a), math.sqrt(norm_b),
                   len(common), shrink)


def adjusted_cosine_items(
    item_a: int,
    item_b: int,
    matrix: Dict[int, Dict[int, float]],
    means: Dict[int, float],
    shrink: float = 3.0,
) -> float:
    """Item-item adjusted cosine using user-mean-centered ratings."""

    numerator = norm_a = norm_b = 0.0
    overlap = 0
    for user, ratings in matrix.items():
        if item_a in ratings and item_b in ratings:
            center = means.get(user, 0.0)
            da = ratings[item_a] - center
            db = ratings[item_b] - center
            numerator += da * db
            norm_a += da * da
            norm_b += db * db
            overlap += 1
    return _shrunk(numerator, math.sqrt(norm_a), math.sqrt(norm_b),
                   overlap, shrink)


def build_user_neighbors(
    matrix: Dict[int, Dict[int, float]],
    k: int = 30,
    shrink: float = 3.0,
    min_overlap: int = 2,
) -> Dict[int, List[Tuple[int, float]]]:
    """Top-k positive-similarity neighbors for every user.

    Candidates are restricted to users sharing at least one item (inverted
    index), which is essential on sparse matrices -- comparing all user
    pairs would mostly produce zero overlap.
    """

    # item -> users who rated it
    raters: Dict[int, List[int]] = defaultdict(list)
    for user, ratings in matrix.items():
        for item in ratings:
            raters[item].append(user)

    candidate_users: Dict[int, set] = defaultdict(set)
    for users in raters.values():
        for user in users:
            candidate_users[user].update(users)

    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for user, ratings_a in matrix.items():
        scored: List[Tuple[int, float]] = []
        for other in candidate_users.get(user, ()):
            if other == user:
                continue
            ratings_b = matrix[other]
            if len(ratings_a.keys() & ratings_b.keys()) < min_overlap:
                continue
            sim = pearson_users(ratings_a, ratings_b, shrink)
            if sim > 0.0:
                scored.append((other, sim))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        neighbors[user] = scored[:k]
    return neighbors


def build_item_neighbors(
    matrix: Dict[int, Dict[int, float]],
    k: int = 30,
    shrink: float = 3.0,
    min_overlap: int = 2,
) -> Dict[int, List[Tuple[int, float]]]:
    """Top-k positive-similarity neighbors for every item (adjusted cosine)."""

    means = user_means(matrix)

    # user -> rated items, then accumulate item-pair stats over each user.
    stats: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for user, ratings in matrix.items():
        center = means.get(user, 0.0)
        items = list(ratings.keys())
        centered = {item: ratings[item] - center for item in items}
        for idx, item_a in enumerate(items):
            for item_b in items[idx + 1:]:
                da = centered[item_a]
                db = centered[item_b]
                key = (item_a, item_b)
                stats[key][0] += da * db
                stats[key][1] += da * da
                stats[key][2] += db * db
                stats[key][3] += 1

    pair_sims: Dict[Tuple[int, int], float] = {}
    for (item_a, item_b), (num, na, nb, overlap) in stats.items():
        if overlap < min_overlap:
            continue
        sim = _shrunk(num, math.sqrt(na), math.sqrt(nb), overlap, shrink)
        if sim > 0.0:
            pair_sims[(item_a, item_b)] = sim

    neighbors: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (item_a, item_b), sim in pair_sims.items():
        neighbors[item_a].append((item_b, sim))
        neighbors[item_b].append((item_a, sim))
    for item in neighbors:
        neighbors[item].sort(key=lambda pair: (-pair[1], pair[0]))
        neighbors[item] = neighbors[item][:k]
    return dict(neighbors)
