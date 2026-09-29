"""Recommendation strategies.

Every recommender exposes ``recommend(user, n, candidates, rng)`` returning
``[(item_id, score), ...]`` sorted best-first. Items the user already rated
in training are excluded.

Strategies
----------
popularity : global rating-count ranking (the "simple fallback" baseline)
user_cf    : user-based collaborative filtering (Pearson neighbors)
item_cf    : item-based collaborative filtering (adjusted-cosine neighbors)
content    : content features (category + tags), weighted-Jaccard profile
cat_pop    : popularity restricted to the user's favorite category
explore    : epsilon-greedy long-tail exploration slots over cat-pop/content
hybrid     : item_cf + content + popularity blend (graceful support loss)
"""

from __future__ import annotations

import math
import random
import zlib
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import Dataset
from .similarities import (
    build_item_neighbors,
    build_user_neighbors,
    user_means,
)

RATING_SCALE = 4.0  # 5 - 1, used to normalize CF predictions to [0, 1]


class BaseRecommender:
    name = "base"

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.matrix = dataset.train_matrix()
        self.all_items = range(dataset.num_items)

        counts: Dict[int, int] = defaultdict(int)
        sums: Dict[int, float] = defaultdict(float)
        for ratings in self.matrix.values():
            for item, rating in ratings.items():
                counts[item] += 1
                sums[item] += rating
        self.counts = dict(counts)
        self.avg_rating = {item: sums[item] / counts[item] for item in sums}
        self.max_count = max(counts.values(), default=1)

        # Items grouped by category, pre-sorted by popularity.
        self.category_items: Dict[str, List[int]] = defaultdict(list)
        for item in range(dataset.num_items):
            self.category_items[dataset.item_categories[item]].append(item)
        for items in self.category_items.values():
            items.sort(key=lambda i: (-self.counts.get(i, 0), i))

    def _candidates(self, user: int, candidates: Optional[Iterable[int]]) -> List[int]:
        seen = self.matrix.get(user, {})
        pool = self.all_items if candidates is None else candidates
        return [item for item in pool if item not in seen]

    def _rank(self, user: int, items: Sequence[int]) -> List[Tuple[int, float]]:
        scored = [(item, self.score(user, item)) for item in items]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored

    def score(self, user: int, item: int) -> float:
        raise NotImplementedError

    def recommend(self, user: int, n: int = 10,
                  candidates: Optional[Iterable[int]] = None,
                  rng: Optional[random.Random] = None) -> List[Tuple[int, float]]:
        return self._rank(user, self._candidates(user, candidates))[:n]

    # Popularity score in [0, 1].
    def pop_score(self, item: int) -> float:
        return math.log1p(self.counts.get(item, 0)) / math.log1p(self.max_count)


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def score(self, user: int, item: int) -> float:
        # Rating count first, average rating breaks ties deterministically.
        return float(self.counts.get(item, 0) * 100) + self.avg_rating.get(item, 0.0)


class UserCFRecommender(BaseRecommender):
    """User-based CF.

    Prediction (positive-similarity neighbors only)::

        p(u, i) = mean(u) + sum_v sim(u,v) * (r(v,i) - mean(v)) / sum_v sim(u,v)

    Users/items without neighbor support get ``None`` (handled by callers /
    hybrid blending) instead of a fabricated global mean.
    """

    name = "user_cf"

    def __init__(self, dataset: Dataset, k: int = 30, shrink: float = 3.0):
        super().__init__(dataset)
        self.means = user_means(self.matrix)
        self.global_mean = (
            sum(self.means.values()) / len(self.means) if self.means else 3.0
        )
        self.neighbors = build_user_neighbors(self.matrix, k=k, shrink=shrink)

    def predict(self, user: int, item: int) -> Optional[float]:
        if user not in self.matrix:
            return None
        numerator = denominator = 0.0
        for other, sim in self.neighbors.get(user, ()):  # positive sims only
            rating = self.matrix.get(other, {}).get(item)
            if rating is None:
                continue
            numerator += sim * (rating - self.means.get(other, self.global_mean))
            denominator += abs(sim)
        if denominator <= 0.0:
            return None
        return self.means.get(user, self.global_mean) + numerator / denominator

    def score(self, user: int, item: int) -> float:
        pred = self.predict(user, item)
        return pred if pred is not None else -1.0

    def recommend(self, user: int, n: int = 10,
                  candidates=None, rng=None) -> List[Tuple[int, float]]:
        scored = [
            (item, pred)
            for item in self._candidates(user, candidates)
            if (pred := self.predict(user, item)) is not None
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:n]


class ItemCFRecommender(BaseRecommender):
    """Item-based CF with adjusted-cosine item neighbors.

    Prediction uses the active user's own ratings centered by her mean::

        p(u, i) = mean(u) + sum_j sim(i,j) * (r(u,j) - mean(u)) / sum_j sim(i,j)

    An item with no co-rating history has no neighbors and therefore no
    prediction -- the key cold-item signal the hybrid reacts to.
    """

    name = "item_cf"

    def __init__(self, dataset: Dataset, k: int = 30, shrink: float = 3.0):
        super().__init__(dataset)
        self.means = user_means(self.matrix)
        self.global_mean = (
            sum(self.means.values()) / len(self.means) if self.means else 3.0
        )
        self.neighbors = build_item_neighbors(self.matrix, k=k, shrink=shrink)

    def predict(self, user: int, item: int) -> Optional[float]:
        user_ratings = self.matrix.get(user)
        if not user_ratings:
            return None
        user_mean = self.means.get(user, self.global_mean)
        numerator = denominator = 0.0
        for other, sim in self.neighbors.get(item, ()):  # positive sims only
            rating = user_ratings.get(other)
            if rating is None:
                continue
            numerator += sim * (rating - user_mean)
            denominator += abs(sim)
        if denominator <= 0.0:
            return None
        return user_mean + numerator / denominator

    def score(self, user: int, item: int) -> float:
        pred = self.predict(user, item)
        return pred if pred is not None else -1.0

    def recommend(self, user: int, n: int = 10,
                  candidates=None, rng=None) -> List[Tuple[int, float]]:
        scored = [
            (item, pred)
            for item in self._candidates(user, candidates)
            if (pred := self.predict(user, item)) is not None
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:n]


class ContentRecommender(BaseRecommender):
    """Content-based profile: weighted Jaccard over category + tag features.

    Profile weight for a feature = sum of (rating - 3) the user gave items
    carrying that feature (negative ratings shrink the feature). Item score
    is the weighted-Jaccard overlap between the profile and the item's
    feature set, tie-broken by popularity. Requires no co-rating, so it
    works for cold items; a cold (empty) user falls back to popularity.
    """

    name = "content"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.profiles: Dict[int, Dict[str, float]] = {}
        for user, ratings in self.matrix.items():
            profile: Dict[str, float] = defaultdict(float)
            for item, rating in ratings.items():
                weight = rating - 3.0
                profile[self.dataset.item_categories[item]] += weight
                for tag in self.dataset.item_tags[item]:
                    profile[tag] += weight * 0.5
            self.profiles[user] = {f: w for f, w in profile.items() if w > 0.0}
        self._jaccard_cache: Dict[Tuple[int, int], float] = {}

    def _features(self, item: int) -> List[str]:
        return [self.dataset.item_categories[item], *self.dataset.item_tags[item]]

    def content_score(self, user: int, item: int) -> float:
        key = (user, item)
        cached = self._jaccard_cache.get(key)
        if cached is not None:
            return cached
        profile = self.profiles.get(user)
        if not profile:
            result = 0.0
        else:
            features = self._features(item)
            intersection = sum(min(profile.get(f, 0.0), 1.0) for f in set(features))
            union = sum(max(profile.get(f, 0.0), 1.0 if f in features else 0.0)
                        for f in set(profile) | set(features))
            result = intersection / union if union > 0.0 else 0.0
        self._jaccard_cache[key] = result
        return result

    def score(self, user: int, item: int) -> float:
        # Tiny popularity term provides a deterministic, useful tie-breaker.
        return self.content_score(user, item) * 10.0 + self.pop_score(item) * 0.01


class CategoryPopularRecommender(BaseRecommender):
    """Most popular items within the user's favorite category.

    Favorite category = the one the user rated most (highest avg on ties).
    Cold users with no history fall back to the global head category, i.e.
    plain popularity -- the strategy cannot help a truly empty profile.
    """

    name = "cat_pop"

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        head_category = max(
            dataset.categories,
            key=lambda c: sum(self.counts.get(i, 0) for i in self.category_items[c]),
        )
        self.head_category = head_category
        self.favorite: Dict[int, str] = {}
        for user, ratings in self.matrix.items():
            cat_count: Dict[str, int] = defaultdict(int)
            cat_sum: Dict[str, float] = defaultdict(float)
            for item, rating in ratings.items():
                cat = dataset.item_categories[item]
                cat_count[cat] += 1
                cat_sum[cat] += rating
            self.favorite[user] = max(
                cat_count,
                key=lambda c: (cat_count[c], cat_sum[c] / cat_count[c], c),
            )

    def score(self, user: int, item: int) -> float:
        category = self.favorite.get(user, self.head_category)
        in_favorite = self.dataset.item_categories[item] == category
        if not in_favorite:
            return -1.0
        return float(self.counts.get(item, 0) * 100) + self.avg_rating.get(item, 0.0)


class ExplorationRecommender(CategoryPopularRecommender):
    """Epsilon-greedy long-tail exploration.

    ``explore_ratio`` of the slots are filled uniformly at random from the
    long-tail pool (items at or below median popularity, zero-rating items
    first); the rest exploit the category-popularity ranking. The RNG is
    seeded per user so results are reproducible. This is the only strategy
    that surfaces brand-new items/users' blind spots systematically.
    """

    name = "explore"

    def __init__(self, dataset: Dataset, explore_ratio: float = 0.4):
        super().__init__(dataset)
        self.explore_ratio = explore_ratio
        popularities = sorted(self.counts.values())
        self.median_pop = (
            popularities[len(popularities) // 2] if popularities else 0
        )
        self.long_tail = [
            item for item in range(dataset.num_items)
            if self.counts.get(item, 0) <= self.median_pop
        ]

    def recommend(self, user: int, n: int = 10,
                  candidates: Optional[Iterable[int]] = None,
                  rng: Optional[random.Random] = None) -> List[Tuple[int, float]]:
        rng = rng or random.Random(zlib.crc32(("explore:%d" % user).encode()))
        pool = self._candidates(user, candidates)
        pool_set = set(pool)
        tail = [item for item in self.long_tail if item in pool_set]
        rng.shuffle(tail)

        n_explore = int(math.ceil(n * self.explore_ratio))
        picks: List[int] = tail[:n_explore]
        picked = set(picks)
        for item, _score in self._rank(user, [i for i in pool if i not in picked]):
            picks.append(item)
            if len(picks) >= n:
                break
        return [(item, self.score(user, item)) for item in picks[:n]]


class HybridRecommender(BaseRecommender):
    """Tiered blend of item-CF, content, category popularity and exploration.

    Linear blending is unsafe on a sparse matrix: CF emits no prediction
    for ~90% of candidate cells, so a normalized linear mix would let the
    always-present popularity term dominate and reproduce the hot-list
    fallback. Scoring is therefore split into explicit tiers:

    * CF-supported cell (user and item have co-rating history):
      ``2.0 + 0.8*s_cf + 0.15*s_ct + 0.05*s_pop`` -- always ranked above
      unsupported cells, so a real CF signal is never diluted.
    * Unsupported cell (cold item or thin history):
      ``0.8*s_ct + 0.2*s_pop`` -- content match dominates, category-pop
      effects live inside ``s_ct``/``s_pop``; zero-history cold items stay
      reachable via the content term instead of being banned.
    * Exploration slots: ``explore_ratio`` of every list is reserved for
      seeded long-tail sampling (see :class:`ExplorationRecommender`),
      guaranteeing brand-new items exposure offline and online.

    A brand-new user has no CF and no content profile, so the unsupported
    tier degenerates to popularity -- the honest limit of history-free
    recommendation; the exploration slots still diversify her list.
    """

    name = "hybrid"

    def __init__(self, dataset: Dataset, k: int = 30,
                 weights: Optional[Dict[str, float]] = None,
                 explore_ratio: float = 0.2):
        super().__init__(dataset)
        self.item_cf = ItemCFRecommender(dataset, k=k)
        self.content = ContentRecommender(dataset)
        self.explorer = ExplorationRecommender(dataset, explore_ratio=explore_ratio)
        self.explore_ratio = explore_ratio
        # Tier-internal mixing weights (kept for inspection).
        self.weights = weights or {"cf": 0.8, "content": 0.15, "pop": 0.05,
                                   "cold_content": 0.8, "cold_pop": 0.2}

    def score_parts(self, user: int, item: int) -> Dict[str, float]:
        parts: Dict[str, float] = {}
        pred = self.item_cf.predict(user, item)
        if pred is not None:
            parts["cf"] = max(0.0, min(1.0, (pred - 1.0) / RATING_SCALE))
        content = self.content.content_score(user, item)
        if content > 0.0:
            parts["content"] = content
        parts["pop"] = self.pop_score(item)
        return parts

    def score(self, user: int, item: int) -> float:
        parts = self.score_parts(user, item)
        s_ct = parts.get("content", 0.0)
        s_pop = parts.get("pop", 0.0)
        if "cf" in parts:
            w = self.weights
            return (2.0 + w["cf"] * parts["cf"]
                    + w["content"] * s_ct + w["pop"] * s_pop)
        return self.weights["cold_content"] * s_ct + self.weights["cold_pop"] * s_pop

    def recommend(self, user: int, n: int = 10,
                  candidates: Optional[Iterable[int]] = None,
                  rng: Optional[random.Random] = None) -> List[Tuple[int, float]]:
        rng = rng or random.Random(zlib.crc32(("hybrid:%d" % user).encode()))
        pool = self._candidates(user, candidates)
        n_explore = int(math.ceil(n * self.explore_ratio))

        long_tail = [item for item in pool
                     if item in set(self.explorer.long_tail)]
        rng.shuffle(long_tail)
        explore_picks = long_tail[:n_explore]
        reserved = set(explore_picks)

        exploit = self._rank(user, [i for i in pool if i not in reserved])
        picks = explore_picks + [item for item, _ in exploit[: n - len(explore_picks)]]
        return [(item, self.score(user, item)) for item in picks[:n]]


STRATEGIES = {
    cls.name: cls
    for cls in (
        PopularityRecommender,
        UserCFRecommender,
        ItemCFRecommender,
        ContentRecommender,
        CategoryPopularRecommender,
        ExplorationRecommender,
        HybridRecommender,
    )
}
