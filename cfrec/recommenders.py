"""Recommenders.

Two collaborative-filtering paths (user-based / item-based) plus three
cold-start strategies (content features, category popularity, exploration
slots) and a hybrid that blends them. Standard library only.

All recommenders share the interface:
    fit(ratings, catalog)
    predict(user_id, item_id) -> float
    recommend(user_id, k, exclude=()) -> [(item_id, score), ...]
Unknown users/items are handled explicitly in each class instead of raising.
"""

from collections import defaultdict
import math
import random
import zlib

from .similarity import (
    adjusted_cosine_item_similarity,
    binary_cosine,
    minmax_normalize,
    pearson_user_similarity,
)


class Recommender(object):
    name = "base"

    def fit(self, ratings, catalog):
        self.ratings = ratings
        self.catalog = catalog
        self.global_mean = ratings.global_mean()
        self.lo = ratings.min_rating if ratings.min_rating != float("inf") else 1.0
        self.hi = ratings.max_rating if ratings.max_rating != float("-inf") else 5.0
        return self

    def _clip(self, value):
        return max(self.lo, min(self.hi, value))

    def predict(self, user_id, item_id):
        return self.global_mean

    def score_catalog(self, user_id, exclude=frozenset()):
        """Return {item_id: score} for every non-excluded catalog item."""
        return {item_id: self.predict(user_id, item_id)
                for item_id in self.catalog.ids() if item_id not in exclude}

    def recommend(self, user_id, k=10, exclude=frozenset()):
        # Never recommend items the user has already interacted with.
        exclude = set(exclude) | set(self.ratings.user_ratings(user_id))
        scores = self.score_catalog(user_id, exclude)
        return self._topk(scores, k)

    @staticmethod
    def _topk(scores, k):
        # Deterministic tie-break: higher score first, then smaller item id.
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:k]


class PopularityRecommender(Recommender):
    """Baseline: rank items by (smoothed) popularity / mean rating."""

    name = "popularity"

    def fit(self, ratings, catalog):
        super(PopularityRecommender, self).fit(ratings, catalog)
        self.popularity = ratings.item_popularity()
        self.item_mean = {}
        for item_id in catalog.ids():
            self.item_mean[item_id] = ratings.item_mean(item_id)
        return self

    def predict(self, user_id, item_id):
        return self.item_mean.get(item_id, self.global_mean)

    def score_catalog(self, user_id, exclude=frozenset()):
        return {item_id: float(self.popularity.get(item_id, 0))
                for item_id in self.catalog.ids() if item_id not in exclude}


class UserCFRecommender(Recommender):
    """User-based CF: Pearson correlation over co-rated items.

    Prediction: global-mean-centered weighted average of neighbor ratings.
    Neighbors with non-positive similarity are ignored; if no neighbor rated
    the target item, fall back to the user's mean (then global mean).
    """

    name = "user_cf"

    def __init__(self, n_neighbors=40, min_common=2, reg=10.0,
                 case_amplification=2.5):
        self.n_neighbors = n_neighbors
        self.min_common = min_common
        self.reg = reg
        self.case_amplification = case_amplification

    def fit(self, ratings, catalog):
        super(UserCFRecommender, self).fit(ratings, catalog)
        self.user_mean = {u: ratings.user_mean(u) for u in ratings.by_user}
        # Inverse user frequency: log(N / n_item), so co-ratings on rare
        # items contribute more than agreement on blockbusters.
        n_users = max(ratings.n_users, 1)
        popularity = ratings.item_popularity()
        self.item_weight = {
            i: math.log(n_users / float(n)) for i, n in popularity.items()}

        def weight_of(item_id):
            return self.item_weight.get(item_id, 1.0)

        users = sorted(ratings.by_user)
        self.sim = {u: {} for u in users}
        for idx, u in enumerate(users):
            ru = ratings.by_user[u]
            for v in users[idx + 1:]:
                s = pearson_user_similarity(
                    ru, ratings.by_user[v],
                    self.user_mean[u], self.user_mean[v],
                    min_common=self.min_common, reg=self.reg,
                    item_weight=weight_of)
                if s > 0.0:
                    self.sim[u][v] = s
                    self.sim[v][u] = s
        return self

    def _amplify(self, sim):
        # Case amplification (Breese et al. 1998): sharpen neighbor weights.
        rho = self.case_amplification
        return sim ** rho if sim >= 0.0 else -((-sim) ** rho)

    def predict(self, user_id, item_id):
        if user_id not in self.ratings.by_user:
            return self.global_mean
        neighbors = self.sim.get(user_id, {})
        num = 0.0
        den = 0.0
        for other, sim in neighbors.items():
            rating = self.ratings.by_user[other].get(item_id)
            if rating is None:
                continue
            w = self._amplify(sim)
            num += w * (rating - self.user_mean[other])
            den += abs(w)
        if den == 0.0:
            return self.user_mean[user_id]
        return self._clip(self.user_mean[user_id] + num / den)

    def score_catalog(self, user_id, exclude=frozenset()):
        if user_id not in self.ratings.by_user:
            return {}
        neighbors = sorted(self.sim.get(user_id, {}).items(),
                           key=lambda kv: -kv[1])[:self.n_neighbors]
        scores = defaultdict(float)
        weights = defaultdict(float)
        for other, sim in neighbors:
            w = self._amplify(sim)
            for item_id, rating in self.ratings.by_user[other].items():
                if item_id in exclude:
                    continue
                scores[item_id] += w * (rating - self.user_mean[other])
                weights[item_id] += abs(w)
        # Only items with at least one neighbor rating get a score; items
        # without evidence (e.g. brand-new items) are simply not recommendable
        # by pure CF and are left to the cold-start strategies.
        return {item_id: self.user_mean[user_id] + scores[item_id] / weights[item_id]
                for item_id in scores if weights[item_id] > 0.0}


class ItemCFRecommender(Recommender):
    """Item-based CF: adjusted-cosine similarity between items.

    Centering uses each *user's* mean rating, which removes user scale
    differences; missing entries are skipped (co-rated users only).
    """

    name = "item_cf"

    def __init__(self, n_neighbors=40, min_common=3, reg=6.0):
        self.n_neighbors = n_neighbors
        self.min_common = min_common
        self.reg = reg

    def fit(self, ratings, catalog):
        super(ItemCFRecommender, self).fit(ratings, catalog)
        user_means = {u: ratings.user_mean(u) for u in ratings.by_user}

        def mean_of(user_id):
            return user_means[user_id]

        items = sorted(ratings.by_item)
        self.sim = {i: {} for i in items}
        for idx, i in enumerate(items):
            ri = ratings.by_item[i]
            for j in items[idx + 1:]:
                s = adjusted_cosine_item_similarity(
                    ri, ratings.by_item[j], mean_of,
                    min_common=self.min_common, reg=self.reg)
                if s > 0.0:
                    self.sim[i][j] = s
                    self.sim[j][i] = s
        return self

    def predict(self, user_id, item_id):
        history = self.ratings.user_ratings(user_id)
        if not history:
            return self.global_mean
        user_mean = self.ratings.user_mean(user_id)
        neighbors = self.sim.get(item_id, {})
        num = 0.0
        den = 0.0
        for other, rating in history.items():
            sim = neighbors.get(other)
            if sim:
                num += sim * (rating - user_mean)
                den += abs(sim)
        if den == 0.0:
            return self.ratings.item_mean(item_id)
        return self._clip(user_mean + num / den)

    def score_catalog(self, user_id, exclude=frozenset()):
        history = self.ratings.user_ratings(user_id)
        if not history:
            return {}
        user_mean = self.ratings.user_mean(user_id)
        out = {}
        for item_id in self.catalog.ids():
            if item_id in exclude:
                continue
            num = 0.0
            den = 0.0
            neighbors = self.sim.get(item_id, {})
            for other, rating in history.items():
                sim = neighbors.get(other)
                if sim:
                    num += sim * (rating - user_mean)
                    den += abs(sim)
            if den > 0.0:
                out[item_id] = user_mean + num / den
        return out


class ContentRecommender(Recommender):
    """Content-based cold start: match item feature tags to a user profile.

    The user profile accumulates the features of positively-rated items
    (rating >= like_threshold); item-item content similarity uses binary
    cosine over feature sets. Works for users with zero interactions only
    through the popularity prior (score 0 -> falls back in the hybrid).
    """

    name = "content"

    def __init__(self, like_threshold=4.0):
        self.like_threshold = like_threshold

    def fit(self, ratings, catalog):
        super(ContentRecommender, self).fit(ratings, catalog)
        self.item_sim = {}
        items = catalog.ids()
        for idx, i in enumerate(items):
            fi = catalog.get(i).features
            for j in items[idx + 1:]:
                s = binary_cosine(fi, catalog.get(j).features)
                if s > 0.0:
                    self.item_sim.setdefault(i, {})[j] = s
                    self.item_sim.setdefault(j, {})[i] = s
        return self

    def profile(self, user_id):
        profile = defaultdict(float)
        for item_id, rating in self.ratings.user_ratings(user_id).items():
            if rating < self.like_threshold:
                continue
            item = self.catalog.get(item_id)
            if item is None:
                continue
            for feature in item.features:
                profile[feature] += 1.0
        return dict(profile)

    def score_catalog(self, user_id, exclude=frozenset()):
        profile = self.profile(user_id)
        if not profile:
            return {}
        norm = sum(profile.values()) or 1.0
        out = {}
        for item_id in self.catalog.ids():
            if item_id in exclude:
                continue
            item = self.catalog.get(item_id)
            score = sum(profile[f] for f in sorted(item.features) if f in profile)
            if score > 0.0:
                out[item_id] = score / norm
        return out

    def predict(self, user_id, item_id):
        # Content similarity is a ranking signal, not a calibrated rating;
        # for rating prediction fall back to the user's own average.
        return self.ratings.user_mean(user_id)

    def similar_items(self, item_id, k=10):
        sims = self.item_sim.get(item_id, {})
        return self._topk(sims, k)


class CategoryPopularityRecommender(Recommender):
    """Category-popularity cold start.

    For users with some history: rank categories by the user's own ratings,
    then recommend the most popular unseen items from those categories.
    For users with zero history: round-robin across categories of the global
    popularity list, so the fallback is not a single flat popular list.
    """

    name = "category_pop"

    def fit(self, ratings, catalog):
        super(CategoryPopularityRecommender, self).fit(ratings, catalog)
        popularity = ratings.item_popularity()
        self.by_category = defaultdict(list)
        for item_id in catalog.ids():
            item = catalog.get(item_id)
            self.by_category[item.category].append(item_id)
        for category in self.by_category:
            self.by_category[category].sort(
                key=lambda i: (-popularity.get(i, 0), i))
        self.global_popular = sorted(
            catalog.ids(), key=lambda i: (-popularity.get(i, 0), i))
        # Round-robin interleave of per-category lists -> diversified fallback.
        self.mixed_popular = []
        categories = sorted(self.by_category)
        idx = 0
        while len(self.mixed_popular) < len(self.global_popular):
            progressed = False
            for category in categories:
                lst = self.by_category[category]
                if idx < len(lst):
                    self.mixed_popular.append(lst[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1
        return self

    def _category_affinity(self, user_id):
        affinity = defaultdict(float)
        counts = defaultdict(int)
        for item_id, rating in self.ratings.user_ratings(user_id).items():
            item = self.catalog.get(item_id)
            if item is None:
                continue
            affinity[item.category] += rating
            counts[item.category] += 1
        return {c: affinity[c] / counts[c] for c in affinity}

    def recommend(self, user_id, k=10, exclude=frozenset()):
        exclude = set(exclude) | set(self.ratings.user_ratings(user_id))
        affinity = self._category_affinity(user_id)
        if not affinity:
            ranked = self.mixed_popular
        else:
            ranked = []
            categories = sorted(affinity, key=lambda c: (-affinity[c], c))
            idx = 0
            seen = set()
            while len(ranked) < len(self.global_popular):
                progressed = False
                for category in categories:
                    lst = self.by_category[category]
                    if idx < len(lst) and lst[idx] not in seen:
                        ranked.append(lst[idx])
                        seen.add(lst[idx])
                        progressed = True
                if not progressed:
                    break
                idx += 1
        out = []
        for item_id in ranked:
            if item_id in exclude:
                continue
            out.append((item_id, float(len(ranked) - len(out))))
            if len(out) >= k:
                break
        return out

    def predict(self, user_id, item_id):
        item = self.catalog.get(item_id)
        if item is None:
            return self.global_mean
        affinity = self._category_affinity(user_id)
        if item.category in affinity:
            return self._clip(affinity[item.category])
        return self.ratings.item_mean(item_id)


class HybridColdStartRecommender(Recommender):
    """Blend CF + content + category popularity, with exploration slots.

    * Warm users (>= warm_threshold ratings): CF (item/user average) dominates.
    * Semi-cold users (1..warm_threshold-1 ratings): content + category lead.
    * Cold users (0 ratings): category-popular mix plus exploration slots
      reserved for long-tail items (few interactions), so brand-new items get
      exposure instead of a flat popular list.
    """

    name = "hybrid"

    def __init__(self, user_cf, item_cf, content, category_pop,
                 warm_threshold=5, exploration_slots=2, seed=0):
        self.user_cf = user_cf
        self.item_cf = item_cf
        self.content = content
        self.category_pop = category_pop
        self.warm_threshold = warm_threshold
        self.exploration_slots = exploration_slots
        self.seed = seed

    def fit(self, ratings, catalog):
        super(HybridColdStartRecommender, self).fit(ratings, catalog)
        self.popularity = ratings.item_popularity()
        return self

    def _history_size(self, user_id):
        return len(self.ratings.user_ratings(user_id))

    def _weighted_sum(self, weights, user_id, exclude):
        blended = defaultdict(float)
        for model, weight in weights:
            if isinstance(model, CategoryPopularityRecommender):
                recs = model.recommend(user_id, k=len(self.catalog),
                                       exclude=exclude)
                scores = {item_id: rank for item_id, rank in recs}
            else:
                scores = model.score_catalog(user_id, exclude)
            for item_id, value in minmax_normalize(scores).items():
                blended[item_id] += weight * value
        return blended

    def _blend(self, user_id, exclude):
        """Route by history size.

        * warm: weighted sum of all four signals (min-max normalized);
        * semi-cold: cascade -- items with direct item-CF evidence always
          outrank blended fallbacks, because with 1-3 ratings the item-CF
          ranking is the only high-precision signal;
        * cold: category popularity only (exploration slots added later).
        """
        n = self._history_size(user_id)
        if n >= self.warm_threshold:
            return self._weighted_sum(
                ((self.user_cf, 0.3), (self.item_cf, 0.3),
                 (self.content, 0.2), (self.category_pop, 0.2)),
                user_id, exclude)
        if n > 0:
            primary = minmax_normalize(
                self.item_cf.score_catalog(user_id, exclude))
            rest = self._weighted_sum(
                ((self.user_cf, 0.05), (self.item_cf, 0.6),
                 (self.content, 0.15), (self.category_pop, 0.2)),
                user_id, exclude)
            blended = {item_id: 1.0 + score
                       for item_id, score in primary.items()}
            for item_id, score in rest.items():
                if item_id not in blended:
                    blended[item_id] = 0.5 * score
            return blended
        return self._weighted_sum(((self.category_pop, 1.0),),
                                  user_id, exclude)

    def recommend(self, user_id, k=10, exclude=frozenset()):
        exclude = set(exclude) | set(self.ratings.user_ratings(user_id))
        blended = self._blend(user_id, exclude)
        recs = self._topk(blended, k)
        if self._history_size(user_id) == 0:
            n_explore = min(self.exploration_slots, k)
        else:
            n_explore = min(max(self.exploration_slots - 1, 1), k)
        # Backfill with category-popular items so the list is always full.
        if len(recs) < k - n_explore:
            have = {item_id for item_id, _ in recs}
            for item_id, _ in self.category_pop.recommend(
                    user_id, k=len(self.catalog), exclude=exclude | have):
                recs.append((item_id, 0.0))
                if len(recs) >= k - n_explore:
                    break
        if n_explore <= 0:
            return recs[:k]
        # Exploration: replace the tail slots with weighted-random long-tail
        # items (weight 1/(1+interactions)), deterministic via seeded RNG.
        rng = random.Random(zlib.crc32(('%s|%s' % (self.seed, user_id)).encode('utf-8')))
        picked = {item_id for item_id, _ in recs}
        candidates = [i for i in self.catalog.ids()
                      if i not in exclude and i not in picked]
        # Steer exploration toward the user's preferred categories when we
        # know them; cold users explore the whole long tail.
        affinity = self.category_pop._category_affinity(user_id)
        if affinity:
            top_cats = {c for c, _ in sorted(
                affinity.items(), key=lambda kv: -kv[1])[:2]}
            preferred = [i for i in candidates
                         if self.catalog.get(i).category in top_cats]
            if preferred:
                candidates = preferred
        if not candidates:
            return recs
        weights = [1.0 / (1.0 + self.popularity.get(i, 0)) for i in candidates]
        chosen = []
        pool = list(zip(candidates, weights))
        for _ in range(min(n_explore, len(pool))):
            total = sum(w for _, w in pool)
            r = rng.random() * total
            acc = 0.0
            for idx, (item_id, w) in enumerate(pool):
                acc += w
                if r <= acc:
                    chosen.append(item_id)
                    pool.pop(idx)
                    break
        keep = recs[:k - len(chosen)]
        return keep + [(item_id, 0.0) for item_id in chosen]

    def predict(self, user_id, item_id):
        n = self._history_size(user_id)
        if n >= self.warm_threshold:
            return self._clip(0.5 * self.user_cf.predict(user_id, item_id)
                              + 0.5 * self.item_cf.predict(user_id, item_id))
        if n > 0:
            return self._clip(0.5 * self.content.predict(user_id, item_id)
                              + 0.5 * self.category_pop.predict(user_id, item_id))
        return self.category_pop.predict(user_id, item_id)
