"""Similarity functions and the missing-value policy.

Missing-value policy (documented here because every metric relies on it):

* Ratings live in a sparse dict-of-dicts; a missing rating is an absent key.
* Similarity between two users/items is computed ONLY on their co-rated
  entries (the intersection of observed keys). Missing entries are never
  imputed with zeros or means -- imputation would fabricate agreement in an
  extremely sparse matrix.
* If the number of co-rated entries is below ``min_common`` the similarity
  is 0.0 (no evidence).
* Significance weighting shrinks similarities supported by few common
  entries: ``sim *= n / (n + reg)``.
* If either side has zero variance on the common entries (e.g. all ratings
  identical), the centered similarity is undefined and returns 0.0.
"""

import math


def centered_cosine(vec_a, vec_b, center_a, center_b, min_common=1, reg=0.0,
                    key_weight=None):
    """Cosine similarity on the intersection of keys, after centering.

    ``center_a``/``center_b`` are callables mapping a key to the center value
    used for that key (e.g. the user's mean for user-based Pearson, or the
    item's mean for item-based adjusted cosine). ``key_weight`` optionally
    maps a key to an importance weight (e.g. inverse user frequency).
    """
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
        center_a, center_b = center_b, center_a
    common = [key for key in vec_a if key in vec_b]
    n = len(common)
    if n < min_common:
        return 0.0
    num = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for key in common:
        w = key_weight(key) if key_weight is not None else 1.0
        da = vec_a[key] - center_a(key)
        db = vec_b[key] - center_b(key)
        num += w * da * db
        sum_a += w * da * da
        sum_b += w * db * db
    if sum_a <= 0.0 or sum_b <= 0.0:
        return 0.0
    sim = num / math.sqrt(sum_a * sum_b)
    if reg > 0.0:
        sim *= n / (n + reg)
    return sim


def pearson_user_similarity(ratings_a, ratings_b, user_mean_a, user_mean_b,
                            min_common=2, reg=10.0, item_weight=None):
    """Pearson correlation between two users over co-rated items.

    ``item_weight`` (Breese et al. 1998 inverse user frequency) down-weights
    universally popular items so agreement on rare items counts more.
    """
    return centered_cosine(
        ratings_a, ratings_b,
        lambda _key: user_mean_a, lambda _key: user_mean_b,
        min_common=min_common, reg=reg, key_weight=item_weight,
    )


def adjusted_cosine_item_similarity(ratings_i, ratings_j, user_means,
                                    min_common=2, reg=10.0):
    """Adjusted cosine between two items: center each rating by the *user's* mean."""
    return centered_cosine(
        ratings_i, ratings_j,
        lambda user_id: user_means(user_id), lambda user_id: user_means(user_id),
        min_common=min_common, reg=reg,
    )


def binary_cosine(features_a, features_b):
    """Cosine similarity between two sets of binary features."""
    if not features_a or not features_b:
        return 0.0
    inter = len(features_a & features_b)
    if inter == 0:
        return 0.0
    return inter / math.sqrt(len(features_a) * len(features_b))


def jaccard(features_a, features_b):
    if not features_a and not features_b:
        return 0.0
    union = features_a | features_b
    if not union:
        return 0.0
    return len(features_a & features_b) / float(len(union))


def minmax_normalize(scores):
    """Scale a {key: score} dict to [0, 1]; constant input maps to 0.5."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo < 1e-12:
        return {key: 0.5 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}
