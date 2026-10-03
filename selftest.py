#!/usr/bin/env python3
"""Self-tests (standard-library unittest).

Covers the required edge cases:
    * single user
    * single item
    * all ratings identical (zero variance -> similarity undefined)
    * extremely sparse matrix
    * unknown user / unknown item
plus similarity/metric unit checks and determinism.

Run:  python3 selftest.py   (or: python3 -m unittest selftest -v)
"""

import unittest

from cfrec.data import Item, ItemCatalog, Ratings
from cfrec.metrics import (
    aggregate_ranking,
    catalog_coverage,
    category_entropy,
    intra_list_diversity,
    novelty,
    rmse,
)
from cfrec.recommenders import (
    CategoryPopularityRecommender,
    ContentRecommender,
    HybridColdStartRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    UserCFRecommender,
)
from cfrec.similarity import (
    adjusted_cosine_item_similarity,
    binary_cosine,
    jaccard,
    minmax_normalize,
    pearson_user_similarity,
)


def make_catalog(n_items=6, n_categories=2):
    catalog = ItemCatalog()
    for i in range(n_items):
        catalog.add(Item("i%d" % i, "c%d" % (i % n_categories),
                         features={"f%d" % i, "shared"}))
    return catalog


def make_models(train, catalog):
    user_cf = UserCFRecommender().fit(train, catalog)
    item_cf = ItemCFRecommender().fit(train, catalog)
    content = ContentRecommender().fit(train, catalog)
    cat_pop = CategoryPopularityRecommender().fit(train, catalog)
    popularity = PopularityRecommender().fit(train, catalog)
    hybrid = HybridColdStartRecommender(
        user_cf, item_cf, content, cat_pop).fit(train, catalog)
    return [popularity, user_cf, item_cf, content, cat_pop, hybrid]


class TestSimilarity(unittest.TestCase):
    def test_pearson_identical_users(self):
        a = {"i1": 5.0, "i2": 1.0, "i3": 3.0}
        b = {"i1": 4.0, "i2": 2.0, "i3": 3.0}
        sim = pearson_user_similarity(a, b, 3.0, 3.0, min_common=2, reg=0.0)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_pearson_missing_values_use_intersection_only(self):
        # Only i1/i2 are co-rated; i3/i4 must be ignored entirely.
        a = {"i1": 5.0, "i2": 1.0, "i3": 5.0}
        b = {"i1": 4.0, "i2": 2.0, "i4": 1.0}
        sim = pearson_user_similarity(a, b, 3.0, 3.0, min_common=2, reg=0.0)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_pearson_insufficient_overlap_is_zero(self):
        a = {"i1": 5.0}
        b = {"i1": 4.0}
        self.assertEqual(
            pearson_user_similarity(a, b, 5.0, 4.0, min_common=2), 0.0)

    def test_pearson_zero_variance_is_zero(self):
        # All ratings identical -> centered vectors are all zero -> undefined.
        a = {"i1": 3.0, "i2": 3.0, "i3": 3.0}
        b = {"i1": 5.0, "i2": 1.0, "i3": 3.0}
        self.assertEqual(
            pearson_user_similarity(a, b, 3.0, 3.0, min_common=2), 0.0)

    def test_significance_weighting_shrinks(self):
        a = {"i1": 5.0, "i2": 1.0}
        b = {"i1": 4.0, "i2": 2.0}
        raw = pearson_user_similarity(a, b, 3.0, 3.0, min_common=2, reg=0.0)
        shrunk = pearson_user_similarity(a, b, 3.0, 3.0, min_common=2, reg=10.0)
        self.assertAlmostEqual(raw, 1.0, places=6)
        self.assertAlmostEqual(shrunk, 2.0 / 12.0, places=6)

    def test_adjusted_cosine_centers_by_user(self):
        # Two items rated identically relative to each user's mean -> +1.
        ri = {"u1": 5.0, "u2": 1.0}
        rj = {"u1": 5.0, "u2": 1.0}
        means = {"u1": 4.0, "u2": 2.0}
        sim = adjusted_cosine_item_similarity(
            ri, rj, lambda u: means[u], min_common=2, reg=0.0)
        self.assertAlmostEqual(sim, 1.0, places=6)

    def test_binary_cosine_and_jaccard(self):
        self.assertAlmostEqual(binary_cosine({"a", "b"}, {"b", "c"}),
                               0.5, places=6)
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}),
                               1.0 / 3.0, places=6)
        self.assertEqual(binary_cosine(set(), {"a"}), 0.0)

    def test_minmax_constant_maps_to_half(self):
        self.assertEqual(minmax_normalize({"a": 2.0, "b": 2.0}),
                         {"a": 0.5, "b": 0.5})


class TestEdgeCases(unittest.TestCase):
    def test_single_user(self):
        catalog = make_catalog()
        train = Ratings()
        train.add("u1", "i0", 5.0)
        train.add("u1", "i1", 4.0)
        for model in make_models(train, catalog):
            recs = model.recommend("u1", k=3)
            self.assertTrue(len(recs) <= 3)
            self.assertTrue(all(i in catalog for i, _ in recs))
            pred = model.predict("u1", "i2")
            self.assertTrue(1.0 <= pred <= 5.0)

    def test_single_item(self):
        catalog = make_catalog(n_items=1, n_categories=1)
        train = Ratings()
        train.add("u1", "i0", 4.0)
        train.add("u2", "i0", 5.0)
        for model in make_models(train, catalog):
            recs = model.recommend("u1", k=5, exclude={"i0"})
            self.assertEqual(recs, [])  # nothing left to recommend
            recs = model.recommend("u3", k=5)
            self.assertTrue(len(recs) <= 1)

    def test_all_ratings_identical(self):
        catalog = make_catalog()
        train = Ratings()
        for u in ("u1", "u2", "u3"):
            for i in ("i0", "i1", "i2"):
                train.add(u, i, 3.0)
        user_cf = UserCFRecommender().fit(train, catalog)
        item_cf = ItemCFRecommender().fit(train, catalog)
        # Zero variance everywhere -> no usable similarity, no crash.
        self.assertEqual(user_cf.sim, {"u1": {}, "u2": {}, "u3": {}})
        self.assertEqual(item_cf.sim, {"i0": {}, "i1": {}, "i2": {}})
        self.assertAlmostEqual(user_cf.predict("u1", "i0"), 3.0)
        self.assertAlmostEqual(item_cf.predict("u1", "i0"), 3.0)
        for model in make_models(train, catalog):
            recs = model.recommend("u1", k=4)
            self.assertTrue(len(recs) <= 4)

    def test_extremely_sparse_matrix(self):
        catalog = make_catalog(n_items=300, n_categories=4)
        train = Ratings()
        train.add("u1", "i0", 5.0)
        train.add("u2", "i7", 2.0)
        train.add("u3", "i13", 4.0)
        self.assertLess(len(train) / float(3 * len(catalog)), 0.01)
        for model in make_models(train, catalog):
            recs = model.recommend("u1", k=5)
            self.assertTrue(len(recs) <= 5)
            pred = model.predict("u9", "i9")
            self.assertTrue(1.0 <= pred <= 5.0)

    def test_unknown_user_and_item(self):
        catalog = make_catalog()
        train = Ratings()
        train.add("u1", "i0", 5.0)
        train.add("u2", "i0", 1.0)
        train.add("u2", "i1", 4.0)
        for model in make_models(train, catalog):
            recs = model.recommend("nobody", k=3)
            self.assertTrue(len(recs) <= 3)
            self.assertTrue(all(i in catalog for i, _ in recs))
            pred = model.predict("nobody", "i0")
            self.assertTrue(1.0 <= pred <= 5.0)
            pred = model.predict("u1", "ghost_item")
            self.assertTrue(1.0 <= pred <= 5.0)

    def test_pure_cf_cannot_serve_cold_user_or_item(self):
        catalog = make_catalog()
        train = Ratings()
        train.add("u1", "i0", 5.0)
        train.add("u2", "i0", 1.0)
        user_cf = UserCFRecommender().fit(train, catalog)
        item_cf = ItemCFRecommender().fit(train, catalog)
        self.assertEqual(user_cf.recommend("cold_user", k=3), [])
        self.assertEqual(item_cf.recommend("cold_user", k=3), [])
        # Brand-new item (no ratings) never gets a CF score.
        scores = item_cf.score_catalog("u1")
        self.assertNotIn("i5", scores)

    def test_hybrid_serves_cold_user_with_exploration(self):
        catalog = make_catalog(n_items=12, n_categories=3)
        train = Ratings()
        for u in ("u1", "u2", "u3"):
            train.add(u, "i0", 5.0)
            train.add(u, "i1", 4.0)
        models = make_models(train, catalog)
        hybrid = models[-1]
        recs = hybrid.recommend("brand_new_user", k=5)
        self.assertEqual(len(recs), 5)
        # Deterministic across repeated calls.
        self.assertEqual(recs, hybrid.recommend("brand_new_user", k=5))

    def test_recommend_excludes_seen_and_is_deterministic(self):
        catalog = make_catalog(n_items=10, n_categories=2)
        train = Ratings()
        for u in ("u1", "u2", "u3", "u4"):
            for i in ("i0", "i1", "i2", "i3"):
                train.add(u, i, 2.0 + (hash(u + i) % 3))
        for model in make_models(train, catalog):
            seen = set(train.user_ratings("u1"))
            recs = model.recommend("u1", k=4)
            self.assertFalse(seen & {i for i, _ in recs})
            self.assertEqual(recs, model.recommend("u1", k=4))


class TestMetrics(unittest.TestCase):
    def test_rmse(self):
        self.assertAlmostEqual(rmse([(4.0, 5.0), (2.0, 1.0)]),
                               (2.0 / 2.0) ** 0.5, places=6)
        self.assertEqual(rmse([]), 0.0)

    def test_aggregate_ranking_and_user_coverage(self):
        test = {"u1": {"i1": 5.0}, "u2": {"i2": 5.0}, "u3": {"i3": 1.0}}
        recs = {"u1": [("i1", 1.0)], "u2": [("i9", 1.0)], "u3": []}
        out = aggregate_ranking(recs, test, k=1, pos_threshold=4.0)
        self.assertAlmostEqual(out["precision@1"], 0.5, places=6)
        self.assertAlmostEqual(out["user_coverage"], 2.0 / 3.0, places=6)

    def test_coverage_diversity_novelty(self):
        catalog = make_catalog(n_items=4, n_categories=2)
        recs = {"u1": [("i0", 1.0), ("i1", 0.9)],
                "u2": [("i2", 1.0), ("i3", 0.9)]}
        self.assertEqual(catalog_coverage(recs, len(catalog)), 1.0)
        sim = {"i0": {"i1": 1.0}, "i1": {"i0": 1.0}}
        self.assertAlmostEqual(
            intra_list_diversity({"u1": [("i0", 1), ("i1", 1)]}, sim), 0.0)
        self.assertGreater(category_entropy(recs, catalog), 0.0)
        pop = {"i0": 100, "i1": 100, "i2": 1, "i3": 1}
        self.assertGreater(novelty(recs, pop), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
