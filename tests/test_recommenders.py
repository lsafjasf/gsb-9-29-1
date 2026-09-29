"""Boundary and behavior tests.

Covers: single user, single item, all-identical ratings, extremely sparse
matrix, brand-new user/item, no train leakage, determinism, and cold-start
strategy differences.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.data import CATEGORIES, Dataset
from recsys.metrics import evaluate
from recsys.recommenders import (
    CategoryPopularRecommender,
    ContentRecommender,
    ExplorationRecommender,
    HybridRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    UserCFRecommender,
)
from recsys.similarities import adjusted_cosine_items, pearson_users


def make_dataset(train, test=None, num_users=None, num_items=None,
                 cold_users=None, cold_items=None):
    user_ids = {u for u, _i, _r in train} | {u for u, _i, _r in (test or [])}
    item_ids = {i for _u, i, _r in train} | {i for _u, i, _r in (test or [])}
    num_users = num_users or (max(user_ids) + 1 if user_ids else 1)
    num_items = num_items or (max(item_ids) + 1 if item_ids else 1)
    item_categories = {}
    item_tags = {}
    for item in range(num_items):
        item_categories[item] = CATEGORIES[item % len(CATEGORIES)]
        item_tags[item] = [f"tag{item % 3}"]
    return Dataset(
        num_users=num_users,
        num_items=num_items,
        categories=list(CATEGORIES),
        item_categories=item_categories,
        item_tags=item_tags,
        train=list(train),
        test=list(test or []),
        cold_users=set(cold_users or []),
        cold_items=set(cold_items or []),
    )


class SimilarityBoundaryTests(unittest.TestCase):
    def test_no_overlap_is_zero(self):
        self.assertEqual(pearson_users({0: 5.0}, {1: 5.0}), 0.0)

    def test_all_identical_ratings_is_zero_variance(self):
        # Perfect agreement with zero variance: correlation is undefined,
        # and we deliberately return 0 instead of NaN.
        sim = pearson_users({0: 3.0, 1: 3.0, 2: 3.0},
                            {0: 3.0, 1: 3.0, 2: 3.0})
        self.assertEqual(sim, 0.0)

    def test_perfect_agreement_positive(self):
        sim = pearson_users({0: 1.0, 1: 5.0, 2: 3.0},
                            {0: 1.0, 1: 5.0, 2: 3.0}, shrink=0.0)
        self.assertGreater(sim, 0.999)

    def test_adjusted_cosine_removes_user_bias(self):
        # Both users rate item 0 one star above their own mean: items 0/1
        # should be positively similar even though raw stars differ.
        # Opposite directional preferences -> negative similarity.
        matrix = {0: {0: 5.0, 1: 1.0}, 1: {0: 4.0, 1: 0.0}}
        means = {0: 3.0, 1: 2.0}
        sim = adjusted_cosine_items(0, 1, matrix, means, shrink=0.0)
        self.assertAlmostEqual(sim, -1.0, places=6)
        # Same directional preference pattern -> positive similarity;
        # global mean-centering removes the easy/hard-rater offset.
        # A third anchor item sets each user's mean so that items 0 and 1
        # deviate in the *same* direction for both users (a hard rater and
        # a generous rater agree on item similarity).
        matrix = {0: {0: 5.0, 1: 4.0, 2: 1.0},
                  1: {0: 4.0, 1: 3.0, 2: 0.5}}
        means = {0: 10.0 / 3.0, 1: 2.5}
        sim = adjusted_cosine_items(0, 1, matrix, means, shrink=0.0)
        self.assertGreater(sim, 0.99)


class ShapeBoundaryTests(unittest.TestCase):
    def test_single_user(self):
        # One user, a few ratings: no other user to collaborate with.
        dataset = make_dataset([(0, 0, 5.0), (0, 1, 4.0), (0, 2, 2.0)],
                               [(0, 3, 5.0)], num_users=1, num_items=5)
        # Pure CF has no second user / no item co-rating signal.
        for cls in (UserCFRecommender, ItemCFRecommender):
            self.assertEqual(cls(dataset).recommend(0, 3), [])
        # Hybrid degrades to content + popularity and still ranks items.
        recs = HybridRecommender(dataset).recommend(0, 3)
        self.assertEqual(len(recs), 2)
        self.assertNotIn(0, [i for i, _ in recs])
        pop = PopularityRecommender(dataset).recommend(0, 3)
        self.assertTrue(all(score >= 0 for _i, score in pop))

    def test_single_item(self):
        # One item in the catalog, rated by several users; user 0 saw it.
        dataset = make_dataset(
            [(0, 0, 5.0), (1, 0, 4.0), (2, 0, 2.0)],
            test=[(1, 0, 5.0)], num_users=3, num_items=1,
        )
        for cls in (UserCFRecommender, ItemCFRecommender):
            self.assertEqual(cls(dataset).recommend(0, 5), [])
            self.assertEqual(cls(dataset).recommend(1, 5), [])
        # Content/hybrid have nothing to offer user 0 on a 1-item catalog.
        for cls in (ContentRecommender, HybridRecommender):
            self.assertEqual(cls(dataset).recommend(0, 5), [])

    def test_all_identical_ratings(self):
        # Every rating equals 4: correlation is undefined everywhere.
        train = [(u, i, 4.0) for u in range(5) for i in range(4)]
        dataset = make_dataset(train, test=[(0, 5, 4.0)],
                               num_users=5, num_items=6)
        user_cf = UserCFRecommender(dataset)
        self.assertEqual(user_cf.predict(0, 4), None)
        item_cf = ItemCFRecommender(dataset)
        self.assertEqual(item_cf.predict(0, 4), None)
        # Fallbacks still return rankings without errors.
        self.assertEqual(len(PopularityRecommender(dataset).recommend(0, 5)), 2)
        self.assertEqual(len(HybridRecommender(dataset).recommend(0, 5)), 2)

    def test_extremely_sparse_matrix(self):
        # 100 users x 100 items, only 120 ratings, mostly disjoint.
        import random
        rng = random.Random(1)
        train = []
        used = set()
        while len(train) < 120:
            u = rng.randrange(100)
            i = rng.randrange(100)
            if (u, i) not in used:
                used.add((u, i))
                train.append((u, i, float(rng.randint(1, 5))))
        dataset = make_dataset(train, test=[(50, 90, 5.0)],
                               num_users=100, num_items=100)
        for cls in (UserCFRecommender, ItemCFRecommender,
                    PopularityRecommender, ContentRecommender,
                    CategoryPopularRecommender, ExplorationRecommender,
                    HybridRecommender):
            recs = cls(dataset).recommend(50, 10)
            self.assertLessEqual(len(recs), 10)
            self.assertEqual(len(recs), len(set(i for i, _ in recs)))


class ColdStartTests(unittest.TestCase):
    def setUp(self):
        # Users 0/1 have history on items 0..3; item 4 and user 2 are cold.
        train = [
            (0, 0, 5.0), (0, 1, 4.0), (0, 2, 1.0),
            (1, 0, 4.0), (1, 1, 5.0), (1, 3, 2.0),
        ]
        self.dataset = make_dataset(
            train,
            test=[(2, 0, 5.0), (0, 4, 5.0), (2, 4, 5.0)],
            num_users=3, num_items=5, cold_users=[2], cold_items=[4],
        )

    def test_cold_user_cf_has_no_prediction(self):
        self.assertIsNone(UserCFRecommender(self.dataset).predict(2, 1))
        self.assertIsNone(ItemCFRecommender(self.dataset).predict(2, 1))

    def test_cold_item_cf_has_no_prediction(self):
        # Item 4 was never rated in training -> no item neighbors.
        self.assertIsNone(ItemCFRecommender(self.dataset).predict(0, 4))

    def test_content_still_ranks_for_cold_item(self):
        # Item 4 shares a category/tag with items user 0 loved.
        self.dataset.item_categories[4] = self.dataset.item_categories[0]
        self.dataset.item_tags[4] = list(self.dataset.item_tags[0])
        content = ContentRecommender(self.dataset)
        self.assertGreater(content.score(0, 4), content.score(0, 2))

    def test_hybrid_does_not_crash_on_cold_entities(self):
        hybrid = HybridRecommender(self.dataset)
        recs_new_user = hybrid.recommend(2, 5)
        recs_new_item = hybrid.recommend(0, 5)
        self.assertTrue(recs_new_user)
        self.assertIn(4, [i for i, _ in recs_new_item])

    def test_exploration_exposes_zero_rating_items(self):
        explore = ExplorationRecommender(self.dataset, explore_ratio=1.0)
        import random
        picks = [i for i, _ in explore.recommend(0, 4, rng=random.Random(0))]
        self.assertIn(4, picks)  # item 4 has zero training ratings

    def test_exploration_is_reproducible_per_user(self):
        explore = ExplorationRecommender(self.dataset)
        import random
        first = explore.recommend(0, 5, rng=random.Random(123))
        second = explore.recommend(0, 5, rng=random.Random(123))
        self.assertEqual(first, second)


class OutputContractTests(unittest.TestCase):
    def test_no_train_leakage_and_sorted(self):
        dataset = make_dataset(
            [(u, i, float((u + i) % 5 + 1)) for u in range(4) for i in range(3)],
            test=[(0, 3, 5.0)], num_users=4, num_items=5,
        )
        for cls in (PopularityRecommender, UserCFRecommender, ItemCFRecommender,
                    ContentRecommender, CategoryPopularRecommender,
                    HybridRecommender):
            recs = cls(dataset).recommend(0, 10)
            items = [i for i, _ in recs]
            self.assertEqual(items, [i for i in items if i not in dataset.train_matrix()[0]])
            scores = [s for _i, s in recs]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_evaluate_metric_ranges(self):
        dataset = make_dataset(
            [(u, i, 5.0 if (u + i) % 2 else 1.0)
             for u in range(6) for i in range(4)],
            test=[(u, 4, 5.0) for u in range(6)],
            num_users=6, num_items=5, cold_items=[4],
        )
        metrics = evaluate(ItemCFRecommender(dataset), dataset, top_n=3)
        for key in ("precision@3", "recall@3", "ndcg@3", "catalog_coverage",
                    "category_coverage", "diversity"):
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)
        self.assertGreaterEqual(metrics["cold_item_exposure"], 0.0)

    def test_empty_interactions_does_not_raise(self):
        dataset = make_dataset([], test=[], num_users=2, num_items=2)
        for cls in (UserCFRecommender, ItemCFRecommender):
            self.assertEqual(cls(dataset).recommend(0, 5), [])
        # History-free strategies degrade to zero-score rankings, not crashes.
        for cls in (PopularityRecommender, ContentRecommender,
                    CategoryPopularRecommender, HybridRecommender):
            recs = cls(dataset).recommend(0, 5)
            self.assertEqual({i for i, _ in recs}, {0, 1})
        self.assertEqual(evaluate(PopularityRecommender(dataset), dataset)
                         ["users_evaluated"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
