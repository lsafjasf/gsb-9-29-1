"""Unit tests for ranking_metrics: hand-computed values, edge cases,
cross-cutoff self-consistency, macro/micro attribution, bootstrap."""

import math
import random
import unittest
import warnings

import ranking_metrics as rm


class HandComputedTest(unittest.TestCase):
    def setUp(self):
        # Tie between A and B at 0.9 -> A first (doc id asc).
        self.scored = [("A", 0.9), ("B", 0.9), ("C", 0.8)]
        self.rel = {"A": 2, "C": 1, "D": 3}  # D relevant but never ranked
        self.qs = rm.QuerySeries(self.scored, self.rel)

    def test_rank_order_breaks_ties_by_doc_id(self):
        self.assertEqual([d for d, _ in rm.rank_docs(self.scored)],
                         ["A", "B", "C"])

    def test_dcg(self):
        self.assertAlmostEqual(self.qs.value("dcg", 1), 2.0)
        self.assertAlmostEqual(self.qs.value("dcg", 2), 2.0)
        self.assertAlmostEqual(self.qs.value("dcg", 3), 2.0 + 1 / 2.0)

    def test_ndcg(self):
        ideal = 3.0 + 2.0 / math.log2(3) + 1.0 / 2.0
        self.assertAlmostEqual(self.qs.value("ndcg", 3), 2.5 / ideal)

    def test_ap_denominator_is_total_relevant(self):
        # R = 3 (A, C, D). Hits at rank 1 (P=1) and rank 3 (P=2/3).
        self.assertAlmostEqual(self.qs.value("ap", 1), 1.0 / 3)
        self.assertAlmostEqual(self.qs.value("ap", 2), 1.0 / 3)
        self.assertAlmostEqual(self.qs.value("ap", 3), (1 + 2 / 3) / 3)

    def test_rr_and_hit(self):
        self.assertEqual(self.qs.value("rr", 1), 1.0)
        self.assertEqual(self.qs.value("hit", 1), 1.0)

    def test_exp_gain(self):
        qs = rm.QuerySeries(self.scored, self.rel, gain="exp")
        self.assertAlmostEqual(qs.value("dcg", 1), 2.0 ** 2 - 1.0)


class EdgeCaseTest(unittest.TestCase):
    def test_empty_ranking_with_relevant_labels(self):
        qs = rm.QuerySeries([], {"a": 1, "b": 2})
        for k in (1, 5, 100):
            self.assertEqual(qs.value("dcg", k), 0.0)
            self.assertEqual(qs.value("ndcg", k), 0.0)
            self.assertEqual(qs.value("ap", k), 0.0)
            self.assertEqual(qs.value("rr", k), 0.0)
            self.assertEqual(qs.value("hit", k), 0.0)

    def test_empty_ranking_no_labels_zero_rel_convention(self):
        self.assertEqual(rm.QuerySeries([], {}).value("ndcg", 3), 0.0)
        qs = rm.QuerySeries([], {}, zero_rel_ndcg=1.0)
        self.assertEqual(qs.value("ndcg", 3), 1.0)

    def test_no_relevant_items(self):
        qs = rm.QuerySeries([("a", 1.0), ("b", 0.5)], {"a": 0, "b": 0})
        for k in (1, 2):
            self.assertEqual(qs.value("dcg", k), 0.0)
            self.assertEqual(qs.value("ndcg", k), 0.0)  # zero_rel default
            self.assertEqual(qs.value("ap", k), 0.0)
            self.assertEqual(qs.value("rr", k), 0.0)
            self.assertEqual(qs.value("hit", k), 0.0)

    def test_all_relevant(self):
        scored = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        rel = {"a": 1, "b": 1, "c": 1}
        qs = rm.QuerySeries(scored, rel)
        for k in (1, 2, 3):
            self.assertAlmostEqual(qs.value("ndcg", k), 1.0)
            self.assertEqual(qs.value("hit", k), 1.0)
        self.assertEqual(qs.value("rr", 1), 1.0)
        self.assertAlmostEqual(qs.value("ap", 3), 1.0)

    def test_missing_labels_count_as_zero(self):
        qs = rm.QuerySeries([("x", 1.0), ("a", 0.5)], {"a": 1})
        self.assertEqual(qs.value("dcg", 1), 0.0)  # x unlabeled
        self.assertEqual(qs.value("rr", 1), 0.0)
        self.assertEqual(qs.value("rr", 2), 0.5)
        self.assertEqual(qs.value("hit", 1), 0.0)
        self.assertEqual(qs.value("hit", 2), 1.0)

    def test_single_result(self):
        qs = rm.QuerySeries([("a", 1.0)], {"a": 2})
        self.assertEqual(qs.value("dcg", 1), 2.0)
        self.assertEqual(qs.value("ndcg", 1), 1.0)
        self.assertEqual(qs.value("ap", 1), 1.0)
        self.assertEqual(qs.value("rr", 1), 1.0)
        self.assertEqual(qs.value("hit", 1), 1.0)

    def test_cutoff_beyond_length_clamps(self):
        qs = rm.QuerySeries([("a", 1.0)], {"a": 1})
        for metric in ("dcg", "ap", "rr", "hit"):
            self.assertEqual(qs.value(metric, 1), qs.value(metric, 50))
        # When more ideal labels exist than retrieved documents, nDCG keeps
        # decreasing past the ranking length because IDCG keeps growing.
        qs2 = rm.QuerySeries([("a", 1.0)], {"a": 1, "b": 1, "c": 1})
        self.assertLess(qs2.value("ndcg", 3), qs2.value("ndcg", 1))

    def test_negative_relevance_rejected(self):
        with self.assertRaises(ValueError):
            rm.QuerySeries([("a", 1.0)], {"a": -1})

    def test_invalid_cutoff_rejected(self):
        qs = rm.QuerySeries([("a", 1.0)], {"a": 1})
        with self.assertRaises(ValueError):
            qs.value("dcg", 0)


class MonotonicityTest(unittest.TestCase):
    def test_metrics_non_decreasing_in_k(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(0, 12)
            scored = [(f"d{i}", rng.choice([0, 1, 1, 2])) for i in range(n)]
            rel = {f"d{i}": rng.choice([0, 1, 2])
                   for i in range(n) if rng.random() < 0.8}
            qs = rm.QuerySeries(scored, rel)
            for metric in ("dcg", "ap", "rr", "hit"):
                series = [qs.value(metric, k)
                          for k in range(1, max(n, 1) + 1)]
                for prev, curr in zip(series, series[1:]):
                    self.assertGreaterEqual(
                        curr, prev, f"{metric} decreased: {series}")


class AggregationTest(unittest.TestCase):
    def setUp(self):
        # q1: one relevant doc, retrieved at rank 1 (perfect, R=1).
        # q2: nine relevant docs, none retrieved (R=9).
        self.queries = [
            ("q1", [("a", 1.0)], {"a": 1}),
            ("q2", [("x", 1.0)], {f"r{i}": 1 for i in range(9)}),
        ]
        self.result = rm.evaluate(self.queries, ks=[1])

    def test_macro_micro_diverge_and_attribution(self):
        macro_ap = self.result["macro"]["ap"][1]
        micro_ap = self.result["micro"]["ap"][1]
        self.assertAlmostEqual(macro_ap, 0.5)          # (1 + 0) / 2
        self.assertAlmostEqual(micro_ap, 1.0 / 10)     # (1 + 0) / (1 + 9)
        # Micro weights q2 nine times as heavily as q1 because R=9 vs R=1.
        self.assertLess(micro_ap, macro_ap)

    def test_macro_micro_ndcg_diverge(self):
        result5 = rm.evaluate(self.queries, ks=[5])
        macro = result5["macro"]["ndcg"][5]
        micro = result5["micro"]["ndcg"][5]
        self.assertAlmostEqual(macro, 0.5)  # (1 + 0) / 2
        # Micro pools DCG/IDCG: q2's large IDCG@5 dominates the ratio.
        self.assertLess(micro, macro)

    def test_macro_micro_coincide_for_rr_hit_dcg_mean(self):
        for metric in ("rr", "hit", "dcg"):
            self.assertAlmostEqual(self.result["macro"][metric][1],
                                   self.result["micro"][metric][1])

    def test_zero_rel_query_vanishes_from_micro_ratios(self):
        queries = [
            ("q1", [("a", 1.0)], {"a": 1}),
            ("q0", [("b", 1.0)], {}),  # zero-relevance query
        ]
        result = rm.evaluate(queries, ks=[1])
        self.assertAlmostEqual(result["macro"]["ndcg"][1], 0.5)
        self.assertAlmostEqual(result["micro"]["ndcg"][1], 1.0)


class BootstrapTest(unittest.TestCase):
    def test_reproducible_with_seed(self):
        values = [0.0, 1.0] * 100
        ci1 = rm.bootstrap_ci(values, n_boot=500, seed=1)
        ci2 = rm.bootstrap_ci(values, n_boot=500, seed=1)
        self.assertEqual(ci1, ci2)

    def test_interval_contains_point_and_shrinks_with_n(self):
        rng = random.Random(3)
        small = [rng.random() for _ in range(40)]
        large = [rng.random() for _ in range(400)]
        ci_small = rm.bootstrap_ci(small, n_boot=1000, seed=9)
        ci_large = rm.bootstrap_ci(large, n_boot=1000, seed=9)
        self.assertLessEqual(ci_small["lo"], ci_small["point"])
        self.assertGreaterEqual(ci_small["hi"], ci_small["point"])
        self.assertLess(ci_large["se"], ci_small["se"])

    def test_constant_values_degenerate_interval(self):
        ci = rm.bootstrap_ci([0.5] * 50, n_boot=200, seed=5)
        self.assertEqual(ci["lo"], ci["hi"])
        self.assertEqual(ci["point"], 0.5)

    def test_small_sample_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rm.bootstrap_ci([0.0, 1.0, 0.5], n_boot=100, seed=1)
        self.assertTrue(any("unstable" in str(w.message) for w in caught))

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            rm.bootstrap_ci([])


if __name__ == "__main__":
    unittest.main()
