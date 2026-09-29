"""Self-tests for risk_binning. Run: python3 -m unittest discover -s tests -v"""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_binning import (
    BinStat,
    OptimalBinner,
    QuantileBinner,
    find_violations,
    monotonicity_report,
    quantile,
    suggest_merges,
)


def make_monotone_data(n=4000, seed=7):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.random()
        xs.append(x)
        ys.append(1 if rng.random() < x else 0)
    return xs, ys


def make_u_shaped_data(n=6000, seed=11):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.random()
        p = 0.05 + 0.8 * (x - 0.5) ** 2 * 4
        xs.append(x)
        ys.append(1 if rng.random() < p else 0)
    return xs, ys


def normal_rates(report):
    return [b["bad_rate"] for b in report["bins"]
            if b["kind"] == "normal" and b["count"] > 0]


class TestQuantileCore(unittest.TestCase):
    def test_quantile_interpolation(self):
        self.assertEqual(quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(quantile([1, 2, 3, 4], 0.0), 1.0)
        self.assertEqual(quantile([1, 2, 3, 4], 1.0), 4.0)
        self.assertEqual(quantile([5], 0.3), 5.0)

    def test_find_violations(self):
        self.assertEqual(find_violations([0.1, 0.2, 0.3], "increasing"), [])
        self.assertEqual(find_violations([0.3, 0.1, 0.2], "increasing"), [0])
        self.assertEqual(find_violations([0.1, 0.3, 0.2], "decreasing"), [0])

    def test_suggest_merges_restores_monotonicity(self):
        stats = [
            BinStat(0, "b0", bad=9, good=1),
            BinStat(1, "b1", bad=1, good=9),
            BinStat(2, "b2", bad=9, good=1),
        ]
        suggestions, groups, rates = suggest_merges(stats, "increasing")
        self.assertTrue(suggestions)
        self.assertEqual(find_violations(rates, "increasing"), [])

    def test_suggest_merges_blocked_by_min_bins(self):
        stats = [
            BinStat(0, "b0", bad=9, good=1),
            BinStat(1, "b1", bad=1, good=9),
            BinStat(2, "b2", bad=9, good=1),
        ]
        suggestions, _, rates = suggest_merges(stats, "increasing", min_bins=3)
        self.assertEqual(suggestions, [])
        self.assertTrue(find_violations(rates, "increasing"))

    def test_monotonicity_report_direction(self):
        stats = [BinStat(i, "b%d" % i, bad=i + 1, good=10) for i in range(4)]
        rep = monotonicity_report(stats, direction="auto")
        self.assertEqual(rep["direction"], "increasing")
        self.assertTrue(rep["is_monotone"])


class TestBoundaryStability(unittest.TestCase):
    def test_same_sample_same_edges(self):
        xs, ys = make_monotone_data()
        for binner in (QuantileBinner(n_bins=6),
                       OptimalBinner(n_bins=6, monotone="auto")):
            b1 = binner.fit(xs, ys)
            again = type(binner)(**self._params(binner)).fit(xs, ys)
            self.assertEqual(b1.edges, again.edges)
            refit = b1.fit(xs, ys)
            self.assertEqual(refit.edges, again.edges)

    def _params(self, b):
        if isinstance(b, OptimalBinner):
            return {"n_bins": b.n_bins, "prebins": b.prebins,
                    "monotone": b.monotone, "min_bins": b.min_bins}
        return {"n_bins": b.n_bins}

    def test_transform_never_changes_edges(self):
        xs, ys = make_monotone_data()
        b = QuantileBinner(n_bins=5).fit(xs, ys)
        edges_before = b.edges
        fences_before = (b.lower_fence_, b.upper_fence_)
        b.transform([rng_v for rng_v in range(100)])
        b.transform(xs)
        b.fit_transform  # attribute access only, no-op
        self.assertEqual(b.edges, edges_before)
        self.assertEqual((b.lower_fence_, b.upper_fence_), fences_before)

    def test_boundary_assignment(self):
        xs = list(range(1000))
        b = QuantileBinner(n_bins=4, outlier_quantile=0.0).fit(xs)
        self.assertEqual(len(b.edges), 3)
        e0 = b.edges[0]
        # exact edge value -> bin whose upper bound it is: (lower, upper]
        self.assertEqual(b.transform([e0])[0], "bin_00")
        self.assertEqual(b.transform([e0 + 1e-9])[0], "bin_01")
        self.assertEqual(b.transform([0])[0], "bin_00")
        self.assertEqual(b.transform([999])[0], "bin_03")

    def test_fence_boundary_inclusive(self):
        xs = list(range(1000))
        b = QuantileBinner(n_bins=4).fit(xs)
        lo, hi = b.lower_fence_, b.upper_fence_
        self.assertTrue(b.transform([lo])[0].startswith("bin_"))
        self.assertTrue(b.transform([hi])[0].startswith("bin_"))
        self.assertEqual(b.transform([lo - 1e-6])[0], "outlier_low")
        self.assertEqual(b.transform([hi + 1e6])[0], "outlier_high")


class TestMissingAndOutliers(unittest.TestCase):
    def test_missing_and_outliers_counted(self):
        rng = random.Random(3)
        xs = [rng.random() for _ in range(2000)]
        ys = [1 if rng.random() < v else 0 for v in xs]
        xs = xs + [None, float("nan"), 1e9, -1e9]
        ys = ys + [1, 0, 1, 0]
        b = QuantileBinner(n_bins=5).fit(xs, ys)
        rep = b.report()
        self.assertEqual(rep["n_missing"], 2)
        # quantile fences flag ~0.5% of each tail plus the injected extremes
        self.assertGreaterEqual(rep["n_outlier_low"], 1)
        self.assertGreaterEqual(rep["n_outlier_high"], 1)
        by_label = {bin_["label"]: bin_ for bin_ in rep["bins"]}
        self.assertEqual(by_label["missing"]["count"], 2)
        self.assertEqual(by_label["missing"]["bad"], 1)
        self.assertGreaterEqual(by_label["outlier_high"]["bad"], 1)
        self.assertEqual(b.transform([None, float("nan"), 1e9, -1e9]),
                         ["missing", "missing", "outlier_high", "outlier_low"])

    def test_extreme_long_tail_not_collapsed(self):
        rng = random.Random(9)
        xs = [rng.expovariate(1.0) for _ in range(5000)]
        ys = [1 if rng.random() < min(v / 5.0, 0.95) else 0 for v in xs]
        xs += [1e6, 1e7, 1e8]
        ys += [1, 1, 1]
        b = QuantileBinner(n_bins=8).fit(xs, ys)
        rep = b.report()
        used = [bin_ for bin_ in rep["bins"]
                if bin_["kind"] == "normal" and bin_["count"] > 0]
        self.assertGreaterEqual(len(used), 4)
        self.assertGreaterEqual(rep["n_outlier_high"], 3)
        self.assertLess(max(b.edges), 100.0)  # edges immune to huge outliers
        self.assertEqual(b.transform([1e8])[0], "outlier_high")


class TestOptimalBinner(unittest.TestCase):
    def test_requires_y(self):
        with self.assertRaises(ValueError):
            OptimalBinner().fit([1.0, 2.0, 3.0])

    def test_monotone_constraint_enforced(self):
        xs, ys = make_monotone_data()
        b = OptimalBinner(n_bins=5, monotone="increasing").fit(xs, ys)
        rates = normal_rates(b.report())
        self.assertEqual(find_violations(rates, "increasing"), [])

    def test_violation_reported_with_suggestions(self):
        xs, ys = make_u_shaped_data()
        b = OptimalBinner(n_bins=5, monotone=None).fit(xs, ys)
        mono = b.report()["monotonicity"]
        self.assertFalse(mono["is_monotone"])
        self.assertTrue(mono["violations"])
        self.assertTrue(mono["merge_suggestions"])
        self.assertEqual(
            find_violations(mono["rates_after_suggested_merges"],
                            mono["direction"]), [])

    def test_unsatisfiable_constraint_warns_and_suggests(self):
        xs, ys = [], []
        for k in range(6):
            for i in range(100):
                xs.append(k + i * 1e-4)
                ys.append(1 if k % 2 == 0 else 0)
        b = OptimalBinner(n_bins=6, prebins=6, min_bins=6,
                          monotone="increasing").fit(xs, ys)
        rep = b.report()
        self.assertTrue(any("unsatisfiable" in w for w in rep["warnings"]))
        mono = rep["monotonicity"]
        self.assertFalse(mono["is_monotone"])
        self.assertTrue(mono["merge_suggestions"])

    def test_optimal_deterministic(self):
        xs, ys = make_u_shaped_data()
        e1 = OptimalBinner(n_bins=5, monotone="auto").fit(xs, ys).edges
        e2 = OptimalBinner(n_bins=5, monotone="auto").fit(xs, ys).edges
        self.assertEqual(e1, e2)


class TestEdgeCases(unittest.TestCase):
    def test_empty_input(self):
        b = QuantileBinner().fit([])
        self.assertEqual(b.edges, [])
        self.assertEqual(b.transform([None]), ["missing"])
        self.assertEqual(b.transform([1.0]), ["unseen_value"])
        self.assertTrue(any("missing" in w for w in b.report()["warnings"]))

    def test_single_row(self):
        b = QuantileBinner().fit([3.14])
        self.assertEqual(b.edges, [])
        self.assertEqual(b.transform([3.14]), ["bin_00"])

    def test_single_distinct_value(self):
        b = QuantileBinner(n_bins=5).fit([2.0] * 100)
        self.assertEqual(b.edges, [])
        self.assertEqual(b.transform([2.0]), ["bin_00"])
        self.assertTrue(any("single distinct" in w
                            for w in b.report()["warnings"]))

    def test_all_missing(self):
        b = QuantileBinner().fit([None, float("nan"), None])
        rep = b.report()
        self.assertEqual(rep["n_missing"], 3)
        by_label = {bin_["label"]: bin_ for bin_ in rep["bins"]}
        self.assertEqual(by_label["missing"]["count"], 3)
        self.assertEqual(b.transform([None]), ["missing"])
        self.assertEqual(b.transform([7.0]), ["unseen_value"])

    def test_tiny_sample(self):
        b = OptimalBinner(n_bins=5).fit([1.0, 2.0, 3.0], [0, 1, 0])
        rep = b.report()
        normal = [bin_ for bin_ in rep["bins"] if bin_["kind"] == "normal"]
        self.assertGreaterEqual(len(normal), 1)
        self.assertEqual(sum(bin_["count"] for bin_ in normal), 3)

    def test_all_same_class_y(self):
        b = OptimalBinner(n_bins=4).fit([float(i) for i in range(200)],
                                        [0] * 200)
        rep = b.report()
        self.assertEqual(rep["total_iv"], 0.0)


if __name__ == "__main__":
    unittest.main()
