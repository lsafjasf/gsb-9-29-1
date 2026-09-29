"""test_binning.py — 分箱库自测（仅标准库 unittest）

运行：python3 test_binning.py -v
"""

import math
import random
import unittest

from binning import Binner, MISSING_BIN, check_monotonicity


def make_long_tail(n=5000, seed=7):
    """极端长尾数据：指数分布 + 少量离群巨值。"""
    rng = random.Random(seed)
    xs = [rng.expovariate(1.0) for _ in range(n)]
    xs += [1e6, 2e6, 5e5]  # 极端离群
    return xs


def make_monotone_xy(n=3000, seed=11):
    """风险率随 x 单调上升的数据。"""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(0, 10)
        p = 0.05 + 0.9 * (x / 10.0)
        xs.append(x)
        ys.append(1 if rng.random() < p else 0)
    return xs, ys


def make_u_shape_xy(n=4000, seed=13):
    """U 形风险率（中间低两端高），天然非单调。"""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(0, 10)
        p = 0.05 + 0.8 * ((x - 5) / 5.0) ** 2
        xs.append(x)
        ys.append(1 if rng.random() < p else 0)
    return xs, ys


class TestBoundaryStability(unittest.TestCase):
    """边界稳定性：重复分箱一致、顺序无关、打分不改边界。"""

    def test_refit_same_sample_same_edges(self):
        xs = make_long_tail()
        e1 = Binner(n_bins=10, method="quantile").fit(xs).edges_
        e2 = Binner(n_bins=10, method="quantile").fit(xs).edges_
        self.assertEqual(e1, e2)

    def test_shuffle_invariant_edges(self):
        xs = make_long_tail()
        shuffled = list(xs)
        random.Random(99).shuffle(shuffled)
        e1 = Binner(n_bins=10, method="quantile").fit(xs).edges_
        e2 = Binner(n_bins=10, method="quantile").fit(shuffled).edges_
        self.assertEqual(e1, e2)

    def test_optimal_refit_deterministic(self):
        xs, ys = make_monotone_xy()
        e1 = Binner(n_bins=6, method="optimal").fit(xs, ys).edges_
        e2 = Binner(n_bins=6, method="optimal").fit(xs, ys).edges_
        self.assertEqual(e1, e2)

    def test_transform_does_not_mutate_boundaries(self):
        xs, ys = make_monotone_xy()
        b = Binner(n_bins=6, method="optimal", monotonic="auto").fit(xs, ys)
        before = list(b.edges_)
        new_sample = [-100.0, 0.0, 3.3, 999.0, None, float("nan"), 1e9]
        b.transform(new_sample)
        b.transform(new_sample)  # 重复打分
        self.assertEqual(b.edges_, before)

    def test_out_of_range_scores_to_edge_bins(self):
        xs, ys = make_monotone_xy()
        b = Binner(n_bins=5, method="quantile").fit(xs, ys)
        assign = b.transform([-1e9, 1e9])
        self.assertEqual(assign[0], 0)
        self.assertEqual(assign[1], len(b.edges_))


class TestEdgeCases(unittest.TestCase):
    """边界用例：样本极少、单一取值、全空值、极端长尾。"""

    def test_tiny_sample(self):
        b = Binner(n_bins=5, method="quantile").fit([3.0, 1.0, 2.0])
        self.assertEqual(b.transform([1.5])[0] in (0, 1, 2), True)
        self.assertLessEqual(len(b.edges_), 2)

    def test_single_sample(self):
        b = Binner(n_bins=5).fit([42.0])
        self.assertEqual(b.edges_, [])
        self.assertEqual(b.transform([42.0]), [0])

    def test_single_distinct_value(self):
        b = Binner(n_bins=8).fit([7.0] * 1000)
        self.assertEqual(b.edges_, [])
        self.assertEqual(b.transform([7.0] * 5), [0] * 5)

    def test_all_missing(self):
        b = Binner(n_bins=5).fit([None, float("nan"), None])
        self.assertEqual(b.edges_, [])
        self.assertEqual(b.transform([None, float("nan")]),
                         [MISSING_BIN, MISSING_BIN])
        self.assertEqual(b.missing_count_, 3)
        self.assertEqual(b.missing_stats_["count"], 3)

    def test_extreme_long_tail_quantile(self):
        xs = make_long_tail()
        b = Binner(n_bins=10, method="quantile").fit(xs)
        # 分位点分箱在长尾下仍近似等频
        counts = [s["count"] for s in b.bin_stats_]
        self.assertTrue(all(c > 0 for c in counts))
        self.assertLess(max(counts) / min(counts), 2.0)
        # 极端值被打分到最右箱，不撑爆边界
        self.assertEqual(b.transform([1e9])[0], len(b.edges_))

    def test_extreme_long_tail_optimal(self):
        rng = random.Random(5)
        xs = make_long_tail()
        ys = [1 if rng.random() < min(0.9, x / 50.0) else 0 for x in xs]
        b = Binner(n_bins=5, method="optimal", monotonic="auto").fit(xs, ys)
        self.assertLessEqual(len(b.edges_), 4)
        report = b.monotonicity_report()
        self.assertTrue(report["is_monotonic"])

    def test_missing_and_outliers_counted(self):
        xs = list(range(1000)) + [None, float("nan"), -10**9, 10**9]
        ys = [0] * len(xs)
        b = Binner(n_bins=5, outlier_quantile=0.01).fit(xs, ys)
        self.assertEqual(b.missing_count_, 2)
        self.assertGreaterEqual(b.outlier_count_, 2)  # 两个极端值被截断计数
        self.assertEqual(b.n_total_, len(xs))
        # 空值箱计入统计且风险率可算
        self.assertEqual(b.missing_stats_["event_rate"], 0.0)


class TestMonotonicity(unittest.TestCase):
    """单调性检测数据与强制单调约束。"""

    def test_detect_monotone_data(self):
        xs, ys = make_monotone_xy()
        b = Binner(n_bins=6, method="optimal").fit(xs, ys)
        report = b.monotonicity_report()
        self.assertTrue(report["available"])
        self.assertTrue(report["is_monotonic"])
        self.assertEqual(report["violations"], [])

    def test_detect_u_shape_violations(self):
        xs, ys = make_u_shape_xy()
        b = Binner(n_bins=8, method="quantile").fit(xs, ys)
        report = b.monotonicity_report()
        self.assertFalse(report["is_monotonic"])
        self.assertGreater(len(report["violations"]), 0)
        # 违约点对结构：(i, i+1, rate_i, rate_i+1)
        for v in report["violations"]:
            self.assertEqual(len(v), 4)
        # 给出 PAVA 合并建议，且建议组覆盖全部箱
        groups = report["merge_suggestions"]
        self.assertGreater(len(groups), 0)
        flat = [i for g in groups for i in g]
        self.assertEqual(flat, list(range(len(report["event_rates"]))))

    def test_enforced_monotonic_constraint(self):
        xs, ys = make_u_shape_xy()
        b = Binner(n_bins=8, method="optimal",
                   monotonic="auto").fit(xs, ys)
        report = b.monotonicity_report()
        self.assertTrue(report["is_monotonic"])
        # 强制单调后箱数应少于等于目标箱数
        self.assertLessEqual(len(b.edges_) + 1, 8)

    def test_check_monotonicity_direction_auto(self):
        inc = check_monotonicity([0.1, 0.2, 0.3, 0.5], "auto")
        self.assertTrue(inc["is_monotonic"])
        self.assertEqual(inc["direction"], "increasing")
        dec = check_monotonicity([0.5, 0.4, 0.3, 0.1], "auto")
        self.assertTrue(dec["is_monotonic"])
        self.assertEqual(dec["direction"], "decreasing")
        bad = check_monotonicity([0.1, 0.5, 0.2, 0.6], "increasing")
        self.assertFalse(bad["is_monotonic"])
        self.assertEqual(len(bad["violations"]), 1)
        self.assertEqual(bad["violations"][0][0], 1)  # 第1、2箱之间违约

    def test_merge_suggestion_is_poolable(self):
        # 按建议合并后风险率必须单调
        rates = [0.3, 0.1, 0.4, 0.2, 0.9]
        report = check_monotonicity(rates, "increasing")
        self.assertFalse(report["is_monotonic"])
        pooled = []
        for g in report["merge_suggestions"]:
            pooled.append(sum(rates[i] for i in g) / len(g))
        for a, b_ in zip(pooled, pooled[1:]):
            self.assertLessEqual(a, b_ + 1e-9)


class TestStats(unittest.TestCase):
    def test_woe_iv_computed(self):
        xs, ys = make_monotone_xy()
        b = Binner(n_bins=5, method="optimal").fit(xs, ys)
        self.assertIsNotNone(b.iv_)
        self.assertGreater(b.iv_, 0.0)
        for s in b.bin_stats_:
            self.assertIn("woe", s)
            self.assertIn("event_rate", s)

    def test_summary_schema(self):
        xs, ys = make_monotone_xy(500)
        b = Binner(n_bins=4, method="quantile").fit(xs, ys)
        s = b.summary()
        for key in ("method", "edges", "n_total", "missing_count",
                    "outlier_count", "bins", "missing_bin", "iv"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
