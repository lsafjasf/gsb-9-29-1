"""bootstrap_ci 自测。

运行：python3 -m unittest discover -s tests -v
"""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bootstrap_ci import (
    bootstrap_ci, bootstrap_replicates, bootstrap_chunk, merge_chunks,
    jackknife, acceleration, norm_ppf, norm_cdf, default_chunk_size,
    stat_mean, stat_median, stat_variance, stat_quantile, stat_ratio,
    derive_seed, resample,
)

NAN = float("nan")
INF = float("inf")


def stat_max(data):
    return max(data)


class TestStatistics(unittest.TestCase):
    def test_mean_median_variance(self):
        self.assertAlmostEqual(stat_mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(stat_median([3, 1, 2]), 2.0)
        self.assertAlmostEqual(stat_variance([1, 2, 3]), 1.0)          # ddof=1
        self.assertAlmostEqual(stat_variance([1, 2, 3], ddof=0), 2 / 3)

    def test_quantile_type7(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(stat_quantile(data, 0.0), 1.0)
        self.assertEqual(stat_quantile(data, 1.0), 4.0)
        self.assertAlmostEqual(stat_quantile(data, 0.5), 2.5)
        self.assertAlmostEqual(stat_quantile(data, 0.25), 1.75)
        self.assertEqual(stat_quantile([7.0], 0.3), 7.0)               # n=1
        with self.assertRaises(ValueError):
            stat_quantile(data, 1.5)

    def test_ratio(self):
        data = [(1.0, 2.0), (3.0, 4.0), (2.0, 4.0)]
        self.assertAlmostEqual(stat_ratio(data), 6.0 / 10.0)

    def test_ratio_zero_denominator(self):
        self.assertTrue(math.isnan(stat_ratio([(1.0, 0.0), (2.0, 0.0)])))
        self.assertEqual(stat_ratio([(1.0, 0.0)], zero_denom=0.0), 0.0)

    def test_norm_ppf_cdf(self):
        self.assertAlmostEqual(norm_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(norm_cdf(1.959964), 0.975, places=5)
        for p in (0.001, 0.1, 0.5, 0.9, 0.999):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=9)


class TestReproducibility(unittest.TestCase):
    def setUp(self):
        rng = random.Random(2026)
        self.data = [rng.expovariate(1.0) for _ in range(30)]

    def test_same_seed_identical(self):
        r1 = bootstrap_ci(self.data, stat_mean, B=999, seed=123)
        r2 = bootstrap_ci(self.data, stat_mean, B=999, seed=123)
        self.assertEqual(r1["replicates"], r2["replicates"])
        self.assertEqual(r1["intervals"], r2["intervals"])

    def test_different_seed_differs(self):
        r1 = bootstrap_replicates(self.data, stat_mean, 200, seed=1)
        r2 = bootstrap_replicates(self.data, stat_mean, 200, seed=2)
        self.assertNotEqual(r1, r2)

    def test_injected_rng_deterministic(self):
        a = bootstrap_replicates(self.data, stat_mean, 50,
                                 rng=random.Random(9))
        b = bootstrap_replicates(self.data, stat_mean, 50,
                                 rng=random.Random(9))
        self.assertEqual(a, b)

    def test_resample_helper(self):
        rng = random.Random(5)
        out = resample(self.data, rng)
        self.assertEqual(len(out), len(self.data))
        self.assertTrue(all(x in self.data for x in out))


class TestParallelMerge(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.data = [rng.gauss(0, 1) for _ in range(25)]

    def test_parallel_equals_serial(self):
        serial = bootstrap_replicates(self.data, stat_median, 1000,
                                      seed=42, n_jobs=1)
        parallel = bootstrap_replicates(self.data, stat_median, 1000,
                                        seed=42, n_jobs=4)
        self.assertEqual(serial, parallel)

    def test_bootstrap_ci_parallel_equals_serial(self):
        s = bootstrap_ci(self.data, stat_variance, B=999, seed=3, n_jobs=1)
        p = bootstrap_ci(self.data, stat_variance, B=999, seed=3, n_jobs=4)
        self.assertEqual(s["replicates"], p["replicates"])
        self.assertEqual(s["intervals"], p["intervals"])

    def test_chunk_merge_order(self):
        B, seed = 500, 11
        full = bootstrap_replicates(self.data, stat_mean, B, seed=seed)
        size = default_chunk_size(B)
        chunks = []
        start = 0
        c = 0
        while start < B:
            take = min(size, B - start)
            chunks.append(bootstrap_chunk(self.data, stat_mean,
                                          derive_seed(seed, c), take))
            start += take
            c += 1
        self.assertEqual(merge_chunks(chunks), full)

    def test_chunk_seed_independence(self):
        # 同一 chunk_seed 重复执行结果一致（worker 无共享状态）
        a = bootstrap_chunk(self.data, stat_mean, 777, 64)
        b = bootstrap_chunk(self.data, stat_mean, 777, 64)
        self.assertEqual(a, b)


class TestEdgeCases(unittest.TestCase):
    def test_n_equals_1(self):
        res = bootstrap_ci([5.0], stat_mean, B=200, seed=1)
        self.assertEqual(res["theta_hat"], 5.0)
        self.assertEqual(res["intervals"]["percentile"], (5.0, 5.0))
        self.assertTrue(any("n=1" in w for w in res["warnings"]))

    def test_empty_data_raises(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([], stat_mean, B=100, seed=1)

    def test_all_equal_values(self):
        res = bootstrap_ci([3.0] * 10, stat_mean, B=500, seed=2)
        self.assertEqual(res["intervals"]["percentile"], (3.0, 3.0))
        self.assertEqual(res["acceleration"], 0.0)  # jackknife 退化 -> a=0
        self.assertTrue(math.isinf(res["z0"]))
        self.assertTrue(any("z0" in w for w in res["warnings"]))
        self.assertTrue(all(math.isnan(v)
                            for v in res["intervals"]["bca"]))

    def test_extreme_statistic_max(self):
        rng = random.Random(4)
        data = [rng.random() for _ in range(20)]
        res = bootstrap_ci(data, stat_max, B=999, seed=8)
        lo, hi = res["intervals"]["percentile"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(hi, max(data))  # 重采样 max 不会超过样本 max
        # 重采样 max 系统性低于样本 max（约 1-(1-1/n)^n 的比例），z0 显著为负，
        # 说明三种非参数区间对极值统计量都不可信（经典失效情形）
        self.assertLess(res["z0"], -0.2)
        self.assertTrue(math.isfinite(res["z0"]))

    def test_nonfinite_rejected_by_default(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0, NAN, 2.0], stat_mean, B=100, seed=1)
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0, INF, 2.0], stat_mean, B=100, seed=1)

    def test_nonfinite_propagates_when_allowed(self):
        res = bootstrap_ci([1.0, NAN, 2.0, 3.0], stat_mean, B=500,
                           seed=6, allow_nonfinite=True)
        self.assertGreater(res["n_nonfinite_replicates"], 0)
        lo, hi = res["intervals"]["percentile"]
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
        self.assertTrue(any("非有限" in w for w in res["warnings"]))

    def test_ratio_with_zero_denominator_resamples(self):
        data = [(1.0, 0.0), (2.0, 1.0), (3.0, 1.0)]
        res = bootstrap_ci(data, stat_ratio, B=999, seed=10)
        self.assertAlmostEqual(res["theta_hat"], 3.0)
        self.assertGreater(res["n_nonfinite_replicates"], 0)
        lo, hi = res["intervals"]["percentile"]
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))

    def test_degenerate_variance_ddof(self):
        self.assertTrue(math.isnan(stat_variance([1.0])))  # n=1, ddof=1

    def test_jackknife_and_acceleration(self):
        data = [1.0, 2.0, 3.0, 4.0]
        jk = jackknife(data, stat_mean)
        self.assertEqual(len(jk), 4)
        self.assertAlmostEqual(jk[0], 3.0)  # 去掉 1.0 后均值
        self.assertEqual(acceleration([2.0, 2.0, 2.0]), 0.0)  # 无变异
        self.assertEqual(acceleration([1.0]), 0.0)            # 太少


class TestIntervalSmoke(unittest.TestCase):
    """小尺度覆盖率冒烟测试（宽松界，防回归；正式证据见 simulate.py）。"""

    def test_normal_mean_coverage_roughly_nominal(self):
        R, B, n = 300, 499, 20
        hits = 0
        for rep in range(R):
            rng = random.Random(10 ** 6 + rep)
            data = [rng.gauss(0, 1) for _ in range(n)]
            res = bootstrap_ci(data, stat_mean, B=B, seed=1000 + rep,
                               methods=("percentile", "bca"))
            lo, hi = res["intervals"]["percentile"]
            if lo <= 0.0 <= hi:
                hits += 1
        rate = hits / R
        self.assertGreater(rate, 0.85)
        self.assertLessEqual(rate, 1.0)

    def test_methods_dict_keys(self):
        res = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], stat_mean, B=499,
                           seed=1, methods=("percentile",))
        self.assertEqual(set(res["intervals"]), {"percentile"})


if __name__ == "__main__":
    unittest.main()
