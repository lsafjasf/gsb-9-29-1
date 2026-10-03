"""Self-tests for bootstrap_ci: statistics, reproducibility, parallel-merge
consistency, and edge cases.  Run: python3 test_bootstrap.py -v"""

import math
import unittest

from bootstrap_ci import (
    bootstrap,
    make_quantile_stat,
    make_ratio_stat,
    make_variance_stat,
    stat_max,
    stat_mean,
    stat_median,
)

DATA = [3.1, 1.4, 5.9, 2.6, 5.3, 8.9, 7.9, 3.2, 3.8, 4.6, 2.7, 6.1]


class TestStatistics(unittest.TestCase):
    def test_mean_median(self):
        self.assertAlmostEqual(stat_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(stat_median([3.0, 1.0, 2.0]), 2.0)
        self.assertAlmostEqual(stat_median([4.0, 1.0, 3.0, 2.0]), 2.5)

    def test_variance(self):
        var = make_variance_stat(ddof=1)
        self.assertAlmostEqual(var([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 4.5714286, places=6)
        # degenerate: n <= ddof falls back to population variance
        self.assertEqual(var([5.0]), 0.0)
        self.assertEqual(var([5.0, 5.0]), 0.0)

    def test_quantile(self):
        q50 = make_quantile_stat(0.5)
        q90 = make_quantile_stat(0.9)
        xs = [float(i) for i in range(1, 11)]
        self.assertAlmostEqual(q50(xs), 5.5)
        self.assertAlmostEqual(q90(xs), 9.1)
        with self.assertRaises(ValueError):
            make_quantile_stat(1.5)

    def test_ratio(self):
        ratio = make_ratio_stat()
        self.assertAlmostEqual(ratio(([2.0, 4.0], [1.0, 2.0])), 2.0)
        self.assertTrue(math.isnan(ratio(([1.0], [0.0, 0.0]))))


class TestReproducibility(unittest.TestCase):
    def test_same_seed_identical(self):
        r1 = bootstrap(DATA, stat_mean, 1000, seed=42)
        r2 = bootstrap(DATA, stat_mean, 1000, seed=42)
        self.assertEqual(r1.replicates, r2.replicates)
        self.assertEqual(r1.percentile_ci(), r2.percentile_ci())
        self.assertEqual(r1.bc_ci(), r2.bc_ci())
        self.assertEqual(r1.bca_ci(), r2.bca_ci())

    def test_different_seed_differs(self):
        r1 = bootstrap(DATA, stat_mean, 1000, seed=1)
        r2 = bootstrap(DATA, stat_mean, 1000, seed=2)
        self.assertNotEqual(r1.replicates, r2.replicates)

    def test_none_seed_is_recorded_and_reusable(self):
        r1 = bootstrap(DATA, stat_mean, 500, seed=None)
        r2 = bootstrap(DATA, stat_mean, 500, seed=r1.seed)
        self.assertEqual(r1.replicates, r2.replicates)

    def test_injectable_rng_factory(self):
        import random

        class Lcg:  # minimal custom RNG with the random.Random interface
            def __init__(self, seed):
                self.state = seed & ((1 << 64) - 1)

            def randrange(self, n):
                self.state = (self.state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
                return (self.state >> 11) % n

        r1 = bootstrap(DATA, stat_mean, 300, seed=7, rng_factory=Lcg)
        r2 = bootstrap(DATA, stat_mean, 300, seed=7, rng_factory=Lcg)
        r3 = bootstrap(DATA, stat_mean, 300, seed=7, rng_factory=random.Random)
        self.assertEqual(r1.replicates, r2.replicates)
        self.assertNotEqual(r1.replicates, r3.replicates)


class TestParallelMerge(unittest.TestCase):
    def test_parallel_equals_serial(self):
        serial = bootstrap(DATA, stat_mean, 2000, seed=99, chunk_size=256, n_jobs=1)
        parallel = bootstrap(DATA, stat_mean, 2000, seed=99, chunk_size=256, n_jobs=4)
        self.assertEqual(serial.replicates, parallel.replicates)
        self.assertEqual(serial.bca_ci(), parallel.bca_ci())

    def test_chunked_serial_equals_unchunked_layout(self):
        # Same seed + same chunk_size => identical regardless of how the
        # chunks are scheduled; chunk_size is part of the layout contract.
        a = bootstrap(DATA, stat_median, 1024, seed=5, chunk_size=512, n_jobs=1)
        b = bootstrap(DATA, stat_median, 1024, seed=5, chunk_size=512, n_jobs=2)
        self.assertEqual(a.replicates, b.replicates)

    def test_uneven_last_chunk(self):
        r = bootstrap(DATA, stat_mean, 1000, seed=3, chunk_size=300)
        self.assertEqual(len(r.replicates), 1000)


class TestEdgeCases(unittest.TestCase):
    def test_sample_size_one(self):
        r = bootstrap([4.2], stat_mean, 500, seed=0)
        self.assertTrue(all(v == 4.2 for v in r.replicates))
        self.assertEqual(r.percentile_ci(), (4.2, 4.2))
        self.assertEqual(r.bca_ci(), (4.2, 4.2))  # jackknife empty -> accel 0

    def test_constant_sample(self):
        r = bootstrap([2.0] * 20, stat_mean, 500, seed=0)
        self.assertEqual(r.percentile_ci() if False else r.percentile_ci(), (2.0, 2.0))
        self.assertEqual(r.bc_ci(), (2.0, 2.0))
        self.assertEqual(r.bca_ci(), (2.0, 2.0))  # accel denominator 0 -> 0

    def test_extreme_statistic_max(self):
        r = bootstrap(DATA, stat_max, 1000, seed=11)
        lo, hi = r.percentile_ci()
        self.assertLessEqual(lo, max(DATA))
        self.assertLessEqual(lo, hi)
        # z0 clamping keeps BC/BCa finite even though theta_hat is the max
        lo_bc, hi_bc = r.bc_ci()
        lo_bca, hi_bca = r.bca_ci()
        for v in (lo_bc, hi_bc, lo_bca, hi_bca):
            self.assertTrue(math.isfinite(v))

    def test_nonfinite_input_rejected(self):
        for bad in ([1.0, float("nan"), 3.0], [1.0, float("inf")], [float("-inf")]):
            with self.assertRaises(ValueError):
                bootstrap(bad, stat_mean, 100, seed=0)
        with self.assertRaises(ValueError):
            bootstrap(([1.0, 2.0], [float("nan")]), make_ratio_stat(), 100, seed=0)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap([], stat_mean, 100, seed=0)
        with self.assertRaises(ValueError):
            bootstrap(([], [1.0]), make_ratio_stat(), 100, seed=0)

    def test_ratio_zero_denominator(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 0.0, 0.0, 0.0]
        r = bootstrap((xs, ys), make_ratio_stat(), 200, seed=0)
        self.assertEqual(r.n_failed, 200)  # every replicate degenerate
        lo, hi = r.percentile_ci()
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_ratio_partial_zero_denominator(self):
        # Denominator mean is zero only for some resamples: those replicates
        # are dropped, the interval is computed from the finite remainder.
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [-1.0, 1.0, -2.0, 2.0, 0.5]
        r = bootstrap((xs, ys), make_ratio_stat(), 2000, seed=17)
        self.assertGreater(r.n_failed, 0)
        self.assertLess(r.n_failed, 2000)
        lo, hi = r.bca_ci()
        self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
        self.assertLess(lo, hi)

    def test_two_sample_ratio_reproducible(self):
        xs = [2.1, 3.4, 2.8, 4.0, 3.3]
        ys = [1.1, 0.9, 1.4, 1.2, 1.0]
        r1 = bootstrap((xs, ys), make_ratio_stat(), 1000, seed=8)
        r2 = bootstrap((xs, ys), make_ratio_stat(), 1000, seed=8)
        self.assertEqual(r1.replicates, r2.replicates)
        lo, hi = r1.bca_ci()
        self.assertLess(lo, r1.theta_hat)
        self.assertGreater(hi, r1.theta_hat)


class TestIntervalSanity(unittest.TestCase):
    def test_intervals_bracket_estimate_on_well_behaved_data(self):
        r = bootstrap(DATA, stat_mean, 4000, seed=123)
        for ci in (r.percentile_ci(), r.bc_ci(), r.bca_ci()):
            lo, hi = ci
            self.assertLess(lo, r.theta_hat)
            self.assertGreater(hi, r.theta_hat)
            self.assertLess(lo, hi)

    def test_confidence_level_widens_interval(self):
        r = bootstrap(DATA, stat_mean, 4000, seed=123)
        lo90, hi90 = r.bca_ci(0.90)
        lo99, hi99 = r.bca_ci(0.99)
        self.assertLess(lo99, lo90)
        self.assertGreater(hi99, hi90)


if __name__ == "__main__":
    unittest.main()
