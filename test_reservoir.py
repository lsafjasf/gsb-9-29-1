"""Edge-case and statistical self-tests for reservoir.py.

All statistical tests use fixed seeds, so the suite is deterministic.
Run:  python3 test_reservoir.py   (or: python3 -m unittest -v)
"""

import math
import random
import unittest

from reservoir import ReservoirSampler, WeightedReservoirSampler


class TestUnweighted(unittest.TestCase):
    def test_k_must_be_positive(self):
        for bad in (0, -1, 2.5, "3"):
            with self.assertRaises((ValueError, TypeError)):
                ReservoirSampler(bad)

    def test_empty_stream(self):
        s = ReservoirSampler(5, rng=random.Random(1))
        self.assertEqual(s.sample(), [])

    def test_single_element_stream(self):
        s = ReservoirSampler(3, rng=random.Random(1))
        s.add("only")
        self.assertEqual(s.sample(), ["only"])

    def test_stream_shorter_than_k_keeps_everything(self):
        s = ReservoirSampler(10, rng=random.Random(1))
        for i in range(4):
            s.add(i)
        self.assertEqual(sorted(s.sample()), [0, 1, 2, 3])

    def test_sample_size_and_memory_bounded(self):
        s = ReservoirSampler(7, rng=random.Random(2))
        for i in range(100_000):
            s.add(i)
        self.assertEqual(len(s.sample()), 7)
        self.assertEqual(len(set(s.sample())), 7)  # without replacement

    def test_reproducible_with_injected_rng(self):
        def run():
            s = ReservoirSampler(5, rng=random.Random(42))
            for i in range(1000):
                s.add(i)
            return s.sample()
        self.assertEqual(run(), run())

    def test_uniform_inclusion_fixed_seed(self):
        # n=10, k=3 -> theoretical inclusion probability 0.3 for every item.
        n, k, trials = 10, 3, 60_000
        counts = [0] * n
        rng = random.Random(20260930)
        for _ in range(trials):
            s = ReservoirSampler(k, rng=rng)
            for i in range(n):
                s.add(i)
            for i in s.sample():
                counts[i] += 1
        expected = trials * k / n
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        # df = 9; 99.9% critical value ~ 27.9. Fixed seed -> deterministic.
        self.assertLess(chi2, 27.9, "counts=%r chi2=%.2f" % (counts, chi2))


class TestWeighted(unittest.TestCase):
    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            WeightedReservoirSampler(0)

    def test_empty_stream(self):
        s = WeightedReservoirSampler(3, rng=random.Random(1))
        self.assertEqual(s.sample(), [])

    def test_single_element_stream(self):
        s = WeightedReservoirSampler(5, rng=random.Random(1))
        s.add("only", 2.5)
        self.assertEqual(s.sample(), ["only"])

    def test_all_zero_weights(self):
        s = WeightedReservoirSampler(3, rng=random.Random(1))
        for i in range(100):
            s.add(i, 0.0)
        self.assertEqual(s.sample(), [])
        self.assertEqual(s.n, 100)
        self.assertEqual(s.n_positive, 0)

    def test_zero_weight_items_never_selected(self):
        s = WeightedReservoirSampler(2, rng=random.Random(3))
        s.add("zero1", 0.0)
        s.add("heavy", 1.0)
        s.add("zero2", 0.0)
        self.assertEqual(s.sample(), ["heavy"])

    def test_negative_and_nonfinite_weights_rejected(self):
        s = WeightedReservoirSampler(2, rng=random.Random(1))
        for bad in (-1.0, -1e-300, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg="weight=%r" % bad):
                s.add("x", bad)

    def test_integer_and_float_weights_accepted(self):
        s = WeightedReservoirSampler(2, rng=random.Random(1))
        s.add("int", 3)          # int weight
        s.add("float", 0.5)      # float weight
        s.add("big", 10**6)
        self.assertEqual(len(s.sample()), 2)

    def test_extreme_weight_ratio_dominates(self):
        # Ratio 1e6 : 1 -> the heavy item must (essentially) always win.
        wins = 0
        trials = 2_000
        rng = random.Random(7)
        for _ in range(trials):
            s = WeightedReservoirSampler(1, rng=rng)
            s.add("light", 1.0)
            s.add("heavy", 1e6)
            if s.sample() == ["heavy"]:
                wins += 1
        # P(light wins) = 1e-6 per trial; 0 wins in 2000 trials is expected.
        self.assertEqual(wins, trials)

    def test_extreme_magnitudes_no_overflow_or_underflow(self):
        # Weights from 1e-300 to 1e300: log-domain keys stay finite.
        s = WeightedReservoirSampler(3, rng=random.Random(11))
        weights = [1e-300, 1e-100, 1.0, 1e100, 1e300, 5e-324]  # 5e-324 = min denormal
        for i, w in enumerate(weights):
            s.add(i, w)
        self.assertEqual(len(s.sample()), 3)
        # The 1e300 item must be in the sample with overwhelming probability.
        self.assertIn(4, s.sample())

    def test_million_fold_spread_all_present(self):
        # Weights 1, 10, ..., 1e6 in one stream: no crash, exact size k.
        s = WeightedReservoirSampler(4, rng=random.Random(13))
        for i in range(7):
            s.add(i, 10.0 ** i)
        self.assertEqual(len(s.sample()), 4)

    def test_fewer_positive_items_than_k(self):
        s = WeightedReservoirSampler(10, rng=random.Random(1))
        s.add("a", 1.0)
        s.add("b", 0.0)
        s.add("c", 2.0)
        self.assertEqual(sorted(s.sample()), ["a", "c"])

    def test_reproducible_with_injected_rng(self):
        def run():
            s = WeightedReservoirSampler(3, rng=random.Random(99))
            for i in range(500):
                s.add(i, (i % 7) + 0.5)
            return s.sample()
        self.assertEqual(run(), run())

    def test_weighted_k1_distribution_fixed_seed(self):
        # k=1: inclusion probability is exactly w_i / sum(w).
        weights = [1.0, 2.0, 3.0, 4.0]
        total = sum(weights)
        trials = 80_000
        counts = [0] * len(weights)
        rng = random.Random(20260930)
        for _ in range(trials):
            s = WeightedReservoirSampler(1, rng=rng)
            for i, w in enumerate(weights):
                s.add(i, w)
            counts[s.sample()[0]] += 1
        chi2 = sum(
            (c - trials * w / total) ** 2 / (trials * w / total)
            for c, w in zip(counts, weights)
        )
        # df = 3; 99.9% critical value ~ 16.3. Fixed seed -> deterministic.
        self.assertLess(chi2, 16.3, "counts=%r chi2=%.2f" % (counts, chi2))

    def test_rng_returning_zero_is_handled(self):
        # A (hostile) RNG that returns exactly 0.0 must not crash or hang.
        class ZeroThenRandom:
            def __init__(self):
                self.calls = 0
                self.inner = random.Random(5)
            def random(self):
                self.calls += 1
                return 0.0 if self.calls == 1 else self.inner.random()
        s = WeightedReservoirSampler(2, rng=ZeroThenRandom())
        for i in range(10):
            s.add(i, 1.0)
        self.assertEqual(len(s.sample()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
