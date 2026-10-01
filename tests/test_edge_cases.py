import unittest
from fractions import Fraction

from reservoir import (
    UniformReservoirSampler, WeightedOneSampler, merge_weighted_one,
    AResWeightedSampler, PPSReservoirSampler,
    merge_uniform, merge_ares, parse_weight,
)
from tests._scripted_rng import ScriptedRNG


class TestUniformEdgeCases(unittest.TestCase):
    def test_single_element_stream(self):
        s = UniformReservoirSampler(1, ScriptedRNG()).feed("only")
        self.assertEqual(s.sample(), ["only"])

    def test_k_zero(self):
        s = UniformReservoirSampler(0, ScriptedRNG()).feed_all(range(5))
        self.assertEqual(s.sample(), [])

    def test_k_larger_than_stream(self):
        s = UniformReservoirSampler(10, ScriptedRNG()).feed_all([1, 2])
        self.assertEqual(sorted(s.sample()), [1, 2])

    def test_replacement_happens(self):
        # j = randrange(n) == 0 forces replacement of slot 0
        s = UniformReservoirSampler(2, ScriptedRNG([0.0, 0.0]))
        s.feed_all(["a", "b", "c", "d"])
        self.assertEqual(s.sample()[0], "d")


class TestWeightedOneEdgeCases(unittest.TestCase):
    def test_all_zero_weights_raise(self):
        s = WeightedOneSampler(ScriptedRNG())
        s.feed_all([(i, 0) for i in range(5)])
        with self.assertRaises(ValueError):
            s.sample()
        with self.assertRaises(ValueError):
            s.state()

    def test_single_element(self):
        s = WeightedOneSampler(ScriptedRNG()).feed("x", "0.0001")
        self.assertEqual(s.sample(), "x")

    def test_zero_weight_never_selected(self):
        for script in ([0.0], [0.999]):
            s = WeightedOneSampler(ScriptedRNG(script))
            s.feed("zero", 0)
            s.feed("pos", 1)
            self.assertEqual(s.sample(), "pos")

    def test_running_total_is_exact(self):
        s = WeightedOneSampler(ScriptedRNG([1.0] * 100000))
        for _ in range(100000):
            s.feed("x", "0.1")
        self.assertEqual(s.W, Fraction(10000))  # float accumulation would drift

    def test_merge_all_zero_raises(self):
        with self.assertRaises(ValueError):
            merge_weighted_one([("a", Fraction(0))], ScriptedRNG())


class TestAResEdgeCases(unittest.TestCase):
    def test_all_zero_weights_raise(self):
        s = AResWeightedSampler(2, ScriptedRNG())
        s.feed_all([(i, 0) for i in range(5)])
        with self.assertRaises(ValueError):
            s.sample()

    def test_zero_weights_never_selected(self):
        s = AResWeightedSampler(2, ScriptedRNG([0.5, 0.5]))
        for i in range(10):
            s.feed(("zero", i), 0)
        s.feed("pos1", 1)
        s.feed("pos2", 1)
        self.assertEqual(s.sample(), ["pos1", "pos2"])

    def test_single_element(self):
        s = AResWeightedSampler(3, ScriptedRNG([0.5])).feed("solo", "1e-20")
        self.assertEqual(s.sample(), ["solo"])

    def test_k_zero(self):
        s = AResWeightedSampler(0, ScriptedRNG()).feed("a", 1)
        self.assertEqual(s.sample(), [])

    def test_extreme_weights_are_ordered_exactly(self):
        # 10**30 vs 1: small key can never beat huge key with the same draw.
        from reservoir.ares import _Key
        big = _Key(0.5, Fraction(10**30))
        small = _Key(0.5, Fraction(1))
        self.assertGreater(big, small)
        # close huge weights are still distinguished (float64 collapse case)
        k1 = _Key(0.5, Fraction(10**30))
        k2 = _Key(0.5000000001, Fraction(10**30))
        self.assertGreater(k2, k1)

    def test_u_zero_is_valid_key(self):
        from reservoir.ares import _Key
        self.assertEqual(_Key(0.0, Fraction(1)) < _Key(0.5, Fraction(1)), True)
        s = AResWeightedSampler(1, ScriptedRNG([0.0, 0.5]))
        s.feed("a", 1)
        s.feed("b", 1)
        self.assertEqual(s.sample(), ["b"])


class TestPPSEdgeCases(unittest.TestCase):
    def test_all_zero_weights_raise(self):
        s = PPSReservoirSampler(2, ScriptedRNG())
        s.feed_all([(i, 0) for i in range(5)])
        with self.assertRaises(ValueError):
            s.sample()

    def test_single_element(self):
        s = PPSReservoirSampler(1, ScriptedRNG()).feed("only", 3)
        self.assertEqual(s.sample(), ["only"])

    def test_extreme_skew_runs_and_keeps_probabilities_valid(self):
        rng = ScriptedRNG(seed=123)
        s = PPSReservoirSampler(5, rng)
        s.feed("huge", 10**6)
        for i in range(1, 50):
            s.feed(i, i % 7 + 1)
        sample = s.sample()
        self.assertEqual(len(sample), 5)
        for p in s.inclusion_probabilities():
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)

    def test_zero_weight_never_enters_when_full(self):
        # after reservoir fills, a zero-weight arrival cannot enter
        s = PPSReservoirSampler(2, ScriptedRNG(seed=5))
        s.feed("a", 1)
        s.feed("b", 1)
        s.feed("zero", 0)
        self.assertEqual(sorted(s.sample()), ["a", "b"])



    def test_exactness_flag(self):
        from reservoir import PPSReservoirSampler
        # exact regime: first k equal, no certainty units later
        s = PPSReservoirSampler(3, ScriptedRNG(seed=2))
        for i, w in enumerate([12, 12, 12, 9, 10, 11, 9]):
            s.feed(i, w)
        self.assertTrue(s.exactness_holds)
        # certainty unit violates exactness
        s2 = PPSReservoirSampler(3, ScriptedRNG(seed=2))
        for i, w in enumerate([12, 12, 12, 100]):
            s2.feed(i, w)
        self.assertFalse(s2.exactness_holds)
        # unequal first k violates condition (i)
        s3 = PPSReservoirSampler(3, ScriptedRNG(seed=2))
        for i, w in enumerate([12, 12, 1, 9]):
            s3.feed(i, w)
        self.assertFalse(s3.exactness_holds)


class TestMergeEdgeCases(unittest.TestCase):
    def test_uniform_empty_and_passthrough(self):
        self.assertEqual(merge_uniform([([], 0), ([], 0)], 3, ScriptedRNG()), [])
        shards = [([1, 2], 2), ([3], 1)]
        self.assertEqual(sorted(merge_uniform(shards, 5, ScriptedRNG())), [1, 2, 3])

    def test_ares_single_shard_passthrough(self):
        rng = ScriptedRNG(seed=1)
        a = AResWeightedSampler(3, rng)
        for i in range(3):
            a.feed(i, i + 1)
        self.assertEqual(sorted(merge_ares([a.state()], 3)), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
