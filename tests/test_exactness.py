import unittest
from fractions import Fraction

from reservoir import (
    WeightedOneSampler, AResWeightedSampler,
    ares_inclusion_probabilities, pps_targets, parse_weight,
)
from tests._scripted_rng import ScriptedRNG


class TestExactFormulae(unittest.TestCase):
    def test_k1_integral_equals_weight_share(self):
        for weights in ([1, 2, 3], [Fraction(1, 2), Fraction(3, 7), Fraction(10)],
                        [1, 10**6, 3]):
            probs = ares_inclusion_probabilities(weights, 1)
            W = sum(Fraction(w) for w in weights)
            for p, w in zip(probs, weights):
                self.assertEqual(p, Fraction(w) / W)

    def test_probabilities_sum_to_k(self):
        for weights, k in (([1, 2, 3, 4, 5], 2),
                           ([1, 10, 100, 1000], 3),
                           ([Fraction(1, 3), Fraction(2), Fraction(5)], 2)):
            probs = ares_inclusion_probabilities(weights, k)
            self.assertEqual(sum(probs), Fraction(k))
            for p in probs:
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)

    def test_within_0_and_1_with_decimal_weights(self):
        weights = [parse_weight(s) for s in ("0.5", "2", "10", "0.001", "7", "3")]
        probs = ares_inclusion_probabilities(weights, 3)
        self.assertEqual(sum(probs), Fraction(3))

    def test_k_ge_n(self):
        probs = ares_inclusion_probabilities([1, 2], 5)
        self.assertEqual(probs, [Fraction(1), Fraction(1)])

    def test_all_zero_rejected(self):
        with self.assertRaises(ValueError):
            ares_inclusion_probabilities([0, 0], 2)
        with self.assertRaises(ValueError):
            pps_targets([0, 0], 2)

    def test_zero_weight_has_zero_probability(self):
        probs = ares_inclusion_probabilities([1, 2, 0, 3], 2)
        self.assertEqual(probs[2], Fraction(0))
        self.assertEqual(sum(probs), Fraction(2))

    def test_weighted_one_exact_replacement_boundary(self):
        # w/W = 1/3: random() returns 0.3333.. < 1/3 -> keep; > -> reject
        self.assertTrue(ScriptedRNG([float(Fraction(1, 3) - Fraction(1, 10**15))
                                     ]).random() < Fraction(1, 3))
        s_keep = WeightedOneSampler(ScriptedRNG([0.0, 0.3]))
        s_keep.feed("a", 1)
        s_keep.feed("b", 0.5)  # W=1.5, p=1/3; 0.3 < 1/3 -> b kept
        self.assertEqual(s_keep.sample(), "b")
        s_drop = WeightedOneSampler(ScriptedRNG([0.0, 0.4]))
        s_drop.feed("a", 1)
        s_drop.feed("b", 0.5)
        self.assertEqual(s_drop.sample(), "a")


if __name__ == "__main__":
    unittest.main()
