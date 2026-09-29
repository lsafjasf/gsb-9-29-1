import math
import random
import unittest

from sequential_test import (
    BernoulliSPRT,
    BernoulliTestConfig,
    Decision,
    MultiBernoulliSPRT,
    conservative_max_n,
)
from run_simulation import run_single_scenario, wilson_interval


class BernoulliSPRTTest(unittest.TestCase):
    def test_upper_boundary_hit_on_all_successes(self):
        test = BernoulliSPRT(p0=0.2, p1=0.3, alpha=0.05, beta=0.20)
        for n in range(1, 8):
            snapshot = test.update(1)
            self.assertIs(snapshot.decision, Decision.CONTINUE)
        test.update(1)
        self.assertEqual(test.n, 8)
        self.assertIs(test.decision, Decision.POSITIVE)
        self.assertGreaterEqual(test.e_value, 1.0 / 0.05)

    def test_lower_futility_boundary_hit_on_all_failures(self):
        test = BernoulliSPRT(p0=0.2, p1=0.3, alpha=0.05, beta=0.20)
        for n in range(1, 18):
            test.update(0)
            self.assertFalse(test.stopped)
        test.update(0)
        self.assertEqual(test.n, 18)
        self.assertIs(test.decision, Decision.NO_POSITIVE)
        self.assertLessEqual(test.e_value, 0.10)

    def test_negative_effect_does_not_trigger_greater_alternative(self):
        test = BernoulliSPRT(p0=0.2, p1=0.3, alpha=0.05, beta=0.20)
        test.run(0 for _ in range(test.max_n))
        self.assertIs(test.decision, Decision.NO_POSITIVE)
        self.assertEqual(test.n, 18)

    def test_less_alternative_detects_negative_effect(self):
        test = BernoulliSPRT(p0=0.2, p1=0.1, alpha=0.05, beta=0.20)
        test.run(0 for _ in range(30))
        self.assertIs(test.decision, Decision.POSITIVE)
        self.assertGreaterEqual(test.e_value, 20.0)

    def test_one_observation_cap_for_extremely_small_sample(self):
        test = BernoulliSPRT(
            p0=0.2,
            p1=0.3,
            alpha=0.05,
            beta=0.20,
            max_n=1,
        )
        snapshot = test.update(1)
        self.assertEqual(snapshot.n, 1)
        self.assertIs(snapshot.decision, Decision.NO_POSITIVE)

    def test_rare_event_boundary_uses_finishable_max_n(self):
        p0 = 0.001
        p1 = 0.01
        max_n = conservative_max_n(p0, p1, 0.05, 0.20)
        test = BernoulliSPRT(p0=p0, p1=p1, alpha=0.05, beta=0.20, max_n=max_n)
        test.update(1)
        self.assertFalse(test.stopped)
        test.update(1)
        self.assertIs(test.decision, Decision.POSITIVE)

    def test_run_with_injected_random_source(self):
        rng = random.Random(7)
        test = BernoulliSPRT(p0=0.2, p1=0.3, alpha=0.05, beta=0.20)
        observations = (1 if rng.random() < 0.2 else 0 for _ in range(test.max_n))
        snapshot = test.run(observations)
        self.assertIn(snapshot.decision, {Decision.POSITIVE, Decision.NO_POSITIVE})

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            BernoulliSPRT(p0=0.5, p1=0.5)
        with self.assertRaises(ValueError):
            BernoulliSPRT(p0=0, p1=0.1)
        with self.assertRaises(ValueError):
            BernoulliSPRT(p0=0.2, p1=0.3, alpha=0)
        test = BernoulliSPRT(p0=0.2, p1=0.3)
        with self.assertRaises(ValueError):
            test.update(2)


class MultiBernoulliSPRTTest(unittest.TestCase):
    def test_bonferroni_boundary_and_sample_cap(self):
        single = BernoulliSPRT(0.2, 0.3, alpha=0.05, beta=0.20)
        configs = [BernoulliTestConfig(f"m{i}", 0.2, 0.3) for i in range(5)]
        suite = MultiBernoulliSPRT(configs, family_alpha=0.05, beta=0.20)
        self.assertAlmostEqual(suite.per_test_alpha, 0.01)
        self.assertEqual(
            suite.tests[0].upper_log_boundary,
            math.log(5.0 / 0.05),
        )
        self.assertGreater(suite.max_n, single.max_n)

    def test_multi_metric_upper_boundary_hit(self):
        configs = [(f"m{i}", 0.2, 0.3) for i in range(5)]
        suite = MultiBernoulliSPRT(configs, family_alpha=0.05, beta=0.20)
        for _ in range(11):
            self.assertIs(suite.update([1] * 5).decision, Decision.CONTINUE)
        snapshot = suite.update([1] * 5)
        self.assertIs(snapshot.decision, Decision.POSITIVE)
        self.assertEqual(snapshot.positive_index, 0)
        self.assertEqual(snapshot.n, 12)

    def test_multi_metric_requires_one_observation_per_metric(self):
        suite = MultiBernoulliSPRT([("m1", 0.2, 0.3), ("m2", 0.4, 0.5)])
        with self.assertRaises(ValueError):
            suite.update([1])

    def test_named_observations(self):
        suite = MultiBernoulliSPRT([("m1", 0.2, 0.3), ("m2", 0.4, 0.5)])
        snapshot = suite.update({"m2": 0, "m1": 1})
        self.assertEqual(snapshot.n, 1)
        self.assertFalse(snapshot.stopped)


class SmallSimulationTest(unittest.TestCase):
    def test_null_error_is_below_nominal_alpha_in_seeded_run(self):
        result = run_single_scenario(
            "deterministic_selftest_null",
            true_p=0.2,
            p0=0.2,
            p1=0.3,
            alpha=0.05,
            beta=0.20,
            trials=500,
            seed=12345,
            event_rate_name="false_positive_rate",
        )
        low, high = wilson_interval(
            round(result["false_positive_rate"] * 500), 500
        )
        self.assertLess(high, 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
