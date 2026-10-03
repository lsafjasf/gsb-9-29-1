import math
import unittest

from rolling_validation import (
    FoldMetric,
    RollingSplitter,
    SplitConfig,
    accuracy,
    evaluate,
    mae,
    summarize,
)


def fm(fid, value, n=5, n_missing=0):
    return FoldMetric(fid, value, n, n_missing)


class TestMetricFunctions(unittest.TestCase):
    def test_mae_known_value(self):
        value, n, n_missing = mae([1, 2, 3], [1, 4, 1])
        self.assertAlmostEqual(value, 4 / 3)
        self.assertEqual((n, n_missing), (3, 0))

    def test_missing_pairs_skipped(self):
        value, n, n_missing = mae([1, None, 3, float("nan")], [1, 2, 5, 1])
        self.assertAlmostEqual(value, 1.0)
        self.assertEqual((n, n_missing), (2, 2))

    def test_all_missing_returns_none(self):
        value, n, n_missing = mae([None, None], [1, 2])
        self.assertIsNone(value)
        self.assertEqual((n, n_missing), (0, 2))

    def test_accuracy_extreme_imbalance(self):
        y_true = [0] * 99 + [1]
        y_pred = [0] * 100
        value, n, _ = accuracy(y_true, y_pred)
        self.assertAlmostEqual(value, 0.99)
        self.assertEqual(n, 100)


class TestSummarize(unittest.TestCase):
    def test_mean_std_ci_known_values(self):
        s = summarize("mae", [fm(i, v) for i, v in enumerate([1, 2, 3, 4])])
        self.assertAlmostEqual(s.mean, 2.5)
        self.assertAlmostEqual(s.std, math.sqrt(5 / 3))
        half = 3.182 * s.std / 2
        self.assertAlmostEqual(s.ci_low, 2.5 - half)
        self.assertAlmostEqual(s.ci_high, 2.5 + half)
        self.assertEqual(s.n_folds, 4)
        self.assertTrue(any("small fold count" in w for w in s.warnings))

    def test_single_fold_degrades_to_point_estimate(self):
        s = summarize("mae", [fm(0, 1.5)])
        self.assertEqual(s.mean, 1.5)
        self.assertIsNone(s.std)
        self.assertIsNone(s.ci_low)
        self.assertIsNone(s.ci_high)
        self.assertTrue(any("single fold" in w for w in s.warnings))

    def test_all_folds_missing(self):
        s = summarize("mae", [fm(0, None, n=0, n_missing=5),
                              fm(1, None, n=0, n_missing=5)])
        self.assertIsNone(s.mean)
        self.assertIsNone(s.ci_low)
        self.assertEqual(s.n_folds, 0)
        self.assertTrue(any("no usable folds" in w for w in s.warnings))
        self.assertTrue(any("missing" in w for w in s.warnings))

    def test_partially_missing_fold_excluded_with_warning(self):
        s = summarize("mae", [fm(0, 1.0), fm(1, None, n=0, n_missing=5),
                              fm(2, 3.0)])
        self.assertAlmostEqual(s.mean, 2.0)
        self.assertEqual(s.n_folds, 2)
        self.assertTrue(any("excluded" in w for w in s.warnings))

    def test_many_folds_no_small_sample_warning(self):
        s = summarize("mae", [fm(i, float(i)) for i in range(10)])
        self.assertFalse(any("small fold count" in w for w in s.warnings))
        self.assertIsNotNone(s.ci_low)


class TestEvaluate(unittest.TestCase):
    def test_evaluate_over_folds(self):
        n = 40
        ts = list(range(n))
        y = [float(i) for i in range(n)]
        y_pred = [float(i) + 1 for i in range(n)]
        cfg = SplitConfig(mode="expanding", min_train=20, horizon=5, step=5)
        folds = RollingSplitter(cfg).split(ts)
        s = evaluate(folds, y, y_pred, "mae")
        self.assertEqual(len(s.fold_metrics), 4)
        for fold_metric in s.fold_metrics:
            self.assertAlmostEqual(fold_metric.value, 1.0)
        self.assertAlmostEqual(s.mean, 1.0)
        self.assertAlmostEqual(s.std, 0.0)
        self.assertAlmostEqual(s.ci_low, 1.0)

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([], [], [], "r2")


if __name__ == "__main__":
    unittest.main()
