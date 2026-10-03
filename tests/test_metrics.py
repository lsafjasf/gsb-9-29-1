"""指标与区间：缺失处理、单折/全缺失降级、置信区间计算。"""

import math
import unittest

from rolling_validation import (
    RollingValidator, evaluate, mae, rmse, mape, summarize,
)


def const_forecaster(value):
    return lambda train_idx, val_idx: [value] * len(val_idx)


class TestMetricFunctions(unittest.TestCase):
    def test_basic_metrics(self):
        self.assertEqual(mae([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(rmse([0, 0], [3, 4]), math.sqrt(12.5))
        self.assertAlmostEqual(mape([1, 2], [2, 3]), 75.0)

    def test_missing_pairs_ignored(self):
        self.assertEqual(mae([1, None, 3], [1, 999, 3]), 0.0)
        self.assertEqual(mae([1, float("nan")], [1, 2]), 0.0)

    def test_mape_zero_true_returns_none(self):
        self.assertIsNone(mape([0, 0], [1, 2]))
        self.assertAlmostEqual(mape([0, 2], [9, 3]), 50.0)

    def test_summarize_single_and_empty(self):
        s = summarize([2.5])
        self.assertEqual(s["mean"], 2.5)
        self.assertIsNone(s["std"])
        self.assertIsNone(s["ci"])
        s0 = summarize([None])
        self.assertEqual(s0["n"], 0)
        self.assertIsNone(s0["mean"])

    def test_ci_contains_mean_and_width_shrinks(self):
        s2 = summarize([1.0, 3.0])
        lo, hi = s2["ci"]
        self.assertLess(lo, 2.0 < hi)
        many = [float(i) for i in range(1, 21)]
        s20 = summarize(many)
        self.assertLess(s20["ci"][1] - s20["ci"][0], hi - lo)


class TestEvaluationDegradation(unittest.TestCase):
    def test_single_fold_no_ci_with_warning(self):
        ts = list(range(30))
        y = [float(t) for t in ts]
        v = RollingValidator(min_train=20, horizon=5)
        result = evaluate(ts, y, const_forecaster(1.0), validator=v)
        self.assertEqual(len(result.folds), 1)
        self.assertIsNone(result.summary["mae"]["ci"])
        self.assertIsNone(result.summary["mae"]["std"])
        self.assertTrue(any("单折" in w for w in result.warnings))

    def test_partial_missing_fold(self):
        ts = list(range(30))
        y = [float(t) for t in ts]
        y[25] = None  # 落在首折验证段 [24,29) 中
        v = RollingValidator(min_train=20, horizon=5)
        result = evaluate(ts, y, const_forecaster(0.0), validator=v)
        self.assertTrue(any(f.n_missing_val >= 1 for f in result.folds))
        self.assertIsNotNone(result.summary["mae"]["mean"])

    def test_to_dict_serializable(self):
        ts = list(range(30))
        v = RollingValidator(min_train=20, horizon=5)
        result = evaluate(ts, [1.0] * 30, const_forecaster(1.0),
                          validator=v)
        d = result.to_dict()
        self.assertEqual(d["folds"][0]["metrics"]["mae"], 0.0)
        self.assertIn("summary", d)


if __name__ == "__main__":
    unittest.main()
