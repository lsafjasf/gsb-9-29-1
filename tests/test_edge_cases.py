"""边界用例：单折、序列过短、全部缺失、极端不均衡、时间戳全相同。"""

import unittest

from rolling_validation import (
    RollingValidator, evaluate, LeakageError, InsufficientDataError,
)


def zero_forecaster(train_idx, val_idx):
    return [0.0] * len(val_idx)


class TestEdgeCases(unittest.TestCase):
    def test_single_fold_exact_fit(self):
        # 恰好一折：need = min_train + max(gap,horizon-1) + horizon = 29
        ts = list(range(29))
        v = RollingValidator(min_train=20, horizon=5)
        folds = v.split(ts)
        self.assertEqual(len(folds), 1)

    def test_series_too_short_raises(self):
        v = RollingValidator(min_train=20, horizon=5, gap=2)
        with self.assertRaises(InsufficientDataError) as ctx:
            v.split(list(range(10)))
        self.assertIn("序列过短", str(ctx.exception))

    def test_empty_raises(self):
        with self.assertRaises(InsufficientDataError):
            RollingValidator().split([])

    def test_all_missing_values(self):
        ts = list(range(40))
        y = [float("nan")] * 40
        v = RollingValidator(min_train=20, horizon=5)
        result = evaluate(ts, y, zero_forecaster, validator=v)
        self.assertTrue(result.folds)
        for f in result.folds:
            self.assertEqual(f.n_missing_val, f.n_val)
            self.assertTrue(all(x is None for x in f.metrics.values()))
        self.assertTrue(all(s["mean"] is None
                            for s in result.summary.values()))
        self.assertTrue(any("全部缺失" in w for w in result.warnings))

    def test_extreme_imbalance_near_all_zeros(self):
        # 99% 真值为 0：MAPE 无定义但 MAE/RMSE 必须照常产出
        ts = list(range(60))
        y = [0.0] * 59 + [100.0]
        v = RollingValidator(min_train=20, horizon=5)
        result = evaluate(ts, y, zero_forecaster, validator=v,
                          metrics=("mae", "rmse", "mape"))
        # 某折含唯一非零点时 MAPE 仍有定义；其余折 MAPE 为 None
        mape_vals = [f.metrics["mape"] for f in result.folds]
        self.assertTrue(any(x is None for x in mape_vals))
        self.assertTrue(any("无法计算" in w
                            for f in result.folds for w in f.warnings))
        self.assertIsNotNone(result.summary["mae"]["mean"])
        # 全零折 MAE 恰为 0
        self.assertTrue(any(f.metrics["mae"] == 0.0 for f in result.folds))

    def test_identical_timestamps_flagged_by_default(self):
        # 所有时间戳相同：默认严格模式必须报泄漏（指出编号与时间）
        ts = [100] * 40
        v = RollingValidator(min_train=20, horizon=5)
        with self.assertRaises(LeakageError) as ctx:
            v.split(ts)
        # 注：分离量 4 + horizon 5，首折 v=24，边界两侧时间戳同为 100
        msg = str(ctx.exception)
        self.assertIn("100", msg)
        self.assertRegex(msg, r"#\d+")

    def test_identical_timestamps_allow_ties(self):
        # 显式允许并列时退化为按输入顺序切分（用户自担风险）
        ts = [0] * 40
        v = RollingValidator(min_train=20, horizon=5, allow_ties=True)
        folds = v.split(ts)
        self.assertTrue(folds)
        for f in folds:
            self.assertGreaterEqual(len(f.train_indices), 20)

    def test_validation_windows_within_bounds(self):
        # 末折必须完整落在数据范围内（不产生残缺尾折）
        ts = list(range(40))
        v = RollingValidator(min_train=20, horizon=10)
        folds = v.split(ts)
        for f in folds:
            self.assertEqual(len(f.val_indices), 10)
            self.assertLess(max(f.val_indices), 40)


if __name__ == "__main__":
    unittest.main()
