"""泄漏检测用例：重复时间戳 + 乱序输入，报错须含样本编号与时间。"""

import re
import unittest

from rolling_validation import (
    RollingValidator, LeakageError, check_leakage,
)


class TestLeakageDetection(unittest.TestCase):
    def test_future_train_sample_flagged_with_id_and_time(self):
        # 乱序 + 重复时间戳：样本 #2 时间戳 15 晚于验证起点 10
        timestamps = [3, 7, 15, 10, 10, 12, 5]
        train_idx = [0, 1, 2, 6]      # 含样本 #2 (t=15)
        val_idx = [3, 4, 5]           # 验证起点 t=10（含重复时间戳）
        with self.assertRaises(LeakageError) as ctx:
            check_leakage(train_idx, val_idx, timestamps)
        msg = str(ctx.exception)
        self.assertIn("#2", msg)          # 具体样本编号
        self.assertIn("15", msg)          # 具体时间
        self.assertIn("10", msg)          # 验证起点

    def test_multiple_offenders_all_listed(self):
        timestamps = [1, 2, 99, 50, 10, 11]
        with self.assertRaises(LeakageError) as ctx:
            check_leakage([0, 1, 2, 3], [4, 5], timestamps)
        msg = str(ctx.exception)
        self.assertIn("#2", msg)
        self.assertIn("#3", msg)
        self.assertIn("99", msg)
        self.assertIn("50", msg)

    def test_tie_at_boundary_flagged_by_default(self):
        # 训练样本与验证起点时间戳相同：默认视为泄漏
        timestamps = [1, 2, 5, 5, 6]
        with self.assertRaises(LeakageError):
            check_leakage([0, 1, 2], [3, 4], timestamps)
        # allow_ties=True 时放行
        self.assertTrue(check_leakage([0, 1, 2], [3, 4], timestamps,
                                      allow_ties=True))

    def test_clean_split_passes(self):
        timestamps = [5, 1, 3, 9, 7]  # 乱序但时间上训练早于验证
        self.assertTrue(check_leakage([1, 2, 0], [4, 3], timestamps))

    def test_validator_self_check_on_duplicates(self):
        # 边界两侧存在重复时间戳时，切分器自检必须拦截：
        # min_train=12 使首折训练末端越过 t=1 这一组，验证起点也在 t=1
        ts = [0] * 10 + [1] * 10
        v = RollingValidator(min_train=12, horizon=3, gap=0)
        with self.assertRaises(LeakageError):
            v.split(ts)
        # 允许并列后可正常切分
        v_ok = RollingValidator(min_train=12, horizon=3, gap=0,
                                allow_ties=True)
        self.assertTrue(v_ok.split(ts))


if __name__ == "__main__":
    unittest.main()
