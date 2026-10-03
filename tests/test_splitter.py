"""切分器结构测试：扩张/滑动、gap/embargo/purge、乱序等价性。"""

import random
import unittest

from rolling_validation import (
    RollingValidator, InsufficientDataError,
)


class TestExpanding(unittest.TestCase):
    def test_fold_structure_and_monotonic_time(self):
        ts = list(range(100))
        v = RollingValidator(mode="expanding", min_train=30, horizon=10, gap=2)
        folds = v.split(ts)
        self.assertTrue(folds)
        prev_val_end = -1
        for f in folds:
            # 训练集时间全部早于验证起点
            self.assertLess(f.train_end_ts, f.val_start_ts)
            # 隔离量 max(gap=2, horizon-1=9)=9：训练末端到验证起点相距 10 个位置
            self.assertGreaterEqual(f.val_start_ts - f.train_end_ts, 10)
            # 验证集不重叠且时间递增
            self.assertGreater(f.val_start_ts, prev_val_end)
            prev_val_end = f.val_end_ts
            self.assertEqual(len(f.val_indices), 10)
        # 扩张窗口：训练集单调变长
        sizes = [len(f.train_indices) for f in folds]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])

    def test_sliding_window_constant_size(self):
        ts = list(range(120))
        v = RollingValidator(mode="sliding", min_train=20, window=40,
                             horizon=10)
        folds = v.split(ts)
        sizes = [len(f.train_indices) for f in folds]
        # 窗口锚定在净化截止处：每折净化后训练量都应恰好等于 window
        self.assertTrue(all(s == 40 for s in sizes))

    def test_purge_removes_overlapping_label_windows(self):
        # horizon=5, gap=0：训练样本 i 的标签窗口 [i, i+5) 与验证区间
        # [v, v+5) 重叠者（i > v-5）必须被剔除
        ts = list(range(60))
        v = RollingValidator(mode="expanding", min_train=10, horizon=5, gap=0)
        for f in v.split(ts):
            val_start = min(f.val_indices)
            for i in f.train_indices:
                self.assertLessEqual(i + 5, val_start)
            # 被剔除的正是 (v-horizon, v) 这段重叠样本
            for i in f.purged_indices:
                self.assertGreater(i + 5, val_start)
                self.assertLess(i, val_start)

    def test_embargo_excludes_post_validation_band(self):
        # step(15) > horizon(5)+gap(0) 时，上一折验证区之后的隔离带样本
        # 会落入下一折训练窗，必须被剔除
        ts = list(range(120))
        v = RollingValidator(mode="expanding", min_train=20, horizon=5,
                             embargo=4, step=15)
        folds = v.split(ts)
        self.assertGreaterEqual(len(folds), 2)
        prev = folds[0]
        nxt = folds[1]
        band = set(range(max(prev.val_indices) + 1,
                         max(prev.val_indices) + 5))  # 4 个隔离样本
        self.assertEqual(len(band), 4)
        self.assertFalse(band & set(nxt.train_indices))
        self.assertTrue(band <= set(nxt.embargoed_indices))

    def test_shuffled_input_equivalent(self):
        ts = list(range(80))
        idx = list(range(80))
        random.Random(3).shuffle(idx)
        shuffled_ts = [ts[i] for i in idx]  # 乱序输入
        v = RollingValidator(mode="expanding", min_train=20, horizon=8)
        folds_sorted = v.split(ts)
        folds_shuffled = v.split(shuffled_ts)
        self.assertEqual(len(folds_sorted), len(folds_shuffled))
        for a, b in zip(folds_sorted, folds_shuffled):
            # 乱序输入的折，映射回时间值集合后必须与顺序输入一致
            self.assertEqual(sorted(ts[i] for i in a.train_indices),
                             sorted(shuffled_ts[i] for i in b.train_indices))
            self.assertEqual(sorted(ts[i] for i in a.val_indices),
                             sorted(shuffled_ts[i] for i in b.val_indices))
            self.assertEqual(a.val_start_ts, b.val_start_ts)

    def test_step_controls_fold_count(self):
        ts = list(range(100))
        v1 = RollingValidator(min_train=20, horizon=10)
        v2 = RollingValidator(min_train=20, horizon=10, step=20)
        self.assertGreater(len(v1.split(ts)), len(v2.split(ts)))


if __name__ == "__main__":
    unittest.main()
