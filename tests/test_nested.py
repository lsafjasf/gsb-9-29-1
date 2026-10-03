"""嵌套验证断言：内外层验证零重叠、时间隔离、按目标选参。"""

import unittest

from rolling_validation import (
    RollingValidator, nested_splits, assert_nested_integrity, run_nested,
    Fold,
)


class TestNested(unittest.TestCase):
    def setUp(self):
        self.ts = list(range(200))

    def test_integrity_holds_for_every_outer_fold(self):
        outer = RollingValidator(mode="expanding", min_train=40,
                                 horizon=10, step=20)
        inner = RollingValidator(mode="sliding", min_train=20, window=30,
                                 horizon=5, step=10)
        nf_list = nested_splits(self.ts, outer, inner)
        self.assertTrue(nf_list)
        for nf in nf_list:
            self.assertTrue(assert_nested_integrity(nf, self.ts))
            self.assertTrue(nf.inner)
            # 内层验证时间戳全部早于外层验证起点
            for f in nf.inner:
                self.assertLess(f.val_end_ts, nf.outer.val_start_ts)

    def test_tampered_overlap_is_rejected(self):
        outer = RollingValidator(min_train=40, horizon=10, step=40)
        inner = RollingValidator(min_train=20, horizon=5, step=20)
        nf = nested_splits(self.ts, outer, inner)[0]
        # 人为把一个外层验证样本塞进内层验证集，模拟共用验证段
        bad_inner_fold = Fold(
            fold_id=0,
            train_indices=nf.inner[0].train_indices,
            val_indices=tuple(list(nf.inner[0].val_indices[:-1])
                              + [nf.outer.val_indices[0]]),
            train_end_ts=nf.inner[0].train_end_ts,
            val_start_ts=nf.inner[0].val_start_ts,
            val_end_ts=nf.inner[0].val_end_ts,
        )
        from rolling_validation import NestedFold
        bad = NestedFold(outer=nf.outer,
                         inner=(bad_inner_fold,) + nf.inner[1:])
        with self.assertRaises(AssertionError) as ctx:
            assert_nested_integrity(bad, self.ts)
        self.assertIn("共用", str(ctx.exception))

    def test_run_nested_per_target_selection(self):
        outer = RollingValidator(min_train=40, horizon=10, step=40)
        inner = RollingValidator(min_train=20, horizon=5, step=20)
        targets = {"a": [float(t % 3) for t in range(200)],
                   "b": [float(t % 7) for t in range(200)]}

        def score(target, params, train_idx, val_idx, y):
            # params 是预测常数；简单起见返回 val 段均方误差
            c = params
            return sum((y[i] - c) ** 2 for i in val_idx) / len(val_idx)

        records = run_nested(self.ts, targets, score, outer, inner,
                             param_grid=[0.0, 1.0, 3.0])
        self.assertEqual({r["target"] for r in records}, {"a", "b"})
        for r in records:
            self.assertIsNotNone(r["best_param"])
            self.assertIn(r["best_param"], [0.0, 1.0, 3.0])
            # 内层选参只用到内层分数
            self.assertEqual(len(r["inner_scores"]), 3)


if __name__ == "__main__":
    unittest.main()
