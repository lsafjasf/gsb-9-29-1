import unittest

from rolling_validation import (
    RollingSplitter,
    SplitConfig,
    SplitConfigError,
    check_folds,
)


class TestExpanding(unittest.TestCase):
    def test_fold_boundaries(self):
        ts = list(range(50))
        cfg = SplitConfig(mode="expanding", min_train=20, horizon=5, step=5,
                          embargo=2)
        folds = RollingSplitter(cfg).split(ts)
        self.assertEqual(len(folds), 5)
        f0 = folds[0]
        self.assertEqual(f0.train_idx, tuple(range(20)))
        self.assertEqual(f0.val_idx, tuple(range(22, 27)))
        self.assertEqual(f0.n_embargo_dropped, 2)
        last = folds[-1]
        self.assertEqual(last.val_idx, tuple(range(42, 47)))

    def test_step_gap_between_val_windows(self):
        ts = list(range(60))
        cfg = SplitConfig(mode="expanding", min_train=20, horizon=5, step=10)
        folds = RollingSplitter(cfg).split(ts)
        self.assertEqual(folds[0].val_idx, tuple(range(20, 25)))
        self.assertEqual(folds[1].val_idx, tuple(range(30, 35)))

    def test_step_smaller_than_horizon_rejected(self):
        with self.assertRaises(SplitConfigError):
            RollingSplitter(SplitConfig(horizon=10, step=5))

    def test_val_windows_never_overlap(self):
        ts = list(range(100))
        cfg = SplitConfig(mode="expanding", min_train=10, horizon=7, step=7)
        folds = RollingSplitter(cfg).split(ts)
        seen = set()
        for f in folds:
            self.assertFalse(seen & set(f.val_idx))
            seen |= set(f.val_idx)


class TestSliding(unittest.TestCase):
    def test_train_window_fixed_size(self):
        ts = list(range(50))
        cfg = SplitConfig(mode="sliding", min_train=20, horizon=5, step=5,
                          embargo=2, window=10)
        folds = RollingSplitter(cfg).split(ts)
        for f in folds:
            self.assertEqual(len(f.train_idx), 10)
        self.assertEqual(folds[0].train_idx, tuple(range(10, 20)))
        self.assertEqual(folds[1].train_idx, tuple(range(15, 25)))

    def test_sliding_requires_window(self):
        with self.assertRaises(SplitConfigError):
            RollingSplitter(SplitConfig(mode="sliding"))

    def test_train_window_capped_at_start(self):
        ts = list(range(30))
        cfg = SplitConfig(mode="sliding", min_train=5, horizon=5, step=5,
                          window=100)
        folds = RollingSplitter(cfg).split(ts)
        self.assertEqual(folds[0].train_idx, tuple(range(5)))


class TestOverlapPurge(unittest.TestCase):
    def test_duplicate_timestamps_purged_from_train(self):
        ts = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9]
        cfg = SplitConfig(mode="expanding", min_train=4, horizon=2, step=2)
        folds = RollingSplitter(cfg).split(ts)
        for f in folds:
            train_ts = [ts[i] for i in f.train_idx]
            self.assertTrue(all(t < f.val_start for t in train_ts))
        first = folds[0]
        self.assertEqual(first.val_start, 2)
        self.assertEqual(first.n_overlap_dropped, 0)
        dup_fold = folds[1]
        self.assertEqual(dup_fold.val_start, 3)
        self.assertEqual(dup_fold.n_overlap_dropped, 0)
        self.assertNotIn(3, [ts[i] for i in dup_fold.train_idx])

    def test_duplicate_straddling_cut_is_dropped(self):
        ts = [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10]
        cfg = SplitConfig(mode="expanding", min_train=4, horizon=2, step=2)
        folds = RollingSplitter(cfg).split(ts)
        f0 = folds[0]
        self.assertEqual(f0.val_start, 3)
        self.assertEqual(f0.n_overlap_dropped, 1)
        self.assertEqual([ts[i] for i in f0.train_idx], [0, 1, 2])

    def test_shuffled_input_yields_time_ordered_folds(self):
        ts = [9, 3, 0, 7, 1, 8, 2, 6, 4, 5] + list(range(10, 40))
        cfg = SplitConfig(mode="expanding", min_train=10, horizon=5, step=5)
        folds = RollingSplitter(cfg).split(ts)
        self.assertTrue(check_folds(ts, folds))
        for f in folds:
            train_ts = sorted(ts[i] for i in f.train_idx)
            self.assertEqual(train_ts, list(range(len(train_ts))))


class TestEdgeCases(unittest.TestCase):
    def test_series_too_short_returns_no_folds(self):
        ts = list(range(10))
        cfg = SplitConfig(mode="expanding", min_train=20, horizon=5)
        self.assertEqual(RollingSplitter(cfg).split(ts), [])

    def test_exactly_one_fold(self):
        ts = list(range(25))
        cfg = SplitConfig(mode="expanding", min_train=20, horizon=5, step=5)
        folds = RollingSplitter(cfg).split(ts)
        self.assertEqual(len(folds), 1)

    def test_all_identical_timestamps_no_folds(self):
        ts = [5] * 30
        cfg = SplitConfig(mode="expanding", min_train=10, horizon=5)
        self.assertEqual(RollingSplitter(cfg).split(ts), [])

    def test_empty_series(self):
        cfg = SplitConfig(mode="expanding", min_train=1, horizon=1)
        self.assertEqual(RollingSplitter(cfg).split([]), [])

    def test_nan_timestamp_rejected(self):
        cfg = SplitConfig(mode="expanding", min_train=1, horizon=1)
        with self.assertRaises(SplitConfigError):
            RollingSplitter(cfg).split([0.0, float("nan"), 2.0])


if __name__ == "__main__":
    unittest.main()
