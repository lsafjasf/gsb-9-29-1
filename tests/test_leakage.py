import unittest

from rolling_validation import (
    LeakageError,
    RollingSplitter,
    SplitConfig,
    assert_no_leakage,
    check_folds,
)


class TestLeakageDetection(unittest.TestCase):
    def test_clean_split_passes(self):
        ts = list(range(30))
        assert_no_leakage(ts, range(20), range(20, 25))

    def test_future_sample_in_train_reports_index_and_time(self):
        ts = list(range(30))
        train = list(range(20)) + [27]
        with self.assertRaises(LeakageError) as ctx:
            assert_no_leakage(ts, train, range(20, 25))
        msg = str(ctx.exception)
        self.assertIn("sample #27", msg)
        self.assertIn("t=27", msg)
        self.assertIn("t=20", msg)

    def test_duplicate_timestamp_at_boundary_is_leakage(self):
        ts = [0, 1, 2, 3, 3, 4, 5]
        with self.assertRaises(LeakageError) as ctx:
            assert_no_leakage(ts, train_idx=[0, 1, 2, 3], val_idx=[4, 5])
        self.assertIn("sample #3", str(ctx.exception))

    def test_shuffled_input_reports_original_index(self):
        ts = [50, 10, 30, 20, 40, 60, 70]
        train = [1, 3, 2, 0]
        val = [4, 5]
        with self.assertRaises(LeakageError) as ctx:
            assert_no_leakage(ts, train, val)
        msg = str(ctx.exception)
        self.assertIn("sample #0", msg)
        self.assertIn("t=50", msg)

    def test_multiple_violations_all_listed(self):
        ts = list(range(10))
        with self.assertRaises(LeakageError) as ctx:
            assert_no_leakage(ts, train_idx=[0, 6, 8], val_idx=[5, 9])
        self.assertEqual(len(ctx.exception.violations), 2)
        idxs = {v[0] for v in ctx.exception.violations}
        self.assertEqual(idxs, {6, 8})

    def test_empty_validation_rejected(self):
        with self.assertRaises(ValueError):
            assert_no_leakage([1, 2, 3], train_idx=[0], val_idx=[])

    def test_splitter_output_is_leak_free(self):
        ts = [3, 1, 2, 0, 5, 4, 7, 6, 9, 8] + list(range(10, 60))
        for mode, extra in (("expanding", {}),
                            ("sliding", {"window": 15})):
            cfg = SplitConfig(mode=mode, min_train=10, horizon=5, step=5,
                              embargo=1, **extra)
            folds = RollingSplitter(cfg).split(ts)
            self.assertTrue(folds)
            self.assertTrue(check_folds(ts, folds))

    def test_identical_timestamps_flagged(self):
        ts = [7] * 10
        with self.assertRaises(LeakageError):
            assert_no_leakage(ts, train_idx=range(5), val_idx=range(5, 8))


if __name__ == "__main__":
    unittest.main()
