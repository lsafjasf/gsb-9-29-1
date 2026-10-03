import unittest

from rolling_validation import (
    NestedFold,
    RollingSplitter,
    SplitConfig,
    assert_nested_disjoint,
    assert_no_leakage,
    nested_split,
)


def make_nested(n=100):
    ts = list(range(n))
    outer_cfg = SplitConfig(mode="expanding", min_train=40, horizon=10,
                            step=10, embargo=2)
    inner_cfg = SplitConfig(mode="sliding", min_train=15, horizon=5, step=5,
                            embargo=1, window=20)
    return ts, nested_split(ts, outer_cfg, inner_cfg)


class TestNestedSplit(unittest.TestCase):
    def test_structure(self):
        ts, nested = make_nested()
        self.assertEqual(len(nested), 5)
        for nf in nested:
            self.assertIsInstance(nf, NestedFold)
            self.assertTrue(nf.inner)

    def test_disjointness_assertion_passes(self):
        _, nested = make_nested()
        self.assertTrue(assert_nested_disjoint(nested))

    def test_inner_val_strictly_before_outer_val(self):
        ts, nested = make_nested()
        for nf in nested:
            outer_val = set(nf.outer.val_idx)
            for inner in nf.inner:
                self.assertLess(inner.val_end, nf.outer.val_start)
                self.assertFalse(outer_val & set(inner.val_idx))
                self.assertLess(max(ts[i] for i in inner.val_idx),
                                nf.outer.val_start)

    def test_inner_folds_themselves_leak_free(self):
        ts, nested = make_nested()
        for nf in nested:
            for inner in nf.inner:
                self.assertTrue(
                    assert_no_leakage(ts, inner.train_idx, inner.val_idx))

    def test_inner_val_windows_mutually_disjoint(self):
        _, nested = make_nested()
        for nf in nested:
            seen = set()
            for inner in nf.inner:
                self.assertFalse(seen & set(inner.val_idx))
                seen |= set(inner.val_idx)

    def test_inner_empty_when_history_too_short(self):
        ts = list(range(50))
        outer_cfg = SplitConfig(mode="expanding", min_train=20, horizon=10,
                                step=10)
        inner_cfg = SplitConfig(mode="expanding", min_train=40, horizon=5,
                                step=5)
        nested = nested_split(ts, outer_cfg, inner_cfg)
        self.assertEqual(len(nested[0].inner), 0)
        self.assertTrue(assert_nested_disjoint(nested))

    def test_detects_shared_validation_data(self):
        ts, nested = make_nested()
        nf = nested[1]
        poisoned_inner = nf.inner[0]
        object.__setattr__(poisoned_inner, "val_idx", nf.outer.val_idx)
        object.__setattr__(poisoned_inner, "val_end", nf.outer.val_end)
        bad = NestedFold(outer=nf.outer,
                         inner=(poisoned_inner,) + nf.inner[1:])
        with self.assertRaises(AssertionError):
            assert_nested_disjoint([bad])


if __name__ == "__main__":
    unittest.main()
