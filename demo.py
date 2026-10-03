"""Demo: rolling validation report on a synthetic series.

Run: python3 demo.py
"""
import math
import random

from rolling_validation import (
    SplitConfig,
    RollingSplitter,
    check_folds,
    evaluate,
    nested_split,
    assert_nested_disjoint,
)


def main():
    random.seed(7)
    n = 120
    timestamps = list(range(n))
    y = [
        10 + 0.05 * t + 3 * math.sin(t / 8.0) + random.gauss(0, 0.5)
        for t in timestamps
    ]
    y_pred = [y[0]] + [y[t - 1] + random.gauss(0, 0.3) for t in range(1, n)]

    cfg = SplitConfig(mode="expanding", min_train=40, horizon=10, step=10,
                      embargo=2)
    folds = RollingSplitter(cfg).split(timestamps)
    check_folds(timestamps, folds)
    print("expanding split: %d folds (min_train=40, horizon=10, step=10, "
          "embargo=2)" % len(folds))
    for f in folds:
        print("  fold %d: train t=[%g..%g] (%d samples, %d embargo-dropped) "
              "| val t=[%g..%g] (%d samples)"
              % (f.fold_id, f.train_start, f.train_end, len(f.train_idx),
                 f.n_embargo_dropped, f.val_start, f.val_end,
                 len(f.val_idx)))

    for metric in ("mae", "rmse"):
        s = evaluate(folds, y, y_pred, metric)
        print("\n%s per fold: %s"
              % (metric.upper(),
                 ["None" if fm.value is None else round(fm.value, 4)
                  for fm in s.fold_metrics]))
        print("%s mean=%.4f std=%.4f 95%% CI=[%.4f, %.4f] (n=%d folds)"
              % (metric.upper(), s.mean, s.std, s.ci_low, s.ci_high,
                 s.n_folds))
        for w in s.warnings:
            print("  WARNING: %s" % w)

    outer_cfg = SplitConfig(mode="expanding", min_train=60, horizon=10,
                            step=10, embargo=2)
    inner_cfg = SplitConfig(mode="sliding", min_train=20, horizon=5, step=5,
                            embargo=1, window=30)
    nested = nested_split(timestamps, outer_cfg, inner_cfg)
    assert_nested_disjoint(nested)
    print("\nnested validation: %d outer folds, inner folds per outer: %s"
          % (len(nested), [len(nf.inner) for nf in nested]))
    print("nested disjointness assertions: PASS")


if __name__ == "__main__":
    main()
