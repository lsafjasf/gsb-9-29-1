"""演示：合成序列上的扩张/滑动滚动评估，输出每折指标与折间置信区间。

运行：python3 demo.py
"""

import math
import random

from rolling_validation import RollingValidator, evaluate


def make_series(n=120, seed=7):
    rng = random.Random(seed)
    return [10 + 0.05 * t + 2 * math.sin(t / 6.0) + rng.gauss(0, 0.5)
            for t in range(n)]


def naive_forecaster(y):
    """朴素预测：用训练集最后一个值外推。"""
    def _f(train_idx, val_idx):
        return [y[train_idx[-1]]] * len(val_idx)
    return _f


def main():
    y = make_series()
    timestamps = list(range(len(y)))

    for mode, kwargs in [
        ("expanding", {}),
        ("sliding", {"window": 40}),
    ]:
        validator = RollingValidator(
            mode=mode, min_train=30, horizon=10, gap=2, embargo=3, step=15, **kwargs)
        result = evaluate(timestamps, y, naive_forecaster(y),
                          validator=validator,
                          metrics=("mae", "rmse", "mape"))
        print("=" * 64)
        print(result.pretty())
        print()


if __name__ == "__main__":
    main()
