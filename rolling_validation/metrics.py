"""Per-fold metrics, cross-fold dispersion and confidence intervals.

Degradation policy (never silently drop data):
* missing targets/predictions (None or NaN) are skipped pairwise and counted;
* a fold with no usable pairs gets value=None and is excluded from the
  aggregate, with a warning;
* if every fold is unusable, mean/std/CI are None plus a warning;
* a single fold yields a point estimate only (std and CI are None) plus a
  warning, because dispersion is undefined;
* fewer than 5 usable folds triggers a "small sample" warning; the CI then
  uses Student's t quantiles, which are wide but honest.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_crit_95(df: int) -> float:
    if df >= 30:
        return 1.96
    return _T_CRIT_95[max(df, 1)]


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


@dataclass(frozen=True)
class FoldMetric:
    fold_id: int
    value: Optional[float]
    n: int
    n_missing: int


@dataclass(frozen=True)
class MetricSummary:
    name: str
    fold_metrics: Tuple[FoldMetric, ...]
    mean: Optional[float]
    std: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_folds: int
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def _paired(
    y_true: Sequence, y_pred: Sequence
) -> Tuple[List[Tuple[float, float]], int]:
    pairs, n_missing = [], 0
    for yt, yp in zip(y_true, y_pred):
        if _is_missing(yt) or _is_missing(yp):
            n_missing += 1
        else:
            pairs.append((yt, yp))
    return pairs, n_missing


def mae(y_true, y_pred) -> Tuple[Optional[float], int, int]:
    pairs, n_missing = _paired(y_true, y_pred)
    if not pairs:
        return None, 0, n_missing
    value = sum(abs(a - b) for a, b in pairs) / len(pairs)
    return value, len(pairs), n_missing


def mse(y_true, y_pred) -> Tuple[Optional[float], int, int]:
    pairs, n_missing = _paired(y_true, y_pred)
    if not pairs:
        return None, 0, n_missing
    value = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
    return value, len(pairs), n_missing


def rmse(y_true, y_pred) -> Tuple[Optional[float], int, int]:
    value, n, n_missing = mse(y_true, y_pred)
    if value is None:
        return None, n, n_missing
    return math.sqrt(value), n, n_missing


def accuracy(y_true, y_pred) -> Tuple[Optional[float], int, int]:
    pairs, n_missing = _paired(y_true, y_pred)
    if not pairs:
        return None, 0, n_missing
    value = sum(1.0 for a, b in pairs if a == b) / len(pairs)
    return value, len(pairs), n_missing


_METRICS: dict = {"mae": mae, "mse": mse, "rmse": rmse, "accuracy": accuracy}


def summarize(name: str, fold_metrics: Sequence[FoldMetric]) -> MetricSummary:
    warnings: List[str] = []
    values = [fm.value for fm in fold_metrics if fm.value is not None]
    total_missing = sum(fm.n_missing for fm in fold_metrics)
    if total_missing:
        warnings.append(
            "%d missing target/prediction pair(s) skipped" % total_missing
        )
    dropped = len(fold_metrics) - len(values)
    if dropped:
        warnings.append(
            "%d fold(s) had no usable data and were excluded" % dropped
        )
    n = len(values)
    if n == 0:
        warnings.append("no usable folds: metrics unavailable")
        return MetricSummary(
            name, tuple(fold_metrics), None, None, None, None, 0, tuple(warnings)
        )
    mean = statistics.fmean(values)
    if n == 1:
        warnings.append(
            "single fold: variance and confidence interval unavailable"
        )
        return MetricSummary(
            name, tuple(fold_metrics), mean, None, None, None, 1, tuple(warnings)
        )
    std = statistics.stdev(values)
    if n < 5:
        warnings.append(
            "small fold count (n=%d): confidence interval is unreliable" % n
        )
    half = _t_crit_95(n - 1) * std / math.sqrt(n)
    return MetricSummary(
        name, tuple(fold_metrics), mean, std, mean - half, mean + half,
        n, tuple(warnings),
    )


def evaluate(
    folds,
    y_true: Sequence,
    y_pred: Sequence,
    metric: str = "mae",
) -> MetricSummary:
    if metric not in _METRICS:
        raise ValueError("unknown metric %r (have: %s)"
                         % (metric, sorted(_METRICS)))
    fn = _METRICS[metric]
    fold_metrics = []
    for f in folds:
        yt = [y_true[i] for i in f.val_idx]
        yp = [y_pred[i] for i in f.val_idx]
        value, n, n_missing = fn(yt, yp)
        fold_metrics.append(FoldMetric(f.fold_id, value, n, n_missing))
    return summarize(metric, fold_metrics)
