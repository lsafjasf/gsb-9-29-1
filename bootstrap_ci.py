"""Bootstrap confidence intervals: percentile, BC, and BCa methods.

Pure standard library (Python 3.8+). The random source is injectable:
every resampling chunk is driven by ``rng_factory(chunk_seed)`` where
``chunk_seed`` is derived deterministically from the user-supplied
``seed`` and the chunk index.  Because each chunk's randomness depends
only on ``(seed, chunk_index)`` and ``chunk_size``, splitting the same
number of resamples into chunks and running them in parallel, then
concatenating the chunks in order, yields exactly the same replicates
as a serial run with the same seed and chunk_size.
"""

from __future__ import annotations

import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from random import Random
from statistics import NormalDist
from typing import Callable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "bootstrap",
    "BootstrapResult",
    "stat_mean",
    "stat_median",
    "stat_max",
    "make_variance_stat",
    "make_quantile_stat",
    "make_ratio_stat",
]

_NORM = NormalDist()
_MASK64 = (1 << 64) - 1

Data = Union[List[float], Tuple[List[float], List[float]]]


# ---------------------------------------------------------------------------
# Built-in statistics
# ---------------------------------------------------------------------------

def stat_mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs)


def stat_median(xs: Sequence[float]) -> float:
    return statistics.median(xs)


def stat_max(xs: Sequence[float]) -> float:
    return max(xs)


def make_variance_stat(ddof: int = 1) -> Callable[[Sequence[float]], float]:
    """Sample variance.  Degenerate samples (n <= ddof, e.g. n == 1 with the
    default ddof=1) fall back to the population variance, which is 0.0."""

    def stat(xs: Sequence[float]) -> float:
        if len(xs) <= ddof:
            return statistics.pvariance(xs)
        return statistics.variance(xs)

    stat.__name__ = "variance_ddof%d" % ddof
    return stat


def make_quantile_stat(p: float) -> Callable[[Sequence[float]], float]:
    """p-quantile with linear interpolation (numpy 'linear' convention)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    def stat(xs: Sequence[float]) -> float:
        return _quantile_sorted(sorted(xs), p)

    stat.__name__ = "quantile_%g" % p
    return stat


def make_ratio_stat(
    numerator_stat: Callable = stat_mean,
    denominator_stat: Callable = stat_mean,
) -> Callable[[Tuple[Sequence[float], Sequence[float]]], float]:
    """Ratio statistic for two independent samples, e.g. mean(x) / mean(y).

    A zero (or non-finite) denominator yields NaN for that replicate; NaN
    replicates are excluded when the interval quantiles are computed and
    are reported via ``BootstrapResult.n_failed``.
    """

    def stat(pair: Tuple[Sequence[float], Sequence[float]]) -> float:
        xs, ys = pair
        den = denominator_stat(ys)
        if den == 0.0 or not math.isfinite(den):
            return float("nan")
        return numerator_stat(xs) / den

    stat.__name__ = "ratio"
    return stat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_two_sample(data) -> bool:
    return isinstance(data, tuple) and len(data) == 2


def _validate(data) -> Data:
    """Coerce to floats and reject empty samples / non-finite values."""
    if _is_two_sample(data):
        parts = []
        for part in data:
            vals = [float(v) for v in part]
            if not vals:
                raise ValueError("empty sample")
            for v in vals:
                if not math.isfinite(v):
                    raise ValueError("sample contains a non-finite value: %r" % v)
            parts.append(vals)
        return (parts[0], parts[1])
    vals = [float(v) for v in data]
    if not vals:
        raise ValueError("empty sample")
    for v in vals:
        if not math.isfinite(v):
            raise ValueError("sample contains a non-finite value: %r" % v)
    return vals


def _quantile_sorted(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _chunk_seed(seed: int, index: int) -> int:
    """Deterministic 64-bit sub-seed for a chunk (Knuth MMIX LCG step)."""
    return (seed * 6364136223846793005
            + (index + 1) * 1442695040888963407
            + 0x9E3779B97F4A7C15) & _MASK64


def _resample_chunk(data: Data, statistic: Callable, size: int,
                    chunk_seed: int, rng_factory: Callable) -> List[float]:
    rng = rng_factory(chunk_seed)
    out: List[float] = []
    if _is_two_sample(data):
        xs, ys = data
        nx, ny = len(xs), len(ys)
        for _ in range(size):
            sx = [xs[rng.randrange(nx)] for _ in range(nx)]
            sy = [ys[rng.randrange(ny)] for _ in range(ny)]
            out.append(float(statistic((sx, sy))))
    else:
        n = len(data)
        for _ in range(size):
            out.append(float(statistic([data[rng.randrange(n)] for _ in range(n)])))
    return out


def _worker(args):
    data, statistic, size, chunk_seed, rng_factory = args
    return _resample_chunk(data, statistic, size, chunk_seed, rng_factory)


def _bias_correction(reps: Sequence[float], theta_hat: float) -> float:
    m = len(reps)
    if m == 0 or not math.isfinite(theta_hat):
        return 0.0
    less = sum(1 for t in reps if t < theta_hat)
    # Clamp away from 0/1 so z0 stays finite even when theta_hat lies
    # outside the bootstrap distribution (common for extreme statistics).
    p = min(max(less / m, 0.5 / m), 1.0 - 0.5 / m)
    z0 = _NORM.inv_cdf(p)
    return z0 if math.isfinite(z0) else 0.0


def _jackknife(data: Data, statistic: Callable) -> List[float]:
    if _is_two_sample(data):
        xs, ys = data
        vals: List[float] = []
        if len(xs) > 1:
            for i in range(len(xs)):
                vals.append(float(statistic((xs[:i] + xs[i + 1:], ys))))
        if len(ys) > 1:
            for j in range(len(ys)):
                vals.append(float(statistic((xs, ys[:j] + ys[j + 1:]))))
        return [v for v in vals if math.isfinite(v)]
    if len(data) < 2:
        return []
    vals = [float(statistic(data[:i] + data[i + 1:])) for i in range(len(data))]
    return [v for v in vals if math.isfinite(v)]


def _acceleration(jk: Sequence[float]) -> float:
    if len(jk) < 2:
        return 0.0
    mean = statistics.fmean(jk)
    diffs = [mean - v for v in jk]
    denom = 6.0 * (sum(d * d for d in diffs) ** 1.5)
    if denom == 0.0:  # all jackknife values identical (degenerate sample)
        return 0.0
    accel = sum(d ** 3 for d in diffs) / denom
    return accel if math.isfinite(accel) else 0.0


def _adjusted_alphas(z0: float, accel: float, alpha: float) -> Tuple[float, float]:
    out = []
    for a in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = _NORM.inv_cdf(a)
        denom = 1.0 - accel * (z0 + z)
        if denom <= 0.0:
            adj = 0.0 if (z0 + z) > 0 else 1.0
        else:
            adj = _NORM.cdf(z0 + (z0 + z) / denom)
        out.append(min(max(adj, 1e-12), 1.0 - 1e-12))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BootstrapResult:
    """Holds the bootstrap replicates; intervals are computed on demand."""

    theta_hat: float
    replicates: List[float]
    seed: int
    data: Data
    statistic: Callable

    @property
    def n_failed(self) -> int:
        """Replicates that were non-finite (e.g. zero-denominator ratios)."""
        return sum(1 for t in self.replicates if not math.isfinite(t))

    def _finite_sorted(self) -> List[float]:
        return sorted(t for t in self.replicates if math.isfinite(t))

    def percentile_ci(self, confidence: float = 0.95) -> Tuple[float, float]:
        alpha = 1.0 - confidence
        reps = self._finite_sorted()
        return (_quantile_sorted(reps, alpha / 2.0),
                _quantile_sorted(reps, 1.0 - alpha / 2.0))

    def bc_ci(self, confidence: float = 0.95) -> Tuple[float, float]:
        alpha = 1.0 - confidence
        reps = self._finite_sorted()
        z0 = _bias_correction(reps, self.theta_hat)
        lo_a, hi_a = _adjusted_alphas(z0, 0.0, alpha)
        return _quantile_sorted(reps, lo_a), _quantile_sorted(reps, hi_a)

    def bca_ci(self, confidence: float = 0.95) -> Tuple[float, float]:
        alpha = 1.0 - confidence
        reps = self._finite_sorted()
        z0 = _bias_correction(reps, self.theta_hat)
        accel = _acceleration(_jackknife(self.data, self.statistic))
        lo_a, hi_a = _adjusted_alphas(z0, accel, alpha)
        return _quantile_sorted(reps, lo_a), _quantile_sorted(reps, hi_a)

    def normal_ci(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Normal-approximation interval using the bootstrap standard error;
        included only as a baseline for coverage comparisons."""
        reps = self._finite_sorted()
        se = statistics.pstdev(reps) if len(reps) > 1 else 0.0
        z = _NORM.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
        return self.theta_hat - z * se, self.theta_hat + z * se


def bootstrap(
    data,
    statistic: Callable,
    n_resamples: int = 2000,
    *,
    seed: Optional[int] = None,
    chunk_size: int = 512,
    n_jobs: int = 1,
    rng_factory: Callable = Random,
) -> BootstrapResult:
    """Run a nonparametric bootstrap.

    Parameters
    ----------
    data : sequence of numbers, or (xs, ys) tuple for two-sample statistics
        (e.g. ratios).  Two-sample data is resampled independently per group.
    statistic : callable applied to each resample.  For ``n_jobs > 1`` it must
        be picklable (a top-level function, not a lambda/closure).
    n_resamples : total number of bootstrap replicates B.
    seed : master seed.  Given the same ``seed``, ``chunk_size`` and data, the
        replicates are bit-for-bit reproducible.  ``None`` draws a random
        seed, which is stored on the result for later reproduction.
    chunk_size : resamples per chunk.  Chunk ``i`` uses
        ``rng_factory(_chunk_seed(seed, i))``, so a serial run and a
        chunked/parallel run with the same seed and chunk_size produce
        identical replicates after in-order concatenation.
    n_jobs : worker processes (1 = serial).  Only affects wall-clock time.
    rng_factory : builds the per-chunk RNG from a 64-bit seed; must provide
        the ``random.Random`` interface (``randrange``).
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    validated = _validate(data)
    if seed is None:
        seed = Random().randrange(1 << 63)
    seed = int(seed) & _MASK64

    sizes: List[int] = []
    remaining = n_resamples
    while remaining > 0:
        take = min(chunk_size, remaining)
        sizes.append(take)
        remaining -= take

    tasks = [(validated, statistic, size, _chunk_seed(seed, i), rng_factory)
             for i, size in enumerate(sizes)]
    if n_jobs <= 1 or len(tasks) == 1:
        chunks = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            chunks = list(pool.map(_worker, tasks))
    replicates = [v for chunk in chunks for v in chunk]

    return BootstrapResult(
        theta_hat=float(statistic(validated)),
        replicates=replicates,
        seed=seed,
        data=validated,
        statistic=statistic,
    )
