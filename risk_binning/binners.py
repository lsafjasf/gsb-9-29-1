"""Quantile and optimal (IV-maximizing, optionally monotone) binners.

Stability contract:
  * fit() computes edges once, deterministically (no randomness anywhere).
  * transform() only reads the stored edges/fences; scoring new samples
    can never move the boundaries.
"""

from __future__ import annotations

import bisect
import math
from typing import List, Optional, Sequence

from .core import (
    BinStat,
    detect_direction,
    find_violations,
    is_missing,
    monotonicity_report,
    quantile,
    woe_iv,
)

MISSING_LABEL = "missing"
OUTLIER_LOW_LABEL = "outlier_low"
OUTLIER_HIGH_LABEL = "outlier_high"
UNSEEN_LABEL = "unseen_value"


class BaseBinner:
    """Shared preprocessing: missing / outlier handling, stats, transform."""

    kind = "base"

    def __init__(self, n_bins: int = 5, min_bin_pct: float = 0.02,
                 min_bin_size: int = 1, outlier_quantile: float = 0.005):
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if not 0.0 <= outlier_quantile < 0.5:
            raise ValueError("outlier_quantile must be in [0, 0.5)")
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.min_bin_size = min_bin_size
        self.outlier_quantile = outlier_quantile

        self.edges_: Optional[List[float]] = None
        self.lower_fence_: Optional[float] = None
        self.upper_fence_: Optional[float] = None
        self.warnings_: List[str] = []
        self.stats_: List[BinStat] = []
        self.n_samples_ = 0
        self.n_missing_ = 0
        self._has_y = False
        self.fitted_ = False

    # ------------------------------------------------------------------ API
    def fit(self, x: Sequence, y: Optional[Sequence[int]] = None):
        xs = list(x)
        ys = list(y) if y is not None else None
        if ys is not None:
            if len(ys) != len(xs):
                raise ValueError("x and y must have the same length")
            for v in ys:
                if v not in (0, 1):
                    raise ValueError("y must be binary 0/1")
        self._has_y = ys is not None
        self.warnings_ = []
        self.n_samples_ = len(xs)

        pairs = []          # (value, y) for non-missing
        missing_bad = missing_good = 0
        for i, v in enumerate(xs):
            if is_missing(v):
                self.n_missing_ += 0  # placeholder, counted below
                if ys is not None:
                    if ys[i] == 1:
                        missing_bad += 1
                    else:
                        missing_good += 1
            else:
                pairs.append((float(v), ys[i] if ys is not None else None))
        self.n_missing_ = self.n_samples_ - len(pairs)

        inliers, outliers_low, outliers_high = [], [], []
        if not pairs:
            self.edges_ = []
            self.lower_fence_ = self.upper_fence_ = None
            self.warnings_.append(
                "all values missing: only the 'missing' bin is defined")
        else:
            sv = sorted(v for v, _ in pairs)
            # Fences only make sense when the sample is large enough to
            # expect at least one point beyond them; otherwise every
            # extreme-looking value of a tiny sample would be exiled.
            if self.outlier_quantile > 0 and \
                    len(sv) >= 1.0 / self.outlier_quantile:
                lo = quantile(sv, self.outlier_quantile)
                hi = quantile(sv, 1.0 - self.outlier_quantile)
            else:
                lo, hi = sv[0], sv[-1]
            self.lower_fence_, self.upper_fence_ = lo, hi
            for v, t in pairs:
                if v < lo:
                    outliers_low.append((v, t))
                elif v > hi:
                    outliers_high.append((v, t))
                else:
                    inliers.append((v, t))
            distinct = sorted({v for v, _ in inliers})
            if not inliers:
                self.edges_ = []
                self.warnings_.append(
                    "no inlier values: every value fell outside the fences")
            elif len(distinct) == 1:
                self.edges_ = []
                self.warnings_.append(
                    "single distinct inlier value: collapsed to one bin")
            else:
                self.edges_ = self._compute_edges(sorted(inliers), distinct)

        self._build_stats(inliers, outliers_low, outliers_high,
                          missing_bad, missing_good)
        self.fitted_ = True
        return self

    def transform(self, x: Sequence) -> List[str]:
        """Map values to bin labels. Never mutates fitted boundaries."""
        self._check_fitted()
        return [self._label_for(v) for v in x]

    def fit_transform(self, x: Sequence, y: Optional[Sequence[int]] = None):
        return self.fit(x, y).transform(x)

    @property
    def edges(self) -> Optional[List[float]]:
        return list(self.edges_) if self.edges_ is not None else None

    def report(self) -> dict:
        self._check_fitted()
        rep = {
            "binner": self.kind,
            "n_samples": self.n_samples_,
            "n_missing": self.n_missing_,
            "n_outlier_low": self._count_of(OUTLIER_LOW_LABEL),
            "n_outlier_high": self._count_of(OUTLIER_HIGH_LABEL),
            "edges": list(self.edges_),
            "fences": [self.lower_fence_, self.upper_fence_],
            "warnings": list(self.warnings_),
            "bins": [s.to_dict() for s in self.stats_],
        }
        normal = [s for s in self.stats_
                  if s.kind == "normal" and s.count > 0]
        if self._has_y:
            rep["total_iv"] = sum(s.iv for s in self.stats_)
            if len(normal) >= 2:
                rep["monotonicity"] = monotonicity_report(
                    normal, direction=self._report_direction())
        return rep

    # ------------------------------------------------------------- internal
    def _compute_edges(self, sorted_inliers, distinct) -> List[float]:
        raise NotImplementedError

    def _report_direction(self) -> str:
        return "auto"

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("binner is not fitted yet")

    def _count_of(self, label: str) -> int:
        for s in self.stats_:
            if s.label == label:
                return s.count
        return 0

    def _label_for(self, v) -> str:
        if is_missing(v):
            return MISSING_LABEL
        v = float(v)
        if self.lower_fence_ is None:
            return UNSEEN_LABEL
        if v < self.lower_fence_:
            return OUTLIER_LOW_LABEL
        if v > self.upper_fence_:
            return OUTLIER_HIGH_LABEL
        # Convention: bins are (lower, upper]; an exact edge value belongs
        # to the bin whose upper bound it is (bisect_left).
        return "bin_%02d" % bisect.bisect_left(self.edges_, v)

    def _min_bin_size(self, n: int) -> int:
        return max(self.min_bin_size, int(math.ceil(self.min_bin_pct * n)))

    def _build_stats(self, inliers, outliers_low, outliers_high,
                     missing_bad, missing_good):
        stats: List[BinStat] = []
        if self.lower_fence_ is not None:
            n_edges = len(self.edges_)
            for i in range(n_edges + 1):
                lower = self.lower_fence_ if i == 0 else self.edges_[i - 1]
                upper = self.edges_[i] if i < n_edges else self.upper_fence_
                stats.append(BinStat(index=len(stats), label="bin_%02d" % i,
                                     kind="normal", lower=lower, upper=upper))
            for v, t in inliers:
                s = stats[bisect.bisect_left(self.edges_, v)]
                s.count += 1
                if t is not None:
                    if t == 1:
                        s.bad += 1
                    else:
                        s.good += 1
        for label, kind, group in (
            (MISSING_LABEL, "missing", None),
            (OUTLIER_LOW_LABEL, "outlier_low", outliers_low),
            (OUTLIER_HIGH_LABEL, "outlier_high", outliers_high),
        ):
            stat = BinStat(index=len(stats), label=label, kind=kind)
            if group is None:
                stat.count = self.n_missing_
                stat.bad = missing_bad
                stat.good = missing_good
            else:
                stat.count = len(group)
                stat.bad = sum(1 for _, t in group if t == 1)
                stat.good = sum(1 for _, t in group if t == 0)
            stats.append(stat)
        if self._has_y:
            total_bad = sum(s.bad for s in stats)
            total_good = sum(s.good for s in stats)
            for s in stats:
                s.woe, s.iv = woe_iv(s.bad, s.good, total_bad, total_good)
        self.stats_ = stats


class QuantileBinner(BaseBinner):
    """Equal-frequency binning with deterministic, de-duplicated edges."""

    kind = "quantile"

    def _compute_edges(self, sorted_inliers, distinct) -> List[float]:
        values = [v for v, _ in sorted_inliers]
        max_edges = min(self.n_bins - 1, len(distinct) - 1)
        edges: List[float] = []
        collisions = 0
        for k in range(1, max_edges + 1):
            e = quantile(values, k / (max_edges + 1))
            if not edges or e > edges[-1]:
                edges.append(e)
            else:
                collisions += 1
        if collisions:
            self.warnings_.append(
                "quantile collision on skewed data: %d duplicate edge(s) "
                "dropped, %d bin(s) remain" % (collisions, len(edges) + 1))
        return edges


class OptimalBinner(BaseBinner):
    """Supervised binning: fine quantile pre-bins merged to maximize IV.

    With monotone='increasing'|'decreasing'|'auto', adjacent pre-bin groups
    that violate the constraint are merged (smallest IV loss first) until
    the bad-rate sequence is monotone or `min_bins` is reached. If the
    constraint cannot be satisfied, a warning is recorded and the report
    carries concrete merge suggestions.
    """

    kind = "optimal"

    def __init__(self, n_bins: int = 5, prebins: int = 20,
                 monotone: Optional[str] = None, min_bins: int = 2, **kw):
        super().__init__(n_bins=n_bins, **kw)
        if monotone not in (None, "increasing", "decreasing", "auto"):
            raise ValueError("monotone must be None|increasing|decreasing|auto")
        if min_bins < 1:
            raise ValueError("min_bins must be >= 1")
        self.prebins = prebins
        self.monotone = monotone
        self.min_bins = min_bins

    def fit(self, x, y=None):
        if y is None:
            raise ValueError("OptimalBinner is supervised: y is required")
        return super().fit(x, y)

    def _report_direction(self) -> str:
        return self.monotone if self.monotone else "auto"

    def _compute_edges(self, sorted_inliers, distinct) -> List[float]:
        n = len(sorted_inliers)
        min_size = self._min_bin_size(n)
        pre = max(2, min(self.prebins, n // max(min_size, 1) or 2,
                         len(distinct)))

        values = [v for v, _ in sorted_inliers]
        pre_edges: List[float] = []
        for k in range(1, pre):
            e = quantile(values, k / pre)
            if not pre_edges or e > pre_edges[-1]:
                pre_edges.append(e)

        nb = len(pre_edges) + 1
        bads = [0] * nb
        goods = [0] * nb
        uppers: List[Optional[float]] = [None] * nb
        for v, t in sorted_inliers:
            b = bisect.bisect_left(pre_edges, v)
            if t == 1:
                bads[b] += 1
            else:
                goods[b] += 1
            uppers[b] = v
        # groups: [bad, good, upper_edge]
        groups = [[bads[i], goods[i], uppers[i]] for i in range(nb)
                  if bads[i] + goods[i] > 0]
        total_bad = sum(bads)
        total_good = sum(goods)

        def iv_of(bad, good):
            return woe_iv(bad, good, total_bad, total_good)[1]

        def merge_loss(i, j):
            b1, g1, _ = groups[i]
            b2, g2, _ = groups[j]
            return (iv_of(b1, g1) + iv_of(b2, g2)
                    - iv_of(b1 + b2, g1 + g2))

        def do_merge(i, j):
            lo, hi = min(i, j), max(i, j)
            b1, g1, u1 = groups[lo]
            b2, g2, u2 = groups[hi]
            groups[lo:hi + 1] = [[b1 + b2, g1 + g2, max(u1, u2)]]

        # Phase 1: absorb undersized groups into the cheapest neighbour.
        while len(groups) > 1:
            small = [i for i, g in enumerate(groups)
                     if g[0] + g[1] < min_size]
            if not small:
                break
            i = min(small, key=lambda k: groups[k][0] + groups[k][1])
            if i == 0:
                j = 1
            elif i == len(groups) - 1:
                j = i - 1
            else:
                j = i - 1 if merge_loss(i - 1, i) <= merge_loss(i, i + 1) \
                    else i + 1
            do_merge(i, j)

        # Phase 2: merge down to n_bins, always sacrificing the least IV.
        while len(groups) > self.n_bins:
            i = min(range(len(groups) - 1),
                    key=lambda k: merge_loss(k, k + 1))
            do_merge(i, i + 1)

        # Phase 3: enforce the monotone constraint if requested.
        if self.monotone:
            direction = self.monotone
            if direction == "auto":
                direction = detect_direction(
                    [g[0] / (g[0] + g[1]) for g in groups])
            applied = 0
            while len(groups) > self.min_bins:
                rates = [g[0] / (g[0] + g[1]) for g in groups]
                violations = find_violations(rates, direction)
                if not violations:
                    break
                i = min(violations, key=lambda k: merge_loss(k, k + 1))
                do_merge(i, i + 1)
                applied += 1
            rates = [g[0] / (g[0] + g[1]) for g in groups]
            if find_violations(rates, direction):
                self.warnings_.append(
                    "monotone constraint '%s' unsatisfiable with min_bins=%d; "
                    "see report['monotonicity']['merge_suggestions']"
                    % (direction, self.min_bins))
            elif applied:
                self.warnings_.append(
                    "monotone constraint '%s' enforced via %d merge(s)"
                    % (direction, applied))

        return [g[2] for g in groups[:-1]]
