"""A-Res weighted reservoir sampling (Efraimidis & Spirakis), single pass,
O(k) memory, with *certified exact* key comparison.

Each item gets a random key ``u ** (1 / w)`` and the reservoir keeps the
``k`` largest keys.  Keys are only ever *compared*, never printed, so we
compare them exactly: ``ln(u)/w`` is evaluated with ``decimal.Decimal`` at
increasing precision until the ordering is certified (``Decimal.ln`` and
Decimal division are correctly rounded, so an error bound of a few ulps is
rigorous).  A float64 pre-screen with a 1e-9 relative guard (10^7 times
larger than any float64 error) resolves virtually all comparisons; the
exact path is the fallback.  This means arbitrarily large integer weights
and tiny decimal weights never lose precision, and zero-weight items are
skipped before any ``1/w`` is formed (no division by zero).

Shards merge exactly: the merged sample is the top-``k`` of the union of
shard keys, which is *identical* (not just identically distributed) to a
single-pass run with the same random keys.
"""
import heapq
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction

from .weights import parse_weight

__all__ = ["AResWeightedSampler"]

# Precision ladder (decimal digits) for certified comparison.
_PRECISION_LADDER = (50, 100, 200, 400, 800, 1600, 3200)
_CONFIRM_EXTRA = 25
# Relative guard for the float64 pre-screen.  float64 log/division errors
# are ~1e-16 relative, so 1e-9 is a safe certification margin.
_FLOAT_GUARD = 1e-9


class _Key:
    """Lazy sort key ``u ** (1 / w)`` supporting exact comparison."""

    __slots__ = ("u", "w", "_f", "_cache")

    def __init__(self, u, w):
        self.u = u  # float in [0, 1), an exact dyadic rational
        self.w = w  # Fraction > 0
        self._f = None
        self._cache = {}

    def approx(self):
        """float64 approximation of ln(u)/w, or -inf; used only as a
        pre-screen with a wide safety margin."""
        f = self._f
        if f is None:
            if self.u == 0.0:
                f = float("-inf")
            else:
                try:
                    fw = float(self.w)
                except OverflowError:
                    fw = float("inf")
                if fw == 0.0:  # w underflowed float64: key ~ 0
                    f = float("-inf")
                elif math.isinf(fw):
                    f = -0.0  # w overflowed float64: key ~ 1
                else:
                    f = math.log(self.u) / fw
            self._f = f
        return f

    def value(self, prec):
        """ln(u)/w as a Decimal computed at ``prec`` digits (correctly
        rounded components, cached)."""
        v = self._cache.get(prec)
        if v is None:
            with localcontext() as ctx:
                ctx.prec = prec
                u = Decimal(self.u)  # exact
                if u.is_zero():
                    v = Decimal("-Infinity")
                else:
                    w = Decimal(self.w.numerator) / Decimal(self.w.denominator)
                    v = u.ln() / w
            self._cache[prec] = v
        return v

    def __lt__(self, other):
        return compare_keys(self, other) < 0


def _decide(va, vb, prec):
    """Return -1/0/1 if the ordering of va, vb is certified at ``prec``
    digits, else None.  Components are correctly rounded (error <= ~1 ulp
    each), so a gap larger than 16 ulps cannot be an artifact."""
    if va.is_infinite() or vb.is_infinite():
        if va == vb:
            return None
        return -1 if va < vb else 1
    d = va - vb
    if d.is_zero():
        return None
    scale = max(va.adjusted(), vb.adjusted())
    ulp = Decimal(1).scaleb(scale - prec + 1)
    if abs(d) > 16 * ulp:
        return -1 if d < 0 else 1
    return None


def _compare_decimal(a, b):
    for prec in _PRECISION_LADDER:
        r = _decide(a.value(prec), b.value(prec), prec)
        if r is None:
            continue
        rc = _decide(a.value(prec + _CONFIRM_EXTRA), b.value(prec + _CONFIRM_EXTRA),
                     prec + _CONFIRM_EXTRA)
        if rc is not None and rc == r:
            return r
    # Measure-zero fallback (keys equal beyond 3200+25 digits): deterministic
    # tie-break so that heaps and merges stay consistent within a run.
    if a.u != b.u:
        return -1 if a.u < b.u else 1
    if a.w != b.w:
        return -1 if a.w < b.w else 1
    return 0


def compare_keys(a, b):
    """Certified comparison of two keys: -1, 0 or 1."""
    if a is b:
        return 0
    fa, fb = a.approx(), b.approx()
    if math.isfinite(fa) and math.isfinite(fb):
        m = max(1.0, abs(fa), abs(fb))
        if abs(fa - fb) > _FLOAT_GUARD * m:
            return -1 if fa < fb else 1
    return _compare_decimal(a, b)


class AResWeightedSampler:
    """Weighted reservoir of size ``k``.  Items with weight 0 are never
    selected.  For ``k = 1`` the inclusion probability of item ``i`` is
    exactly ``w_i / sum(w)``; for ``k > 1`` the exact inclusion
    probabilities are given by the rational integral in
    ``reservoir.exact.ares_inclusion_probabilities`` (verified
    empirically in ``verify/verify_ares.py``)."""

    def __init__(self, k, rng=None):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self._heap = []  # min-heap of (key, seq, item), top-k largest keys
        self._seq = 0
        self.n = 0           # items seen
        self.n_positive = 0  # items with positive weight

    def feed(self, item, weight=1):
        w = parse_weight(weight)
        self.n += 1
        if self.k == 0 or w == 0:
            return self
        self.n_positive += 1
        key = _Key(self.rng.random(), w)
        self._seq += 1
        entry = (key, self._seq, item)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif compare_keys(key, self._heap[0][0]) > 0:
            heapq.heapreplace(self._heap, entry)
        return self

    def feed_all(self, items_with_weights):
        for item, weight in items_with_weights:
            self.feed(item, weight)
        return self

    def sample(self):
        """The reservoir as a list, sorted by key descending."""
        if not self._heap and self.k > 0:
            raise ValueError("no positive-weight items seen")
        return [item for _, _, item in sorted(self._heap, reverse=True)]

    def state(self):
        """List of ``(key, seq, item)`` used by ``merge.merge_ares``."""
        return list(self._heap)
