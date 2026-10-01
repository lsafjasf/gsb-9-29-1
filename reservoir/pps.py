"""Streaming PPS reservoir: inclusion probabilities proportional to weight,
``pi_i = k * w_i / W``, single pass, O(k) memory, exact rational arithmetic.

Algorithm (exactness-preserving plan):
  * the reservoir holds the first ``k`` positive-weight items;
  * when item ``x`` with weight ``w_x`` arrives and the reservoir is full,
    it enters with probability ``p_x = min(1, k * w_x / W)`` where ``W`` is
    the running total of positive weights; if it enters, a uniformly chosen
    reservoir item is evicted;
  * zero-weight items never enter (their inclusion probability is 0).

Exactness (proved by induction, verified empirically in
``verify/verify_pps.py``): if
  (i) each of the first ``k`` positive-weight items has ``k * w_i >= W_k``
      (its inclusion probability is 1 when the reservoir fills -- e.g. the
      first ``k`` weights are equal), and
  (ii) every later arrival has ``k * w_x < W`` at entry (no certainty
      unit ever occurs),
then every item's final inclusion probability is EXACTLY ``k * w_i / W``.
The sampler exposes ``exactness_holds`` so callers can check the
conditions after the fact.  When a certainty unit occurs (condition ii
is violated), no single-pass fixed-memory algorithm can realise
``min(1, k*w_i/W)`` exactly; the plan then falls back to the standard
approximation (see ``verify/verify_pps.py`` case B), and for exact
results with skewed weights use ``AResWeightedSampler`` /
``WeightedOneSampler`` instead.

All probabilities are ``fractions.Fraction`` and the uniform draws are
compared exactly (float vs Fraction comparison is exact in Python), so
no precision is ever lost and no division by zero occurs.
"""
import random
from fractions import Fraction

from .weights import parse_weight

__all__ = ["PPSReservoirSampler"]


class PPSReservoirSampler:
    """Streaming PPS reservoir of size ``k`` (see module docstring)."""

    def __init__(self, k, rng=None):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self._items = []
        self._w = []
        self.W = Fraction(0)  # exact total of positive weights seen
        self.n = 0            # items seen (any weight)
        self._exact = True

    def feed(self, item, weight=1):
        w = parse_weight(weight)
        self.n += 1
        if w == 0:
            return self  # zero weight: inclusion probability 0, never enters
        if len(self._items) < self.k:
            self._items.append(item)
            self._w.append(w)
            self.W += w
            if len(self._items) == self.k:
                # exactness condition (i): every initial item has pi = 1
                if any(self.k * wi < self.W for wi in self._w):
                    self._exact = False
            return self
        self.W += w
        p_x = self.k * w / self.W
        if p_x >= 1:
            # certainty unit: exactness condition (ii) violated
            self._exact = False
            p_x = Fraction(1)
        if self.rng.random() < p_x:
            j = self.rng.randrange(self.k)
            self._items[j] = item
            self._w[j] = w
        return self

    def feed_all(self, items_with_weights):
        for item, weight in items_with_weights:
            self.feed(item, weight)
        return self

    @property
    def exactness_holds(self):
        """True iff the exactness conditions (i) and (ii) held throughout."""
        return self._exact

    def sample(self):
        if self.k > 0 and self.W <= 0:
            raise ValueError("no positive-weight items seen")
        return list(self._items)

    def inclusion_probabilities(self):
        """Current inclusion probabilities of the reservoir items."""
        if self.W <= 0:
            return [Fraction(0)] * len(self._items)
        return [min(Fraction(1), self.k * w / self.W) for w in self._w]
