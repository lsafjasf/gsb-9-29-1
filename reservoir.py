"""Streaming (one-pass, fixed-memory) reservoir sampling.

Two samplers are provided:

- ``ReservoirSampler``: unweighted, uniform sampling without replacement
  (Algorithm R, Vitter 1985). Every stream item has inclusion probability
  exactly k/n.
- ``WeightedReservoirSampler``: weighted sampling without replacement
  (A-Res, Efraimidis & Spirakis 2005). Each item gets key u^(1/w) and the
  k largest keys are kept; implemented in the log domain as log(u)/w so
  that extreme weights (e.g. ratios of 1e6 or weights like 1e+-300) do
  not cause floating-point under/overflow.

Both are single-pass, O(k) memory, and accept an injectable random source
(any object with ``.random()`` -> float in [0, 1) and, for the unweighted
sampler, ``.randrange(n)`` -- ``random.Random`` satisfies both).
"""

import heapq
import math
import random

__all__ = ["ReservoirSampler", "WeightedReservoirSampler"]


class ReservoirSampler:
    """Unweighted reservoir sampling (Algorithm R).

    Inclusion probability of every seen item is exactly k/n.
    Memory: O(k). Time: O(1) per item.
    """

    def __init__(self, k, rng=None):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self._reservoir = []
        self.n = 0  # number of items seen so far

    def add(self, item):
        """Offer one stream item. O(1)."""
        self.n += 1
        if len(self._reservoir) < self.k:
            self._reservoir.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self._reservoir[j] = item

    def sample(self):
        """Return the current sample (a copy). Size is min(k, n)."""
        return list(self._reservoir)

    def __len__(self):
        return len(self._reservoir)


class WeightedReservoirSampler:
    """Weighted reservoir sampling without replacement (A-Res).

    Each item with weight w > 0 draws u ~ U(0,1) and gets key
    ``log(u) / w`` (the log of the classic key u^(1/w)). The k items with
    the largest keys form the sample. For k = 1 the inclusion probability
    of item i is exactly w_i / sum(w); for k > 1 it is weighted sampling
    without replacement in the sense of Efraimidis & Spirakis.

    Weight rules (see README for rationale):
      - weight must be a finite, non-negative number;
      - NaN / +-inf / negative weights raise ``ValueError``;
      - weight == 0 items are skipped and can never be sampled;
      - if fewer than k positive-weight items are seen, the sample holds
        all of them (size < k is possible).

    Memory: O(k). Time: O(log k) per positive-weight item.
    """

    def __init__(self, k, rng=None):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self._heap = []          # min-heap of (key, seq, item), size <= k
        self._seq = 0            # tie-breaker so items are never compared
        self.n = 0               # items seen
        self.n_positive = 0      # items with weight > 0

    def add(self, item, weight):
        """Offer one stream item with the given weight. O(log k)."""
        w = float(weight)
        if math.isnan(w) or math.isinf(w):
            raise ValueError("weight must be finite, got %r" % (weight,))
        if w < 0.0:
            raise ValueError("weight must be non-negative, got %r" % (weight,))
        self.n += 1
        if w == 0.0:
            return  # zero-weight items can never enter the sample
        self.n_positive += 1

        # Draw u in (0, 1]: guard against an RNG returning exactly 0.0,
        # which would make log(0) raise / produce -inf and silently bias
        # the key distribution.
        u = self.rng.random()
        while u <= 0.0:
            u = self.rng.random()
        # Log-domain key of u^(1/w). With w as small as 1e-300 or as
        # large as 1e300 this stays a finite float, whereas u**(1/w)
        # would underflow to 0.0 or overflow for extreme w.
        key = math.log(u) / w

        entry = (key, self._seq, item)
        self._seq += 1
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif key > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def sample(self):
        """Return the current sample (a copy). Size is min(k, n_positive)."""
        return [item for _, _, item in self._heap]

    def __len__(self):
        return len(self._heap)
