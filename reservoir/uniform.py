"""Equal-probability reservoir sampling (Algorithm R), single pass, O(k) memory."""
import random

__all__ = ["UniformReservoirSampler"]


class UniformReservoirSampler:
    """Keeps a uniform random sample of ``k`` items from a stream of unknown
    length.  Every seen item ends up in the final sample with probability
    exactly ``k / N`` where ``N`` is the stream length.

    ``rng`` is injectable: any object with the ``random.Random`` interface
    (``random()`` and ``randrange()``) works.
    """

    def __init__(self, k, rng=None):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.rng = rng if rng is not None else random.Random()
        self._res = []
        self.n = 0  # number of items seen so far

    def feed(self, item):
        self.n += 1
        if len(self._res) < self.k:
            self._res.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self._res[j] = item
        return self

    def feed_all(self, items):
        for item in items:
            self.feed(item)
        return self

    @property
    def count(self):
        """Number of items seen so far."""
        return self.n

    def sample(self):
        return list(self._res)

    def state(self):
        """Serializable state used by ``merge.merge_uniform``."""
        return (list(self._res), self.n)
