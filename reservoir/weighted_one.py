"""Exact weighted sampling of a single item (k = 1), single pass, O(1) memory.

P(item i is the final sample) = w_i / sum(w) *exactly*, and all arithmetic is
done with ``fractions.Fraction`` so weights never lose precision.
"""
import random
from fractions import Fraction

from .weights import parse_weight

__all__ = ["WeightedOneSampler", "merge_weighted_one"]


class WeightedOneSampler:
    """Classic weighted reservoir of size 1: keeps the running weight total
    ``W`` and replaces the current item with probability ``w_i / W_i``.
    The replacement comparison ``rng.random() < w / W`` is exact because
    Python compares floats and Fractions without rounding."""

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.W = Fraction(0)  # exact running total of positive weights
        self.n = 0            # items seen
        self.n_positive = 0   # items with positive weight
        self.current = None

    def feed(self, item, weight=1):
        w = parse_weight(weight)
        self.n += 1
        if w == 0:
            return self  # zero weight: inclusion probability 0, no division
        self.W += w
        self.n_positive += 1
        if self.rng.random() < w / self.W:
            self.current = item
        return self

    def feed_all(self, items_with_weights):
        for item, weight in items_with_weights:
            self.feed(item, weight)
        return self

    def sample(self):
        if self.n_positive == 0:
            raise ValueError("no positive-weight items seen")
        return self.current

    def state(self):
        if self.n_positive == 0:
            raise ValueError("no positive-weight items seen")
        return (self.current, self.W)


def merge_weighted_one(states, rng=None):
    """Merge k=1 weighted shards exactly.

    ``states`` is an iterable of ``(item, W_shard)`` as returned by
    :meth:`WeightedOneSampler.state`.  The shard item is kept with probability
    ``W_shard / sum(W)``, which reproduces the single-pass distribution
    exactly.
    """
    rng = rng if rng is not None else random.Random()
    states = list(states)
    total = sum((W for _, W in states), Fraction(0))
    if total <= 0:
        raise ValueError("cannot merge: total weight is zero")
    x = rng.random()
    acc = Fraction(0)
    for item, W in states:
        acc += W
        if x < acc / total:
            return item
    return states[-1][0]  # unreachable, guards float boundary
