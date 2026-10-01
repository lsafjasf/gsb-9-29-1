"""Exact shard merging.

* ``merge_uniform``: merges equal-probability reservoirs with an exact
  (multivariate) hypergeometric draw -- the merged sample is a uniform
  k-subset of the concatenated stream, exactly as a single-pass run.
* ``merge_ares``: merges A-Res reservoirs by keeping the top-k keys --
  the result is *identical* to a single-pass run with the same keys.
* k=1 weighted shards merge via ``weighted_one.merge_weighted_one``.
"""
import heapq
import random

__all__ = ["merge_uniform", "merge_ares"]


def merge_uniform(shard_states, k, rng=None):
    """Merge uniform-reservoir shards.

    ``shard_states``: iterable of ``(reservoir_items, stream_count)`` as
    returned by ``UniformReservoirSampler.state()``.  The number of items
    drawn from shard ``s`` follows the multivariate hypergeometric
    distribution of ``k`` draws from ``sum(n_s)`` items grouped by shard;
    combined with a uniform choice inside each shard this yields exactly
    the single-pass distribution.
    """
    rng = rng if rng is not None else random.Random()
    states = [(list(res), int(n)) for res, n in shard_states]
    total_n = sum(n for _, n in states)
    if total_n == 0 or k == 0:
        return []
    if total_n <= k:
        # Every item was kept by its shard; the merged sample is everything.
        return [item for res, _ in states for item in res]
    remaining = [n for _, n in states]
    draws = [0] * len(states)
    left = total_n
    for _ in range(k):
        x = rng.randrange(left)
        acc = 0
        for i, c in enumerate(remaining):
            acc += c
            if x < acc:
                draws[i] += 1
                remaining[i] -= 1
                left -= 1
                break
    out = []
    for (res, _), c in zip(states, draws):
        if c:
            out.extend(rng.sample(res, c))
    return out


def merge_ares(shard_states, k):
    """Merge A-Res shards by keeping the top-``k`` keys across shards.

    ``shard_states``: iterable of states from ``AResWeightedSampler.state()``.
    Deterministic (no randomness needed): the result equals the single-pass
    reservoir over the concatenated stream with the same keys.
    """
    entries = [entry for state in shard_states for entry in state]
    if len(entries) <= k:
        return [item for _, _, item in sorted(entries, reverse=True)]
    best = heapq.nlargest(k, entries, key=lambda e: e[0])
    return [item for _, _, item in best]
