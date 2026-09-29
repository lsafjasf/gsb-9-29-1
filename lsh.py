"""Banded locality-sensitive hashing (LSH) index for approximate near-duplicates.

Uses random-hyperplane LSH, which approximates cosine similarity:
for two vectors with angle theta, a random hyperplane separates them with
probability theta / pi, so each signature bit matches with probability
1 - theta / pi.

The index keeps ``num_tables`` (L) independent bucket tables. Each table signs
a vector with ``num_hashes`` (k) random hyperplanes, producing a k-bit integer
used as the bucket key. Two items become candidates when they collide in at
least one table:

    P(candidate) = 1 - (1 - p^k)^L,   p = 1 - theta/pi

so larger k shrinks the candidate set (fewer false positives, lower recall)
and larger L raises recall at the cost of memory and query time.

Vectors may be dense (any sequence of floats) or sparse (a mapping of
dimension -> value). Projection weights are generated deterministically from a
seed and cached lazily per dimension, so high-dimensional sparse vectors only
ever materialize weights for the dimensions they actually touch.

Pure standard library; works on Python 3.9+.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _weight(seed: int, table: int, hash_idx: int, dim: int) -> float:
    """Deterministic pseudo-random projection weight in [-1, 1)."""
    x = (seed
         ^ (table * 0x9E3779B97F4A7C15)
         ^ (hash_idx * 0xC2B2AE3D27D4EB4F)
         ^ (dim * 0x165667B19E3779F9)) & _MASK64
    return (_splitmix64(x) / 2**64) * 2.0 - 1.0


def _nonzero_items(vector):
    if isinstance(vector, Mapping):
        return [(i, v) for i, v in vector.items() if v != 0.0]
    return [(i, v) for i, v in enumerate(vector) if v != 0.0]


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two dense and/or sparse vectors."""
    if not isinstance(a, Mapping) and not isinstance(b, Mapping):
        dot = norm_a = norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
    else:
        if not isinstance(a, Mapping):
            a = {i: v for i, v in enumerate(a) if v != 0.0}
        if not isinstance(b, Mapping):
            b = {i: v for i, v in enumerate(b) if v != 0.0}
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(i, 0.0) for i, v in a.items())
        norm_a = sum(v * v for v in a.values())
        norm_b = sum(v * v for v in b.values())
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class LSHIndex:
    """Dynamic LSH index supporting insert, delete and approximate queries."""

    def __init__(self, num_hashes: int = 8, num_tables: int = 16, seed: int = 0xC0FFEE):
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1")
        if num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        self.num_hashes = num_hashes
        self.num_tables = num_tables
        self._seed = seed & _MASK64
        # Lazily filled projection weights: one dict per (table, hash) pair.
        self._weights = [dict() for _ in range(num_tables * num_hashes)]
        # Bucket tables: table -> {signature: set of keys}.
        self._buckets = [dict() for _ in range(num_tables)]
        # key -> tuple of per-table signatures (kept so deletes are O(L)).
        self._sigs = {}
        # key -> original vector (for exact re-ranking and deletes).
        self._vectors = {}

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key) -> bool:
        return key in self._vectors

    def _signatures(self, vector):
        items = _nonzero_items(vector)
        num_hashes = self.num_hashes
        sigs = []
        for t in range(self.num_tables):
            bits = 0
            base = t * num_hashes
            for h in range(num_hashes):
                cache = self._weights[base + h]
                dot = 0.0
                for dim, value in items:
                    w = cache.get(dim)
                    if w is None:
                        w = _weight(self._seed, t, h, dim)
                        cache[dim] = w
                    dot += value * w
                if dot >= 0.0:
                    bits |= 1 << h
            sigs.append(bits)
        return tuple(sigs)

    def add(self, key, vector) -> None:
        """Insert a vector; re-adding an existing key replaces its vector."""
        if key in self._vectors:
            self.remove(key)
        sigs = self._signatures(vector)
        self._vectors[key] = vector
        self._sigs[key] = sigs
        for t, bits in enumerate(sigs):
            bucket = self._buckets[t].get(bits)
            if bucket is None:
                self._buckets[t][bits] = {key}
            else:
                bucket.add(key)

    def remove(self, key) -> None:
        """Delete a key. Raises KeyError if the key is not indexed."""
        sigs = self._sigs.pop(key)
        del self._vectors[key]
        for t, bits in enumerate(sigs):
            table = self._buckets[t]
            bucket = table[bits]
            bucket.discard(key)
            if not bucket:
                del table[bits]  # empty buckets are freed immediately

    def discard(self, key) -> None:
        """Delete a key if present, otherwise do nothing."""
        if key in self._vectors:
            self.remove(key)

    def query(self, vector) -> set:
        """Return the candidate keys colliding with the query in any table."""
        sigs = self._signatures(vector)
        candidates = set()
        for t, bits in enumerate(sigs):
            bucket = self._buckets[t].get(bits)
            if bucket:
                candidates.update(bucket)
        return candidates

    def query_topk(self, vector, k: int = 10):
        """Return up to k (similarity, key) pairs, exactly re-ranked by cosine."""
        scored = [
            (cosine_similarity(vector, self._vectors[key]), key)
            for key in self.query(vector)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:k]

    def num_buckets(self) -> int:
        """Total number of non-empty buckets across all tables."""
        return sum(len(table) for table in self._buckets)

    def memory_stats(self) -> dict:
        """Structural memory usage, useful for checking the memory bounds."""
        return {
            "items": len(self._vectors),
            "bucket_memberships": sum(
                len(bucket) for table in self._buckets for bucket in table.values()
            ),
            "nonempty_buckets": self.num_buckets(),
            "max_possible_buckets": self.num_tables * (1 << self.num_hashes),
            "weight_cache_entries": sum(len(cache) for cache in self._weights),
        }
