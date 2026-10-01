#!/usr/bin/env python3
"""Unbiasedness of the equal-probability reservoir sampler.

Fixed-seed repeated experiments: item inclusion frequencies and pair
co-inclusion frequencies are compared against the exact theoretical
probabilities k/N and k(k-1)/(N(N-1)).
"""
import random
from itertools import combinations

from common import report, scaled, Z_LIMIT
from reservoir import UniformReservoirSampler

N, K = 12, 4
R = scaled(200_000)
rng = random.Random(20261001)

item_counts = [0] * N
pair_counts = {(i, j): 0 for i in range(N) for j in range(i + 1, N)}

for _ in range(R):
    s = UniformReservoirSampler(K, rng)
    s.feed_all(range(N))
    samp = sorted(s.sample())
    for i in samp:
        item_counts[i] += 1
    for pair in combinations(samp, 2):
        pair_counts[pair] += 1

p_item = K / N
ok1, _ = report(
    "Uniform reservoir: item inclusion frequency vs k/N = %d/%d" % (K, N),
    [("item %d" % i, c, p_item) for i, c in enumerate(item_counts)],
    R, show_rows=6)

p_pair = K * (K - 1) / (N * (N - 1))
ok2, _ = report(
    "Uniform reservoir: pair co-inclusion vs k(k-1)/(N(N-1)) = %d/%d"
    % (K * (K - 1), N * (N - 1)),
    [("%d&%d" % p, c, p_pair) for p, c in sorted(pair_counts.items())],
    R, show_rows=6)

print("OVERALL: %s" % ("PASS" if ok1 and ok2 else "FAIL"))
