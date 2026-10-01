#!/usr/bin/env python3
"""Shard merge verification.

1. Uniform reservoirs merged from 4 shards: merged frequencies vs
   single-pass frequencies and vs k/N (z-tests on both).
2. A-Res shards: with identical random keys, the merged sample is
   IDENTICAL to the single-pass sample in every repetition; in addition,
   merged frequencies match the exact integral inclusion probabilities.
3. k=1 weighted shards: merged frequencies vs exact w_i/W."""
import random

from common import report, scaled
from reservoir import (
    UniformReservoirSampler, AResWeightedSampler, WeightedOneSampler,
    merge_uniform, merge_ares, merge_weighted_one,
    ares_inclusion_probabilities, parse_weight,
)

ok_all = True

# ---- 1. Uniform: 4 shards ---------------------------------------------------
N, K = 200, 10
BOUNDS = [0, 30, 100, 150, 200]  # shard sizes: 30, 70, 50, 50
R = scaled(20_000)
single_counts = [0] * N
merged_counts = [0] * N
for rep in range(R):
    rng1 = random.Random(10_000_000 + rep)
    s1 = UniformReservoirSampler(K, rng1)
    s1.feed_all(range(N))
    for i in s1.sample():
        single_counts[i] += 1

    rng2 = random.Random(20_000_000 + rep)
    shards = []
    for a, b in zip(BOUNDS[:-1], BOUNDS[1:]):
        sh = UniformReservoirSampler(K, rng2)
        sh.feed_all(range(a, b))
        shards.append(sh.state())
    for i in merge_uniform(shards, K, rng2):
        merged_counts[i] += 1

p = K / N
ok1, _ = report(
    "Uniform merge: single-pass frequencies vs k/N (%d items, 4 shards)" % N,
    [("item %d" % i, single_counts[i], p) for i in range(N)],
    R, z_limit=5.5, show_rows=5)
ok2, _ = report(
    "Uniform merge: shard-merged frequencies vs k/N",
    [("item %d" % i, merged_counts[i], p) for i in range(N)],
    R, z_limit=5.5, show_rows=5)

# two-sample comparison
import math
worst_two = 0.0
for i in range(N):
    p1, p2 = single_counts[i] / R, merged_counts[i] / R
    se = math.sqrt(p1 * (1 - p1) / R + p2 * (1 - p2) / R)
    if se > 0:
        worst_two = max(worst_two, abs(p1 - p2) / se)
print("Uniform merge: max two-sample |z| single vs merged = %.2f" % worst_two)
ok3 = worst_two < 5.5
print("-> %s" % ("PASS" if ok3 else "FAIL"))
print()
ok_all &= ok1 and ok2 and ok3

# ---- 2a. A-Res: identical-result identity ----------------------------------
N2, K2 = 60, 5
R2 = scaled(3_000)
weights2 = [(i * 37) % 100 + 1 for i in range(N2)]
mismatches = 0
for rep in range(R2):
    seed = 30_000_000 + rep
    a = AResWeightedSampler(K2, random.Random(seed))
    for i in range(N2):
        a.feed(i, weights2[i])
    direct = set(a.sample())

    rng = random.Random(seed)  # SAME key stream, sharded below
    states = []
    for lo, hi in ((0, 13), (13, 40), (40, 60)):
        sh = AResWeightedSampler(K2, rng)
        for i in range(lo, hi):
            sh.feed(i, weights2[i])
        states.append(sh.state())
    merged = set(merge_ares(states, K2))
    if direct != merged:
        mismatches += 1
print("A-Res merge identity: identical to single-pass in %d/%d repetitions -> %s"
      % (R2 - mismatches, R2, "PASS" if mismatches == 0 else "FAIL"))
print()
ok_all &= mismatches == 0

# ---- 2b. A-Res: merged distribution vs exact integral ----------------------
ws3 = [parse_weight(w) for w in [1, 2, 3, 4, 5, 6, 7, 8]]
K3 = 3
pis3 = ares_inclusion_probabilities(ws3, K3)
R3 = scaled(100_000)
counts3 = [0] * len(ws3)
for rep in range(R3):
    rng = random.Random(40_000_000 + rep)
    shards = []
    for lo, hi in ((0, 3), (3, 6), (6, 8)):
        sh = AResWeightedSampler(K3, rng)
        for i in range(lo, hi):
            sh.feed(i, ws3[i])
        shards.append(sh.state())
    for i in merge_ares(shards, K3):
        counts3[i] += 1
ok4, _ = report(
    "A-Res merge: 3 shards, weights 1..8, k=3, frequency vs exact integral pi",
    [("w=%d" % (i + 1), counts3[i], float(pis3[i])) for i in range(len(ws3))],
    R3)
ok_all &= ok4

# ---- 3. k=1 weighted merge --------------------------------------------------
weights4 = ["0.5", "2", "10", "0.001", "7", "3"]
ws4 = [parse_weight(w) for w in weights4]
W4 = sum(ws4)
R4 = scaled(200_000)
counts4 = [0] * len(ws4)
for rep in range(R4):
    rng = random.Random(50_000_000 + rep)
    states = []
    for lo, hi in ((0, 2), (2, 4), (4, 6)):
        sh = WeightedOneSampler(rng)
        for i in range(lo, hi):
            sh.feed(i, ws4[i])
        states.append(sh.state())
    counts4[merge_weighted_one(states, rng)] += 1
ok5, _ = report(
    "WeightedOne merge: 3 shards, weights %s, frequency vs exact w_i/W" % weights4,
    [("w=%s" % weights4[i], counts4[i], float(ws4[i] / W4)) for i in range(len(ws4))],
    R4)
ok_all &= ok5

print("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
