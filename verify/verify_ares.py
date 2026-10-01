#!/usr/bin/env python3
"""A-Res weighted reservoir: empirical inclusion frequencies vs EXACT
inclusion probabilities computed by rational integration
(reservoir.exact.ares_inclusion_probabilities).  Fixed seed, so the
experiment is fully reproducible."""
import random

from common import report, scaled
from reservoir import AResWeightedSampler, ares_inclusion_probabilities, parse_weight

ok_all = True

# ---- Case A: k = 1 (exact pi_i = w_i / W) ----------------------------------
weights = ["0.5", "2", "10", "0.001", "7", "3"]
ws = [parse_weight(w) for w in weights]
pis = ares_inclusion_probabilities(ws, 1)
R = scaled(300_000)
rng = random.Random(20261001)
counts = [0] * len(ws)
for _ in range(R):
    s = AResWeightedSampler(1, rng)
    for i, w in enumerate(ws):
        s.feed(i, w)
    counts[s.sample()[0]] += 1
ok, _ = report(
    "A-Res k=1: frequency vs exact integral pi (weights %s)" % weights,
    [("w=%s" % weights[i], counts[i], float(pis[i])) for i in range(len(ws))],
    R)
ok_all &= ok

# ---- Case B: k = 3, weight ratio 1e6 ---------------------------------------
weightsB = [10**6, 1000, 100, 10, 1, 1, 1, 1]
wsB = [parse_weight(w) for w in weightsB]
pisB = ares_inclusion_probabilities(wsB, 3)
RB = scaled(200_000)
rng = random.Random(20261001)
countsB = [0] * len(wsB)
for _ in range(RB):
    s = AResWeightedSampler(3, rng)
    for i, w in enumerate(wsB):
        s.feed(i, w)
    for i in s.sample():
        countsB[i] += 1
ok, _ = report(
    "A-Res k=3: weights %s, frequency vs exact integral pi" % weightsB,
    [("w=%d" % weightsB[i], countsB[i], float(pisB[i])) for i in range(len(wsB))],
    RB)
ok_all &= ok

# ---- Case C: k = 2, tiny decimal weights -----------------------------------
weightsC = ["0.000001", "0.01", "1", "100", "1000000", "0.5"]
wsC = [parse_weight(w) for w in weightsC]
pisC = ares_inclusion_probabilities(wsC, 2)
RC = scaled(200_000)
rng = random.Random(20261001)
countsC = [0] * len(wsC)
for _ in range(RC):
    s = AResWeightedSampler(2, rng)
    for i, w in enumerate(wsC):
        s.feed(i, w)
    for i in s.sample():
        countsC[i] += 1
ok, _ = report(
    "A-Res k=2: weights %s, frequency vs exact integral pi" % weightsC,
    [("w=%s" % weightsC[i], countsC[i], float(pisC[i])) for i in range(len(wsC))],
    RC)
ok_all &= ok

print("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
