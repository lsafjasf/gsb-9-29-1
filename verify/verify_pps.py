#!/usr/bin/env python3
"""PPS reservoir: empirical inclusion frequencies vs the target
pi_i = k*w_i/W (or min(1, k*w_i/W) with certainty units).

Case A (mild weights, no certainty units): the classical regime where
Chao's algorithm tracks the target closely -> strict z-test.
Case B (extreme skew, certainty units present): no single-pass
fixed-memory algorithm can realise min(1, k*w_i/W) exactly; reported for
transparency (use A-Res / WeightedOne when exactness matters)."""
import random

from common import report, scaled
from reservoir import PPSReservoirSampler, pps_targets

ok_all = True

# ---- Case A: exact regime ---------------------------------------------------
# The first k items must have pi_i(k) = 1 (equal largest weights) and no
# later arrival may be a certainty unit (k * w_x < W_t).  In that regime
# Chao's streaming PPS is EXACT for every item, so the strict z-test is
# meaningful.
weightsA = [12, 12, 12, 9, 10, 11, 9, 10, 11, 9]
kA = 3
pisA = [float(p) for p in pps_targets(weightsA, kA)]
R = scaled(100_000)
rng = random.Random(20261001)
counts = [0] * len(weightsA)
for _ in range(R):
    s = PPSReservoirSampler(kA, rng)
    for i, w in enumerate(weightsA):
        s.feed(i, w)
    for i in s.sample():
        counts[i] += 1
ok, _ = report(
    "PPS k=%d: exact regime (equal first k, no certainty units), weights %s"
    % (kA, weightsA),
    [("w=%d" % weightsA[i], counts[i], pisA[i]) for i in range(len(weightsA))],
    R)
ok_all &= ok

# ---- Case B: extreme skew (transparency, no strict assertion) -----------------
weightsB = [1, 1, 1, 1, 1, 1, 1, 10**6]
kB = 3
pisB = [float(p) for p in pps_targets(weightsB, kB)]
RB = scaled(100_000)
rng = random.Random(20261001)
countsB = [0] * len(weightsB)
for _ in range(RB):
    s = PPSReservoirSampler(kB, rng)
    for i, w in enumerate(weightsB):
        s.feed(i, w)
    for i in s.sample():
        countsB[i] += 1
print("PPS k=%d: extreme skew weights %s (reported, not asserted)"
      % (kB, weightsB))
print("repetitions: %d" % RB)
print("%-16s %12s %12s" % ("item", "observed", "target"))
for i in range(len(weightsB)):
    print("%-16s %12.6f %12.6f" % ("w=%d" % weightsB[i], countsB[i] / RB, pisB[i]))
print("note: with certainty units, min(1,k*w/W) is not exactly attainable single-pass;")
print("this is the documented approximate regime (exactness_holds=False).")
print()

print("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
