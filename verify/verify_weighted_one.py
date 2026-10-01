#!/usr/bin/env python3
"""Exactness of the k=1 weighted sampler (WeightedOneSampler).

Case A: mixed decimal/integer weights, frequencies vs exact w_i/W.
Case B: weight ratio 1e6 (extreme skew) -- rare-event sanity check.
Case C: exact running weight total with decimal-string weights, and a
        10**30 vs 10**30+1 case where float64 cannot tell the weights apart.
"""
import random
from fractions import Fraction

from common import report, scaled
from reservoir import WeightedOneSampler, parse_weight

ok_all = True

# ---- Case A: moderate mixed weights, exact z-test -------------------------
weights = ["0.5", "2", "10", "0.001", "7", "3"]
ws = [parse_weight(w) for w in weights]
W = sum(ws)
R = scaled(300_000)
rng = random.Random(20261001)
counts = [0] * len(ws)
for _ in range(R):
    s = WeightedOneSampler(rng)
    for i, w in enumerate(ws):
        s.feed(i, w)
    counts[s.sample()] += 1
ok, _ = report(
    "WeightedOne k=1: frequency vs exact w_i/W (weights %s)" % weights,
    [("w=%s" % weights[i], counts[i], float(ws[i] / W)) for i in range(len(ws))],
    R)
ok_all &= ok

# ---- Case B: ratio 1e6 -----------------------------------------------------
wsB = [Fraction(10**6), Fraction(1)]
WB = sum(wsB)
RB = scaled(200_000)
rng = random.Random(20261001)
countsB = [0, 0]
for _ in range(RB):
    s = WeightedOneSampler(rng)
    s.feed(0, wsB[0])
    s.feed(1, wsB[1])
    countsB[s.sample()] += 1
print("WeightedOne k=1: weights [1000000, 1], R = %d" % RB)
print("  exact pi_small = 1/1000001 = %.8f, expected count %.4f, observed %d"
      % (float(wsB[1] / WB), RB * float(wsB[1] / WB), countsB[1]))
loose_ok = countsB[1] <= 5  # P(>5 | pi=1e-6, R=2e5) is astronomically small
print("  rare-event sanity (observed <= 5): %s" % ("PASS" if loose_ok else "FAIL"))
print()
ok_all &= loose_ok

# ---- Case C: exact arithmetic where float64 fails --------------------------
s = WeightedOneSampler(random.Random(7))
for _ in range(100_000):
    s.feed("x", "0.1")
exact_ok = s.W == Fraction(10_000)
float_total = 0.0
for _ in range(100_000):
    float_total += 0.1
print("WeightedOne: 100000 x weight '0.1'")
print("  exact Fraction total W = %s (== 10000: %s)"
      % (s.W, "PASS" if exact_ok else "FAIL"))
print("  float64 accumulation   = %.17g (drift %.3g)" % (float_total, float_total - 10000))
print()
ok_all &= exact_ok

big1, big2 = Fraction(10**30), Fraction(10**30) + 1
pi_exact = big1 / (big1 + big2)
pi_float = float(big1) / (float(big1) + float(big2))
print("WeightedOne: weights 10**30 vs 10**30+1")
print("  exact pi_1 = %s ~ %.17f" % (pi_exact, float(pi_exact)))
print("  float64 pi = %.17f (weights collapse to the same double)" % pi_float)
print("  library keeps them distinct: %s"
      % ("PASS" if parse_weight(str(10**30)) != parse_weight(str(10**30 + 1)) else "FAIL"))
print()

print("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
