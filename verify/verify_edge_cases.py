#!/usr/bin/env python3
"""Boundary cases: all-zero weights, single-element streams, k=0, k>N,
zero-weight items mixed with positive ones, and 1e6-skew at k=1."""
import os
import sys
import random
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reservoir import (
    UniformReservoirSampler, WeightedOneSampler, AResWeightedSampler,
    PPSReservoirSampler, parse_weight,
)

ok_all = True


def check(name, cond):
    print("  %s: %s" % (name, "PASS" if cond else "FAIL"))
    return cond


print("all-zero weighted streams raise a clear error:")
makers = [
    ("WeightedOne", lambda rng: WeightedOneSampler(rng)),
    ("ARes(k=3)", lambda rng: AResWeightedSampler(3, rng)),
    ("PPS(k=3)", lambda rng: PPSReservoirSampler(3, rng)),
]
for name, maker in makers:
    raised = False
    try:
        s = maker(random.Random(1))
        s.feed_all([(i, 0) for i in range(10)])
        s.sample()
    except ValueError:
        raised = True
    ok_all &= check(name + " raises ValueError", raised)
print()

print("single-element streams:")
ok_all &= check("Uniform(1) -> element",
                UniformReservoirSampler(1, random.Random(1)).feed("x").sample() == ["x"])
ok_all &= check("WeightedOne -> element",
                WeightedOneSampler(random.Random(1)).feed("x", "1e-9").sample() == "x")
ok_all &= check("ARes(3) -> [element]",
                AResWeightedSampler(3, random.Random(1)).feed("x", 1).sample() == ["x"])
ok_all &= check("PPS(1) -> element",
                PPSReservoirSampler(1, random.Random(1)).feed("x", 7).sample() == ["x"])
print()

print("k = 0 / k > N:")
u = UniformReservoirSampler(0, random.Random(1)).feed_all(range(5))
ok_all &= check("Uniform(0) -> []", u.sample() == [])
u2 = UniformReservoirSampler(10, random.Random(1)).feed_all(range(3))
ok_all &= check("Uniform(10) over 3 items -> all 3", sorted(u2.sample()) == [0, 1, 2])
a0 = AResWeightedSampler(0, random.Random(1)).feed("a", 1)
ok_all &= check("ARes(0) -> []", a0.sample() == [])
print()

print("zero-weight items are never selected:")
a = AResWeightedSampler(2, random.Random(1))
for i in range(1000):
    a.feed(("zero", i), 0)
a.feed("p1", 1)
a.feed("p2", "2.5")
ok_all &= check("ARes reservoir contains only positive-weight items",
                sorted(a.sample()) == ["p1", "p2"])
w = WeightedOneSampler(random.Random(1))
w.feed_all([("zero%d" % i, 0) for i in range(1000)])
w.feed("only", "0.000001")
ok_all &= check("WeightedOne with 1000 zeros then a tiny weight -> that item",
                w.sample() == "only")
print()

print("1e6 skew at k=1, R = 1,000,000 (expected small-item count ~ 0.999999):")
R = 1_000_000
s_count = 0
rng = random.Random(20261001)
for _ in range(R):
    w = WeightedOneSampler(rng)
    w.feed("big", 10**6)
    w.feed("small", 1)
    if w.sample() == "small":
        s_count += 1
print("  small selected %d times (exact expectation %.6f)"
      % (s_count, R / 1_000_001))
ok_all &= check("rare count within [0, 10]", 0 <= s_count <= 10)
print()

print("decimal-string weights stay exact:")
ok_all &= check("parse_weight('0.0000000001') == 1e-10 as Fraction",
                parse_weight("0.0000000001") == Fraction(1, 10**10))
ok_all &= check("parse_weight huge integer string exact",
                parse_weight("9" * 60) == Fraction(int("9" * 60)))
ok_all &= check("'1.5e-3' == 3/2000",
                parse_weight("1.5e-3") == Fraction(3, 2000))
print()

print("OVERALL: %s" % ("PASS" if ok_all else "FAIL"))
