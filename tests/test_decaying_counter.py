"""Boundary / semantics regression tests for the fixed counter."""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decaying_counter import DecayingCounter   # noqa: E402
from common import admit, oracle_query, make_events  # noqa: E402

HL = 10.0
LATENESS = 100.0


def build(events, half_life=HL, max_lateness=LATENESS, now_fn=None):
    counter = DecayingCounter(half_life, max_lateness, now_fn=now_fn)
    for event_id, t, value in events:
        counter.add(event_id, t, value)
    return counter


class BoundaryTests(unittest.TestCase):

    def test_empty_window(self):
        counter = DecayingCounter(HL, LATENESS)
        self.assertEqual(counter.query(0.0, 10.0), 0.0)
        self.assertEqual(counter.total(10.0), 0.0)
        counter.add("a", 50.0, 2.0)
        self.assertEqual(counter.query(0.0, 40.0), 0.0)   # before event
        self.assertEqual(counter.query(60.0, 100.0), 0.0)  # after event

    def test_single_event_exact_weight(self):
        counter = DecayingCounter(HL, LATENESS)
        counter.add("only", 30.0, 1.0)
        self.assertEqual(counter.query(30.0, 40.0), 2.0 ** -1.0)
        self.assertEqual(counter.query(0.0, 30.0), 1.0)  # endpoint inclusive
        self.assertEqual(counter.query(30.0, 30.0), 1.0)  # point interval

    def test_endpoint_inclusive(self):
        events = [("s", 10.0, 1.0), ("e", 20.0, 1.0)]
        counter = build(events)
        truth = oracle_query(events, 10.0, 20.0, HL)
        self.assertEqual(counter.query(10.0, 20.0), truth)
        self.assertEqual(counter.query(10.0, 10.0), 1.0)

    def test_lateness_boundary_inclusive(self):
        counter = DecayingCounter(HL, max_lateness=5.0)
        counter.add("w", 100.0, 1.0)
        # Exactly on the limit is accepted.
        self.assertTrue(counter.add("edge", 95.0, 1.0))
        # One epsilon beyond the limit is rejected and recorded.
        self.assertFalse(counter.add("too_late", 95.0 - 1e-12, 1.0))
        self.assertEqual(counter.stats["late"], 1)
        record = counter.journal[-1]
        self.assertEqual(record["kind"], "late")
        self.assertEqual(record["id"], "too_late")
        self.assertEqual(record["watermark"], 100.0)
        self.assertEqual(record["limit"], 95.0)
        self.assertEqual(counter.stats["accepted"], 2)

    def test_zero_lateness(self):
        counter = DecayingCounter(HL, 0.0)
        counter.add("a", 10.0, 1.0)
        self.assertTrue(counter.add("b", 10.0, 1.0))   # equal time is fine
        self.assertFalse(counter.add("c", 9.999999, 1.0))

    def test_duplicate_recorded_once(self):
        counter = DecayingCounter(HL, LATENESS)
        self.assertTrue(counter.add("x", 10.0, 3.0))
        self.assertFalse(counter.add("x", 10.0, 3.0))
        self.assertFalse(counter.add("x", 11.0, 9.0))
        self.assertEqual(counter.stats["accepted"], 1)
        self.assertEqual(counter.stats["duplicates"], 2)
        self.assertEqual(counter.query(0.0, 100.0),
                         3.0 * 2.0 ** (-9.0))

    def test_same_timestamp_many_events(self):
        events = [(f"e{i}", 42.0, 1.0) for i in range(100)]
        counter = build(events)
        self.assertEqual(counter.stats["accepted"], 100)
        self.assertAlmostEqual(counter.query(42.0, 52.0),
                               100.0 * 2.0 ** (-1.0), places=9)

    def test_dense_events(self):
        rng = random.Random(777)
        events = [(f"e{i}", rng.uniform(0.0, 1.0),
                   rng.uniform(0.1, 3.0)) for i in range(5000)]
        counter = build(events)
        kept, dropped = admit(events, LATENESS)
        self.assertEqual(dropped, [])
        for lo, hi in [(0.0, 1.0), (0.2, 0.4), (0.7, 0.70001),
                       (-10.0, 10.0), (0.0, 0.0)]:
            expected = oracle_query(kept, lo, hi, HL)
            self.assertLessEqual(
                abs(counter.query(lo, hi) - expected)
                / max(abs(expected), 1e-12),
                1e-12)

    def test_all_events_out_of_order(self):
        rng = random.Random(31)
        ordered = [(f"e{i}", round(rng.uniform(0.0, 100.0), 6),
                    rng.uniform(0.5, 2.0)) for i in range(300)]
        reversed_events = list(reversed(ordered))
        # Lateness window large enough that every late event is accepted.
        counter = build(reversed_events, max_lateness=200.0)
        control = build(ordered, max_lateness=200.0)
        self.assertEqual(counter.stats["accepted"], 300)
        for lo, hi in [(-50.0, 150.0), (0.0, 100.0), (30.0, 60.0)]:
            # Same final state -> bit-identical answers regardless of order.
            self.assertEqual(counter.query(lo, hi), control.query(lo, hi))

    def test_out_of_order_beyond_boundary_dropped(self):
        events = [("a", 100.0, 1.0), ("b", 110.0, 1.0),
                  ("ancient", 89.0, 1.0), ("edge", 90.0, 1.0)]
        counter = build(events, max_lateness=20.0)
        self.assertTrue(counter.stats["late"] == 1)
        kinds = [r["kind"] for r in counter.journal]
        self.assertEqual(kinds, ["late"])
        kept, dropped = admit(events, 20.0)
        expected = oracle_query(kept, 0.0, 200.0, HL)
        self.assertEqual(counter.query(0.0, 200.0), expected)

    def test_extremely_long_range(self):
        counter = DecayingCounter(half_life=1.0, max_lateness=1e9)
        counter.add("old", 0.0, 1.0)
        counter.add("recent", 1e6, 1.0)
        # Age of 1e6 half-lives underflows to zero instead of exploding.
        result = counter.query(-1e9, 1e6)
        self.assertTrue(math.isfinite(result))
        self.assertEqual(result, 1.0)
        # Huge timestamps must not overflow (exponent <= 0 everywhere).
        counter.add("far", 1e12, 1.0)
        self.assertEqual(counter.query(-1e12, 1e12), 1.0)

    def test_window_composability(self):
        # [a,c) = [a,b) re-weighted to c + [b,c) -- the identity that
        # "cross-window query equals per-window recompute" is based on.
        rng = random.Random(9)
        events = [(f"e{i}", rng.uniform(0.0, 100.0),
                   rng.uniform(0.5, 3.0)) for i in range(200)]
        counter = build(events, max_lateness=200.0)
        a, b, c = 20.0, 50.0, 80.0
        composed = (counter.query(a, b) * 2.0 ** (-(c - b) / HL)
                    + counter.query(b, c))
        self.assertLessEqual(
            abs(composed - counter.query(a, c))
            / max(counter.query(a, c), 1e-12),
            1e-12)

    def test_continuity_between_events(self):
        events = [("a", 10.0, 1.0), ("b", 20.0, 1.0)]
        counter = build(events)
        for end in (14.0, 14.0 + 1e-8):
            self.assertAlmostEqual(
                counter.query(0.0, end), 2.0 ** (-(end - 10.0) / HL),
                places=12)
        # The change over 10 ns is tiny (no bucket-style jump).
        self.assertLess(abs(counter.query(0.0, 14.0 + 1e-8)
                            - counter.query(0.0, 14.0)), 1e-9)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DecayingCounter(0, 1)
        counter = DecayingCounter(HL, LATENESS)
        with self.assertRaises(ValueError):
            counter.query(10.0, 9.0)


class ClockRegressionTests(unittest.TestCase):

    def test_backwards_clock_is_clamped(self):
        schedule = [100.0, 110.0, 105.0, 120.0, 90.0, 130.0]
        index = {"i": 0}

        def clock():
            value = schedule[index["i"]]
            index["i"] += 1
            return value

        counter = DecayingCounter(HL, LATENESS, now_fn=clock)
        seen_now = [counter.now() for _ in schedule]
        self.assertEqual(seen_now, [100.0, 110.0, 110.0, 120.0, 120.0, 130.0])
        self.assertEqual(counter.stats["clock_regressions"], 2)
        self.assertTrue(all(r["kind"] == "clock_regression"
                            for r in counter.journal))

    def test_total_never_decreases_on_regression(self):
        schedule = iter([10.0, 20.0, 30.0, 40.0, 35.0, 50.0])
        counter = DecayingCounter(half_life=5.0, max_lateness=1000.0,
                                  now_fn=lambda: next(schedule))
        totals = []
        counter.add("a", 10.0, 10.0)
        totals.append(counter.total())          # clock 10
        counter.add("b", 20.0, 10.0)
        totals.append(counter.total())          # clock 20
        counter.add("c", 30.0, 10.0)
        totals.append(counter.total())          # clock 30
        counter.add("d", 40.0, 10.0)
        totals.append(counter.total())          # clock 40
        totals.append(counter.total())          # clock 35 -> clamped to 40
        counter.add("e", 50.0, 10.0)
        totals.append(counter.total())          # clock 50
        # The regressed tick reproduces the value at the clamped time:
        # no backwards jump, and everything stays non-negative.
        self.assertEqual(totals[-2], totals[-3])
        self.assertTrue(all(t >= 0.0 for t in totals))
        self.assertGreaterEqual(counter.stats["clock_regressions"], 1)

    def test_event_from_regressed_clock_not_late_dropped(self):
        # When the injected clock regresses, an event timestamped "now"
        # must be accepted: the clamped monotone time is used.
        schedule = [100.0, 80.0]
        index = {"i": 0}
        counter = DecayingCounter(HL, max_lateness=10.0,
                                  now_fn=lambda: schedule[index["i"]])
        counter.add("a", timestamp=None, value=1.0)
        index["i"] += 1
        self.assertTrue(counter.add("b", timestamp=None, value=1.0))
        self.assertEqual(counter.stats["late"], 0)
        self.assertEqual(counter.stats["clock_regressions"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
