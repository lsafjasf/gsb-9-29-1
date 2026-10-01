"""Edge-case and contract tests for the fixed DecayCounter."""

import itertools
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decay_counter import DecayCounter
from oracle import Oracle


class QueryEdgeCases(unittest.TestCase):
    def test_empty_counter_and_empty_window(self):
        counter = DecayCounter(10.0, bucket_width=5.0)
        self.assertEqual(counter.query(0.0, 10.0), 0.0)
        self.assertEqual(counter.value_at(10.0), 0.0)

        counter.add("a", 0.0, 1.0)
        counter.add("b", 100.0, 1.0)
        self.assertEqual(counter.query(10.0, 20.0), 0.0)
        self.assertEqual(counter.query(100.0001, 200.0), 0.0)

    def test_single_event(self):
        counter = DecayCounter(10.0, bucket_width=3.0)
        counter.add("a", 7.0, 2.0)
        self.assertAlmostEqual(counter.query(7.0, 7.0), 2.0, places=12)
        self.assertAlmostEqual(counter.query(0.0, 17.0), 1.0, places=12)
        self.assertEqual(counter.query(7.0001, 100.0), 0.0)
        self.assertEqual(counter.query(0.0, 6.9999), 0.0)

    def test_interval_endpoints_are_inclusive(self):
        counter = DecayCounter(10.0, bucket_width=2.0)
        counter.add("s", 4.0, 1.0)
        counter.add("e", 8.0, 1.0)
        self.assertAlmostEqual(counter.query(4.0, 8.0), 1.0 + 2.0 ** (-0.4), places=12)

    def test_events_exactly_on_bucket_edges_all_alignments(self):
        half_life = 7.0
        width = 2.5
        oracle = Oracle(half_life)
        counter = DecayCounter(half_life, bucket_width=width)
        for i in range(8):
            ts = i * width
            oracle.add("e%d" % i, ts, 1.0)
            counter.add("e%d" % i, ts, 1.0)
        for start in (0.0, width, 3 * width):
            for end in (start, start + width, start + 2 * width, 7 * width):
                self.assertAlmostEqual(
                    counter.query(start, end), oracle.query(start, end), places=12
                )

    def test_super_long_intervals(self):
        counter = DecayCounter(10.0, bucket_width=5.0)
        oracle = Oracle(10.0)
        rng = random.Random(4242)
        for i in range(100):
            ts = rng.uniform(0.0, 50.0)
            counter.add("e%d" % i, ts, 1.0)
            oracle.add("e%d" % i, ts, 1.0)
        for start, end in ((-1e6, 1e6), (-100.0, 50.0), (0.0, 1e9)):
            got = counter.query(start, end)
            expected = oracle.query(start, end)
            self.assertGreaterEqual(got, 0.0)
            self.assertAlmostEqual(got, expected, places=12)

    def test_dense_events(self):
        half_life = 3.0
        counter = DecayCounter(half_life, bucket_width=1.0)
        oracle = Oracle(half_life)
        rng = random.Random(1337)
        for i in range(5000):
            ts = rng.uniform(0.0, 2.0)
            value = rng.uniform(0.1, 2.0)
            counter.add("e%d" % i, ts, value)
            oracle.add("e%d" % i, ts, value)
        for end in (0.5, 2.0, 7.0):
            self.assertAlmostEqual(counter.query(0.0, end), oracle.query(0.0, end), places=8)

    def test_arrival_permutations_give_identical_answers(self):
        half_life = 6.0
        events = [(float(t), (i % 3) + 0.5) for i, t in enumerate([0, 1, 2, 5, 9, 12])]
        answers = {}
        for order in itertools.permutations(range(len(events))):
            counter = DecayCounter(half_life, bucket_width=3.0, max_lateness=100.0)
            for pos, (ts, value) in [(i, events[i]) for i in order]:
                counter.add("e%d" % pos, ts, value)
            answers[order] = (
                counter.query(0.0, 12.0),
                counter.query(2.0, 9.0),
                counter.value_at(20.0),
            )
        reference = answers[tuple(range(len(events)))]
        for value in answers.values():
            for got, expected in zip(value, reference):
                self.assertAlmostEqual(got, expected, places=12)

    def test_all_events_reversed_within_bounded_lateness(self):
        half_life = 8.0
        timestamps = [0.1 * i for i in range(30)]
        oracle = Oracle(half_life, max_lateness=100.0)
        counter = DecayCounter(half_life, bucket_width=0.7, max_lateness=100.0)
        for ts in reversed(timestamps):
            oracle.add("e%s" % ts, ts, 1.0)
            counter.add("e%s" % ts, ts, 1.0)
        for end in (0.5, 1.0, 2.9, 5.0):
            self.assertAlmostEqual(counter.query(0.0, end), oracle.query(0.0, end), places=12)


class LatenessAndDedup(unittest.TestCase):
    def test_events_beyond_lateness_boundary_are_dropped_and_recorded(self):
        dropped = []
        counter = DecayCounter(10.0, bucket_width=5.0, max_lateness=5.0,
                               on_drop=lambda eid, ts, value: dropped.append((eid, ts)))
        self.assertEqual(counter.add("a", 0.0, 1.0), "accepted")
        self.assertEqual(counter.add("b", 10.0, 1.0), "accepted")
        self.assertEqual(counter.add("late1", 4.9999, 1.0), "dropped")
        self.assertEqual(counter.add("edge", 5.0, 1.0), "accepted")
        self.assertEqual(counter.add("edge2", 5.0 + 1e-12, 1.0), "accepted")
        self.assertEqual(counter.stats["dropped_late"], 1)
        self.assertEqual(dropped, [("late1", 4.9999)])
        self.assertEqual(counter.dropped_events[0][0], "late1")

        oracle = Oracle(10.0, max_lateness=5.0)
        oracle.add("a", 0.0, 1.0)
        oracle.add("b", 10.0, 1.0)
        oracle.add("late1", 4.9999, 1.0)
        oracle.add("edge", 5.0, 1.0)
        oracle.add("edge2", 5.0 + 1e-12, 1.0)
        for start, end in ((0.0, 10.0), (4.0, 6.0), (0.0, 20.0)):
            self.assertAlmostEqual(
                counter.query(start, end), oracle.query(start, end), places=12
            )

    def test_duplicate_id_first_copy_wins(self):
        counter = DecayCounter(10.0, bucket_width=5.0)
        self.assertEqual(counter.add("x", 0.0, 1.0), "accepted")
        self.assertEqual(counter.add("x", 3.0, 9.0), "duplicate")
        self.assertEqual(counter.stats["accepted"], 1)
        self.assertEqual(counter.stats["duplicates"], 1)
        self.assertAlmostEqual(counter.query(0.0, 10.0), 0.5, places=12)


class ClockRewind(unittest.TestCase):
    def test_backwards_clock_is_clamped_and_recorded(self):
        readings = [100.0, 105.0, 102.0, 108.0]
        ticks = iter(readings)
        rewinds = []
        counter = DecayCounter(10.0, bucket_width=5.0,
                               clock=lambda: next(ticks),
                               on_clock_rewind=lambda raw, last: rewinds.append((raw, last)))
        counter.add("a", 100.0, 1.0)

        self.assertEqual(counter.value_at(), 1.0)
        self.assertAlmostEqual(counter.value_at(), 2.0 ** (-0.5), places=12)
        self.assertAlmostEqual(counter.value_at(), 2.0 ** (-0.5), places=12)
        self.assertAlmostEqual(counter.value_at(), 2.0 ** (-0.8), places=12)
        self.assertEqual(rewinds, [(102.0, 105.0)])
        self.assertEqual(counter.stats["clock_rewinds"], 1)
        self.assertGreaterEqual(counter.value_at(0.0), 0.0)

    def test_invalid_construction_and_queries(self):
        with self.assertRaises(ValueError):
            DecayCounter(0.0)
        with self.assertRaises(ValueError):
            DecayCounter(10.0, max_lateness=-1.0)
        counter = DecayCounter(10.0)
        with self.assertRaises(ValueError):
            counter.query(10.0, 5.0)
        with self.assertRaises(ValueError):
            counter.add("x", math.nan)


if __name__ == "__main__":
    unittest.main()
