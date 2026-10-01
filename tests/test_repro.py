"""Stable reproductions of the historical defects.

Each test first asserts the buggy implementation (``buggy_decay_counter``)
reproduces the described failure, then asserts the fixed implementation
(``DecayCounter``) agrees with the full-recompute ``Oracle``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buggy_decay_counter import BuggyDecayCounter
from decay_counter import DecayCounter
from oracle import Oracle


class BoundaryJumpRepro(unittest.TestCase):
    def test_event_exactly_on_window_boundary_is_lost_then_jumps(self):
        half_life = 10.0
        window = 10.0

        buggy = BuggyDecayCounter(half_life, window)
        buggy.add("e0", 0.0, 1.0)
        buggy.add("e1", 20.0, 1.0)

        oracle = Oracle(half_life)
        oracle.add("e0", 0.0, 1.0)
        oracle.add("e1", 20.0, 1.0)
        expected = oracle.query(0.0, 20.0)

        self.assertEqual(buggy.query(0.0, 20.0), 0.0)
        self.assertAlmostEqual(expected, 1.25, places=12)

        fixed = DecayCounter(half_life, bucket_width=window)
        fixed.add("e0", 0.0, 1.0)
        fixed.add("e1", 20.0, 1.0)
        self.assertAlmostEqual(fixed.query(0.0, 20.0), expected, places=12)

        for boundary in (10.0, 30.0):
            before = fixed.query(0.0, boundary - 1e-9)
            at = fixed.query(0.0, boundary)
            after = fixed.query(0.0, boundary + 1e-9)
            self.assertLess(abs(at - before), 1e-7)
            self.assertLess(abs(after - at), 1e-7)

        step = fixed.query(0.0, 20.0) - fixed.query(0.0, 20.0 - 1e-12)
        self.assertAlmostEqual(step, 1.0, places=9)

    def test_multiple_events_at_the_same_timestamp(self):
        half_life = 5.0
        window = 10.0
        buggy = BuggyDecayCounter(half_life, window)
        for name in ("a", "b", "c"):
            buggy.add(name, 5.0, 1.0)
        self.assertEqual(buggy.query(5.0, 5.0), 0.0)

        oracle = Oracle(half_life)
        fixed = DecayCounter(half_life, bucket_width=window)
        for name in ("a", "b", "c"):
            oracle.add(name, 5.0, 1.0)
            fixed.add(name, 5.0, 1.0)
        expected = oracle.query(5.0, 5.0)
        self.assertEqual(expected, 3.0)
        self.assertEqual(fixed.query(5.0, 5.0), 3.0)


class OutOfOrderDriftRepro(unittest.TestCase):
    def test_reversed_arrival_drifts_in_buggy_implementation(self):
        half_life = 10.0
        timestamps = list(range(10))
        oracle = Oracle(half_life)
        for ts in timestamps:
            oracle.add("e%d" % ts, ts, 1.0)
        expected = oracle.value_at(9.0)

        buggy = BuggyDecayCounter(half_life, 10.0)
        for ts in reversed(timestamps):
            buggy.add("e%d" % ts, ts, 1.0)
        self.assertGreater(abs(buggy.value_at() - expected), 0.5)

        in_order = DecayCounter(half_life, bucket_width=5.0, max_lateness=100.0)
        reversed_order = DecayCounter(half_life, bucket_width=5.0, max_lateness=100.0)
        for ts in timestamps:
            in_order.add("e%d" % ts, ts, 1.0)
        for ts in reversed(timestamps):
            reversed_order.add("e%d" % ts, ts, 1.0)
        self.assertAlmostEqual(in_order.value_at(9.0), expected, places=12)
        self.assertAlmostEqual(reversed_order.value_at(9.0), expected, places=12)


class DuplicateRepro(unittest.TestCase):
    def test_redelivered_event_double_counted_by_buggy_implementation(self):
        half_life = 10.0
        buggy = BuggyDecayCounter(half_life, 10.0)
        buggy.add("same", 0.0, 1.0)
        buggy.add("same", 0.0, 1.0)
        self.assertEqual(buggy.query(0.0, 10.0), 2.0)

        fixed = DecayCounter(half_life, bucket_width=10.0)
        oracle = Oracle(half_life)
        for target in (fixed, oracle):
            target.add("same", 0.0, 1.0)
            target.add("same", 0.0, 1.0)
        self.assertEqual(fixed.stats["duplicates"], 1)
        self.assertEqual(oracle.stats["duplicates"], 1)
        self.assertAlmostEqual(fixed.query(0.0, 10.0), oracle.query(0.0, 10.0), places=12)
        self.assertEqual(fixed.query(0.0, 10.0), 0.5)


class RestartRepro(unittest.TestCase):
    def test_buggy_snapshot_loses_events_and_restarts_decay_clock(self):
        half_life = 10.0
        buggy = BuggyDecayCounter(half_life, 10.0)
        buggy.add("e0", 0.0, 1.0)
        restored = BuggyDecayCounter.restore(buggy.snapshot())
        restored.add("e1", 10.0, 1.0)

        uninterrupted = BuggyDecayCounter(half_life, 10.0)
        uninterrupted.add("e0", 0.0, 1.0)
        uninterrupted.add("e1", 10.0, 1.0)
        self.assertNotAlmostEqual(
            restored.value_at(), uninterrupted.value_at(), places=3
        )

        fixed = DecayCounter(half_life, bucket_width=10.0)
        fixed.add("e0", 0.0, 1.0)
        revived = DecayCounter.restore(fixed.snapshot_json())
        revived.add("e1", 10.0, 1.0)

        continuous = DecayCounter(half_life, bucket_width=10.0)
        continuous.add("e0", 0.0, 1.0)
        continuous.add("e1", 10.0, 1.0)

        oracle = Oracle(half_life)
        oracle.add("e0", 0.0, 1.0)
        oracle.add("e1", 10.0, 1.0)
        for end in (10.0, 20.0, 100.0):
            self.assertAlmostEqual(revived.query(0.0, end), continuous.query(0.0, end), places=12)
            self.assertAlmostEqual(revived.query(0.0, end), oracle.query(0.0, end), places=12)


if __name__ == "__main__":
    unittest.main()
