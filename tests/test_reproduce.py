"""Reproduction tests for the incidents seen in production.

For every incident we run the same scenario against the legacy
BuggyCounter (the bug must be observable, otherwise the reproduction is
useless) and against the fixed DecayingCounter (the bug must be gone).
"""

import math
import unittest

from common import DecayingCounter, BuggyCounter, oracle_query, admit

WINDOW = 10.0
HALF_LIFE = 5.0
MAX_LATENESS = 1000.0  # repro scenarios: nothing rejected as late


def feed(counter, events):
    for event_id, t, value in events:
        if isinstance(counter, DecayingCounter):
            counter.add(event_id, t, value)
        else:
            counter.add(event_id, t, value)
    return counter


class ReproduceIncidents(unittest.TestCase):

    def test_window_switch_jump(self):
        """A query jumps when ingestion crosses a bucket boundary."""
        events = [(f"e{i}", float(i + 1), 1.0) for i in range(9)]

        # Legacy: bucket 0 is in-progress, hence invisible.
        buggy_before = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy_before, events)
        before = buggy_before.query(0.0, 20.0)

        buggy_after = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy_after, events + [("e9", 10.01, 1.0)])
        after = buggy_after.query(0.0, 20.0)

        self.assertEqual(before, 0.0)            # all events hidden
        self.assertGreater(after - before, 1.0)  # whole bucket pops in

        # Fixed: adding one event changes the query by at most its weight.
        fixed_before = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed_before, events)
        before_f = fixed_before.query(0.0, 20.0)
        fixed_before.add("e9", 10.01, 1.0)
        after_f = fixed_before.query(0.0, 20.0)
        new_event_weight = 2.0 ** (-(20.0 - 10.01) / HALF_LIFE)
        self.assertLessEqual(abs(after_f - before_f - new_event_weight), 1e-12)

    def test_events_at_same_timestamp(self):
        """Several events at exactly the same time."""
        events = [(f"e{i}", 100.0, 1.0) for i in range(5)]

        buggy = BuggyCounter(HALF_LIFE, WINDOW)
        # Force bucket 0 to seal so the data is visible.
        feed(buggy, events + [("seal", 110.0, 0.0)])
        buggy_total = buggy.query(0.0, 200.0)
        # The timestamp-keyed bucket silently keeps only 1 of 5 events.
        self.assertAlmostEqual(buggy_total, 9.5367431640625e-07, places=15)

        fixed = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed, events)
        kept, _ = admit(events, MAX_LATENESS)
        expected = oracle_query(kept, 0.0, 200.0, HALF_LIFE)
        self.assertAlmostEqual(fixed.query(0.0, 200.0), expected, places=12)
        self.assertAlmostEqual(expected, 5.0 * 2.0 ** (-(200.0 - 100.0) / 5.0),
                               places=12)

    def test_out_of_order_drift(self):
        """An event from an old bucket arrives after the bucket sealed."""
        in_order = [("late", 5.0, 1.0), ("a", 12.0, 1.0), ("b", 22.0, 1.0)]
        out_of_order = [("a", 12.0, 1.0), ("b", 22.0, 1.0),
                        ("late", 5.0, 1.0)]

        buggy = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy, out_of_order)
        buggy_control = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy_control, in_order)
        self.assertNotAlmostEqual(buggy.query(0.0, 40.0),
                                  buggy_control.query(0.0, 40.0),
                                  places=6)

        fixed = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed, out_of_order)
        kept, _ = admit(out_of_order, MAX_LATENESS)
        expected = oracle_query(kept, 0.0, 40.0, HALF_LIFE)
        self.assertLessEqual(
            abs(fixed.query(0.0, 40.0) - expected) / expected, 1e-12)

    def test_duplicate_events(self):
        """The same event id is delivered twice."""
        # The redelivery arrives after bucket 1 sealed, so the legacy
        # code forces it into bucket 2 and counts it twice.
        events = [("dup", 12.0, 1.0), ("seal1", 20.0, 0.0),
                  ("dup", 12.0, 1.0), ("seal2", 30.0, 0.0)]

        buggy = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy, events)
        buggy_total = buggy.query(0.0, 40.0)
        truth_events = [("dup", 12.0, 1.0), ("seal1", 20.0, 0.0),
                        ("seal2", 30.0, 0.0)]
        truth = oracle_query(admit(truth_events, MAX_LATENESS)[0],
                             0.0, 40.0, HALF_LIFE)
        self.assertGreater(abs(buggy_total - truth) / truth, 0.5)

        fixed = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed, events)
        self.assertEqual(fixed.stats["duplicates"], 1)
        self.assertEqual(fixed.journal[-1]["kind"], "duplicate")
        self.assertLessEqual(
            abs(fixed.query(0.0, 40.0) - truth) / truth, 1e-12)

    def test_cross_window_query_vs_recompute(self):
        """A query spanning several windows must match per-event recompute."""
        events = [(f"e{i}", t, round(0.5 + (i % 3), 3))
                  for i, t in enumerate([1.0, 9.9, 10.01, 19.9, 20.01,
                                         29.99, 30.5])]

        buggy = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy, events + [("seal", 45.0, 0.0)])
        # Query end at 15 cuts bucket 1; the legacy code counts all of it.
        truth = oracle_query(admit(events, MAX_LATENESS)[0],
                             0.0, 15.0, HALF_LIFE)
        self.assertGreater(abs(buggy.query(0.0, 15.0) - truth) / truth, 0.3)

        fixed = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed, events)
        self.assertLessEqual(
            abs(fixed.query(0.0, 15.0) - truth) / truth, 1e-12)

    def test_restart_recovery(self):
        """Snapshot/restore must preserve answers across a restart."""
        part1 = [("a", 5.0, 1.0), ("b", 15.0, 2.0),
                 ("seal1", 25.0, 0.0)]
        part2 = [("late-ish", 3.0, 1.0), ("c", 35.0, 1.0),
                 ("seal2", 45.0, 0.0)]

        # Legacy: the watermark is dropped on snapshot.  After restart the
        # late event lands in bucket 0 instead of bucket 2; a query
        # starting at 17 then sees it only in one of the two runs.
        buggy1 = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy1, part1)
        buggy2 = BuggyCounter.restore(buggy1.snapshot())
        feed(buggy2, part2)
        buggy_full = BuggyCounter(HALF_LIFE, WINDOW)
        feed(buggy_full, part1 + part2)
        self.assertNotAlmostEqual(buggy2.query(17.0, 50.0),
                                  buggy_full.query(17.0, 50.0),
                                  places=9)

        # Fixed: restarted run is identical to the uninterrupted run.
        fixed1 = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed1, part1)
        fixed2 = DecayingCounter.restore(fixed1.snapshot())
        feed(fixed2, part2)
        fixed_full = DecayingCounter(HALF_LIFE, MAX_LATENESS)
        feed(fixed_full, part1 + part2)
        for lo, hi in [(0.0, 50.0), (17.0, 50.0), (0.0, 1_000.0)]:
            self.assertEqual(fixed2.query(lo, hi), fixed_full.query(lo, hi))


if __name__ == "__main__":
    unittest.main(verbosity=2)
