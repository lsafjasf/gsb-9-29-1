"""Restart / persistence tests.

A counter is snapshot-restored (through a real json.dumps/loads round
trip, as it would cross a process boundary) at random cut points while a
random out-of-order/duplicate stream is being ingested.  The restarted
counter must produce bit-identical query results, stats and journal
entries to the uninterrupted run.  Clock regressions are injected too.
"""

import json
import os
import random
import unittest

from common import (DecayingCounter, make_events, RESULTS_DIR)


def split_points(rng, length):
    # 1..3 random cut points, sorted and unique.
    k = rng.randint(1, 3)
    points = sorted(set(rng.randrange(1, length) for _ in range(k)))
    return points


def run_with_restarts(events, points, half_life, max_lateness,
                      clock_at=None):
    """Feed events, restarting from a JSON-round-tripped snapshot."""
    counter = DecayingCounter(half_life, max_lateness)
    cuts = set(points)
    for index, event in enumerate(events):
        if index in cuts:
            blob = json.dumps(counter.snapshot())
            counter = DecayingCounter.restore(json.loads(blob))
        counter.add(*event)
    return counter


class RecoveryTests(unittest.TestCase):

    def _compare(self, seed):
        rng = random.Random(seed)
        half_life = rng.choice([1.0, 7.5, 60.0])
        span = rng.uniform(100.0, 5_000.0)
        max_lateness = rng.choice([0.0, span * 0.05, span * 0.2, span])
        events = make_events(
            rng, rng.randint(50, 500), span=span,
            out_of_order=rng.choice([0.2, 0.6, 1.0]),
            duplicate_rate=rng.choice([0.0, 0.1]),
            jitter=max_lateness * rng.choice([0.5, 1.5, 5.0]) + 1e-9,
        )
        points = split_points(rng, len(events))

        restarted = run_with_restarts(events, points, half_life,
                                      max_lateness)
        uninterrupted = DecayingCounter(half_life, max_lateness)
        for event in events:
            uninterrupted.add(*event)

        # Snapshot round-trips through JSON; internal state is identical.
        self.assertEqual(restarted.snapshot(), uninterrupted.snapshot())

        probe = random.Random(seed + 1)
        time_lo = min(t for _, t, _ in events)
        time_hi = max(t for _, t, _ in events)
        for _ in range(30):
            a = probe.uniform(time_lo - span * 0.1, time_hi + span * 0.1)
            b = probe.uniform(a, time_hi + span * 0.1)
            self.assertEqual(restarted.query(a, b),
                             uninterrupted.query(a, b))
        return len(events), len(points), restarted.stats

    def test_restart_matches_no_restart(self):
        totals = {"events": 0, "restarts": 0}
        sample = None
        for seed in range(300):
            events, restarts, stats = self._compare(seed)
            totals["events"] += events
            totals["restarts"] += restarts
            if seed == 0:
                sample = stats
        os.makedirs(RESULTS_DIR, exist_ok=True)
        report = os.path.join(RESULTS_DIR, "recovery_report.txt")
        with open(report, "w", encoding="utf-8") as fh:
            fh.write("Restart recovery report\n")
            fh.write("=" * 48 + "\n")
            fh.write("random streams          : 300\n")
            fh.write(f"events fed              : {totals['events']}\n")
            fh.write(f"restart cut points      : {totals['restarts']}\n")
            fh.write("comparison              : snapshot() equality + "
                     "30 interval probes each\n")
            fh.write("mismatches              : 0\n")
            fh.write("equality                : bitwise identical (==)\n")
            fh.write(f"sample stream stats     : {sample}\n")

    def test_restart_with_clock_regression(self):
        def make_counter():
            schedule = iter([100.0, 200.0, 150.0, 300.0, 250.0, 400.0])
            return DecayingCounter(half_life=10.0, max_lateness=1e9,
                                   now_fn=lambda: next(schedule))

        counter = make_counter()
        counter.add("a", timestamp=None, value=1.0)  # 100
        counter.add("b", timestamp=None, value=2.0)  # 200
        counter.add("c", timestamp=None, value=1.0)  # 150 -> regression

        blob = json.dumps(counter.snapshot())
        rest_of_schedule = iter([300.0, 250.0, 400.0])
        restored = DecayingCounter.restore(
            json.loads(blob),
            now_fn=lambda: next(rest_of_schedule))
        counter_full = make_counter()
        for event_id, value in [("a", 1.0), ("b", 2.0), ("c", 1.0),
                                ("d", 1.0), ("e", 3.0), ("f", 1.0)]:
            counter_full.add(event_id, timestamp=None, value=value)

        restored.add("d", timestamp=None, value=1.0)   # 300
        restored.add("e", timestamp=None, value=3.0)   # 250 -> regression
        restored.add("f", timestamp=None, value=1.0)   # 400

        self.assertEqual(restored.snapshot(), counter_full.snapshot())
        for end in (0.0, 200.0, 350.0, 1_000.0):
            self.assertEqual(restored.query(0.0, end),
                             counter_full.query(0.0, end))
        self.assertEqual(restored.stats["clock_regressions"],
                         counter_full.stats["clock_regressions"])

    def test_snapshot_json_safe(self):
        counter = DecayingCounter(5.0, 50.0)
        counter.add("a", 10.0, 1.0)
        counter.add("a", 11.0, 1.0)  # duplicate record in journal
        counter.add("late", -100.0, 1.0)  # late record in journal
        blob = json.dumps(counter.snapshot())
        restored = DecayingCounter.restore(json.loads(blob))
        self.assertEqual(restored.snapshot(), counter.snapshot())


if __name__ == "__main__":
    unittest.main(verbosity=2)
