"""Restart-recovery consistency: a counter that snapshots to JSON, dies and
restores mid-stream must answer identically to one that never restarted."""

import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decay_counter import DecayCounter
from oracle import Oracle
from scenarios import KINDS, build_scenario

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS_PER_KIND = 40
TOLERANCE = 1e-12


def scaled_error(got, expected):
    return abs(got - expected) / max(1.0, abs(expected))


class RestartRecoveryTest(unittest.TestCase):
    def test_restart_mid_stream_matches_uninterrupted_run(self):
        report = {"tolerance": TOLERANCE, "cases": [], "totals": {
            "cases": 0, "restarts": 0, "events": 0,
            "queries": 0, "value_checks": 0, "max_error": 0.0,
        }}
        worst = 0.0
        for kind in KINDS:
            for seed in range(SEEDS_PER_KIND):
                scenario = build_scenario(1000 + seed, kind)
                records = scenario["records"]
                rng = random.Random(9000 + seed)

                baseline = DecayCounter(
                    scenario["half_life"],
                    bucket_width=scenario["bucket_width"],
                    max_lateness=scenario["max_lateness"],
                )
                for event_id, ts, value in records:
                    baseline.add(event_id, ts, value)

                restarted = DecayCounter(
                    scenario["half_life"],
                    bucket_width=scenario["bucket_width"],
                    max_lateness=scenario["max_lateness"],
                )
                cuts = sorted(rng.sample(range(1, len(records)), 2))
                cursor = 0
                restarts = 0
                for cut in cuts + [len(records)]:
                    for event_id, ts, value in records[cursor:cut]:
                        restarted.add(event_id, ts, value)
                    cursor = cut
                    if cut < len(records):
                        restarted = DecayCounter.restore(restarted.snapshot_json())
                        restarts += 1

                self.assertEqual(restarted.stats, baseline.stats)
                self.assertEqual(restarted.dropped_events, baseline.dropped_events)

                max_error = 0.0
                for start, end in scenario["queries"]:
                    got = restarted.query(start, end)
                    expected = baseline.query(start, end)
                    error = scaled_error(got, expected)
                    max_error = max(max_error, error)
                    self.assertLessEqual(error, TOLERANCE,
                                         "%s seed=%d query=(%r, %r)" % (kind, seed, start, end))
                for T in scenario["value_times"]:
                    got = restarted.value_at(T)
                    expected = baseline.value_at(T)
                    error = scaled_error(got, expected)
                    max_error = max(max_error, error)
                    self.assertLessEqual(error, TOLERANCE,
                                         "%s seed=%d value_at(%r)" % (kind, seed, T))

                oracle = Oracle(scenario["half_life"], scenario["max_lateness"])
                for event_id, ts, value in records:
                    oracle.add(event_id, ts, value)
                for start, end in scenario["queries"][:10]:
                    self.assertLessEqual(
                        scaled_error(restarted.query(start, end), oracle.query(start, end)),
                        1e-9,
                    )

                worst = max(worst, max_error)
                report["cases"].append({
                    "kind": kind, "seed": 1000 + seed, "restarts": restarts,
                    "events": len(records), "queries": len(scenario["queries"]),
                    "value_checks": len(scenario["value_times"]),
                    "max_error": max_error,
                })
                totals = report["totals"]
                totals["cases"] += 1
                totals["restarts"] += restarts
                totals["events"] += len(records)
                totals["queries"] += len(scenario["queries"])
                totals["value_checks"] += len(scenario["value_times"])
                totals["max_error"] = max(totals["max_error"], max_error)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "restart_report.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=1)
        print("\nrestart: %d cases, %d restarts, %d events, worst scaled error %.3g"
              % (report["totals"]["cases"], report["totals"]["restarts"],
                 report["totals"]["events"], worst))
        print("restart: report -> %s" % path)


if __name__ == "__main__":
    unittest.main()
