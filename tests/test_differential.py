"""Randomized differential test: DecayCounter vs full-recompute Oracle.

Covers random event streams with out-of-order and duplicate delivery across
five scenario families, comparing arbitrary interval queries and value_at.
Writes the comparison data to results/differential_report.json.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decay_counter import DecayCounter
from oracle import Oracle
from scenarios import KINDS, build_scenario

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS_PER_KIND = 60
TOLERANCE = 1e-9


def scaled_error(got, expected):
    return abs(got - expected) / max(1.0, abs(expected))


class DifferentialTest(unittest.TestCase):
    def test_randomized_streams_match_full_recompute(self):
        report = {
            "tolerance": TOLERANCE,
            "seeds_per_kind": SEEDS_PER_KIND,
            "scenarios": [],
            "totals": {
                "scenarios": 0,
                "events": 0,
                "queries": 0,
                "value_checks": 0,
                "max_query_error": 0.0,
                "max_value_error": 0.0,
            },
        }
        worst = 0.0
        for kind in KINDS:
            for seed in range(SEEDS_PER_KIND):
                scenario = build_scenario(seed, kind)
                counter = DecayCounter(
                    scenario["half_life"],
                    bucket_width=scenario["bucket_width"],
                    max_lateness=scenario["max_lateness"],
                )
                oracle = Oracle(scenario["half_life"], scenario["max_lateness"])
                for event_id, ts, value in scenario["records"]:
                    got_status = counter.add(event_id, ts, value)
                    expected_status = oracle.add(event_id, ts, value)
                    self.assertEqual(got_status, expected_status)
                self.assertEqual(counter.stats, oracle.stats)
                self.assertEqual(counter.dropped_events, oracle.dropped_events)

                max_query_error = 0.0
                for start, end in scenario["queries"]:
                    got = counter.query(start, end)
                    expected = oracle.query(start, end)
                    error = scaled_error(got, expected)
                    max_query_error = max(max_query_error, error)
                    self.assertLessEqual(error, TOLERANCE,
                                         "%s seed=%d query=(%r, %r)" % (kind, seed, start, end))

                max_value_error = 0.0
                for T in scenario["value_times"]:
                    got = counter.value_at(T)
                    expected = oracle.value_at(T)
                    error = scaled_error(got, expected)
                    max_value_error = max(max_value_error, error)
                    self.assertLessEqual(error, TOLERANCE,
                                         "%s seed=%d value_at(%r)" % (kind, seed, T))

                worst = max(worst, max_query_error, max_value_error)
                report["scenarios"].append({
                    "kind": kind,
                    "seed": seed,
                    "events": len(scenario["records"]),
                    "accepted": counter.stats["accepted"],
                    "duplicates": counter.stats["duplicates"],
                    "dropped_late": counter.stats["dropped_late"],
                    "queries": len(scenario["queries"]),
                    "value_checks": len(scenario["value_times"]),
                    "max_query_error": max_query_error,
                    "max_value_error": max_value_error,
                })
                totals = report["totals"]
                totals["scenarios"] += 1
                totals["events"] += len(scenario["records"])
                totals["queries"] += len(scenario["queries"])
                totals["value_checks"] += len(scenario["value_times"])
                totals["max_query_error"] = max(totals["max_query_error"], max_query_error)
                totals["max_value_error"] = max(totals["max_value_error"], max_value_error)

        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "differential_report.json")
        with open(path, "w") as handle:
            json.dump(report, handle, indent=1)
        print("\ndifferential: %d scenarios, %d events, %d queries, %d value checks"
              % (report["totals"]["scenarios"], report["totals"]["events"],
                 report["totals"]["queries"], report["totals"]["value_checks"]))
        print("differential: worst scaled error %.3g (tolerance %.3g), report -> %s"
              % (worst, TOLERANCE, path))


if __name__ == "__main__":
    unittest.main()
