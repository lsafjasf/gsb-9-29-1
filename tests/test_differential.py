"""Differential testing: fixed counter vs full per-event recomputation.

Runs many seeded random event streams (out-of-order arrivals,
duplicates, events beyond the lateness bound) and many random intervals
(empty / point / dense / ultra-long).  Asserts the error stays inside a
tight tolerance and writes the aggregate numbers to
results/differential_report.txt.
"""

import json
import os
import random
import time
import unittest

from common import (DecayingCounter, admit, oracle_query, make_events,
                    random_queries, rel_error, RESULTS_DIR)

TRIALS = int(os.environ.get("FUZZ_TRIALS", "400"))
REL_TOL = 1e-9
ABS_TOL = 1e-12


def run_one_trial(seed):
    rng = random.Random(seed)
    half_life = rng.choice([0.5, 1.0, 5.0, 17.0, 100.0])
    span = rng.uniform(10.0, 10_000.0)
    max_lateness = rng.choice([0.0, span * 0.01, span * 0.1, span])
    count = rng.randint(1, 600)

    events = make_events(
        rng, count, span=span,
        out_of_order=rng.choice([0.0, 0.3, 0.8, 1.0]),
        duplicate_rate=rng.choice([0.0, 0.05, 0.2]),
        base_time=rng.choice([0.0, 1_000.0, 1e8]),
        # Displacement comparable to the lateness bound: exercises both
        # accepted-late and rejected-late paths.
        jitter=max_lateness * rng.choice([0.5, 1.0, 2.0, 10.0]) + 1e-9,
    )

    counter = DecayingCounter(half_life, max_lateness)
    for event_id, t, value in events:
        counter.add(event_id, t, value)

    kept, dropped = admit(events, max_lateness)
    # Admission must agree with the reference policy.
    assert counter.stats["accepted"] == len(kept), seed
    assert counter.stats["duplicates"] + counter.stats["late"] == len(dropped), seed

    time_lo = min(t for _, t, _ in events)
    time_hi = max(t for _, t, _ in events)
    queries = random_queries(rng, 60, time_lo, time_hi, half_life)

    worst = 0.0
    worst_case = None
    mismatches = 0
    for start, end in queries:
        expected = oracle_query(kept, start, end, half_life)
        actual = counter.query(start, end)
        error = rel_error(actual, expected)
        if error > worst:
            worst = error
            worst_case = (start, end, actual, expected)
        if error > REL_TOL or (expected == 0.0 and abs(actual) > ABS_TOL):
            mismatches += 1
    dupes = counter.stats["duplicates"]
    late = counter.stats["late"]
    return {
        "seed": seed, "count": len(events), "kept": len(kept),
        "dropped": len(dropped), "duplicates": dupes, "late": late,
        "queries": len(queries), "worst": worst,
        "worst_case": worst_case, "mismatches": mismatches,
    }


class DifferentialTests(unittest.TestCase):

    def test_many_random_streams(self):
        started = time.time()
        summaries = [run_one_trial(10_000 + seed)
                     for seed in range(TRIALS)]
        elapsed = time.time() - started

        total_events = sum(s["count"] for s in summaries)
        total_kept = sum(s["kept"] for s in summaries)
        total_dropped = sum(s["dropped"] for s in summaries)
        total_dupes = sum(s["duplicates"] for s in summaries)
        total_late = sum(s["late"] for s in summaries)
        total_queries = sum(s["queries"] for s in summaries)
        worst_overall = max(s["worst"] for s in summaries)
        total_mismatches = sum(s["mismatches"] for s in summaries)
        mean_worst = sum(s["worst"] for s in summaries) / len(summaries)
        worst_seed = max(summaries, key=lambda s: s["worst"])

        self.assertEqual(total_mismatches, 0,
                         f"{total_mismatches} queries outside tolerance")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        report = os.path.join(RESULTS_DIR, "differential_report.txt")
        with open(report, "w", encoding="utf-8") as fh:
            fh.write("Decaying counter differential test report\n")
            fh.write("=" * 48 + "\n")
            fh.write(f"trials                 : {TRIALS}\n")
            fh.write(f"events generated       : {total_events}\n")
            fh.write(f"events admitted        : {total_kept}\n")
            fh.write(f"events rejected (dup+late): {total_dropped}\n")
            fh.write(f"  of which duplicates    : {total_dupes}\n")
            fh.write(f"  of which too late      : {total_late}\n")
            fh.write(f"interval queries       : {total_queries}\n")
            fh.write(f"tolerance              : rel <= {REL_TOL}, "
                     f"abs <= {ABS_TOL} (zero truth)\n")
            fh.write(f"queries out of tolerance: {total_mismatches}\n")
            fh.write(f"max relative error     : {worst_overall:.3e}\n")
            fh.write(f"mean per-trial max err : {mean_worst:.3e}\n")
            fh.write(f"worst trial seed       : {worst_seed['seed']} "
                     f"(events={worst_seed['count']}, "
                     f"dropped={worst_seed['dropped']})\n")
            if worst_seed["worst_case"] is not None:
                start, end, actual, expected = worst_seed["worst_case"]
                fh.write(f"worst query            : [{start:.6f}, {end:.6f}]\n")
                fh.write(f"  recomputed           : {expected!r}\n")
                fh.write(f"  counter              : {actual!r}\n")
            fh.write(f"elapsed seconds        : {elapsed:.2f}\n")
        print("\n" + open(report, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
