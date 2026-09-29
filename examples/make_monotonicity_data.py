"""Generate monotonicity-detection data and reports.

Run from the repo root:
    python3 examples/make_monotonicity_data.py

Writes data/monotonicity_cases.json with, per scenario, the fitted bin
table and the monotonicity diagnostic (direction, violations, suggested
merges). Deterministic: fixed seeds, no randomness in the binners.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_binning import OptimalBinner, QuantileBinner

OUT_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "data", "monotonicity_cases.json")


def scenario_clean_increasing():
    rng = random.Random(20260930)
    xs, ys = [], []
    for _ in range(8000):
        x = rng.random()
        xs.append(x)
        ys.append(1 if rng.random() < 0.02 + 0.5 * x else 0)
    binner = OptimalBinner(n_bins=6, monotone="auto")
    return xs, ys, binner, "bad rate truly increases with x; auto direction"


def scenario_u_shaped_violation():
    rng = random.Random(20260930)
    xs, ys = [], []
    for _ in range(8000):
        x = rng.random()
        p = 0.05 + 3.2 * (x - 0.5) ** 2
        xs.append(x)
        ys.append(1 if rng.random() < p else 0)
    binner = OptimalBinner(n_bins=6, monotone=None)
    return xs, ys, binner, "U-shaped risk: violations must be reported with merge suggestions"


def scenario_forced_monotone_repair():
    rng = random.Random(20260930)
    xs, ys = [], []
    for _ in range(8000):
        x = rng.random()
        p = 0.05 + 3.2 * (x - 0.5) ** 2
        xs.append(x)
        ys.append(1 if rng.random() < p else 0)
    binner = OptimalBinner(n_bins=6, monotone="increasing")
    return xs, ys, binner, "same U-shape but monotone='increasing' forces repair merges"


def scenario_long_tail_quantile():
    rng = random.Random(20260930)
    xs = [rng.expovariate(1.0) for _ in range(8000)] + [1e6, 1e7, 1e8]
    ys = [1 if rng.random() < min(v / 5.0, 0.95) else 0 for v in xs]
    binner = QuantileBinner(n_bins=8)
    return xs, ys, binner, "extreme long tail: outliers isolated, inlier bins stay spread"


def main():
    scenarios = {
        "clean_increasing": scenario_clean_increasing,
        "u_shaped_violation": scenario_u_shaped_violation,
        "forced_monotone_repair": scenario_forced_monotone_repair,
        "long_tail_quantile": scenario_long_tail_quantile,
    }
    out = {}
    for name, build in scenarios.items():
        xs, ys, binner, note = build()
        binner.fit(xs, ys)
        rep = binner.report()
        rep["scenario"] = note
        out[name] = rep
        mono = rep.get("monotonicity", {})
        rates = [b["bad_rate"] for b in rep["bins"]
                 if b["kind"] == "normal" and b["count"] > 0]
        print("[%s] %s" % (name, note))
        print("  edges: %s" % [round(e, 4) for e in rep["edges"]])
        print("  bad rates: %s" % [round(r, 4) for r in rates])
        if mono:
            print("  direction=%s is_monotone=%s violations=%d suggestions=%d"
                  % (mono["direction"], mono["is_monotone"],
                     len(mono["violations"]),
                     len(mono["merge_suggestions"])))
        for w in rep["warnings"]:
            print("  warning: %s" % w)
        print()
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print("wrote %s" % os.path.relpath(OUT_PATH))


if __name__ == "__main__":
    main()
