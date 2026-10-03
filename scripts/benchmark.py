#!/usr/bin/env python3
"""Compare the aligner with human gold annotations on every dataset.

Outputs:
  outputs/benchmark_report.json  per-dataset + aggregate metrics
  outputs/error_distribution.csv error-type distribution
  stdout                         summary table

Run:  python3 scripts/benchmark.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentalign import AlignConfig, align, evaluate, load_dataset
from sentalign.evaluate import ERROR_TYPES, aggregate

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".json")
    )
    rows = []
    report = {"datasets": []}

    header = (
        f"{'dataset':22s} {'n':>3s} {'m':>3s} {'acc':>6s} {'P':>6s} "
        f"{'R':>6s} {'F1':>6s} {'flagged':>7s}  errors"
    )
    print(header)
    print("-" * len(header))

    for path in paths:
        ds = load_dataset(path)
        result = align(ds.src, ds.tgt, AlignConfig())
        ev = evaluate(result, ds)

        flagged_beads = sum(1 for b in result.beads if b.flagged)
        errors = {t: ev.counts[t] for t in ERROR_TYPES if t not in ("correct",) and ev.counts[t]}
        print(
            f"{ds.name:22s} {len(ds.src):3d} {len(ds.tgt):3d} {ev.accuracy:6.3f} "
            f"{ev.link_precision:6.3f} {ev.link_recall:6.3f} "
            f"{ev.link_f1:6.3f} {flagged_beads:7d}  {json.dumps(errors, ensure_ascii=False)}"
        )

        rows.append(
            {
                "dataset": ds.name,
                "n_src": len(ds.src),
                "n_tgt": len(ds.tgt),
                "accuracy": round(ev.accuracy, 4),
                "link_precision": round(ev.link_precision, 4),
                "link_recall": round(ev.link_recall, 4),
                "link_f1": round(ev.link_f1, 4),
                "beads": len(result.beads),
                "flagged_beads": flagged_beads,
                **{f"err_{t}": ev.counts[t] for t in ERROR_TYPES},
            }
        )
        report["datasets"].append(
            {
                "name": ds.name,
                "n_src": len(ds.src),
                "n_tgt": len(ds.tgt),
                "total_cost": result.total_cost,
                "length_ratio": round(result.length_ratio, 4),
                "accuracy": round(ev.accuracy, 4),
                "link_precision": round(ev.link_precision, 4),
                "link_recall": round(ev.link_recall, 4),
                "link_f1": round(ev.link_f1, 4),
                "beads": len(result.beads),
                "flagged_beads": flagged_beads,
                "error_counts": ev.counts,
                "flagged_precision": (
                    round(ev.flagged_correct / ev.flagged_total, 4) if ev.flagged_total else None
                ),
                "unflagged_precision": (
                    round(ev.unflagged_correct / ev.unflagged_total, 4) if ev.unflagged_total else None
                ),
                "path": [list(b.key) for b in result.beads],
            }
        )

    all_results = []
    for path in paths:
        ds = load_dataset(path)
        all_results.append(evaluate(align(ds.src, ds.tgt, AlignConfig()), ds))
    agg = aggregate(all_results)
    report["aggregate"] = {
        k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) and k == "error_distribution" else v)
        for k, v in agg.items()
    }
    print("-" * len(header))
    print(
        f"AGGREGATE acc={agg['accuracy']:.3f}  "
        f"distribution={ {k: round(v, 3) for k, v in agg['error_distribution'].items()} }"
    )

    with open(os.path.join(OUT_DIR, "benchmark_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    fieldnames = [
        "dataset", "n_src", "n_tgt", "accuracy",
        "link_precision", "link_recall", "link_f1", "beads", "flagged_beads",
        *[f"err_{t}" for t in ERROR_TYPES],
    ]
    with open(os.path.join(OUT_DIR, "error_distribution.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        agg_row = {
            "dataset": "AGGREGATE",
            "n_src": "",
            "n_tgt": "",
            "accuracy": round(agg["accuracy"], 4),
            "link_precision": "",
            "link_recall": "",
            "link_f1": "",
            "beads": "",
            "flagged_beads": "",
            **{f"err_{t}": agg["error_counts"][t] for t in ERROR_TYPES},
        }
        writer.writerow(agg_row)

    print(f"\nWrote {os.path.join('outputs', 'benchmark_report.json')}")
    print(f"Wrote {os.path.join('outputs', 'error_distribution.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
