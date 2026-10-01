"""demo_ci.py

Builds a synthetic evaluation dataset (seeded, reproducible), computes every
metric at several cutoffs, and reports percentile-bootstrap confidence
intervals for the macro (per-query mean) aggregation. Writes ci_data.json.

Also demonstrates the small-sample regime: with only a handful of queries
the percentile bootstrap is unreliable, so the library warns and the report
flags the interval as not robust.
"""

import json
import random
import warnings

import ranking_metrics as rm

KS = (1, 3, 5, 10)


def make_dataset(seed, n_queries):
    rng = random.Random(seed)
    queries = []
    for i in range(n_queries):
        n_docs = rng.randint(5, 20)
        scored = [(f"d{j}", rng.random()) for j in range(n_docs)]
        relevance = {}
        for j in range(n_docs):
            if rng.random() < 0.6:
                relevance[f"d{j}"] = rng.choice([0, 1, 1, 2, 3])
        queries.append((f"q{i}", scored, relevance))
    return queries


def report(result, n_boot=2000, seed=20261001):
    rows = []
    for metric in rm.METRICS:
        for k in KS:
            values = rm.metric_values(result, metric, k)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ci = rm.bootstrap_ci(values, n_boot=n_boot, seed=seed)
            ci["metric"] = metric
            ci["k"] = k
            ci["macro"] = result["macro"][metric][k]
            ci["micro"] = result["micro"][metric][k]
            ci["small_sample_warning"] = bool(caught)
            rows.append(ci)
    return rows


def main():
    queries = make_dataset(seed=11, n_queries=60)
    result = rm.evaluate(queries, ks=KS)
    rows = report(result)

    print(f"queries: {result['n_queries']}, bootstrap replicates: 2000, "
          "95% percentile interval")
    header = f"{'metric':<7}{'k':>4} {'point':>8} {'lo':>8} {'hi':>8} " \
             f"{'se':>8} {'macro':>8} {'micro':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['metric']:<7}{r['k']:>4} {r['point']:>8.4f} "
              f"{r['lo']:>8.4f} {r['hi']:>8.4f} {r['se']:>8.4f} "
              f"{r['macro']:>8.4f} {r['micro']:>8.4f}")

    # Small-sample regime: only 6 queries -> warning + flagged rows.
    small_queries = queries[:6]
    small_result = rm.evaluate(small_queries, ks=KS)
    small_rows = report(small_result, n_boot=1000)
    print(f"\nsmall-sample run ({small_result['n_queries']} queries): "
          "intervals flagged as not robust; report per-query values instead")

    with open("ci_data.json", "w") as fh:
        json.dump({
            "dataset": {"n_queries": result["n_queries"], "seed": 11},
            "bootstrap": {"n_boot": 2000, "seed": 20261001, "alpha": 0.05},
            "rows": rows,
            "small_sample": {
                "n_queries": small_result["n_queries"],
                "note": "percentile bootstrap not robust at this size",
                "rows": small_rows,
            },
        }, fh, indent=2)
    print("wrote ci_data.json")


if __name__ == "__main__":
    main()
