"""difftest.py

Differential test: compare ranking_metrics against the independent
reference_impl on thousands of random rankings / relevance labels, including
tied scores, missing annotations, zero-relevance queries and empty rankings.

Agreement is asserted with EXACT floating equality (not an epsilon): both
implementations fold per-position contributions in the same left-to-right
order, so identical conventions must produce bit-identical numbers.

Also enforces the cross-cutoff self-consistency rules on every random query:
  DCG@k, AP@k, RR@k, Hit@k are non-decreasing as k grows;
  value(k) for k beyond the ranking length stays constant.
"""

import math
import random
import sys

import ranking_metrics as rm
import reference_impl as ref


def gen_case(rng):
    """One random (scored_docs, relevance) pair with messy edge conditions."""
    n_pool = rng.randint(0, 14)
    pool = [f"d{i}" for i in range(n_pool)]

    scored_docs = []
    for doc_id in pool:
        if rng.random() < 0.85:
            # Small discrete score space forces frequent ties.
            score = rng.choice([0, 0, 1, 1, 2, 3])
            if rng.random() < 0.3:
                score += 0.5
            scored_docs.append((doc_id, score))

    relevance = {}
    for doc_id in pool:
        if rng.random() < 0.7:  # some documents stay unlabeled
            relevance[doc_id] = rng.choice([0, 1, 1, 2, 3])
    if rng.random() < 0.3:
        # Relevant document that can never be retrieved (not in the ranking).
        relevance["ghost"] = rng.choice([1, 2])
    return scored_docs, relevance


def assert_exact(a, b, context):
    assert isinstance(a, float) and isinstance(b, float), context
    assert a == b, f"{context}: {a!r} != {b!r}"


def check_query(qid, scored_docs, relevance, gain, ks):
    qs = rm.QuerySeries(scored_docs, relevance, gain=gain)

    # Per-cutoff exact agreement with the reference.
    for k in ks:
        assert_exact(qs.value("dcg", k),
                     ref.dcg_at_k(scored_docs, relevance, k, gain),
                     f"{qid} dcg@{k} gain={gain}")
        assert_exact(qs.value("ndcg", k),
                     ref.ndcg_at_k(scored_docs, relevance, k, gain),
                     f"{qid} ndcg@{k} gain={gain}")
        assert_exact(qs.value("ap", k),
                     ref.ap_at_k(scored_docs, relevance, k),
                     f"{qid} ap@{k}")
        assert_exact(qs.value("rr", k),
                     ref.rr_at_k(scored_docs, relevance, k),
                     f"{qid} rr@{k}")
        assert_exact(qs.value("hit", k),
                     ref.hit_at_k(scored_docs, relevance, k),
                     f"{qid} hit@{k}")
        assert_exact(qs.idcg_value(k),
                     ref.idcg_at_k(relevance, k, gain),
                     f"{qid} idcg@{k}")

    # Monotonicity / clamping self-consistency.
    n = len(scored_docs)
    if n > 0:
        for metric in ("dcg", "ap", "rr", "hit"):
            series = [qs.value(metric, k) for k in range(1, n + 1)]
            for prev, curr in zip(series, series[1:]):
                assert curr >= prev, (
                    f"{qid} {metric} decreased: {prev} -> {curr}")
        for metric in ("dcg", "ap", "rr", "hit"):
            assert qs.value(metric, n) == qs.value(metric, n + 5), (
                f"{qid} {metric} not constant past ranking length")
        # nDCG past the ranking length can decrease: IDCG keeps counting
        # ideal labels that the ranking never retrieved. Reference agrees.


def check_aggregation(queries, gain, ks):
    main_result = rm.evaluate(queries, ks, gain=gain)
    ref_result = ref.evaluate(queries, ks, gain=gain)

    for qid, _, _ in queries:
        for metric in rm.METRICS:
            for k in ks:
                assert_exact(
                    main_result["per_query"][qid][metric][k],
                    ref_result["per_query"][qid][metric][k],
                    f"agg per-query {qid} {metric}@{k}")
    for scope in ("macro", "micro"):
        for metric in rm.METRICS:
            for k in ks:
                assert_exact(
                    main_result[scope][metric][k],
                    ref_result[scope][metric][k],
                    f"agg {scope} {metric}@{k}")


def main():
    rng = random.Random(42)
    n_batches = 3000
    ks = [1, 2, 3, 5, 20]
    total_queries = 0

    for batch in range(n_batches):
        gain = "linear" if batch % 2 == 0 else "exp"
        batch_size = rng.randint(1, 8)
        queries = []
        for j in range(batch_size):
            scored_docs, relevance = gen_case(rng)
            qid = f"q{batch}_{j}"
            queries.append((qid, scored_docs, relevance))
            per_query_ks = ks + list(range(1, len(scored_docs) + 1))
            check_query(qid, scored_docs, relevance, gain,
                        sorted(set(per_query_ks)))
            total_queries += 1
        check_aggregation(queries, gain, ks)

    print(f"PASS: {n_batches} batches / {total_queries} random queries, "
          "all metrics at all cutoffs bit-identical to reference, "
          "monotonicity and clamping assertions held.")


if __name__ == "__main__":
    sys.exit(main())
