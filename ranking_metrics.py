"""ranking_metrics.py

Deterministic ranking metrics for retrieval / recommendation evaluation.
Python 3 standard library only.

Deterministic conventions (see README.md for the full discussion):

1. Ties in score are broken by ascending document id. Two implementations
   therefore cannot disagree merely because of tie handling.
2. A ranked document that has no relevance label is treated as relevance 0
   (missing annotation = non-relevant, never "skip this position").
3. Relevance values must be non-negative; a negative value raises ValueError.
4. AP@k divides by R, the total number of relevant documents for the query
   (not min(R, k)). Consequently AP@k is non-decreasing in k.
5. Queries with no relevant document get AP = RR = Hit = DCG = 0 and
   nDCG = `zero_rel_ndcg` (default 0.0).
6. For DCG/AP/RR/Hit a cutoff k larger than the ranking length clamps to
   the ranking length. For nDCG the denominator still counts ideal labels
   up to k (missing retrieved positions gain 0), so nDCG@k may keep
   decreasing past the ranking length; nDCG is not monotone in general.

Metrics implemented (all at arbitrary cutoffs k):

  DCG@k   gain linear or exponential (2^r - 1), discount 1 / log2(rank + 1)
  nDCG@k  DCG@k / IDCG@k (IDCG counts ideal labels up to k)
  AP@k    sum_{i<=k, rel_i>0} P@i / R
  RR@k    1 / rank of the first relevant document, 0 if none
  Hit@k   1 if any relevant document appears in the top k, else 0
"""

from __future__ import annotations

import math
import random
import warnings
from itertools import accumulate

METRICS = ("dcg", "ndcg", "ap", "rr", "hit")

GAIN_FUNCTIONS = {
    "linear": lambda r: r,
    "exp": lambda r: 2.0 ** r - 1.0,
}


def rank_docs(scored_docs):
    """Return document ids in deterministic rank order.

    Sort key: score descending, then document id ascending. `scored_docs` is
    an iterable of (doc_id, score) pairs; doc_id must be orderable (str/int).
    """
    return sorted(scored_docs, key=lambda item: (-item[1], item[0]))


def _discounted_gains(gains):
    """Per-position DCG contributions g_i / log2(i + 1), i is 0-based here."""
    return [g / math.log2(i + 2) for i, g in enumerate(gains)]


class QuerySeries:
    """All metric values for one query, precomputed for every position.

    Each stored series is indexed by (k - 1), i.e. series[k - 1] is the @k
    value. Querying k > ranking length clamps to the last position.
    """

    def __init__(self, scored_docs, relevance, gain="linear",
                 zero_rel_ndcg=0.0):
        if gain not in GAIN_FUNCTIONS:
            raise ValueError(f"unknown gain: {gain!r}")
        for value in relevance.values():
            if value < 0:
                raise ValueError("relevance values must be non-negative")

        gain_fn = GAIN_FUNCTIONS[gain]
        ranked = rank_docs(scored_docs)
        rels = [float(relevance.get(doc_id, 0.0)) for doc_id, _ in ranked]
        gains = [gain_fn(r) for r in rels]
        binary = [1.0 if r > 0 else 0.0 for r in rels]

        # DCG / nDCG
        self.dcg = list(accumulate(_discounted_gains(gains)))
        ideal_gains = sorted(
            (gain_fn(float(v)) for v in relevance.values()), reverse=True
        )
        self._idcg = list(accumulate(_discounted_gains(ideal_gains)))
        self._zero_rel_ndcg = float(zero_rel_ndcg)

        # AP: numerator = sum over hit positions i of (#hits_so_far / i)
        self.R = sum(1 for v in relevance.values() if v > 0)
        hits_so_far = list(accumulate(binary))
        ap_contrib = [
            b * (hits_so_far[i] / (i + 1)) for i, b in enumerate(binary)
        ]
        self._ap_num = list(accumulate(ap_contrib))
        self.ap = [
            (v / self.R if self.R > 0 else 0.0) for v in self._ap_num
        ]

        # RR / Hit
        self.rr = []
        self.hit = []
        first_hit = None
        for i, b in enumerate(binary, start=1):
            if b > 0 and first_hit is None:
                first_hit = i
            self.rr.append(0.0 if first_hit is None else 1.0 / first_hit)
            self.hit.append(0.0 if first_hit is None else 1.0)

    def _clamped_index(self, series, k):
        if k < 1:
            raise ValueError("cutoff k must be >= 1")
        if not series:
            return None
        return min(k, len(series)) - 1

    def value(self, metric, k):
        if metric == "ndcg":
            # IDCG counts up to k ideal labels even when the ranking has
            # fewer than k documents; missing retrieved positions gain 0.
            if k < 1:
                raise ValueError("cutoff k must be >= 1")
            idx = self._clamped_index(self.dcg, k)
            dcg = 0.0 if idx is None else self.dcg[idx]
            idcg = self.idcg_value(k)
            return dcg / idcg if idcg > 0 else self._zero_rel_ndcg
        series = getattr(self, metric)
        idx = self._clamped_index(series, k)
        if idx is None:
            return self._empty_ranking_value(metric)
        return series[idx]

    def idcg_value(self, k):
        idx = self._clamped_index(self._idcg, k)
        return 0.0 if idx is None else self._idcg[idx]

    def ap_num_value(self, k):
        idx = self._clamped_index(self._ap_num, k)
        return 0.0 if idx is None else self._ap_num[idx]

    def _empty_ranking_value(self, metric):
        if metric == "ndcg":
            # No retrieved documents: nDCG is 0 when relevant docs existed,
            # otherwise the configured zero-relevance convention applies.
            return 0.0 if self._idcg else 0.0
        return 0.0


def evaluate(queries, ks, gain="linear", zero_rel_ndcg=0.0):
    """Evaluate a collection of queries at the given cutoffs.

    `queries` is an iterable of (qid, scored_docs, relevance) tuples.

    Returns a dict:
      per_query[qid][metric][k]  per-query metric values
      macro[metric][k]           mean over queries (each query equal weight)
      micro[metric][k]           global pooling across queries
      n_queries                  number of queries
    """
    ks = sorted(set(ks))
    per_query = {}
    series = {}
    for qid, scored_docs, relevance in queries:
        qs = QuerySeries(
            scored_docs, relevance, gain=gain, zero_rel_ndcg=zero_rel_ndcg
        )
        series[qid] = qs
        per_query[qid] = {
            metric: {k: qs.value(metric, k) for k in ks}
            for metric in METRICS
        }

    n = len(series)
    macro = {m: {} for m in METRICS}
    micro = {m: {} for m in METRICS}
    for k in ks:
        for metric in METRICS:
            values = [qs.value(metric, k) for qs in series.values()]
            macro[metric][k] = sum(values) / n if n else 0.0
            micro[metric][k] = _micro_value(metric, series, k, n)

    return {
        "per_query": per_query,
        "macro": macro,
        "micro": micro,
        "n_queries": n,
    }


def _micro_value(metric, series, k, n):
    """Global ("micro") pooling across queries.

    * dcg / rr / hit: identical to the macro mean (they are already
      additive or binary per query).
    * ndcg: pool numerator and denominator first: sum DCG / sum IDCG.
    * ap:  sum AP numerators / sum R across queries.
    Zero-relevance queries carry no weight in the pooled ratios.
    """
    if n == 0:
        return 0.0
    if metric == "ndcg":
        total_dcg = sum(qs.value("dcg", k) for qs in series.values())
        total_idcg = sum(qs.idcg_value(k) for qs in series.values())
        return total_dcg / total_idcg if total_idcg > 0 else 0.0
    if metric == "ap":
        total_num = sum(qs.ap_num_value(k) for qs in series.values())
        total_r = sum(qs.R for qs in series.values())
        return total_num / total_r if total_r > 0 else 0.0
    return sum(qs.value(metric, k) for qs in series.values()) / n


def metric_values(result, metric, k):
    """Extract the list of per-query values for metric@k (macro input)."""
    return [result["per_query"][qid][metric][k]
            for qid in result["per_query"]]


def bootstrap_ci(values, n_boot=2000, seed=20261001, alpha=0.05,
                 small_sample_threshold=30):
    """Percentile bootstrap confidence interval for the mean over queries.

    Resamples queries (rows) with replacement, recomputes the macro mean for
    each replicate, and returns the alpha/2 and 1-alpha/2 percentiles.
    Results are reproducible for a fixed `seed`.

    With fewer than `small_sample_threshold` queries a warning is emitted:
    the percentile bootstrap is unstable for tiny query counts. In that
    regime prefer reporting every per-query value, a larger alpha is not a
    remedy -- collect more queries, or use exact/permutation methods.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    if n < small_sample_threshold:
        warnings.warn(
            f"bootstrap over only {n} queries: percentile intervals are "
            "unstable at this size; report per-query values alongside and "
            "prefer collecting more queries",
            stacklevel=2,
        )

    point = sum(vals) / n
    rng = random.Random(seed)
    replicates = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        replicates.append(total / n)
    replicates.sort()

    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    boot_mean = sum(replicates) / n_boot
    variance = sum((r - boot_mean) ** 2 for r in replicates) / (n_boot - 1)
    return {
        "point": point,
        "lo": replicates[lo_idx],
        "hi": replicates[hi_idx],
        "se": math.sqrt(variance),
        "n": n,
        "n_boot": n_boot,
        "alpha": alpha,
    }
