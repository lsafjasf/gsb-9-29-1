"""Unbiasedness verification for reservoir.py.

Repeats each experiment many times with a fixed-seed random source and
prints empirical inclusion frequencies next to theoretical probabilities.
Fully deterministic: same output on every run.

For the weighted sampler the theoretical inclusion probabilities are
computed exactly (no Monte Carlo) via the exponential-race view of A-Res:
item i draws T_i ~ Exp(w_i) independently and the k smallest T_i are
sampled, so

    P(i in sample) = integral_0^inf  w_i e^{-w_i t}
                     * P(at most k-1 of the other T_j are < t)  dt,

which expands (inclusion-exclusion over subsets) into a closed-form sum
of terms  w_i / (w_i + sum of a subset of weights).

Run:  python3 verify_unbiasedness.py
"""

import itertools
import random

from reservoir import ReservoirSampler, WeightedReservoirSampler

SEED = 20260930


def exact_inclusion_probs(weights, k):
    """Exact P(item i in sample) for A-Res / exponential race."""
    n = len(weights)
    probs = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        total = 0.0
        # Sum over subsets A of "others" of size <= k-1 (the items below t).
        for r in range(0, min(k - 1, n - 1) + 1):
            for A in itertools.combinations(others, r):
                A = set(A)
                rest = [j for j in others if j not in A]
                base = sum(weights[j] for j in rest)
                # Expand prod_{j in A} (1 - e^{-w_j t}) by inclusion-exclusion.
                for s in range(0, r + 1):
                    for B in itertools.combinations(sorted(A), s):
                        c = base + sum(weights[j] for j in B)
                        term = weights[i] / (weights[i] + c)
                        total += term if s % 2 == 0 else -term
        probs.append(total)
    return probs


def chi2(counts, expected):
    return sum((c - e) ** 2 / e for c, e in zip(counts, expected))


def report(title, labels, counts, expected, trials, df, crit):
    print(title)
    print("  %-14s %12s %12s %12s" % ("item", "empirical", "theoretical", "diff"))
    for label, c, e in zip(labels, counts, expected):
        pe, pt = c / trials, e / trials
        print("  %-14s %12.5f %12.5f %+12.5f" % (label, pe, pt, pe - pt))
    stat = chi2(counts, expected)
    verdict = "PASS" if stat < crit else "FAIL"
    print("  chi2 = %.2f  (df=%d, 99.9%% critical = %.1f)  -> %s\n"
          % (stat, df, crit, verdict))


def run_unweighted(n, k, trials):
    counts = [0] * n
    rng = random.Random(SEED)
    for _ in range(trials):
        s = ReservoirSampler(k, rng=rng)
        for i in range(n):
            s.add(i)
        for i in s.sample():
            counts[i] += 1
    return counts


def run_weighted(weights, k, trials):
    counts = [0] * len(weights)
    rng = random.Random(SEED)
    for _ in range(trials):
        s = WeightedReservoirSampler(k, rng=rng)
        for i, w in enumerate(weights):
            s.add(i, w)
        for i in s.sample():
            counts[i] += 1
    return counts


def experiment_unweighted():
    n, k, trials = 10, 3, 200_000
    counts = run_unweighted(n, k, trials)
    expected = [trials * k / n] * n  # exact: every item has prob k/n
    report("1) Unweighted reservoir, n=%d, k=%d, trials=%d (theory: p = k/n = %.3f)"
           % (n, k, trials, k / n),
           ["item %d" % i for i in range(n)], counts, expected, trials,
           df=9, crit=27.9)


def experiment_weighted_k1():
    weights = [1.0, 2.0, 3.0, 4.0]
    trials = 200_000
    counts = run_weighted(weights, 1, trials)
    total = sum(weights)
    expected = [trials * w / total for w in weights]  # exact for k=1
    report("2) Weighted reservoir, k=1, weights=%s, trials=%d (theory: p_i = w_i/sum(w))"
           % (weights, trials),
           ["w=%.1f" % w for w in weights], counts, expected, trials,
           df=3, crit=16.3)


def experiment_weighted_k2():
    weights = [1.0, 2.0, 3.0, 4.0]
    k, trials = 2, 200_000
    counts = run_weighted(weights, k, trials)
    exact = exact_inclusion_probs(weights, k)
    expected = [trials * p for p in exact]
    report("3) Weighted reservoir, k=2, weights=%s, trials=%d (theory: exact inclusion probs)"
           % (weights, trials),
           ["w=%.1f" % w for w in weights], counts, expected, trials,
           df=3, crit=16.3)


def experiment_weighted_k2_extreme():
    # 1e6-fold weight spread; theory is still the exact inclusion
    # probability, which saturates toward 1 for the heavy items.
    weights = [1.0, 10.0, 100.0, 1e6]
    k, trials = 2, 200_000
    counts = run_weighted(weights, k, trials)
    exact = exact_inclusion_probs(weights, k)
    expected = [trials * p for p in exact]
    report("4) Weighted reservoir, k=2, weights=%s, trials=%d (extreme 1e6-fold spread)"
           % (weights, trials),
           ["w=%g" % w for w in weights], counts, expected, trials,
           df=3, crit=16.3)


if __name__ == "__main__":
    print("Unbiasedness verification (fixed seed = %d, fully reproducible)\n" % SEED)
    experiment_unweighted()
    experiment_weighted_k1()
    experiment_weighted_k2()
    experiment_weighted_k2_extreme()
