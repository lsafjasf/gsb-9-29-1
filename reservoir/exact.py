"""Exact rational inclusion probabilities, used to differential-test the
samplers against ground truth computed with ``fractions.Fraction``."""
from fractions import Fraction
from itertools import combinations

__all__ = ["ares_inclusion_probabilities", "pps_targets"]


def ares_inclusion_probabilities(weights, k):
    """Exact P(item i in the A-Res top-k sample), as a Fraction.

    Key ``K_j = U_j ** (1/w_j)`` has CDF ``x ** w_j`` on [0, 1], so

        P(i in top-k) = sum over S subset of others, |S| <= k-1 of
            integral_0^1 prod_{j in S}(1 - x**w_j) prod_{j not in S} x**w_j
                          * w_i * x**(w_i - 1) dx

    Expanding the products gives only monomials, and
    ``integral_0^1 w_i * x**(r - 1) dx = w_i / r`` is an exact rational.
    Feasible for small streams (cost ~ 2**(n-1) * binomial terms).

    Zero-weight items have inclusion probability 0 by convention (the
    sampler never selects them).  Raises ValueError if all weights are 0.
    """
    ws = [Fraction(w) for w in weights]
    n = len(ws)
    if k < 0:
        raise ValueError("k must be >= 0")
    if all(w == 0 for w in ws):
        raise ValueError("all weights are zero")
    if k >= n:
        return [Fraction(1) if w > 0 else Fraction(0) for w in ws]
    if k == 0:
        return [Fraction(0)] * n
    out = []
    for i in range(n):
        wi = ws[i]
        if wi == 0:
            out.append(Fraction(0))
            continue
        others = [ws[j] for j in range(n) if j != i]
        total = Fraction(0)
        for r in range(0, min(k - 1, n - 1) + 1):
            for S in combinations(range(n - 1), r):
                in_S = set(S)
                rest = sum(w for j, w in enumerate(others) if j not in in_S)
                for mask in range(1 << r):
                    add = sum(others[S[b]] for b in range(r) if (mask >> b) & 1)
                    sign = -1 if (bin(mask).count("1") & 1) else 1
                    total += sign * wi / (wi + rest + add)
        out.append(total)
    return out


def pps_targets(weights, k):
    """Target PPS inclusion probabilities ``min(1, k * w_i / W)``."""
    ws = [Fraction(w) for w in weights]
    W = sum(ws)
    if W <= 0:
        raise ValueError("all weights are zero")
    return [min(Fraction(1), k * w / W) for w in ws]
