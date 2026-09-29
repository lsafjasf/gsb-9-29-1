"""Core statistics for risk-control feature binning.

Standard library only. Everything here is deterministic: no randomness,
no wall-clock dependence, same input always yields the same output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

SMOOTHING = 0.5


def is_missing(value) -> bool:
    """None and NaN are treated as missing."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def quantile(sorted_values: Sequence[float], p: float) -> float:
    """Deterministic linear-interpolation quantile on pre-sorted data."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    n = len(sorted_values)
    if n == 0:
        raise ValueError("quantile of an empty sequence")
    if n == 1:
        return float(sorted_values[0])
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo]) * (1.0 - frac) + float(sorted_values[hi]) * frac


def woe_iv(bad: int, good: int, total_bad: int, total_good: int):
    """Smoothed WOE / IV for one bin. Returns (woe, iv)."""
    if total_bad <= 0 or total_good <= 0:
        return 0.0, 0.0
    bad_dist = (bad + SMOOTHING) / (total_bad + SMOOTHING)
    good_dist = (good + SMOOTHING) / (total_good + SMOOTHING)
    woe = math.log(bad_dist / good_dist)
    return woe, (bad_dist - good_dist) * woe


@dataclass
class BinStat:
    index: int
    label: str
    kind: str = "normal"  # normal | missing | outlier_low | outlier_high
    lower: Optional[float] = None
    upper: Optional[float] = None
    count: int = 0
    bad: int = 0
    good: int = 0
    woe: float = 0.0
    iv: float = 0.0

    @property
    def bad_rate(self) -> Optional[float]:
        return self.bad / self.count if self.count else None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "bad": self.bad,
            "good": self.good,
            "bad_rate": self.bad_rate,
            "woe": self.woe,
            "iv": self.iv,
        }


def detect_direction(rates: Sequence[float]) -> str:
    """Guess the monotone direction by comparing head vs tail averages."""
    valid = [r for r in rates if r is not None]
    if len(valid) < 2:
        return "increasing"
    mid = max(1, len(valid) // 2)
    head = sum(valid[:mid]) / len(valid[:mid])
    tail = sum(valid[-mid:]) / len(valid[-mid:])
    return "increasing" if tail >= head else "decreasing"


def find_violations(rates: Sequence[Optional[float]], direction: str,
                    tol: float = 1e-12) -> List[int]:
    """Return positions i where (rate[i], rate[i+1]) breaks `direction`."""
    violations = []
    for i in range(len(rates) - 1):
        a, b = rates[i], rates[i + 1]
        if a is None or b is None:
            continue
        if direction == "increasing" and a > b + tol:
            violations.append(i)
        elif direction == "decreasing" and a < b - tol:
            violations.append(i)
    return violations


def _group_counts(stats: Sequence[BinStat], group: Sequence[int]):
    bad = sum(stats[i].bad for i in group)
    good = sum(stats[i].good for i in group)
    return bad, good


def suggest_merges(stats: Sequence[BinStat], direction: str, min_bins: int = 1):
    """Greedy adjacent merges that restore monotonicity with minimal IV loss.

    Returns (suggestions, merged_groups, final_rates). If `min_bins` prevents
    full repair, `final_rates` may still violate monotonicity; callers should
    re-check with find_violations.
    """
    total_bad = sum(s.bad for s in stats)
    total_good = sum(s.good for s in stats)
    groups = [[i] for i in range(len(stats))]
    suggestions = []
    floor = max(1, min_bins)

    def rates_of(current):
        rates = []
        for g in current:
            bad, good = _group_counts(stats, g)
            total = bad + good
            rates.append(bad / total if total else None)
        return rates

    while len(groups) > floor:
        violations = find_violations(rates_of(groups), direction)
        if not violations:
            break
        best_idx, best_loss = None, None
        for i in violations:
            b1, g1 = _group_counts(stats, groups[i])
            b2, g2 = _group_counts(stats, groups[i + 1])
            _, iv1 = woe_iv(b1, g1, total_bad, total_good)
            _, iv2 = woe_iv(b2, g2, total_bad, total_good)
            _, ivm = woe_iv(b1 + b2, g1 + g2, total_bad, total_good)
            loss = iv1 + iv2 - ivm
            if best_loss is None or loss < best_loss:
                best_idx, best_loss = i, loss
        merged = groups[best_idx] + groups[best_idx + 1]
        suggestions.append({
            "merge_bins": merged,
            "iv_loss": round(best_loss, 10),
            "reason": "bad-rate violates %s order between positions %d and %d"
                      % (direction, best_idx, best_idx + 1),
        })
        groups[best_idx:best_idx + 2] = [merged]

    return suggestions, groups, rates_of(groups)


def monotonicity_report(stats: Sequence[BinStat], direction: str = "auto",
                        min_bins: int = 1) -> dict:
    """Full monotonicity diagnostic for a sequence of (normal) bins."""
    rates = [s.bad_rate for s in stats]
    if direction == "auto":
        direction = detect_direction([r for r in rates if r is not None])
    violations = [
        {
            "left": i,
            "right": i + 1,
            "left_rate": rates[i],
            "right_rate": rates[i + 1],
        }
        for i in find_violations(rates, direction)
    ]
    suggestions, _, final_rates = suggest_merges(stats, direction, min_bins=min_bins)
    return {
        "direction": direction,
        "is_monotone": not violations,
        "violations": violations,
        "merge_suggestions": suggestions,
        "rates_after_suggested_merges": final_rates,
    }
