"""Quantified trade-off report: naive vs fixed smoothing.

For every scenario prints path length, max curvature, minimum clearance
and collision-check counts before/after, plus the degradation record.
"""

import scenarios
import naive_smoother as ns
from smoother import (LEVEL_NAMES, min_path_clearance, path_length,
                      segment_clearances,
                      max_curvature, smooth_path_safely)


def fmt(v):
    if v == float("inf"):
        return "   inf"
    return f"{v:6.3f}"


def main():
    print(f"{'scenario':<18} {'result':<22} {'length':>15} {'max curv':>15} "
          f"{'min clearance':>17} {'checks':>7}")
    print("-" * 100)
    for fn in scenarios.ALL_SCENARIOS:
        s = fn()
        margin = s["min_clearance"]
        iterations = s.get("many_iterations", 60)
        closed = s["closed"]

        naive_path = ns.naive_smooth(s["path"], iterations, 0.6, closed)
        naive_len = path_length(naive_path, closed)
        naive_curv = max_curvature(naive_path, closed)
        naive_clr = min_path_clearance(naive_path, s["obstacles"], closed)

        res = smooth_path_safely(s["path"], s["obstacles"], iterations=iterations,
                                 min_clearance=margin, closed=closed)
        b, a = res.metrics_before, res.metrics_after

        print(f"{s['name']:<18} {'original':<22} "
              f"{fmt(b['length']):>15} {fmt(b['max_curvature']):>15} "
              f"{fmt(b['min_clearance']):>17} {'-':>7}")
        print(f"{'':<18} {'naive (buggy)':<22} "
              f"{fmt(naive_len):>15} {fmt(naive_curv):>15} "
              f"{fmt(naive_clr):>17} {'0':>7}")
        print(f"{'':<18} {res.degradation_level_name:<22} "
              f"{fmt(a['length']):>15} {fmt(a['max_curvature']):>15} "
              f"{fmt(a['min_clearance']):>17} {res.collision_checks:>7}")

        record = " -> ".join(
            f"L{at['level']}:{at['strategy']}"
            f"({'ok' if at['accepted'] else 'reject'})"
            for at in res.attempts)
        print(f"{'':<18} degradation: {record or 'none (trivial path)'}"
              f"  => final {LEVEL_NAMES[res.degradation_level]}")
        clearances = segment_clearances(res.path, s["obstacles"], closed)
        per_segment = " ".join(f"{c:+.2f}" for _, c in clearances)
        print(f"{'':<18} per-segment clearance (fixed): {per_segment}")
        print()


if __name__ == "__main__":
    main()
