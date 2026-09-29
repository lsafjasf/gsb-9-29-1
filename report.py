"""Print the smoothing trade-off report: path length, max curvature and
minimum obstacle clearance for the raw path, the legacy (buggy) smoother
and the fixed smoother, per scenario.

Run:  python3 report.py
"""

from geometry import path_length, path_max_curvature, path_min_clearance, validate_path
from smoother import legacy_smooth_path, smooth_path
import scenarios


def fmt(value):
    if value == float("inf"):
        return "   inf"
    return f"{value:7.4f}"


def row(label, path, obstacles):
    collisions = len(validate_path(path, obstacles))
    return (
        f"  {label:<8} "
        f"{path_length(path):8.3f} "
        f"{fmt(path_max_curvature(path))} "
        f"{fmt(path_min_clearance(path, obstacles))} "
        f"{collisions:>6} "
        f"{'COLLIDES' if collisions else 'safe'}"
    )


def main():
    print("Smoothing trade-off report")
    print("(length / max curvature / min clearance / colliding segment-obstacle pairs)")
    for scenario_fn in scenarios.ALL:
        name, path, obstacles = scenario_fn()
        legacy = legacy_smooth_path(path)
        fixed = smooth_path(path, obstacles)
        print(f"\n[{name}]")
        print(f"  {'variant':<8} {'length':>8} {'maxcurv':>7} {'minclear':>7} {'hits':>6}  status")
        print(row("raw", path, obstacles))
        print(row("legacy", legacy, obstacles))
        print(row("fixed", fixed, obstacles))


if __name__ == "__main__":
    main()
