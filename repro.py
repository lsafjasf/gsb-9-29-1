"""Reproduce the original defect: smoothed paths cutting into obstacles.

Runs the buggy (pre-fix) naive smoother on every reproduction shape and
verifies the collision with a *continuous* ground-truth check. Then runs
the fixed smoother on the same input to show it stays safe.

Exit code 0 = defect reproduced on all shapes and fix verified safe.
"""

import sys

import naive_smoother as ns
import scenarios
from smoother import (assert_path_safe, min_path_clearance,
                      smooth_path_safely)


def main():
    failures = 0
    print("=== BUG REPRODUCTION (naive smoother, pre-fix) ===")
    for fn in scenarios.REPRO_SHAPES:
        s = fn()
        margin = s["min_clearance"]
        iterations = s.get("many_iterations", 60)

        naive_path = ns.naive_smooth(s["path"], iterations, 0.6, s["closed"])
        true_clearance = min_path_clearance(naive_path, s["obstacles"], s["closed"])
        collides = true_clearance < 0.0
        print(f"\n[{s['name']}]")
        print(f"  naive smoothing ({iterations} iters): "
              f"continuous clearance = {true_clearance:+.4f} "
              f"-> {'COLLISION' if collides else 'no collision'}")

        if s["name"] == "sparse_obstacles":
            # the buggy pipeline's own sampled check reports 'safe'
            cand, reported_safe, checks = ns.naive_smooth_with_sampled_check(
                s["path"], s["obstacles"], iterations=iterations,
                min_clearance=margin, sample_step=s["sample_step"])
            real = min_path_clearance(cand, s["obstacles"], s["closed"])
            print(f"  buggy sampled check (step={s['sample_step']}): "
                  f"reported_safe={reported_safe} ({checks} sample checks), "
                  f"but continuous clearance = {real:+.4f} -> collision missed")

        if s["name"] == "multi_iteration":
            few = s["few_iterations"]
            early = ns.naive_smooth(s["path"], few, 0.6, s["closed"])
            c_early = min_path_clearance(early, s["obstacles"], s["closed"])
            print(f"  drift accumulates: after {few} iter(s) clearance "
                  f"{c_early:+.4f}, after {iterations} iters {true_clearance:+.4f}")

        if not collides:
            print("  !! defect NOT reproduced")
            failures += 1

        # the fix on the same input
        res = smooth_path_safely(s["path"], s["obstacles"], iterations=iterations,
                                 min_clearance=margin, closed=s["closed"])
        assert_path_safe(res.path, s["obstacles"], margin, s["closed"])
        print(f"  fixed smoother: clearance "
              f"{res.metrics_after['min_clearance']:+.4f} (>= {margin}), "
              f"degradation={res.degradation_level_name}, "
              f"checks={res.collision_checks}")

    print("\n=== RESULT ===")
    if failures:
        print(f"{failures} shape(s) did not reproduce the defect")
        return 1
    print("defect reproduced on all shapes; fixed smoother stays safe everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
