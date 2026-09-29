"""Regression tests for the path smoother.

Run:  python3 -m unittest test_smoother -v   (or: python3 test_smoother.py)
"""

import unittest

from geometry import (
    path_length,
    path_max_curvature,
    path_min_clearance,
    segment_clearance,
    validate_path,
)
from smoother import legacy_smooth_path, smooth_path
import scenarios


class SmootherTestCase(unittest.TestCase):
    def assert_path_valid(self, path, obstacles, margin=0.0):
        """Per-segment assertion: every segment must clear every obstacle."""
        for i, (a, b) in enumerate(zip(path, path[1:])):
            for ob in obstacles:
                clearance = segment_clearance(a, b, ob)
                self.assertGreaterEqual(
                    clearance, margin - 1e-9,
                    msg=(
                        f"segment {i} {a}->{b} violates obstacle {ob}: "
                        f"clearance {clearance:.4f} < {margin}"
                    ),
                )

    def assert_endpoints_kept(self, original, smoothed):
        self.assertEqual(original[0], smoothed[0], "start point must not move")
        self.assertEqual(original[-1], smoothed[-1], "goal point must not move")


class TestBugReproduction(SmootherTestCase):
    """These tests document the ORIGINAL bug: the legacy smoother produces
    paths that intersect obstacles. They guard the repro, not the fix."""

    def _check_legacy_collides(self, scenario_fn):
        name, path, obstacles = scenario_fn()
        # Precondition: the raw path itself is collision-free.
        self.assertEqual(validate_path(path, obstacles), [],
                         f"{name}: raw path should be valid")
        legacy = legacy_smooth_path(path)
        violations = validate_path(legacy, obstacles)
        self.assertTrue(
            violations,
            f"{name}: expected legacy smoother to cut into an obstacle",
        )
        return legacy, violations

    def test_legacy_collides_narrow_corridor(self):
        _, violations = self._check_legacy_collides(scenarios.narrow_corridor)
        print(f"\n[repro] narrow_corridor: legacy has "
              f"{len(violations)} colliding segment/obstacle pairs, "
              f"worst clearance {min(v[2] for v in violations):.4f}")

    def test_legacy_collides_sharp_turns(self):
        _, violations = self._check_legacy_collides(scenarios.sharp_turns)
        print(f"\n[repro] sharp_turns: legacy has "
              f"{len(violations)} colliding segment/obstacle pairs, "
              f"worst clearance {min(v[2] for v in violations):.4f}")

    def test_legacy_collides_obstacle_hugging(self):
        _, violations = self._check_legacy_collides(scenarios.obstacle_hugging)
        print(f"\n[repro] obstacle_hugging: legacy has "
              f"{len(violations)} colliding segment/obstacle pairs, "
              f"worst clearance {min(v[2] for v in violations):.4f}")


class TestFixedSmoother(SmootherTestCase):
    """The fix: smoothed paths must never intersect obstacles."""

    def _check_fixed(self, scenario_fn, expect_shorter=True):
        name, path, obstacles = scenario_fn()
        smoothed = smooth_path(path, obstacles)
        # Per-segment validation of the whole result.
        self.assert_path_valid(smoothed, obstacles, margin=0.0)
        self.assertEqual(validate_path(smoothed, obstacles), [])
        self.assert_endpoints_kept(path, smoothed)
        if expect_shorter:
            self.assertLessEqual(
                path_length(smoothed), path_length(path) + 1e-9,
                f"{name}: smoothing must not lengthen the path",
            )
            self.assertLess(
                path_max_curvature(smoothed), path_max_curvature(path) + 1e-9,
                f"{name}: smoothing must reduce curvature",
            )
        return path, smoothed, obstacles

    def test_narrow_corridor(self):
        path, smoothed, obstacles = self._check_fixed(scenarios.narrow_corridor)
        print(f"\n[fixed] narrow_corridor: len {path_length(path):.3f} -> "
              f"{path_length(smoothed):.3f}, min clearance "
              f"{path_min_clearance(smoothed, obstacles):.4f}")

    def test_sharp_turns(self):
        path, smoothed, obstacles = self._check_fixed(scenarios.sharp_turns)
        print(f"\n[fixed] sharp_turns: len {path_length(path):.3f} -> "
              f"{path_length(smoothed):.3f}, min clearance "
              f"{path_min_clearance(smoothed, obstacles):.4f}")

    def test_obstacle_hugging(self):
        path, smoothed, obstacles = self._check_fixed(scenarios.obstacle_hugging)
        print(f"\n[fixed] obstacle_hugging: len {path_length(path):.3f} -> "
              f"{path_length(smoothed):.3f}, min clearance "
              f"{path_min_clearance(smoothed, obstacles):.4f}")

    def test_single_point(self):
        name, path, obstacles = scenarios.single_point()
        smoothed = smooth_path(path, obstacles)
        self.assertEqual(smoothed, path, "single point must be returned as-is")
        self.assert_path_valid(smoothed, obstacles)

    def test_two_points(self):
        path = [(0.0, 0.0), (5.0, 5.0)]
        smoothed = smooth_path(path, [(2.0, 8.0, 1.0)])
        self.assertEqual(smoothed, path, "two-point path must be returned as-is")

    def test_straight_line_unchanged(self):
        name, path, obstacles = scenarios.straight_line()
        smoothed = smooth_path(path, obstacles)
        self.assertEqual(smoothed, path,
                         "straight collision-free line needs no smoothing")
        self.assert_path_valid(smoothed, obstacles)

    def test_unsmoothable_returns_original(self):
        name, path, obstacles = scenarios.unsmoothable()
        # Precondition: original is valid but tighter than the safety margin.
        self.assertEqual(validate_path(path, obstacles), [])
        self.assertLess(path_min_clearance(path, obstacles), 0.05)
        smoothed = smooth_path(path, obstacles, margin=0.05)
        self.assertEqual(
            smoothed, path,
            "when no safe smoothing move exists, the ORIGINAL path must be "
            "returned (never a colliding 'improvement')",
        )
        self.assert_path_valid(smoothed, obstacles)

    def test_never_returns_colliding_path_even_if_input_collides(self):
        # Pathological input: the raw path already clips an obstacle.
        # The smoother must not make it worse, and must still validate.
        path = [(0.0, 0.0), (5.0, 0.5), (10.0, 0.0)]
        obstacles = [(5.0, 0.0, 1.0)]  # raw path already inside
        smoothed = smooth_path(path, obstacles)
        self.assertEqual(smoothed, [tuple(p) for p in path],
                         "unfixable input must fall back to the original path")


if __name__ == "__main__":
    unittest.main(verbosity=2)
