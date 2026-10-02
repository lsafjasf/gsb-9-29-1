"""Regression tests for the path-smoothing safety fix.

Part 1 proves the original bug exists (naive smoother collides on every
reproduction shape). Part 2 proves the fix never collides, exercises the
degradation ladder and all boundary cases, and asserts the reported
metrics / check counts.
"""

import math
import unittest

import naive_smoother as ns
import scenarios
from smoother import (CollisionChecker, assert_path_safe, laplacian_smooth,
                      max_curvature, min_path_clearance, path_length,
                      point_segment_distance, segment_clearances,
                      smooth_path_safely)


def _params(s, iterations=None):
    return dict(
        obstacles=s["obstacles"],
        iterations=iterations or s.get("many_iterations", 60),
        min_clearance=s["min_clearance"],
        closed=s["closed"],
    )


class ReproduceBugTests(unittest.TestCase):
    """Stable regression cases: the pre-fix implementation must fail."""

    def test_repro_narrow_corridor(self):
        s = scenarios.narrow_corridor()
        out = ns.naive_smooth(s["path"], 60, 0.6)
        self.assertLess(min_path_clearance(out, s["obstacles"]), 0.0)

    def test_repro_sharp_turns(self):
        s = scenarios.sharp_turns()
        out = ns.naive_smooth(s["path"], 60, 0.6)
        self.assertLess(min_path_clearance(out, s["obstacles"]), 0.0)

    def test_repro_hugging_obstacle(self):
        s = scenarios.hugging_obstacle()
        out = ns.naive_smooth(s["path"], 60, 0.6)
        self.assertLess(min_path_clearance(out, s["obstacles"]), 0.0)

    def test_repro_sparse_records_sampled_check_misses_obstacle(self):
        s = scenarios.sparse_obstacles()
        out = ns.naive_smooth(s["path"], 60, 0.6)
        # ground truth: real penetration
        self.assertLess(min_path_clearance(out, s["obstacles"]), 0.0)
        # the buggy sampled check claims it is safe
        _, reported_safe, _ = ns.naive_smooth_with_sampled_check(
            s["path"], s["obstacles"], iterations=60,
            min_clearance=s["min_clearance"], sample_step=s["sample_step"])
        self.assertTrue(reported_safe)

    def test_repro_multi_iteration_accumulation(self):
        s = scenarios.multi_iteration()
        early = ns.naive_smooth(s["path"], s["few_iterations"], 0.6)
        late = ns.naive_smooth(s["path"], s["many_iterations"], 0.6)
        c_early = min_path_clearance(early, s["obstacles"])
        c_late = min_path_clearance(late, s["obstacles"])
        self.assertGreaterEqual(c_early, 0.0)       # one pass still outside
        self.assertLess(c_late, 0.0)                # accumulated drift collides

    def test_continuous_check_catches_what_sampling_misses(self):
        # a tiny disc sitting exactly between two sample points
        a, b = (0.0, 0.0), (2.0, 0.0)
        obstacles = [(1.0, 0.05, 0.1)]
        sampled = ns.sampled_segment_clearance(a, b, obstacles, 2.0)
        exact = min([point_segment_distance((1.0, 0.05), a, b) - 0.1])
        self.assertGreater(sampled, 0.0)   # sampling reports clearance
        self.assertLess(exact, 0.0)        # continuous check proves collision


class FixedSmootherSafetyTests(unittest.TestCase):
    def test_all_scenarios_pass_per_segment_assertions(self):
        for fn in scenarios.ALL_SCENARIOS:
            with self.subTest(scenario=fn.__name__):
                s = fn()
                res = smooth_path_safely(s["path"], **_params(s))
                # per-segment assertion (raises on any violating segment)
                assert_path_safe(res.path, s["obstacles"],
                                 s["min_clearance"], s["closed"])
                # per-segment clearance data: every segment above margin
                clearances = segment_clearances(res.path, s["obstacles"], s["closed"])
                if s["path"] and len(s["path"]) > 1:
                    self.assertTrue(clearances)
                for _, c in clearances:
                    self.assertGreaterEqual(c + 1e-9, s["min_clearance"])
                # reported minimum matches the segment data
                if clearances:
                    self.assertAlmostEqual(
                        res.metrics_after["min_clearance"],
                        min(c for _, c in clearances), places=9)

    def test_endpoints_preserved_open_path(self):
        s = scenarios.narrow_corridor()
        res = smooth_path_safely(s["path"], **_params(s))
        self.assertEqual(res.path[0], tuple(map(float, s["path"][0])))
        self.assertEqual(res.path[-1], tuple(map(float, s["path"][-1])))

    def test_closed_loop_remains_closed_and_safe(self):
        s = scenarios.closed_loop()
        res = smooth_path_safely(s["path"], **_params(s))
        self.assertTrue(res.closed)
        self.assertEqual(len(res.path), len(s["path"]))
        assert_path_safe(res.path, s["obstacles"], s["min_clearance"], closed=True)
        # the naive version deeply penetrates the central disc
        naive = ns.naive_smooth(s["path"], 60, 0.6, closed=True)
        self.assertLess(min_path_clearance(naive, s["obstacles"], closed=True), 0.0)

    def test_single_point_path(self):
        s = scenarios.single_point()
        res = smooth_path_safely(s["path"], **_params(s))
        self.assertEqual(res.path, [(1.0, 2.0)])
        self.assertEqual(res.degradation_level, 0)
        self.assertEqual(res.collision_checks, 0)

    def test_two_point_path(self):
        res = smooth_path_safely([(0.0, 0.0), (1.0, 1.0)], [], min_clearance=0.1)
        self.assertEqual(len(res.path), 2)
        self.assertEqual(res.degradation_level, 0)

    def test_straight_line_unchanged(self):
        s = scenarios.straight_line()
        res = smooth_path_safely(s["path"], **_params(s))
        self.assertEqual(res.degradation_level, 0)
        self.assertAlmostEqual(path_length(res.path), path_length(s["path"]), places=9)
        self.assertEqual(max_curvature(res.path), 0.0)
        for p0, p1 in zip(s["path"], res.path):
            self.assertAlmostEqual(p0[0], p1[0], places=9)
            self.assertAlmostEqual(p0[1], p1[1], places=9)

    def test_enclosed_returns_original_at_l3(self):
        s = scenarios.enclosed()
        res = smooth_path_safely(s["path"], **_params(s))
        self.assertEqual(res.degradation_level, 3)
        self.assertEqual(res.degradation_reason,
                         "constraints_unsatisfiable_returned_original_path")
        for p0, p1 in zip(s["path"], res.path):
            self.assertEqual(p0, p1)
        # every lower level must have been tried and rejected first
        tried = [at["level"] for at in res.attempts]
        self.assertEqual(sorted(tried), [0, 1, 1, 1, 2, 3])
        self.assertFalse(any(at["accepted"] for at in res.attempts
                             if at["level"] in (0, 1, 2)))

    def test_degradation_ladder_recorded_in_order(self):
        for fn in scenarios.ALL_SCENARIOS:
            with self.subTest(scenario=fn.__name__):
                s = fn()
                res = smooth_path_safely(s["path"], **_params(s))
                levels = [at["level"] for at in res.attempts if at["accepted"]]
                if levels:
                    self.assertEqual(min(levels), res.degradation_level)
                non_final = [at["level"] for at in res.attempts
                             if not at["accepted"]]
                self.assertEqual(non_final, sorted(non_final))

    def test_each_degradation_level_has_a_real_case(self):
        observed = set()
        for fn in scenarios.ALL_SCENARIOS:
            s = fn()
            res = smooth_path_safely(s["path"], **_params(s))
            observed.add(res.degradation_level)
        self.assertEqual(observed, {0, 1, 2, 3})

    def test_metrics_reported_and_consistent(self):
        for fn in scenarios.ALL_SCENARIOS:
            with self.subTest(scenario=fn.__name__):
                s = fn()
                res = smooth_path_safely(s["path"], **_params(s))
                b, a = res.metrics_before, res.metrics_after
                self.assertEqual(set(b), {"length", "max_curvature", "min_clearance"})
                self.assertLessEqual(a["length"] - b["length"], 1e-9)
                # metrics recomputed from the returned path agree
                self.assertAlmostEqual(a["length"], path_length(res.path, res.closed), places=9)
                self.assertAlmostEqual(a["max_curvature"],
                                       max_curvature(res.path, res.closed), places=9)
                self.assertAlmostEqual(a["min_clearance"],
                                       min_path_clearance(res.path, s["obstacles"], res.closed),
                                       places=9)

    def test_open_path_curvature_not_increased_when_smoothing_succeeds(self):
        for fn in scenarios.narrow_corridor, scenarios.hugging_obstacle, \
                  scenarios.sparse_obstacles, scenarios.sharp_turns:
            with self.subTest(scenario=fn.__name__):
                s = fn()
                res = smooth_path_safely(s["path"], **_params(s))
                self.assertLessEqual(
                    res.metrics_after["max_curvature"],
                    res.metrics_before["max_curvature"] + 1e-9)

    def test_collision_checks_are_counted(self):
        s = scenarios.narrow_corridor()
        checker = CollisionChecker(s["obstacles"], s["min_clearance"])
        res = smooth_path_safely(s["path"], s["obstacles"], checker=checker,
                                 iterations=60, min_clearance=s["min_clearance"])
        self.assertGreater(res.collision_checks, 0)
        self.assertEqual(res.collision_checks, checker.checks)
        # 5 segments, 5 obstacles per whole-path validation... exact
        # lower bound: whole-path validations before L1 acceptance x
        # segments x obstacles, plus greedy not needed here (L1)
        self.assertGreaterEqual(checker.checks,
                                4 * 5 * len(s["obstacles"]))

    def test_smoothing_is_deterministic(self):
        s = scenarios.narrow_corridor()
        r1 = smooth_path_safely(s["path"], **_params(s))
        r2 = smooth_path_safely(s["path"], **_params(s))
        self.assertEqual(r1.path, r2.path)
        self.assertEqual(r1.degradation_level, r2.degradation_level)

    def test_injected_obstacle_is_detected_on_accepted_segment(self):
        # even a tiny disc exactly in the middle of an accepted segment
        # must be caught by the continuous check (not only at endpoints)
        s = scenarios.narrow_corridor()
        res = smooth_path_safely(s["path"], **_params(s))
        a = res.path[1]
        b = res.path[2]
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        injected = s["obstacles"] + [(mid[0], mid[1], 0.001)]
        with self.assertRaises(AssertionError):
            assert_path_safe(res.path, injected, s["min_clearance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
