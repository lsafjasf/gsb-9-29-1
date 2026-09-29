"""Self-tests for field_solver.  Run:  python3 -m unittest -v

Covers: residual-based convergence, acceleration (SOR / multigrid) vs.
basic iteration, and the boundary cases: uniform boundary, single point
source, fine grid, and non-convergent parameters.
"""

import math
import unittest

from field_solver import (
    CONVERGED, DIVERGED, MAX_ITER_REACHED,
    max_error, new_problem, optimal_sor_omega, residual_norms, solve,
)


def manufactured(n):
    """u = sin(pi x) sin(pi y), f = 2 pi^2 sin(pi x) sin(pi y), g = 0."""
    f = lambda x, y: 2.0 * math.pi ** 2 * math.sin(math.pi * x) * math.sin(math.pi * y)
    exact = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)
    return new_problem(n, f=f, g=0.0) + (exact,)


class TestResidualCriterion(unittest.TestCase):
    def test_converged_means_residual_below_tol(self):
        u, fv, h, _ = manufactured(31)
        res = solve(u, fv, 31, method="sor", tol=1e-9)
        self.assertEqual(res.status, CONVERGED)
        self.assertLessEqual(res.final_relative_residual, 1e-9)
        # residuals recorded for every iteration, monotonically shrinking
        self.assertEqual(len(res.rel_residuals), res.iterations)
        self.assertTrue(all(b <= a * 1.0001 for a, b in
                            zip(res.rel_residuals, res.rel_residuals[1:])))

    def test_max_iter_is_not_success(self):
        u, fv, h, _ = manufactured(31)
        res = solve(u, fv, 31, method="gauss_seidel", tol=1e-12, max_iter=3)
        self.assertEqual(res.status, MAX_ITER_REACHED)
        self.assertEqual(res.iterations, 3)
        self.assertGreater(res.final_relative_residual, 1e-12)

    def test_zero_initial_residual_converges_immediately(self):
        # u == exact constant field already: residual is exactly zero
        u, fv, h = new_problem(15, f=0.0, g=1.0)
        for k in range(len(u)):
            u[k] = 1.0
        res = solve(u, fv, 15, method="jacobi")
        self.assertEqual(res.status, CONVERGED)
        self.assertEqual(res.iterations, 0)


class TestBasicMethods(unittest.TestCase):
    def test_jacobi_reduces_residual(self):
        u, fv, h, _ = manufactured(15)
        r0, _ = residual_norms(u, fv, 15, h)
        res = solve(u, fv, 15, method="jacobi", tol=1e-4, max_iter=10000)
        self.assertEqual(res.status, CONVERGED)
        r1, _ = residual_norms(u, fv, 15, h)
        self.assertLess(r1, 1e-4 * r0)

    def test_gauss_seidel_matches_analytic(self):
        u, fv, h, exact = manufactured(31)
        res = solve(u, fv, 31, method="gauss_seidel", tol=1e-8)
        self.assertEqual(res.status, CONVERGED)
        self.assertLess(max_error(u, 31, exact), 2e-3)  # O(h^2), h = 1/32


class TestAcceleration(unittest.TestCase):
    def test_optimal_omega_in_range_and_grows_with_n(self):
        for n in (7, 31, 127, 1023):
            om = optimal_sor_omega(n)
            self.assertGreater(om, 1.0)
            self.assertLess(om, 2.0)
        self.assertLess(optimal_sor_omega(7), optimal_sor_omega(127))

    def test_sor_needs_far_fewer_iterations_than_gauss_seidel(self):
        u1, fv1, _, _ = manufactured(31)
        u2, fv2, _, _ = manufactured(31)
        gs = solve(u1, fv1, 31, method="gauss_seidel", tol=1e-6)
        sor = solve(u2, fv2, 31, method="sor", tol=1e-6)
        self.assertEqual(gs.status, CONVERGED)
        self.assertEqual(sor.status, CONVERGED)
        self.assertLess(sor.iterations * 5, gs.iterations)

    def test_multigrid_iterations_mesh_independent(self):
        iters = []
        for n in (15, 31, 63, 127):
            u, fv, _, _ = manufactured(n)
            res = solve(u, fv, n, method="multigrid", tol=1e-8)
            self.assertEqual(res.status, CONVERGED)
            iters.append(res.iterations)
        # grid is 8x finer in each direction, iteration count stays flat
        self.assertLessEqual(max(iters) - min(iters), 3)

    def test_multigrid_rejects_bad_grid(self):
        u, fv, _ = new_problem(10)
        with self.assertRaises(ValueError):
            solve(u, fv, 10, method="multigrid")


class TestBoundaryCases(unittest.TestCase):
    def test_uniform_boundary_gives_constant_field(self):
        # all four boundaries identical (g = 2.5), no source: u == 2.5
        u, fv, h = new_problem(31, f=0.0, g=2.5)
        res = solve(u, fv, 31, method="sor", tol=1e-12)
        self.assertEqual(res.status, CONVERGED)
        self.assertLess(max_error(u, 31, lambda x, y: 2.5), 1e-9)

    def test_single_point_source(self):
        n = 31
        h = 1.0 / (n + 1)
        u, fv, _ = new_problem(n, f=0.0, g=0.0)
        c = n // 2 + 1  # centre node
        fv[c * (n + 2) + c] = 1.0 / (h * h)  # unit total source
        res = solve(u, fv, n, method="multigrid", tol=1e-8)
        self.assertEqual(res.status, CONVERGED)
        w = n + 2
        # maximum at the source, symmetric, boundary untouched
        self.assertEqual(max(u), u[c * w + c])
        for k in range(1, n + 1):
            self.assertAlmostEqual(u[k * w + 1], u[1 * w + k], places=12)
            self.assertAlmostEqual(u[k * w + n], u[n * w + k], places=12)
        self.assertTrue(all(u[i * w + j] >= 0.0
                            for i in range(1, n + 1) for j in range(1, n + 1)))

    def test_fine_grid_multigrid_still_converges_quickly(self):
        n = 255  # "very fine" for a pure-Python solver
        u, fv, _, exact = manufactured(n)
        res = solve(u, fv, n, method="multigrid", tol=1e-8)
        self.assertEqual(res.status, CONVERGED)
        self.assertLessEqual(res.iterations, 20)
        self.assertLess(max_error(u, n, exact), 2e-5)  # O(h^2), h = 1/256

    def test_nonconvergent_parameter_is_detected(self):
        # SOR with omega > 2 diverges for this SPD problem
        for omega in (2.05, 2.1):
            u, fv, _, _ = manufactured(15)
            res = solve(u, fv, 15, method="sor", omega=omega,
                        tol=1e-8, max_iter=100000)
            self.assertEqual(res.status, DIVERGED)
            self.assertGreater(res.final_relative_residual, 1.0)

    def test_omega_exactly_two_never_converges(self):
        # omega = 2 is the boundary of the convergence region: the
        # iteration oscillates and the residual never reaches tol
        u, fv, _, _ = manufactured(15)
        res = solve(u, fv, 15, method="sor", omega=2.0,
                    tol=1e-8, max_iter=20000)
        self.assertNotEqual(res.status, CONVERGED)
        self.assertGreater(res.final_relative_residual, 1e-8)

    def test_omega_just_below_two_still_converges(self):
        u, fv, _, _ = manufactured(15)
        res = solve(u, fv, 15, method="sor", omega=1.99, tol=1e-6)
        self.assertEqual(res.status, CONVERGED)


class TestAccuracyOrder(unittest.TestCase):
    def test_second_order_accuracy_vs_analytic(self):
        errs = []
        for n in (31, 63):
            u, fv, _, exact = manufactured(n)
            res = solve(u, fv, n, method="multigrid", tol=1e-10)
            self.assertEqual(res.status, CONVERGED)
            errs.append(max_error(u, n, exact))
        ratio = errs[0] / errs[1]
        self.assertGreater(ratio, 3.0)  # expect ~4 for O(h^2)
        self.assertLess(ratio, 5.0)

    def test_harmonic_nonzero_boundary(self):
        # u = e^x sin(y) is harmonic (f = 0); boundary from exact solution
        exact = lambda x, y: math.exp(x) * math.sin(y)
        u, fv, h = new_problem(31, f=0.0, g=exact)
        res = solve(u, fv, 31, method="multigrid", tol=1e-10)
        self.assertEqual(res.status, CONVERGED)
        self.assertLess(max_error(u, 31, exact), 5e-3)


if __name__ == "__main__":
    unittest.main()
