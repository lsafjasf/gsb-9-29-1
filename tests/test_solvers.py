"""steady_field 自测：收敛性、精度阶、边界/点源/细网格/不收敛参数。"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from steady_field import (
    optimal_omega,
    jacobi,
    gauss_seidel,
    sor,
    multigrid,
    max_error,
    max_diff,
    sine_problem,
    uniform_boundary_problem,
    point_source_problem,
)


class TestBasicConvergence(unittest.TestCase):
    """四种方法在解析解问题上都应收敛，且收敛判据基于残差。"""

    def setUp(self):
        self.n = 31
        self.f, self.bc, self.exact = sine_problem(self.n)

    def test_all_methods_converge(self):
        results = [
            jacobi(self.n, self.f, self.bc),
            gauss_seidel(self.n, self.f, self.bc),
            sor(self.n, self.f, self.bc),
            multigrid(self.n, self.f, self.bc),
        ]
        for res in results:
            self.assertTrue(res.converged, res.method)
            self.assertEqual(res.reason, "converged")
            # 收敛判据是残差：最终相对残差必须 <= tol
            self.assertLessEqual(res.residuals[-1], 1e-6, res.method)
            # 残差历史完整：iterations + 1 个记录（含初始）
            self.assertEqual(len(res.residuals), res.iterations + 1)
            # 与解析解对拍：离散误差 O(h^2)，h=1/32 -> ~1e-3 量级
            self.assertLess(max_error(res.u, self.exact, self.n), 2e-3,
                            res.method)

    def test_residual_decreases(self):
        res = gauss_seidel(self.n, self.f, self.bc)
        for a, b in zip(res.residuals, res.residuals[1:]):
            self.assertLess(b, a)

    def test_acceleration_order(self):
        """加速方法迭代轮数应显著少于基本迭代。"""
        it_jacobi = jacobi(self.n, self.f, self.bc).iterations
        it_gs = gauss_seidel(self.n, self.f, self.bc).iterations
        it_sor = sor(self.n, self.f, self.bc).iterations
        it_mg = multigrid(self.n, self.f, self.bc).iterations
        self.assertLess(it_gs, it_jacobi)
        self.assertLess(it_sor * 5, it_gs)
        self.assertLess(it_mg * 5, it_sor)


class TestAccuracyOrder(unittest.TestCase):
    """网格加密一倍，最大误差应约缩小为 1/4（二阶精度）。"""

    def test_second_order(self):
        errs = []
        for n in (15, 31, 63):
            f, bc, exact = sine_problem(n)
            res = multigrid(n, f, bc, tol=1e-10)
            self.assertTrue(res.converged)
            errs.append(max_error(res.u, exact, n))
        for e1, e2 in zip(errs, errs[1:]):
            ratio = e1 / e2
            self.assertGreater(ratio, 3.5)
            self.assertLess(ratio, 4.5)


class TestUniformBoundary(unittest.TestCase):
    """边界全同（四周 u≡1，f=0）：解析解 u≡1，所有方法都应收敛到它。"""

    def test_constant_solution(self):
        n = 31
        f, bc, exact = uniform_boundary_problem(n, value=1.0)
        for res in (jacobi(n, f, bc), gauss_seidel(n, f, bc),
                    sor(n, f, bc), multigrid(n, f, bc)):
            self.assertTrue(res.converged, res.method)
            self.assertLess(max_error(res.u, exact, n), 1e-4, res.method)

    def test_multigrid_few_cycles(self):
        """常数解几乎全在低频，多重网格应在极少的循环内收敛。"""
        n = 31
        f, bc, _ = uniform_boundary_problem(n)
        res = multigrid(n, f, bc)
        self.assertTrue(res.converged)
        self.assertLessEqual(res.iterations, 10)


class TestPointSource(unittest.TestCase):
    """单点源：无解析解，各方法结果互相对拍。"""

    def test_methods_agree(self):
        n = 63
        f, bc, c = point_source_problem(n)
        ref = multigrid(n, f, bc, tol=1e-10)
        self.assertTrue(ref.converged)
        for res in (gauss_seidel(n, f, bc, tol=1e-8),
                    sor(n, f, bc, tol=1e-8),
                    multigrid(n, f, bc, tol=1e-8)):
            self.assertTrue(res.converged, res.method)
            self.assertLess(max_diff(res.u, ref.u, n), 1e-5, res.method)
        # 点源处场值最大（物理合理性）
        self.assertEqual(max(max(row) for row in ref.u), ref.u[c][c])


class TestFineGrid(unittest.TestCase):
    """极细网格：多重网格迭代轮数应与网格尺寸基本无关。"""

    def test_mg_iteration_independent_of_h(self):
        counts = []
        for n in (63, 127, 255):
            f, bc, exact = sine_problem(n)
            res = multigrid(n, f, bc, tol=1e-8)
            self.assertTrue(res.converged)
            counts.append(res.iterations)
        # 网格加密 4 倍，V 循环数不应明显增长
        self.assertLessEqual(max(counts) - min(counts), 2)


class TestBadParameters(unittest.TestCase):
    """不收敛参数：ω >= 2 必须被检测出来，而不是悄悄跑满迭代。"""

    def setUp(self):
        self.n = 31
        self.f, self.bc, _ = sine_problem(self.n)

    def test_omega_2_does_not_converge(self):
        res = sor(self.n, self.f, self.bc, omega=2.0, max_iter=500)
        self.assertFalse(res.converged)
        self.assertIn(res.reason, ("diverged", "max_iter"))

    def test_omega_above_2_diverges(self):
        res = sor(self.n, self.f, self.bc, omega=2.05, max_iter=10000)
        self.assertFalse(res.converged)
        self.assertEqual(res.reason, "diverged")
        self.assertLess(res.iterations, 10000)  # 提前截获，没有白跑

    def test_max_iter_path(self):
        res = jacobi(self.n, self.f, self.bc, max_iter=5)
        self.assertFalse(res.converged)
        self.assertEqual(res.reason, "max_iter")
        self.assertEqual(res.iterations, 5)

    def test_optimal_omega_formula(self):
        # ω_opt = 2/(1+sin(πh))，应落在 (1, 2) 且随 h 减小趋近 2
        w31 = optimal_omega(31)
        w127 = optimal_omega(127)
        self.assertGreater(w31, 1.0)
        self.assertLess(w31, 2.0)
        self.assertGreater(w127, w31)
        self.assertAlmostEqual(w31, 2.0 / (1.0 + math.sin(math.pi / 32.0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
