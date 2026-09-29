"""边界用例自测：起始点即最优、非凸多峰、梯度接近零、线搜索降级等。"""

import math
import unittest

from bfgs import minimize_bfgs, minimize_gd, _norm_inf, _bfgs_update, _eye
from problems import (make_quadratic, make_rosenbrock, make_himmelblau,
                      make_quartic)


class TestEdgeCases(unittest.TestCase):

    def test_start_at_optimum(self):
        """起始点即最优：0 次迭代、1 次函数求值、立即收敛。"""
        f, g, _, f_star, _ = make_quadratic(n=5, cond=10.0)
        # 精确解：x_i = b_i / d_i，由 make_quadratic 内部构造保证
        # 这里通过梯度反解最优点：g(x)=0 => 直接取 f 的极小点
        # 用 bfgs 先求一次解，再以其为起点
        r0 = minimize_bfgs(f, g, [0.0] * 5, tol=1e-14)
        r = minimize_bfgs(f, g, r0.x, tol=1e-8)
        self.assertEqual(r.status, "converged")
        self.assertEqual(r.n_iter, 0)
        self.assertEqual(r.n_fev, 1)
        self.assertIn("initial point", r.message)
        r_gd = minimize_gd(f, g, r0.x, tol=1e-8)
        self.assertEqual(r_gd.n_iter, 0)

    def test_nonconvex_multimodal(self):
        """非凸多峰 Himmelblau：多个起点都收敛到某个全局极小 (f*=0)。"""
        for x0 in [(0.0, 0.0), (-4.0, 0.0), (4.0, 4.0), (-2.0, 3.0)]:
            f, g, x0l, f_star, _ = make_himmelblau(x0)
            r = minimize_bfgs(f, g, x0l, tol=1e-8)
            self.assertEqual(r.status, "converged", f"x0={x0}: {r.message}")
            self.assertAlmostEqual(r.fun, 0.0, places=8)
            self.assertLessEqual(_norm_inf(r.grad), 1e-8)

    def test_near_zero_gradient_start(self):
        """起点梯度已接近零（平坦四次函数）：立即判定收敛，不崩溃。"""
        f, g, _, _, _ = make_quartic(n=3, x0_val=1e-4)
        # |g|_inf = 4e-12 < tol
        r = minimize_bfgs(f, g, [1e-4] * 3, tol=1e-8)
        self.assertEqual(r.status, "converged")
        self.assertEqual(r.n_iter, 0)

    def test_near_zero_gradient_region(self):
        """最优点附近梯度趋于零且 Hessian 奇异：仍能收敛到高精度。"""
        f, g, x0, f_star, _ = make_quartic(n=5, x0_val=1.3)
        r = minimize_bfgs(f, g, x0, tol=1e-8)
        self.assertEqual(r.status, "converged")
        self.assertLessEqual(r.fun, 1e-9)

    def test_rosenbrock(self):
        f, g, x0, _, _ = make_rosenbrock(n=2)
        r = minimize_bfgs(f, g, x0, tol=1e-8)
        self.assertEqual(r.status, "converged")
        self.assertLessEqual(r.fun, 1e-14)

    def test_quadratic_accuracy(self):
        f, g, x0, f_star, _ = make_quadratic(n=20, cond=1e3)
        r = minimize_bfgs(f, g, x0, tol=1e-8)
        self.assertEqual(r.status, "converged")
        self.assertLessEqual(abs(r.fun - f_star), 1e-10)

    def test_hess_inv_stays_spd(self):
        """非凸问题上近似逆矩阵始终保持对称正定（Sylvester 判据）。"""
        f, g, x0, _, _ = make_himmelblau((0.0, 0.0))
        r = minimize_bfgs(f, g, x0, tol=1e-10)
        H = r.hess_inv
        self.assertAlmostEqual(H[0][1], H[1][0], places=12)   # 对称
        self.assertGreater(H[0][0], 0.0)                       # 一阶主子式
        det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
        self.assertGreater(det, 0.0)                           # 二阶主子式

    def test_line_search_failure_recorded(self):
        """目标函数间断（沿任何步长都无法充分下降）时：
        Wolfe -> Armijo -> 最速下降重试 全部失败，原因被完整记录。"""
        # 在 x=1 处有跳跃间断：除当前点外函数值整体抬高 10，
        # 任何步长都无法满足 Armijo 充分下降条件。
        def f(v):
            x = v[0]
            return x * x if x == 1.0 else x * x + 10.0
        g = lambda v: [2.0 * v[0]]
        r = minimize_bfgs(f, g, [1.0], tol=1e-12, max_iter=50)
        self.assertEqual(r.status, "line_search_failed")
        self.assertTrue(r.fallback_log, "降级/失败原因应被记录")
        joined = " ".join(r.fallback_log)
        self.assertIn("Wolfe", joined)
        self.assertIn("Armijo", joined)
        self.assertIn("steepest", r.message)

    def test_gd_baseline_consistency(self):
        """GD 基线在同一问题上也能收敛（用于对拍的正确性兜底）。"""
        f, g, x0, f_star, _ = make_quadratic(n=10, cond=100.0)
        r = minimize_gd(f, g, x0, tol=1e-6)
        self.assertEqual(r.status, "converged")
        self.assertLessEqual(abs(r.fun - f_star), 1e-8)


def _leading_minors_positive(H):
    """Sylvester 判据：全部顺序主子式为正 <=> 对称正定。"""
    n = len(H)
    for k in range(1, n + 1):
        # 小维度直接用 LU 无主元消元求行列式（主子式）
        M = [row[:k] for row in H[:k]]
        det = 1.0
        for i in range(k):
            pivot = M[i][i]
            if pivot == 0.0:
                return False
            det *= pivot
            for r in range(i + 1, k):
                factor = M[r][i] / pivot
                for c in range(i, k):
                    M[r][c] -= factor * M[i][c]
        if det <= 0.0:
            return False
    return True


class TestBFGSUpdate(unittest.TestCase):

    def test_update_preserves_spd(self):
        """曲率条件满足时，更新后 H 仍对称正定。"""
        log = []
        H = _eye(3)
        s = [1.0, -0.5, 2.0]
        y = [0.8, 0.3, 1.5]          # y^T s > 0
        _bfgs_update(H, s, y, True, log, 0)
        self.assertFalse(any("damped" in l or "skip" in l for l in log))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(H[i][j], H[j][i], places=12)
        self.assertTrue(_leading_minors_positive(H))

    def test_damped_update_on_curvature_violation(self):
        """y^T s <= 0（非凸区域）时阻尼更新仍保持 SPD 并记录日志。"""
        log = []
        H = _eye(2)
        s = [1.0, 0.5]
        y = [-0.6, 0.1]              # y^T s < 0：穿越负曲率区
        _bfgs_update(H, s, y, True, log, 3)
        self.assertTrue(any("damped" in l for l in log))
        self.assertTrue(_leading_minors_positive(H))
        self.assertAlmostEqual(H[0][1], H[1][0], places=12)

    def test_degenerate_pair_skipped(self):
        """y == 0（梯度无变化）时跳过更新，H 不变。"""
        log = []
        H = _eye(2)
        H_before = [row[:] for row in H]
        _bfgs_update(H, [1.0, 1.0], [0.0, 0.0], True, log, 5)
        self.assertTrue(any("skip" in l for l in log))
        self.assertEqual(H, H_before)



if __name__ == "__main__":
    unittest.main(verbosity=2)
