"""heat1d 自测（标准库 unittest）。

运行: python3 test_heat1d.py [-v]
"""

import math
import unittest

from heat1d import DIRICHLET, NEUMANN, energy, solve, stability_limit

L = 1.0
ALPHA = 1.0


def grid(n):
    return [i / n for i in range(n + 1)]


def analytic_dirichlet(x, t, alpha=ALPHA, length=L):
    """两端 0 度定温, 初值 sin(pi*x/L) 的解析解。"""
    return math.sin(math.pi * x / length) * math.exp(
        -alpha * (math.pi / length) ** 2 * t)


def analytic_adiabatic(x, t, alpha=ALPHA, length=L):
    """两端绝热, 初值 0.5 + cos(pi*x/L) 的解析解。"""
    return 0.5 + math.cos(math.pi * x / length) * math.exp(
        -alpha * (math.pi / length) ** 2 * t)


class TestStability(unittest.TestCase):
    def test_explicit_rejects_unstable_dt(self):
        n = 20
        dx = L / n
        dt_max = stability_limit(ALPHA, dx)
        with self.assertRaises(ValueError):
            solve([0.0] * (n + 1), ALPHA, dx, dt_max * 1.01, 1,
                  scheme="explicit")

    def test_explicit_accepts_boundary_dt(self):
        n = 20
        dx = L / n
        dt = stability_limit(ALPHA, dx)  # r 恰好 1/2, 允许
        u = solve([math.sin(math.pi * x) for x in grid(n)],
                  ALPHA, dx, dt, 10, scheme="explicit")
        self.assertTrue(all(math.isfinite(v) for v in u))

    def test_large_alpha_explicit_rejected_implicit_ok(self):
        n = 20
        dx = L / n
        alpha = 1e4  # 大系数: 显式在常规 dt 下必然越界
        with self.assertRaises(ValueError):
            solve([1.0] * (n + 1), alpha, dx, 1e-3, 1, scheme="explicit")
        u = solve([math.sin(math.pi * x) for x in grid(n)],
                  alpha, dx, 1e-3, 5, scheme="implicit")
        self.assertTrue(all(math.isfinite(v) for v in u))


class TestAnalyticAgreement(unittest.TestCase):
    def _check(self, scheme, r, n=64, t_final=0.05):
        dx = L / n
        dt = r * dx * dx / ALPHA
        steps = round(t_final / dt)
        u = solve([math.sin(math.pi * x) for x in grid(n)],
                  ALPHA, dx, dt, steps, scheme=scheme)
        err = max(abs(u[i] - analytic_dirichlet(i * dx, steps * dt))
                  for i in range(n + 1))
        self.assertLess(err, 2e-3)

    def test_explicit_dirichlet(self):
        self._check("explicit", r=0.4)

    def test_implicit_dirichlet(self):
        self._check("implicit", r=2.0)


class TestEnergyConservation(unittest.TestCase):
    """两端绝热时离散能量必须守恒（不允许热量凭空出现/消失）。"""

    def _assert_conserved(self, scheme, r):
        n = 50
        dx = L / n
        dt = r * dx * dx / ALPHA
        u0 = [0.5 + math.cos(math.pi * x) for x in grid(n)]
        e0 = energy(u0, dx)
        u = solve(u0, ALPHA, dx, dt, 200,
                  bc_left=(NEUMANN, 0.0), bc_right=(NEUMANN, 0.0),
                  scheme=scheme)
        e1 = energy(u, dx)
        self.assertAlmostEqual(e0, e1, places=12,
                               msg="%s 格式绝热边界能量不守恒" % scheme)

    def test_explicit_conserves_energy(self):
        self._assert_conserved("explicit", r=0.4)

    def test_implicit_conserves_energy(self):
        self._assert_conserved("implicit", r=5.0)

    def test_adiabatic_matches_analytic(self):
        n = 64
        dx = L / n
        dt = 0.4 * dx * dx / ALPHA
        steps = round(0.05 / dt)
        u0 = [0.5 + math.cos(math.pi * x) for x in grid(n)]
        u = solve(u0, ALPHA, dx, dt, steps,
                  bc_left=(NEUMANN, 0.0), bc_right=(NEUMANN, 0.0),
                  scheme="explicit")
        err = max(abs(u[i] - analytic_adiabatic(i * dx, steps * dt))
                  for i in range(n + 1))
        self.assertLess(err, 2e-3)


class TestBoundaryBehavior(unittest.TestCase):
    def test_dirichlet_holds_value(self):
        n = 20
        dx = L / n
        dt = 0.4 * dx * dx
        u = solve([0.0] * (n + 1), ALPHA, dx, dt, 50,
                  bc_left=(DIRICHLET, 100.0), bc_right=(DIRICHLET, -20.0))
        self.assertEqual(u[0], 100.0)
        self.assertEqual(u[-1], -20.0)
        self.assertTrue(all(-20.0 - 1e-12 <= v <= 100.0 + 1e-12 for v in u))

    def test_mixed_boundary(self):
        n = 20
        dx = L / n
        dt = 0.4 * dx * dx
        u = solve([1.0] * (n + 1), ALPHA, dx, dt, 2000,  # t=2.0
                  bc_left=(DIRICHLET, 2.0), bc_right=(NEUMANN, 0.0))
        self.assertEqual(u[0], 2.0)
        # 长时间后应趋于均匀 2.0（右端绝热, 左端定温）
        self.assertAlmostEqual(u[-1], 2.0, delta=0.02)


class TestEdgeCases(unittest.TestCase):
    def test_zero_diffusion(self):
        n = 30
        dx = L / n
        u0 = [math.sin(3.0 * math.pi * x) + x for x in grid(n)]
        for scheme in ("explicit", "implicit"):
            u = solve(u0, 0.0, dx, 1e-3, 100, scheme=scheme)
            self.assertEqual(u, [float(v) for v in u0])

    def test_step_initial(self):
        """阶跃初值 + 两端绝热: 能量守恒, 且满足最大值原理（无过冲）。"""
        n = 100
        dx = L / n
        u0 = [1.0 if i * dx < 0.5 else 0.0 for i in range(n + 1)]
        e0 = energy(u0, dx)
        for scheme, r in (("explicit", 0.4), ("implicit", 10.0)):
            dt = r * dx * dx
            u = solve(u0, ALPHA, dx, dt, 500,
                      bc_left=(NEUMANN, 0.0), bc_right=(NEUMANN, 0.0),
                      scheme=scheme)
            self.assertAlmostEqual(energy(u, dx), e0, places=10)
            self.assertGreaterEqual(min(u), -1e-12)
            self.assertLessEqual(max(u), 1.0 + 1e-12)

    def test_long_time_decay_dirichlet(self):
        """长时间演化: 两端 0 度定温, 温度场应衰减到 0。"""
        n = 40
        dx = L / n
        dt = 0.4 * dx * dx
        t_final = 2.0  # >> 扩散时间尺度 1/alpha
        steps = round(t_final / dt)
        u = solve([math.sin(math.pi * x) for x in grid(n)],
                  ALPHA, dx, dt, steps, scheme="explicit")
        self.assertLess(max(abs(v) for v in u), 1e-8)

    def test_long_time_relaxes_to_mean_adiabatic(self):
        """长时间演化: 两端绝热, 温度场应趋于初值平均且能量守恒。"""
        n = 40
        dx = L / n
        dt = 2.0 * dx * dx  # 隐式, 大步长
        u0 = [0.5 + math.cos(math.pi * x) for x in grid(n)]
        e0 = energy(u0, dx)
        steps = round(2.0 / dt)
        u = solve(u0, ALPHA, dx, dt, steps,
                  bc_left=(NEUMANN, 0.0), bc_right=(NEUMANN, 0.0),
                  scheme="implicit")
        mean = e0 / L
        self.assertLess(max(abs(v - mean) for v in u), 1e-6)
        self.assertAlmostEqual(energy(u, dx), e0, places=10)


if __name__ == "__main__":
    unittest.main()
