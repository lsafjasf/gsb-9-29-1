"""wave1d 自测:  python3 -m unittest test_wave1d -v  (或 python3 test_wave1d.py)

覆盖: CFL 稳定条件、三种边界的反射率、封闭边界能量守恒、
      二阶收敛性、极窄脉冲、高频模态、长时间演化。
"""

import math
import unittest

from wave1d import (Wave1D, Wave1D as W, DIRICHLET, NEUMANN, MUR,
                    traveling_gaussian, cfl_max_dt, dispersion_omega,
                    group_velocity, mur_reflection_coeff)


def make(nx=201, r=0.9, left=DIRICHLET, right=DIRICHLET, c=1.0, strict=True):
    dx = 1.0 / (nx - 1)
    return Wave1D(nx, c, dx, r * dx / c, left=left, right=right, strict=strict)


def reflection_energy(right_bc, width, r=0.9, nx=501):
    """右行脉冲打右边界, 返回能量反射率。"""
    c = 1.0
    dx = 1.0 / (nx - 1)
    sim = Wave1D(nx, c, dx, r * dx / c, left=MUR, right=right_bc)
    sim.set_right_traveling_gaussian(0.25, width)
    e0 = sim.energy()
    sim.step(round(1.10 / sim.dt))
    return sim.energy() / e0


class TestStability(unittest.TestCase):
    def test_cfl_violation_raises(self):
        with self.assertRaises(ValueError):
            Wave1D(101, 1.0, 0.01, 1.01 * cfl_max_dt(1.0, 0.01))

    def test_cfl_boundary_accepted(self):
        make(r=1.0)  # r = 1 恰好稳定, 不应抛异常

    def test_supercritical_blows_up(self):
        sim = make(r=1.01, strict=False)
        sim.set_right_traveling_gaussian(0.5, 0.05)
        sim.step(2000)
        self.assertGreater(sim.max_abs(), 1e6)

    def test_critical_stays_bounded(self):
        sim = make(r=1.0)
        sim.set_right_traveling_gaussian(0.5, 0.05)
        sim.step(5000)
        self.assertLess(sim.max_abs(), 2.0)


class TestReflection(unittest.TestCase):
    def test_dirichlet_total_reflection_inverted(self):
        sim = make(right=DIRICHLET, left=MUR)
        sim.set_right_traveling_gaussian(0.25, 0.04)
        sim.step(round(0.85 / sim.dt))  # 反射波回到域内
        self.assertAlmostEqual(reflection_energy(DIRICHLET, 0.04), 1.0, places=6)
        self.assertLess(min(sim.u), -0.5)  # 反相

    def test_neumann_total_reflection_not_inverted(self):
        sim = make(right=NEUMANN, left=MUR)
        sim.set_right_traveling_gaussian(0.25, 0.04)
        sim.step(round(0.85 / sim.dt))
        self.assertAlmostEqual(reflection_energy(NEUMANN, 0.04), 1.0, places=6)
        self.assertGreater(max(sim.u), 0.5)  # 不反相

    def test_mur_low_reflection(self):
        self.assertLess(reflection_energy(MUR, 0.04), 1e-6)

    def test_mur_exact_at_r1(self):
        self.assertLess(reflection_energy(MUR, 0.04, r=1.0), 1e-24)

    def test_mur_narrow_pulse_higher_reflection(self):
        # 极窄脉冲含高频分量, 反射更强, 但仍应远小于 1
        r_narrow = reflection_energy(MUR, 0.004)
        r_wide = reflection_energy(MUR, 0.04)
        self.assertGreater(r_narrow, r_wide)
        self.assertLess(r_narrow, 1e-2)

    def test_mur_coeff_matches_theory(self):
        # 单色波包实测反射系数与离散理论值同量级
        kdx, c, nx = 0.2, 1.0, 1001
        dx = 2.0 / (nx - 1)
        dt = 0.9 * dx / c
        k = kdx / dx
        width = 8.0 / k
        sim = Wave1D(nx, c, dx, dt, left=MUR, right=MUR)
        u0, v0 = [], []
        for i in range(nx):
            x = i * dx
            s = (x - 0.5) / width
            env = math.exp(-s * s)
            u0.append(env * math.cos(k * x))
            v0.append(c * (2 * s / width * env * math.cos(k * x)
                           + k * env * math.sin(k * x)))
        sim.set_initial(u0, v0)
        e0 = sim.energy()
        sim.step(round(1.8 / dt))
        r_meas = math.sqrt(sim.energy() / e0)
        r_theo = mur_reflection_coeff(k, c, dx, dt)
        self.assertLess(abs(r_meas / r_theo - 1.0), 0.3)


class TestEnergy(unittest.TestCase):
    def run_drift(self, bc, nsteps):
        sim = make(left=bc, right=bc)
        sim.set_right_traveling_gaussian(0.5, 0.05)
        e0 = sim.energy()
        drift = 0.0
        for _ in range(nsteps // 1000):
            sim.step(1000)
            drift = max(drift, abs(sim.energy() / e0 - 1.0))
        return drift

    def test_energy_conserved_dirichlet(self):
        self.assertLess(self.run_drift(DIRICHLET, 20000), 1e-10)

    def test_energy_conserved_neumann(self):
        self.assertLess(self.run_drift(NEUMANN, 20000), 1e-10)

    def test_energy_long_time(self):
        # 长时间演化: 5 万步 (~112 次往返) 能量仍守恒
        self.assertLess(self.run_drift(DIRICHLET, 50000), 1e-10)

    def test_naive_energy_bounded(self):
        sim = make()
        sim.set_right_traveling_gaussian(0.5, 0.05)
        e0 = sim.energy_naive()
        for _ in range(20):
            sim.step(1000)
            self.assertLess(abs(sim.energy_naive() / e0 - 1.0), 0.1)


class TestAccuracy(unittest.TestCase):
    def test_second_order_convergence(self):
        # 驻波 u = sin(2 pi x) cos(2 pi t), 固定 r, dx 减半误差应约 4 倍
        errs = []
        for nx in (51, 101, 201):
            c, r = 1.0, 0.9
            dx = 1.0 / (nx - 1)
            dt = r * dx / c
            k = 2.0 * math.pi
            sim = Wave1D(nx, c, dx, dt)
            sim.set_initial([math.sin(k * i * dx) for i in range(nx)])
            nsteps = round(0.625 / dt)  # sin(w t) != 0, 相位误差可被观测
            t_end = nsteps * dt  # 用整步到达的终点, 避免时间量化误差
            sim.step(nsteps)
            err = math.sqrt(sum(
                (ui - math.sin(k * i * dx) * math.cos(k * t_end)) ** 2
                for i, ui in enumerate(sim.u)) * dx)
            errs.append(err)
        rate = math.log(errs[0] / errs[2]) / math.log(4.0)
        self.assertGreater(rate, 1.8)
        self.assertLess(rate, 2.2)

    def test_dispersion_relation(self):
        # 数值频率须满足 sin(w dt/2) = r sin(k dx/2)
        c, dx, r = 1.0, 0.01, 0.9
        dt = r * dx / c
        for m in (1, 8, 32):
            k = m * math.pi
            lhs = math.sin(0.5 * dispersion_omega(k, c, dx, dt) * dt)
            self.assertAlmostEqual(lhs, r * math.sin(0.5 * k * dx), places=12)

    def test_group_velocity_decreases_with_k(self):
        c, dx, dt = 1.0, 0.01, 0.009
        v_low = group_velocity(0.1 / dx, c, dx, dt)
        v_high = group_velocity(1.0 / dx, c, dx, dt)
        self.assertLess(v_high, v_low)


class TestStressCases(unittest.TestCase):
    def test_very_narrow_pulse_stable(self):
        # 极窄脉冲 (半宽 2dx): 会色散变形, 但封闭边界下能量必须守恒
        sim = make()
        sim.set_right_traveling_gaussian(0.5, 2.0 * sim.dx)
        e0 = sim.energy()
        sim.step(10000)
        self.assertTrue(math.isfinite(sim.max_abs()))
        self.assertAlmostEqual(sim.energy() / e0, 1.0, places=10)

    def test_high_frequency_mode(self):
        # 高频模态 k*dx ~ 1: 相速度失真 ~1%, 与色散关系预测一致
        nx, c, r = 101, 1.0, 0.9
        dx = 1.0 / (nx - 1)
        dt = r * dx / c
        m = 32
        k = m * math.pi
        mode = [math.sin(k * i * dx) for i in range(nx)]
        sim = Wave1D(nx, c, dx, dt)
        sim.set_initial(mode)
        nsteps = int(10.0 * (2.0 * math.pi / (c * k)) / dt)
        crossings, a_prev = [], sum(u * s for u, s in zip(sim.u, mode))
        for _ in range(nsteps):
            sim.step()
            a = sum(u * s for u, s in zip(sim.u, mode))
            if a_prev < 0.0 <= a:
                crossings.append((sim.n - 1 + -a_prev / (a - a_prev)) * dt)
            a_prev = a
        period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
        w_meas = 2.0 * math.pi / period
        w_theo = dispersion_omega(k, c, dx, dt)
        self.assertLess(abs(w_meas / w_theo - 1.0), 1e-3)
        self.assertLess(w_meas / (c * k), 0.995)  # 确实有色散失真


if __name__ == "__main__":
    unittest.main()
