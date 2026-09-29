"""Self-tests for wave1d: stability, energy, boundaries, dispersion.

Run:  python3 -m unittest -v test_wave1d
"""

import math
import unittest

from wave1d import (DIRICHLET, NEUMANN, MUR, WaveSolver1D,
                    gaussian_pulse, wave_packet, numerical_omega,
                    numerical_group_velocity)
from run_experiments import measure_reflection, measure_group_velocity


def make_solver(nx=401, r=0.9, left=DIRICHLET, right=DIRICHLET, c=1.0):
    dx = 1.0 / (nx - 1)
    return WaveSolver1D(nx, dx, c, r * dx / c, left=left, right=right)


class TestStability(unittest.TestCase):
    """CFL condition: r <= 1 stable, r > 1 unstable."""

    def test_stable_below_cfl(self):
        s = make_solver(r=0.9)
        u0, v0 = gaussian_pulse(s.nx, s.dx, 0.5, 0.05)
        s.set_initial(u0, v0)
        a0 = s.max_abs()
        s.step(5000)
        self.assertLess(s.max_abs(), 5.0 * a0)
        self.assertTrue(math.isfinite(s.max_abs()))

    def test_stable_at_cfl(self):
        s = make_solver(r=1.0)
        u0, v0 = gaussian_pulse(s.nx, s.dx, 0.5, 0.05)
        s.set_initial(u0, v0)
        a0 = s.max_abs()
        s.step(5000)
        self.assertLess(s.max_abs(), 5.0 * a0)

    def test_unstable_above_cfl(self):
        s = make_solver(r=1.05)
        u0, v0 = gaussian_pulse(s.nx, s.dx, 0.5, 0.05)
        s.set_initial(u0, v0)
        a0 = s.max_abs()
        blown = False
        for _ in range(30):
            s.step(100)
            a = s.max_abs()
            if not math.isfinite(a) or a > 10.0 * a0:
                blown = True
                break
        self.assertTrue(blown, "r=1.05 should blow up")


class TestEnergy(unittest.TestCase):
    """Closed boundaries must conserve the discrete energy."""

    def check_energy(self, boundary):
        s = make_solver(r=0.9, left=boundary, right=boundary)
        u0, v0 = gaussian_pulse(s.nx, s.dx, 0.5, 0.05, moving=False)
        s.set_initial(u0, v0)
        e0c, e0s = s.conserved_energy(), s.energy()
        drift_c = drift_s = 0.0
        for _ in range(4000):
            s.step()
            drift_c = max(drift_c, abs(s.conserved_energy() / e0c - 1.0))
            drift_s = max(drift_s, abs(s.energy() / e0s - 1.0))
        return drift_c, drift_s

    def test_dirichlet_energy(self):
        drift_c, drift_s = self.check_energy(DIRICHLET)
        self.assertLess(drift_c, 1e-8)   # exactly conserved quantity
        self.assertLess(drift_s, 0.05)   # simple form only oscillates

    def test_neumann_energy(self):
        drift_c, drift_s = self.check_energy(NEUMANN)
        self.assertLess(drift_c, 1e-8)
        self.assertLess(drift_s, 0.05)


class TestBoundaries(unittest.TestCase):
    """Reflection coefficients: closed ends reflect ~all, Mur ~none."""

    def test_dirichlet_reflects_all(self):
        self.assertGreater(measure_reflection(DIRICHLET), 0.99)

    def test_neumann_reflects_all(self):
        self.assertGreater(measure_reflection(NEUMANN), 0.99)

    def test_mur_absorbs(self):
        self.assertLess(measure_reflection(MUR, sigma_over_dx=8.0), 0.01)

    def test_mur_absorbs_narrow_pulse(self):
        # even a 4-points-wide pulse keeps reflected energy small
        self.assertLess(measure_reflection(MUR, sigma_over_dx=4.0), 0.05)

    def test_dirichlet_flips_sign(self):
        # u=0 end inverts the pulse; a Neumann end does not
        def reflected_peak(boundary):
            s = make_solver(nx=801, r=0.9, left=MUR, right=boundary)
            u0, v0 = gaussian_pulse(s.nx, s.dx, 0.25, 8.0 * s.dx)
            s.set_initial(u0, v0)
            s.step(round(2.0 * 0.75 / s.dt))
            i = round(0.25 / s.dx)
            return s.u_cur[i]
        self.assertLess(reflected_peak(DIRICHLET), -0.5)
        self.assertGreater(reflected_peak(NEUMANN), 0.5)


class TestDispersion(unittest.TestCase):
    """Grid dispersion: discrete theory matches the solver, not u_tt=c^2u_xx."""

    def test_dispersion_relation_residual(self):
        # plane wave exp(i(kx-wt)) must satisfy the discrete relation
        c, dt, dx = 1.0, 9e-4, 1e-3
        k = 2.0 * math.pi / (10.0 * dx)
        om = numerical_omega(k, c, dt, dx)
        r = c * dt / dx
        res = math.sin(0.5 * om * dt) - r * math.sin(0.5 * k * dx)
        self.assertAlmostEqual(res, 0.0, places=12)
        # and it must differ from the continuum omega = c k
        self.assertNotAlmostEqual(om, c * k, places=3)

    def test_group_velocity_matches_theory(self):
        for lam in (10.0, 20.0):
            vg, vg_th, _ = measure_group_velocity(lam)
            self.assertLess(abs(vg / vg_th - 1.0), 0.05)

    def test_short_waves_are_slower(self):
        c, dt, dx = 1.0, 9e-4, 1e-3
        slow = numerical_group_velocity(2 * math.pi / (5 * dx), c, dt, dx)
        fast = numerical_group_velocity(2 * math.pi / (40 * dx), c, dt, dx)
        self.assertLess(slow, 0.97 * c)
        self.assertGreater(fast, 0.99 * c)


class TestConvergence(unittest.TestCase):
    """Second-order accuracy against the exact standing-wave solution."""

    def error(self, nx, r=0.9, periods=1.25):
        c, L = 1.0, 1.0
        dx = L / (nx - 1)
        dt = r * dx / c
        s = WaveSolver1D(nx, dx, c, dt, left=DIRICHLET, right=DIRICHLET)
        k = math.pi / L
        u0 = [math.sin(k * i * dx) for i in range(nx)]
        s.set_initial(u0, [0.0] * nx)
        # compare with the continuum exact solution sin(kx) cos(c k t);
        # the error is then dominated by the O(dx^2) frequency defect
        # of the discrete dispersion relation.  1.25 periods puts
        # |sin(c k t)| = 1 so the phase defect shows at full strength.
        t = periods * 2.0 * math.pi / (c * k)
        s.step(round(t / dt))
        err = 0.0
        for i in range(nx):
            exact = math.sin(k * i * dx) * math.cos(c * k * s.t)
            err += (s.u_cur[i] - exact) ** 2 * dx
        return math.sqrt(err)

    def test_second_order(self):
        e1 = self.error(101)
        e2 = self.error(201)
        ratio = e1 / e2
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 5.0)


class TestLongTime(unittest.TestCase):
    """Long closed-box run: no drift, no blow-up."""

    def test_longtime_bounded(self):
        s = make_solver(nx=201, r=0.9, left=NEUMANN, right=NEUMANN)
        u0, v0 = gaussian_pulse(s.nx, s.dx, 0.3, 0.05, moving=False)
        s.set_initial(u0, v0)
        e0 = s.conserved_energy()
        a0 = s.max_abs()
        s.step(20000)
        self.assertLess(abs(s.conserved_energy() / e0 - 1.0), 1e-8)
        self.assertLess(s.max_abs(), 5.0 * a0)


if __name__ == "__main__":
    unittest.main()
