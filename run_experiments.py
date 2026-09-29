"""Numerical experiments for the 1D wave solver.

Generates CSV data under results/ and prints summary tables:

  reflection.csv   reflected-energy fraction R per boundary / pulse width / r
  energy.csv       energy drift vs time for closed boundaries
  dispersion.csv   numerical group velocity vs discrete theory
  narrow_pulse.csv dispersive distortion vs pulse width
  longtime.csv     energy / amplitude over a long closed-box run
  stability.csv    amplitude growth below/above the CFL limit

Standard library only.  Run:  python3 run_experiments.py
"""

import csv
import math
import os

from wave1d import (DIRICHLET, NEUMANN, MUR, WaveSolver1D,
                    gaussian_pulse, wave_packet, numerical_group_velocity)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _write_csv(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# 1. Boundary reflectivity
# ---------------------------------------------------------------------------

def measure_reflection(boundary, sigma_over_dx=8.0, r=0.9, nx=801):
    """Reflected-energy fraction for a Gaussian pulse hitting the right end.

    Left end is Mur (absorbing).  A right-moving Gaussian starts at x=L/4.
    R = max_t E_left(t) / E0, where E_left is the energy in [0, L/2] and
    the max is taken after the incident pulse has left that region, so
    only energy returning from the right boundary is counted.
    """
    c, L = 1.0, 1.0
    dx = L / (nx - 1)
    dt = r * dx / c
    x0 = 0.25 * L
    solver = WaveSolver1D(nx, dx, c, dt, left=MUR, right=boundary)
    u0, v0 = gaussian_pulse(nx, dx, x0, sigma_over_dx * dx, c)
    solver.set_initial(u0, v0)
    half = solver.nx // 2

    def left_energy():
        e = 0.0
        u, up = solver.u_cur, solver.u_prev
        for i in range(half):
            v = (u[i] - up[i]) / dt
            e += 0.5 * v * v * dx
        for i in range(half - 1):
            g = (u[i + 1] - u[i]) / dx
            e += 0.5 * c * c * g * g * dx
        return e

    t_hit = (L - x0) / c
    n_hit = round(t_hit / dt)
    solver.step(n_hit // 2)
    e0 = solver.energy()
    # incident pulse fully out of the left half well before t_hit
    n_leave = round((0.5 * L - x0 + 4.0 * sigma_over_dx * dx) / c / dt)
    n_end = round(2.2 * t_hit / dt)
    r_frac = 0.0
    for n in range(n_hit // 2, n_end):
        solver.step()
        if n >= n_leave and n % 5 == 0:
            r_frac = max(r_frac, left_energy() / e0)
    return r_frac


def run_reflection():
    rows = []
    for boundary in (DIRICHLET, NEUMANN, MUR):
        for s in (2.0, 4.0, 8.0, 16.0):
            rows.append([boundary, 0.9, s, "%.6g" % measure_reflection(boundary, s)])
    for r in (0.5, 0.7, 0.99):
        rows.append([MUR, r, 8.0, "%.6g" % measure_reflection(MUR, 8.0, r)])
    path = _write_csv("reflection.csv",
                      ["boundary", "courant_r", "sigma_over_dx", "R_energy"], rows)
    print("\n== Boundary reflectivity (R = reflected energy / incident energy) ==")
    print("%-10s %-6s %-12s %-12s" % ("boundary", "r", "sigma/dx", "R"))
    for b, r, s, R in rows:
        print("%-10s %-6s %-12s %-12s" % (b, r, s, R))
    print("->", path)


# ---------------------------------------------------------------------------
# 2. Energy conservation with closed boundaries
# ---------------------------------------------------------------------------

def run_energy():
    rows, summary = [], []
    for boundary in (DIRICHLET, NEUMANN):
        for r in (0.5, 0.9):
            c, L, nx = 1.0, 1.0, 401
            dx = L / (nx - 1)
            dt = r * dx / c
            solver = WaveSolver1D(nx, dx, c, dt, left=boundary, right=boundary)
            u0, v0 = gaussian_pulse(nx, dx, 0.5 * L, 0.05 * L, c, moving=False)
            solver.set_initial(u0, v0)
            e0s, e0c = solver.energy(), solver.conserved_energy()
            drift_s = drift_c = 0.0
            nsteps = 10000
            for n in range(1, nsteps + 1):
                solver.step()
                if n % 100 == 0:
                    ds = abs(solver.energy() / e0s - 1.0)
                    dc = abs(solver.conserved_energy() / e0c - 1.0)
                    drift_s, drift_c = max(drift_s, ds), max(drift_c, dc)
                    rows.append([boundary, r, n, "%.3e" % ds, "%.3e" % dc])
            summary.append([boundary, r, nsteps, "%.3e" % drift_s, "%.3e" % drift_c])
    path = _write_csv("energy.csv",
                      ["boundary", "courant_r", "step",
                       "rel_drift_simple", "rel_drift_conserved"], rows)
    print("\n== Energy conservation, closed box (max relative drift over run) ==")
    print("%-10s %-6s %-8s %-18s %-18s" %
          ("boundary", "r", "steps", "simple E", "conserved E"))
    for b, r, n, ds, dc in summary:
        print("%-10s %-6s %-8s %-18s %-18s" % (b, r, n, ds, dc))
    print("->", path)


# ---------------------------------------------------------------------------
# 3. Numerical dispersion: measured group velocity vs discrete theory
# ---------------------------------------------------------------------------

def measure_group_velocity(lam_over_dx, r=0.9, nx=2001):
    """Group velocity of a narrow-band packet from detector crossing times."""
    c, L = 1.0, 2.0
    dx = L / (nx - 1)
    dt = r * dx / c
    lam = lam_over_dx * dx
    k = 2.0 * math.pi / lam
    sigma = 4.0 * lam
    x0, x1, x2 = 0.4, 0.7, 1.5
    solver = WaveSolver1D(nx, dx, c, dt, left=MUR, right=MUR)
    u0, v0 = wave_packet(nx, dx, x0, sigma, k, c)
    solver.set_initial(u0, v0)
    i1, i2 = round(x1 / dx), round(x2 / dx)
    om = 2.0 / dt * math.asin(r * math.sin(0.5 * k * dx))
    s1, s2 = [], []
    nsteps = round((x2 - x0 + 6.0 * sigma) / c / dt)
    for _ in range(nsteps):
        solver.step()
        s1.append(solver.u_cur[i1])
        s2.append(solver.u_cur[i2])

    def peak_time(s):
        # envelope^2 ~ u^2 + (u_t/omega)^2, then parabolic peak interpolation
        env = []
        for n in range(1, len(s)):
            ut = (s[n] - s[n - 1]) / dt
            env.append(s[n] ** 2 + (ut / om) ** 2)
        n = max(range(1, len(env) - 1), key=lambda i: env[i])
        y0, y1, y2 = env[n - 1], env[n], env[n + 1]
        shift = 0.5 * (y0 - y2) / (y0 - 2.0 * y1 + y2)
        return (n + 1 + shift) * dt

    vg = (x2 - x1) / (peak_time(s2) - peak_time(s1))
    return vg, numerical_group_velocity(k, c, dt, dx), k * dx


def run_dispersion():
    rows = []
    print("\n== Grid dispersion: group velocity vs discrete theory ==")
    print("%-10s %-10s %-14s %-14s %-10s" %
          ("lambda/dx", "k*dx", "vg_theory/c", "vg_meas/c", "rel_err"))
    for lam in (5.0, 10.0, 20.0, 40.0):
        vg, vg_th, kdx = measure_group_velocity(lam)
        err = abs(vg / vg_th - 1.0)
        rows.append([lam, "%.4f" % kdx, "%.6f" % vg_th, "%.6f" % vg,
                     "%.3e" % err])
        print("%-10s %-10s %-14s %-14s %-10s" % tuple(rows[-1]))
    path = _write_csv("dispersion.csv",
                      ["lambda_over_dx", "k_dx", "vg_theory_over_c",
                       "vg_measured_over_c", "rel_err"], rows)
    print("->", path)


# ---------------------------------------------------------------------------
# 4. Narrow-pulse dispersive distortion
# ---------------------------------------------------------------------------

def run_narrow_pulse():
    rows = []
    print("\n== Narrow-pulse distortion after propagating 0.5 L ==")
    print("%-10s %-16s %-16s" % ("sigma/dx", "tail_energy_frac", "width_growth"))
    c, L, nx = 1.0, 2.0, 2001
    dx = L / (nx - 1)
    dt = 0.9 * dx / c
    for s in (2.0, 4.0, 8.0, 16.0):
        sigma = s * dx
        x0 = 0.3
        solver = WaveSolver1D(nx, dx, c, dt, left=MUR, right=MUR)
        u0, v0 = gaussian_pulse(nx, dx, x0, sigma, c)
        solver.set_initial(u0, v0)
        solver.step(round(0.5 / c / dt))
        u = solver.u_cur
        i_pk = max(range(nx), key=lambda i: abs(u[i]))
        # energy in the trailing tail, more than 6 sigma behind the peak
        i_cut = i_pk - round(6.0 * sigma / dx)
        etot = sum(v * v for v in u)
        etail = sum(v * v for v in u[:max(i_cut, 0)])
        # second-moment width vs initial width
        mean = sum(i * u[i] ** 2 for i in range(nx)) / etot
        var = sum((i - mean) ** 2 * u[i] ** 2 for i in range(nx)) / etot
        width = math.sqrt(2.0 * var) * dx
        rows.append([s, "%.6g" % (etail / etot), "%.4f" % (width / sigma)])
        print("%-10s %-16s %-16s" % tuple(rows[-1]))
    path = _write_csv("narrow_pulse.csv",
                      ["sigma_over_dx", "tail_energy_frac", "width_growth"],
                      rows)
    print("->", path)


# ---------------------------------------------------------------------------
# 5. Long-time evolution in a closed box
# ---------------------------------------------------------------------------

def run_longtime():
    c, L, nx, r = 1.0, 1.0, 201, 0.9
    dx = L / (nx - 1)
    dt = r * dx / c
    solver = WaveSolver1D(nx, dx, c, dt, left=NEUMANN, right=NEUMANN)
    u0, v0 = gaussian_pulse(nx, dx, 0.3 * L, 0.05 * L, c, moving=False)
    solver.set_initial(u0, v0)
    e0 = solver.conserved_energy()
    rows, nsteps = [], 50000
    for n in range(1, nsteps + 1):
        solver.step()
        if n % 1000 == 0:
            rows.append([n, "%.6g" % solver.max_abs(),
                         "%.3e" % abs(solver.conserved_energy() / e0 - 1.0)])
    path = _write_csv("longtime.csv",
                      ["step", "max_abs_u", "rel_drift_conserved_E"], rows)
    print("\n== Long-time evolution, Neumann box, r=%.2f, %d steps ==" % (r, nsteps))
    print("%-8s %-12s %-14s" % ("step", "max|u|", "consE drift"))
    for row in rows[::5]:
        print("%-8s %-12s %-14s" % tuple(row))
    print("->", path)


# ---------------------------------------------------------------------------
# 6. Stability demonstration around the CFL limit
# ---------------------------------------------------------------------------

def run_stability():
    rows = []
    print("\n== Stability vs Courant number (peak |u| growth over 3000 steps) ==")
    print("%-8s %-12s" % ("r", "growth"))
    for r in (0.9, 1.0, 1.01, 1.05):
        c, L, nx = 1.0, 1.0, 201
        dx = L / (nx - 1)
        dt = r * dx / c
        solver = WaveSolver1D(nx, dx, c, dt, left=DIRICHLET, right=DIRICHLET)
        u0, v0 = gaussian_pulse(nx, dx, 0.5 * L, 0.05 * L, c, moving=False)
        solver.set_initial(u0, v0)
        a0 = solver.max_abs()
        amax = a0
        for _ in range(30):
            solver.step(100)
            a = solver.max_abs()
            if not math.isfinite(a):
                amax = float("inf")
                break
            amax = max(amax, a)
            if amax > 1e6 * a0:
                break
        growth = amax / a0
        rows.append([r, "%.6g" % growth])
        print("%-8s %-12s" % (r, "%.6g" % growth))
    path = _write_csv("stability.csv", ["courant_r", "growth_3000_steps"], rows)
    print("->", path)


def main():
    run_reflection()
    run_energy()
    run_dispersion()
    run_narrow_pulse()
    run_longtime()
    run_stability()
    print("\nAll CSV data written under", RESULTS)


if __name__ == "__main__":
    main()
