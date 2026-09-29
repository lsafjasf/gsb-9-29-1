"""Benchmarks and verification runs for field_solver.

Generates CSV data into ./artifacts and prints summary tables:

  accuracy_vs_grid.csv      max error vs. analytic solution, n = 15..255
  acceleration.csv          Jacobi / GS / SOR / multigrid on the same problem
  residual_history.csv      per-iteration relative residual of every method
  point_source.csv          single point source, method comparison
  fine_grid.csv             very fine grids (n = 255, 511)
  edge_cases.csv            uniform boundary + non-convergent parameters
  divergence_history.csv    residual growth of SOR with omega >= 2

Run:  python3 run_benchmarks.py            (everything, a few minutes)
      python3 run_benchmarks.py quick      (skip the slowest sections)
"""

import csv
import math
import os
import sys

from field_solver import (CONVERGED, max_error, new_problem,
                          optimal_sor_omega, solve)

ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "artifacts")


def manufactured(n):
    """Analytic solution u = sin(pi x) sin(pi y) (zero boundary)."""
    f = lambda x, y: (2.0 * math.pi ** 2 * math.sin(math.pi * x)
                      * math.sin(math.pi * y))
    exact = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)
    u, fv, h = new_problem(n, f=f, g=0.0)
    return u, fv, h, exact


def write_csv(name, header, rows):
    path = os.path.join(ARTIFACTS, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (path, len(rows)))


def bench_accuracy_vs_grid():
    """Multigrid vs. analytic solution on successively refined grids."""
    print("\n== accuracy vs. grid (multigrid, tol=1e-10) ==")
    print("%6s %8s %6s %12s %12s %6s %8s"
          % ("n", "h", "iters", "final_relres", "max_error", "ratio", "time_s"))
    rows = []
    prev_err = None
    for n in (15, 31, 63, 127, 255):
        u, fv, h, exact = manufactured(n)
        res = solve(u, fv, n, method="multigrid", tol=1e-10)
        err = max_error(u, n, exact)
        ratio = prev_err / err if prev_err else float("nan")
        rows.append([n, "%.6f" % h, res.iterations,
                     "%.3e" % res.final_relative_residual,
                     "%.6e" % err, "%.2f" % ratio,
                     "%.3f" % res.wall_time])
        print("%6d %8.5f %6d %12.3e %12.3e %6.2f %8.3f"
              % (n, h, res.iterations, res.final_relative_residual,
                 err, ratio, res.wall_time))
        prev_err = err
    write_csv("accuracy_vs_grid.csv",
              ["n", "h", "iterations", "final_rel_residual",
               "max_error", "error_ratio", "wall_time_s"], rows)


def bench_acceleration(n=63, tol=1e-6):
    """All four methods on the same problem; per-iteration residuals."""
    print("\n== acceleration comparison (n=%d, tol=%.0e) ==" % (n, tol))
    print("%-14s %8s %12s %12s %8s" % ("method", "iters", "final_relres",
                                       "max_error", "time_s"))
    rows = []
    hist_rows = []
    for method in ("jacobi", "gauss_seidel", "sor", "multigrid"):
        u, fv, h, exact = manufactured(n)
        res = solve(u, fv, n, method=method, tol=tol, max_iter=100000)
        err = max_error(u, n, exact)
        rows.append([method, res.status, res.iterations,
                     "%.3e" % res.final_relative_residual,
                     "%.6e" % err, "%.3f" % res.wall_time])
        print("%-14s %8d %12.3e %12.3e %8.2f"
              % (method, res.iterations, res.final_relative_residual,
                 err, res.wall_time))
        for k, rel in enumerate(res.rel_residuals, 1):
            hist_rows.append([method, k, "%.6e" % rel])
    write_csv("acceleration.csv",
              ["method", "status", "iterations", "final_rel_residual",
               "max_error", "wall_time_s"], rows)
    write_csv("residual_history.csv",
              ["method", "iteration", "rel_residual"], hist_rows)


def bench_point_source(n=63, tol=1e-6):
    """Single unit point source at the domain centre, zero boundary."""
    print("\n== single point source (n=%d, tol=%.0e) ==" % (n, tol))
    print("%-14s %8s %12s %12s %8s" % ("method", "iters", "final_relres",
                                       "u_centre", "time_s"))
    rows = []
    for method in ("jacobi", "gauss_seidel", "sor", "multigrid"):
        h = 1.0 / (n + 1)
        u, fv, _ = new_problem(n, f=0.0, g=0.0)
        c = n // 2 + 1
        fv[c * (n + 2) + c] = 1.0 / (h * h)  # unit total source
        res = solve(u, fv, n, method=method, tol=tol, max_iter=100000)
        umax = u[c * (n + 2) + c]
        rows.append([method, res.status, res.iterations,
                     "%.3e" % res.final_relative_residual,
                     "%.6e" % umax, "%.3f" % res.wall_time])
        print("%-14s %8d %12.3e %12.6f %8.2f"
              % (method, res.iterations, res.final_relative_residual,
                 umax, res.wall_time))
    write_csv("point_source.csv",
              ["method", "status", "iterations", "final_rel_residual",
               "u_centre", "wall_time_s"], rows)


def bench_fine_grid():
    """Very fine grids: multigrid stays flat, SOR grows like O(1/h)."""
    print("\n== very fine grids (tol=1e-6) ==")
    print("%6s %-10s %8s %12s %12s %8s"
          % ("n", "method", "iters", "final_relres", "max_error", "time_s"))
    rows = []
    for n, methods in ((255, ("sor", "multigrid")), (511, ("multigrid",))):
        for method in methods:
            u, fv, h, exact = manufactured(n)
            res = solve(u, fv, n, method=method, tol=1e-6, max_iter=100000)
            err = max_error(u, n, exact)
            rows.append([n, method, res.status, res.iterations,
                         "%.3e" % res.final_relative_residual,
                         "%.6e" % err, "%.3f" % res.wall_time])
            print("%6d %-10s %8d %12.3e %12.3e %8.2f"
                  % (n, method, res.iterations,
                     res.final_relative_residual, err, res.wall_time))
    write_csv("fine_grid.csv",
              ["n", "method", "status", "iterations", "final_rel_residual",
               "max_error", "wall_time_s"], rows)


def bench_edge_cases():
    """Uniform boundary field and non-convergent SOR parameters."""
    print("\n== edge case: uniform boundary (g = 2.5 everywhere, f = 0) ==")
    rows = []
    for n in (31, 63):
        u, fv, h = new_problem(n, f=0.0, g=2.5)
        res = solve(u, fv, n, method="sor", tol=1e-12)
        err = max_error(u, n, lambda x, y: 2.5)
        rows.append(["uniform_boundary", n, res.status, res.iterations,
                     "%.3e" % res.final_relative_residual, "%.3e" % err])
        print("  n=%-4d status=%-9s iters=%-4d final_relres=%.2e "
              "max_error_vs_2.5=%.2e"
              % (n, res.status, res.iterations,
                 res.final_relative_residual, err))

    print("\n== edge case: non-convergent SOR parameters (n=31) ==")
    print("  omega* (optimal) = %.4f" % optimal_sor_omega(31))
    hist_rows = []
    for omega in (optimal_sor_omega(31), 2.0, 2.05, 2.1):
        u, fv, _, _ = manufactured(31)
        res = solve(u, fv, 31, method="sor", omega=omega,
                    tol=1e-8, max_iter=20000)
        rows.append(["sor_omega=%.4f" % omega, 31, res.status,
                     res.iterations,
                     "%.3e" % res.final_relative_residual, ""])
        print("  omega=%.4f  status=%-15s iters=%-6d final_relres=%.2e"
              % (omega, res.status, res.iterations,
                 res.final_relative_residual))
        for k, rel in enumerate(res.rel_residuals[:200], 1):
            hist_rows.append(["omega=%.4f" % omega, k, "%.6e" % rel])
    write_csv("edge_cases.csv",
              ["case", "n", "status", "iterations", "final_rel_residual",
               "max_error"], rows)
    write_csv("divergence_history.csv",
              ["case", "iteration", "rel_residual"], hist_rows)


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    quick = "quick" in sys.argv[1:]
    bench_accuracy_vs_grid()
    bench_acceleration(n=31 if quick else 63)
    bench_point_source(n=31 if quick else 63)
    if not quick:
        bench_fine_grid()
    bench_edge_cases()
    print("\ndone. CSV data in %s" % ARTIFACTS)


if __name__ == "__main__":
    main()
