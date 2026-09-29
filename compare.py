"""BFGS 与梯度下降对拍：收敛轮数 / 函数求值次数 / 最终精度。"""

from bfgs import minimize_bfgs, minimize_gd
from problems import make_quadratic, make_rosenbrock, make_himmelblau, make_quartic

TOL = 1e-6


def run():
    cases = [
        make_quadratic(n=20, cond=1e3),
        make_rosenbrock(n=2),
        make_rosenbrock(n=10),
        make_himmelblau((0.0, 0.0)),
        make_quartic(n=5, x0_val=1.3),
    ]
    header = (f"{'problem':<26}{'method':<7}{'iter':>7}{'f_eval':>8}{'g_eval':>8}"
              f"{'f_final':>14}{'|f-f*|':>12}{'|g|_inf':>11}  status")
    print(header)
    print("-" * len(header))
    for f, g, x0, f_star, name in cases:
        for method, solver, kw in (
            ("BFGS", minimize_bfgs, dict(tol=TOL, max_iter=2000)),
            ("GD", minimize_gd, dict(tol=TOL, max_iter=100000)),
        ):
            r = solver(f, g, x0, **kw)
            print(f"{name:<26}{method:<7}{r.n_iter:>7}{r.n_fev:>8}{r.n_gev:>8}"
                  f"{r.fun:>14.6e}{abs(r.fun - f_star):>12.3e}"
                  f"{max(abs(v) for v in r.grad):>11.3e}  {r.status}")
        print()


if __name__ == "__main__":
    run()
