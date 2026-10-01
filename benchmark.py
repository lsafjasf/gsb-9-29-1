"""BFGS vs 梯度下降对拍：python3 benchmark.py

输出收敛轮数(nit)、函数求值次数(nfev)、梯度求值次数(ngev)、
最终 |f-f*| 与 ||g||inf，并同时写 convergence_data.md。
"""

import math

from quasinewton.bfgs import minimize_bfgs, minimize_gd
from quasinewton import problems


CASES = [
    # name, builder, gtol, maxiter
    ("二次函数(n=10, cond=1e4)",
     lambda: problems.quadratic_spd(10, 1e4), 1e-8, 300),
    ("Rosenbrock(n=2)", lambda: problems.rosenbrock(2), 1e-8, 1000),
    ("Rosenbrock(n=10)", lambda: problems.rosenbrock(10), 1e-6, 3000),
    ("Rastrigin 非凸多峰(n=2)", lambda: problems.rastrigin(2), 1e-8, 1000),
    ("四次函数近零梯度(n=3)", lambda: problems.quartic(3), 1e-13, 500),
    ("起点即最优", lambda: problems.shifted_quadratic(), 1e-8, 500),
]


def run():
    rows = []
    print("%-24s | %-7s | %5s | %6s | %6s | %11s | %10s | %s"
          % ("问题", "方法", "轮数", "nfev", "ngev", "|f-f*|", "||g||inf",
             "状态"))
    print("-" * 110)
    for name, build, gtol, maxit in CASES:
        f, g, x0, fstar = build()

        def solve(method):
            return method(f, x0, g, gtol=gtol, maxiter=maxit)

        rb = solve(minimize_bfgs)
        rg = solve(minimize_gd)
        for tag, r in (("BFGS", rb), ("GD", rg)):
            ferr = abs(r.fun - fstar) if (fstar is not None and r.fun is not None
                                          and math.isfinite(r.fun)) else float("nan")
            gnorm = r.grad_norm
            rows.append((name, tag, r, ferr, gnorm))
            print("%-24s | %-7s | %5d | %6d | %6d | %11.3e | %10.3e | %s"
                  % (name if tag == "BFGS" else "", tag, r.nit, r.nfev,
                     r.ngev, ferr, gnorm, r.status))
        print("-" * 110)

    # 边界情形的事件链也打印出来
    print("\n边界用例事件：")
    f, g, x0, _ = problems.shifted_quadratic()
    ev = []
    minimize_bfgs(f, x0, g, events=ev)
    print("  起点即最优:", ev)

    f, g, x0, _ = problems.quartic(3)
    ev = []
    r = minimize_bfgs(f, x0, g, gtol=1e-13, maxiter=500, events=ev)
    kinds = {}
    for k, _ in ev:
        kinds[k] = kinds.get(k, 0) + 1
    print("  近零梯度起步事件计数:", kinds, " nit=%d" % r.nit)

    with open("convergence_data.md", "w", encoding="utf-8") as fh:
        fh.write("# 收敛对比数据（BFGS vs 梯度下降）\n\n")
        fh.write("同一套强 Wolfe/Armijo 线搜索，c1=1e-4, c2=0.9；"
                 "停机 \\|g\\|_inf <= gtol。\n\n")
        fh.write("| 问题 | 方法 | 收敛轮数 | f 求值 | g 求值 | "
                 "\\|f-f*\\| | \\|g\\|inf | 状态 |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for name, tag, r, ferr, gnorm in rows:
            ferr_s = "%.3e" % ferr if math.isfinite(ferr) else "—"
            fh.write("| %s | %s | %d | %d | %d | %s | %.3e | %s |\n"
                     % (name, tag, r.nit, r.nfev, r.ngev, ferr_s, gnorm,
                        r.status))
    print("\n已写入 convergence_data.md")


if __name__ == "__main__":
    run()
