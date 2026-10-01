"""自测：python3 selftest.py  （仅标准库）"""

import math
import sys

from quasinewton.bfgs import minimize_bfgs, minimize_gd
from quasinewton.linesearch import line_search
from quasinewton import problems

PASS, FAIL = "PASS", "FAIL"
failures = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    print("[%s] %-42s %s" % (tag, name, detail))
    if not cond:
        failures.append(name)


def t_bfgs_quadratic():
    f, g, x0, fstar = problems.quadratic_spd(10, 1e4)
    r = minimize_bfgs(f, x0, g, gtol=1e-8)
    check("bfgs_quadratic", r.success and abs(r.fun - fstar) < 1e-8,
          "nit=%d nfev=%d f=%.3e" % (r.nit, r.nfev, r.fun))


def t_bfgs_rosenbrock():
    f, g, x0, fstar = problems.rosenbrock(2)
    r = minimize_bfgs(f, x0, g, gtol=1e-8)
    check("bfgs_rosenbrock2d", r.success and abs(r.fun - fstar) < 1e-10,
          "nit=%d nfev=%d f=%.3e" % (r.nit, r.nfev, r.fun))


def t_bfgs_rosenbrock10():
    f, g, x0, fstar = problems.rosenbrock(10)
    r = minimize_bfgs(f, x0, g, gtol=1e-6, maxiter=2000)
    # n>=4 的 Rosenbrock 存在非全局驻点，BFGS 收敛到任一驻点均合法
    check("bfgs_rosenbrock10d", r.success and r.grad_norm <= 1e-6,
          "nit=%d nfev=%d f=%.3e ginf=%.2e"
          % (r.nit, r.nfev, r.fun, r.grad_norm))


def t_optimal_start():
    f, g, x0, _ = problems.shifted_quadratic()
    r = minimize_bfgs(f, x0, g)
    check("optimal_start", r.success and r.nit == 0 and r.nfev == 1,
          "nit=%d nfev=%d status=%s" % (r.nit, r.nfev, r.status))


def t_nonconvex_multimodal():
    f, g, x0, fstar = problems.rastrigin(2)
    r = minimize_bfgs(f, x0, g, gtol=1e-8)
    check("nonconvex_rastrigin", r.success and abs(r.fun - fstar) < 1e-6,
          "nit=%d nfev=%d f=%.3e" % (r.nit, r.nfev, r.fun))


def t_near_zero_gradient():
    f, g, x0, fstar = problems.quartic(3)
    r = minimize_bfgs(f, x0, g, gtol=1e-13, maxiter=200)
    check("near_zero_gradient", r.success and r.fun < 1e-16,
          "nit=%d nfev=%d f=%.3e" % (r.nit, r.nfev, r.fun))


def t_saddle_escape():
    f, g, x0, _ = problems.saddle_start()
    r = minimize_bfgs(f, x0, g, gtol=1e-8, maxiter=50)
    ok = (not r.success) or r.fun < f(x0)
    check("saddle_no_false_converge", ok,
          "status=%s f=%.3e (f0=%.3e)" % (r.status, r.fun, f(x0)))


def t_spd_maintained():
    """曲率条件被违反时跳过更新，且 H 保持对称（数值上）。"""
    f, g, x0, _ = problems.quadratic_spd(5, 100.0)
    events = []
    r = minimize_bfgs(f, x0, g, events=events)
    check("spd_via_curvature", r.success, "nit=%d" % r.nit)
    # 人为构造违反曲率的情形：线性函数 y^T s = 0
    from quasinewton.bfgs import _bfgs_inverse_update
    H = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    ok, sy = _bfgs_inverse_update(H, [1.0, 0, 0], [0.0, 0, 0], 1e-10)
    sym = all(H[i][j] == H[j][i] for i in range(3) for j in range(3))
    check("curvature_skip_keeps_spd", (not ok) and sym, "sy=%r" % sy)


def t_linesearch_forced_wolfe_fail():
    """max_bracket=0 强制 Wolfe 失败，应降级到 Armijo 并记录原因。"""
    f, g, x0, _ = problems.quadratic_spd(3, 10.0)
    grad = g(x0)
    p = [-v for v in grad]
    events = []
    r = line_search(f, g, x0, p, f(x0), grad,
                    max_bracket=0, events=events)
    kinds = [e[0] for e in events]
    check("ls_wolfe_fail_fallback",
          r.success and r.method == "armijo_fallback"
          and "wolfe_failed" in kinds,
          "method=%s events=%s" % (r.method, kinds))


def t_linesearch_total_failure():
    """误导方向 + 极小 alpha_min 卡死：Wolfe/Armijo/-grad 三级全失败。"""
    f = lambda x: math.exp(1e6 * x[0]) if x[0] > 0 else 1.0 + x[0]
    g = lambda x: [-1.0]
    events = []
    r = line_search(f, g, [0.0], [1.0], 1.0, [-1.0],
                    alpha_min=0.6, max_armijo=3, max_bracket=3,
                    events=events)
    kinds = [e[0] for e in events]
    check("ls_total_failure",
          (not r.success) and r.reason and "line_search_failed" in kinds,
          "reason=%s" % r.reason)


def t_gd_baseline():
    f, g, x0, fstar = problems.quadratic_spd(10, 100.0)
    r = minimize_gd(f, x0, g, gtol=1e-8, maxiter=20000)
    check("gd_quadratic", r.success and abs(r.fun - fstar) < 1e-8,
          "nit=%d nfev=%d f=%.3e" % (r.nit, r.nfev, r.fun))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for t in tests:
        t()
    print("-" * 60)
    if failures:
        print("FAILED: %d/%d -> %s" % (len(failures), len(tests), failures))
        return 1
    print("ALL %d TESTS PASSED" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
