"""最小用例：python3 example.py"""

import math

from quasinewton import minimize_bfgs, minimize_gd


def rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def rosenbrock_grad(x):
    return [
        -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] ** 2),
    ]


if __name__ == "__main__":
    x0 = [-1.2, 1.0]
    events = []
    r = minimize_bfgs(rosenbrock, x0, rosenbrock_grad, events=events)
    print("BFGS  : nit=%d nfev=%d f=%.3e status=%s"
          % (r.nit, r.nfev, r.fun, r.status))
    print("  x =", [round(v, 6) for v in r.x])
    print("  事件:", events)

    r2 = minimize_gd(rosenbrock, x0, rosenbrock_grad, maxiter=5000)
    print("GD    : nit=%d nfev=%d f=%.3e status=%s"
          % (r2.nit, r2.nfev, r2.fun, r2.status))
