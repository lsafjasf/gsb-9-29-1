"""标准测试问题集：目标函数、梯度、起点、最优值。"""

import math
import random


def make_quadratic(n=20, cond=1e3, seed=0):
    """病态二次型 f = 0.5 x^T A x - b^T x，A 对角、条件数 cond。"""
    rng = random.Random(seed)
    d = [math.exp(math.log(cond) * i / (n - 1)) for i in range(n)]
    b = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    x_star = [bi / di for bi, di in zip(b, d)]
    f_star = -0.5 * sum(bi * xi for bi, xi in zip(b, x_star))

    def f(x):
        return 0.5 * sum(di * xi * xi for di, xi in zip(d, x)) \
            - sum(bi * xi for bi, xi in zip(b, x))

    def g(x):
        return [di * xi - bi for di, xi, bi in zip(d, x, b)]

    return f, g, [0.0] * n, f_star, f"quad(n={n},cond={cond:g})"


def make_rosenbrock(n=2):
    """扩展 Rosenbrock，经典香蕉谷，最优 f* = 0 于全 1。"""
    def f(x):
        return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
                   for i in range(n - 1))

    def g(x):
        grad = [0.0] * n
        for i in range(n - 1):
            grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
            grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return grad

    x0 = [-1.2 if i % 2 == 0 else 1.0 for i in range(n)]
    return f, g, x0, 0.0, f"rosenbrock(n={n})"


def make_himmelblau(x0=(0.0, 0.0)):
    """Himmelblau 函数：非凸、4 个全局极小 f* = 0。"""
    def f(v):
        x, y = v
        return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2

    def g(v):
        x, y = v
        return [4.0 * x * (x * x + y - 11.0) + 2.0 * (x + y * y - 7.0),
                2.0 * (x * x + y - 11.0) + 4.0 * y * (x + y * y - 7.0)]

    return f, g, list(x0), 0.0, f"himmelblau(x0={x0})"


def make_quartic(n=5, x0_val=1.0):
    """f = sum x_i^4：最优处 Hessian 奇异、梯度三阶小量。"""
    def f(x):
        return sum(xi ** 4 for xi in x)

    def g(x):
        return [4.0 * xi ** 3 for xi in x]

    return f, g, [x0_val] * n, 0.0, f"quartic(n={n},x0={x0_val:g})"
