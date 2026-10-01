"""标准测试问题：全部解析梯度，覆盖要求的边界情形。"""

import math


def quadratic_spd(n=10, kappa=1e4):
    """良态但病态条件数的二次函数，f* = 0 于 x=0。"""
    lam = [1.0 + (kappa - 1.0) * i / (n - 1) for i in range(n)]

    def f(x):
        return 0.5 * sum(li * xi * xi for li, xi in zip(lam, x))

    def g(x):
        return [li * xi for li, xi in zip(lam, x)]

    return f, g, [1.0] * n, 0.0


def rosenbrock(n=2):
    def f(x):
        return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
                   for i in range(n - 1))

    def g(x):
        gr = [0.0] * n
        for i in range(n - 1):
            gr[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
            gr[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
        return gr

    return f, g, [-1.2] + [1.0] * (n - 1), 0.0


def rastrigin(n=2):
    """非凸多峰，全局最小 f*=0 于 x=0。"""
    def f(x):
        return 10.0 * n + sum(xi * xi - 10.0 * math.cos(2.0 * math.pi * xi)
                              for xi in x)

    def g(x):
        return [2.0 * xi + 20.0 * math.pi * math.sin(2.0 * math.pi * xi)
                for xi in x]

    return f, g, [0.25, -0.25] if n == 2 else [0.25] * n, 0.0


def quartic(n=3):
    """sum x_i^4：原点梯度为零但非最优点（测试近零梯度起步）。"""
    def f(x):
        return sum(xi ** 4 for xi in x)

    def g(x):
        return [4.0 * xi ** 3 for xi in x]

    return f, g, [1e-4] * n, 0.0


def saddle_start():
    """f = x^2 - y^2，起点在鞍点方向分量极小处。"""
    def f(x):
        return x[0] ** 2 - x[1] ** 2

    def g(x):
        return [2.0 * x[0], -2.0 * x[1]]

    return f, g, [1e-8, 1.0], None


def shifted_quadratic():
    """起点即最优：f = (x-1)^2 + (y+2)^2，x0 = (1, -2)。"""
    def f(x):
        return (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2

    def g(x):
        return [2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)]

    return f, g, [1.0, -2.0], 0.0
