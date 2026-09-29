"""测试问题构造：右端项、边界条件与解析解。"""

import math


def build_f(n, func):
    """把右端函数 func(x, y) 采样到 (n+2)x(n+2) 网格上（边界处不用，置 0）。"""
    h = 1.0 / (n + 1)
    f = [[0.0] * (n + 2) for _ in range(n + 2)]
    for i in range(1, n + 1):
        fi = f[i]
        x = i * h
        for j in range(1, n + 1):
            fi[j] = func(x, j * h)
    return f


def max_error(u, u_exact, n):
    """数值解与解析解 u_exact(x, y) 的最大模误差（内部点）。"""
    h = 1.0 / (n + 1)
    err = 0.0
    for i in range(1, n + 1):
        ui = u[i]
        x = i * h
        for j in range(1, n + 1):
            e = abs(ui[j] - u_exact(x, j * h))
            if e > err:
                err = e
    return err


def max_diff(u, v, n):
    """两个数值解在内部点上的最大模差。"""
    d = 0.0
    for i in range(1, n + 1):
        ui = u[i]
        vi = v[i]
        for j in range(1, n + 1):
            e = abs(ui[j] - vi[j])
            if e > d:
                d = e
    return d


# ------------------------------------------------------------ 问题 1：光滑解析解
# u = sin(πx)sin(πy)，-Δu = 2π² sin(πx)sin(πy)，零边界。
def sine_problem(n):
    f = build_f(n, lambda x, y: 2.0 * math.pi ** 2
                * math.sin(math.pi * x) * math.sin(math.pi * y))
    bc = lambda x, y: 0.0
    exact = lambda x, y: math.sin(math.pi * x) * math.sin(math.pi * y)
    return f, bc, exact


# ------------------------------------------------------------ 问题 2：边界全同
# 四周 u ≡ 1，f = 0，解析解 u ≡ 1。
def uniform_boundary_problem(n, value=1.0):
    f = build_f(n, lambda x, y: 0.0)
    bc = lambda x, y: value
    exact = lambda x, y: value
    return f, bc, exact


# ------------------------------------------------------------ 问题 3：单点源
# 中心一个强度 1/h² 的点源（δ 函数离散），零边界，无解析解，
# 用多重网格高精度解作为参考解对拍。
def point_source_problem(n):
    f = build_f(n, lambda x, y: 0.0)
    c = n // 2 + 1 if n % 2 == 0 else (n + 1) // 2
    h = 1.0 / (n + 1)
    f[c][c] = 1.0 / (h * h)
    bc = lambda x, y: 0.0
    return f, bc, c
