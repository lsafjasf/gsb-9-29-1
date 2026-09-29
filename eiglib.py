"""eiglib: 小型稠密实对称矩阵的全部特征值/特征向量求解器（仅标准库）。

为什么限定实对称矩阵
--------------------
"特征向量必须正交归一" 这一要求只在（实）对称矩阵上有保证 —— 谱定理：
实对称矩阵存在完备的正交归一特征向量系。非对称矩阵一般没有正交特征向量系，
因此本库只接受实对称输入，并对非对称输入抛出 ValueError。

算法
----
主求解器 ``qr_eigen``（正交迭代的标准实用形式）：

1. Householder 正交相似变换把 A 三对角化：T = Q1^T A Q1。
2. 对三对角阵做带 Wilkinson 位移的 QR 迭代（等价于对 A - mu I 做正交迭代
   并累计正交变换），每步用 Givens 旋转完成 QR，保持正交性。
3. 收缩（deflation）：当次对角元满足
       |e_i| <= eps * (|d_i| + |d_{i+1}|)        (eps = 双精度机器 epsilon)
   时置零，矩阵分裂为更小的块分别处理；2x2 块用 Jacobi 旋转解析对角化。
   这是 LAPACK 同款判据：按局部对角元尺度做相对判断，因此对
   "量级差异极大" 的谱是稳健的；对重根/近重根也不需要特殊处理 ——
   重根只是让某些次对角元衰减到判据以下后自然分裂，Wilkinson 位移
   保证对称三对角情形下的快速（局部三次）收敛，不存在需要
   "双位移" 的非对称情形。

多重特征值的处理
----------------
重根对应的特征子空间维数 > 1，其内任意正交归一基都是合法特征向量系。
因此：
- 算法层面不做任何重根检测，QR 迭代自然收敛；
- 验证层面不能逐向量比较夹角，而要比较 **特征子空间**：
  对按间隙聚类后的每个簇，计算两组向量张成子空间之间的
  主夹角（principal angle，取最大者），见 test_eig.py。

参照实现 ``jacobi_eigen``
-------------------------
经典循环 Jacobi 扫描（另一族正交相似迭代），与 QR 完全独立，
用于对拍。收敛判据：off(A) <= eps * ||A||_F。

返回值约定
----------
两个求解器都返回 ``(eigenvalues, eigenvectors)``：
- eigenvalues: 升序 list[float]
- eigenvectors: 按列存储的特征向量，即 eigenvectors[i][j] 是第 j 个
  特征向量的第 i 个分量；满足 V^T V = I、A V = V diag(eigenvalues)。
"""

import math

_EPS = 2.220446049250313e-16  # 双精度机器 epsilon


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _check_symmetric(a, tol=1e-10):
    n = len(a)
    for row in a:
        if len(row) != n:
            raise ValueError("matrix must be square")
    scale = max((abs(x) for row in a for x in row), default=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > tol * max(1.0, scale):
                raise ValueError(
                    "matrix is not symmetric; orthonormal eigenvectors only "
                    "exist for (real) symmetric matrices"
                )


def _sort_eigenpairs(values, vectors):
    n = len(values)
    order = sorted(range(n), key=lambda k: values[k])
    sorted_values = [values[k] for k in order]
    sorted_vectors = [[vectors[i][k] for k in order] for i in range(len(vectors))]
    return sorted_values, sorted_vectors


# ---------------------------------------------------------------------------
# Householder 三对角化
# ---------------------------------------------------------------------------

def _tridiagonalize(a):
    """对称矩阵 -> (d, e, Q)，T = Q^T A Q 为三对角，d 对角、e 次对角。"""
    n = len(a)
    a = [row[:] for row in a]
    q = _identity(n)
    for k in range(n - 2):
        m = n - k - 1
        x = [a[k + 1 + i][k] for i in range(m)]
        nx = math.sqrt(sum(v * v for v in x))
        if nx == 0.0:
            continue
        sign = 1.0 if x[0] >= 0.0 else -1.0
        u = x[:]
        u[0] += sign * nx
        nu = math.sqrt(sum(v * v for v in u))
        v = [t / nu for t in u]
        # 子块 B <- (I - 2 v v^T) B (I - 2 v v^T)，利用对称性只做一次 B v
        sub = [[a[k + 1 + i][k + 1 + j] for j in range(m)] for i in range(m)]
        w = [sum(sub[i][j] * v[j] for j in range(m)) for i in range(m)]
        vw = sum(v[i] * w[i] for i in range(m))
        for i in range(m):
            vi = v[i]
            wi = w[i]
            for j in range(m):
                sub[i][j] += -2.0 * vi * w[j] - 2.0 * wi * v[j] \
                    + 4.0 * vw * vi * v[j]
        for i in range(m):
            for j in range(m):
                a[k + 1 + i][k + 1 + j] = sub[i][j]
        for i in range(k + 2, n):
            a[i][k] = 0.0
            a[k][i] = 0.0
        a[k + 1][k] = a[k][k + 1] = -sign * nx
        # 累计 Q <- Q (I - 2 v v^T)（作用在 k+1..n-1 列上）
        for i in range(n):
            dot = sum(q[i][k + 1 + j] * v[j] for j in range(m))
            for j in range(m):
                q[i][k + 1 + j] -= 2.0 * dot * v[j]
    d = [a[i][i] for i in range(n)]
    e = [a[i][i + 1] for i in range(n - 1)]
    return d, e, q


# ---------------------------------------------------------------------------
# 三对角 QR 迭代（Wilkinson 位移 + 收缩）
# ---------------------------------------------------------------------------

def _wilkinson_shift(a, b, c):
    """[[a, c], [c, b]] 中离 b 最近的特征值（稳定公式）。"""
    delta = 0.5 * (a - b)
    s = math.sqrt(delta * delta + c * c)
    if delta >= 0.0:
        denom = delta + s
    else:
        denom = delta - s
    if denom == 0.0:
        return b
    return b - c * c / denom


def _qr_step(d, e, q, p, r, mu):
    """对三对角块 [p..r] 做一步显式位移 QR：T <- RQ + mu I，并累计 Q。"""
    m = r - p + 1
    t = [[0.0] * m for _ in range(m)]
    for i in range(m):
        t[i][i] = d[p + i] - mu
    for i in range(m - 1):
        t[i][i + 1] = e[p + i]
        t[i + 1][i] = e[p + i]
    qb = _identity(m)
    # Givens QR：G_i 作用在 (i, i+1) 行上消去次对角元
    for i in range(m - 1):
        a = t[i][i]
        b = t[i + 1][i]
        if b == 0.0:
            continue
        hyp = math.hypot(a, b)
        c = a / hyp
        s = b / hyp
        for j in range(m):
            ti = t[i][j]
            tj = t[i + 1][j]
            t[i][j] = c * ti + s * tj
            t[i + 1][j] = -s * ti + c * tj
        # qb <- qb G_i^T（作用在 (i, i+1) 列上）
        for j in range(m):
            qi = qb[j][i]
            qj = qb[j][i + 1]
            qb[j][i] = c * qi + s * qj
            qb[j][i + 1] = -s * qi + c * qj
    # T_new = R Qb + mu I（R 即变换后的 t）
    tn = [[sum(t[i][k] * qb[k][j] for k in range(m)) + (mu if i == j else 0.0)
           for j in range(m)] for i in range(m)]
    for i in range(m):
        d[p + i] = tn[i][i]
    for i in range(m - 1):
        e[p + i] = 0.5 * (tn[i][i + 1] + tn[i + 1][i])
    # 全局特征向量矩阵 Q <- Q diag(I, Qb, I)
    n = len(q)
    for row in range(n):
        new = [sum(q[row][p + k] * qb[k][j] for k in range(m))
               for j in range(m)]
        for j in range(m):
            q[row][p + j] = new[j]


def _jacobi_rotation_params(app, aqq, apq):
    """返回 (c, s, t)，使 J=[[c,s],[-s,c]] 满足 J^T A J 消去 apq。"""
    theta = (aqq - app) / (2.0 * apq)
    if theta >= 0.0:
        t = 1.0 / (theta + math.sqrt(1.0 + theta * theta))
    else:
        t = -1.0 / (-theta + math.sqrt(1.0 + theta * theta))
    c = 1.0 / math.sqrt(1.0 + t * t)
    return c, t * c, t


def qr_eigen(a, max_iter_per_value=100):
    """主求解器：返回实对称矩阵 a 的全部 (升序特征值, 正交归一特征向量)。"""
    _check_symmetric(a)
    n = len(a)
    if n == 0:
        return [], []
    d, e, q = _tridiagonalize(a)
    max_iterations = max_iter_per_value * max(1, n)
    iterations = 0
    r = n - 1
    while r > 0:
        # 收缩判据：次对角元相对相邻对角元可忽略则置零
        for i in range(r):
            if abs(e[i]) <= _EPS * (abs(d[i]) + abs(d[i + 1])):
                e[i] = 0.0
        while r > 0 and e[r - 1] == 0.0:
            r -= 1
        if r <= 0:
            break
        p = r - 1
        while p > 0 and e[p - 1] != 0.0:
            p -= 1
        if r - p == 1:
            # 2x2 块解析对角化（Jacobi 旋转），避免无谓迭代
            app, aqq, apq = d[p], d[r], e[p]
            if apq != 0.0:
                c, s, t = _jacobi_rotation_params(app, aqq, apq)
                d[p] = app - t * apq
                d[r] = aqq + t * apq
                for i in range(n):
                    qi = q[i][p]
                    qj = q[i][r]
                    q[i][p] = c * qi - s * qj
                    q[i][r] = s * qi + c * qj
            e[p] = 0.0
            continue
        mu = _wilkinson_shift(d[r - 1], d[r], e[r - 1])
        _qr_step(d, e, q, p, r, mu)
        iterations += 1
        if iterations > max_iterations:
            raise ArithmeticError(
                "QR iteration did not converge after %d steps" % iterations)
    return _sort_eigenpairs(d, q)


# ---------------------------------------------------------------------------
# 参照实现：经典循环 Jacobi（独立方法，用于对拍）
# ---------------------------------------------------------------------------

def jacobi_eigen(a, max_sweeps=100):
    """循环 Jacobi 扫描求全部特征值/特征向量（参照实现）。"""
    _check_symmetric(a)
    n = len(a)
    if n == 0:
        return [], []
    a = [row[:] for row in a]
    v = _identity(n)
    fro = math.sqrt(sum(x * x for row in a for x in row))
    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] ** 2
                            for i in range(n) for j in range(i + 1, n)))
        if off <= _EPS * max(1.0, fro):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                c, s, _ = _jacobi_rotation_params(a[p][p], a[q][q], apq)
                for i in range(n):
                    gp = a[i][p]
                    hq = a[i][q]
                    a[i][p] = c * gp - s * hq
                    a[i][q] = s * gp + c * hq
                for i in range(n):
                    gp = a[p][i]
                    hq = a[q][i]
                    a[p][i] = c * gp - s * hq
                    a[q][i] = s * gp + c * hq
                for i in range(n):
                    gp = v[i][p]
                    hq = v[i][q]
                    v[i][p] = c * gp - s * hq
                    v[i][q] = s * gp + c * hq
    else:
        raise ArithmeticError("Jacobi iteration did not converge")
    d = [a[i][i] for i in range(n)]
    return _sort_eigenpairs(d, v)
