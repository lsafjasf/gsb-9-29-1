"""共轭梯度法（CG/PCG）求解稀疏对称正定线性系统 Ax = b。仅使用 Python 标准库。

矩阵用 dict-of-dicts 稀疏存储。提供：
  - SparseMatrix            稀疏矩阵（含对称性检查）
  - IdentityPreconditioner  无预条件（普通 CG）
  - JacobiPreconditioner    对角（Jacobi）预条件
  - SSORPreconditioner      简单分解（SSOR）预条件
  - pcg()                   预条件共轭梯度主过程

前提检查（不满足即抛异常，绝不返回错误结果）：
  - 非方阵 / 维度不匹配        -> ValueError
  - 非对称                     -> NonSymmetricError
  - 非正定（含零矩阵、不定阵） -> NotPositiveDefiniteError
"""

import math

__all__ = [
    "CGError", "NonSymmetricError", "NotPositiveDefiniteError",
    "SparseMatrix", "IdentityPreconditioner", "JacobiPreconditioner",
    "SSORPreconditioner", "CGResult", "pcg",
]


class CGError(Exception):
    """CG 求解器所有前提/运行时错误的基类。"""


class NonSymmetricError(CGError):
    """系数矩阵不满足对称性。"""


class NotPositiveDefiniteError(CGError):
    """系数矩阵（或预条件子）非正定。"""


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(math.fsum(x * x for x in a))


class SparseMatrix:
    """dict-of-dicts 稀疏矩阵，只存非零元。"""

    def __init__(self, n, entries=None):
        if n <= 0:
            raise ValueError("矩阵阶数必须为正")
        self.n = n
        self._rows = [dict() for _ in range(n)]
        if entries:
            for i, j, v in entries:
                self.set(i, j, v)

    @classmethod
    def from_dense(cls, rows):
        n = len(rows)
        if n == 0:
            raise ValueError("矩阵不能为空")
        for row in rows:
            if len(row) != n:
                raise ValueError("系数矩阵必须是方阵")
        mat = cls(n)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if v != 0.0:
                    mat.set(i, j, v)
        return mat

    def set(self, i, j, v):
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise IndexError("矩阵下标越界")
        if v == 0.0:
            self._rows[i].pop(j, None)
        else:
            self._rows[i][j] = float(v)

    def get(self, i, j):
        return self._rows[i].get(j, 0.0)

    def to_dense(self):
        return [[self._rows[i].get(j, 0.0) for j in range(self.n)]
                for i in range(self.n)]

    def matvec(self, x):
        if len(x) != self.n:
            raise ValueError("矩阵与向量维度不匹配")
        out = [0.0] * self.n
        for i, row in enumerate(self._rows):
            out[i] = math.fsum(v * x[j] for j, v in row.items())
        return out

    def diagonal(self):
        return [self._rows[i].get(i, 0.0) for i in range(self.n)]

    def check_symmetric(self, tol=1e-10):
        """逐元素校验 A[i][j] == A[j][i]（相对容差），违反即抛 NonSymmetricError。"""
        for i in range(self.n):
            for j, vij in self._rows[i].items():
                vji = self._rows[j].get(i, 0.0)
                scale = max(1.0, abs(vij), abs(vji))
                if abs(vij - vji) > tol * scale:
                    raise NonSymmetricError(
                        "系数矩阵非对称: A[%d][%d]=%r 而 A[%d][%d]=%r"
                        % (i, j, vij, j, i, vji))


class IdentityPreconditioner:
    """无预条件，即普通 CG。"""

    name = "无预条件 (CG)"

    def apply(self, r):
        return list(r)


class JacobiPreconditioner:
    """对角预条件 M = diag(A)。

    适用场景：系数矩阵各行/列尺度差异巨大（对角元量级悬殊）时，
    M 与 A 的对角完全一致，可显著降低 M^{-1}A 的条件数。
    """

    name = "对角预条件 (Jacobi)"

    def __init__(self, A):
        diag = A.diagonal()
        for i, d in enumerate(diag):
            if d <= 0.0:
                # 正定矩阵的对角元必为正，反之则必不正定
                raise NotPositiveDefiniteError(
                    "A[%d][%d] = %r <= 0，矩阵非正定，不能使用 Jacobi 预条件"
                    % (i, i, d))
        self._inv = [1.0 / d for d in diag]

    def apply(self, r):
        return [ri * mi for ri, mi in zip(r, self._inv)]


class SSORPreconditioner:
    """SSOR 预条件（简单分解）：M = (D + wL) D^{-1} (D + wL)^T，0 < omega < 2。

    L 为 A 的严格下三角部分。对非对角占主导的病态矩阵通常比 Jacobi 更有效。
    """

    name = "SSOR 预条件"

    def __init__(self, A, omega=1.0):
        if not (0.0 < omega < 2.0):
            raise ValueError("SSOR 松弛因子必须满足 0 < omega < 2")
        n = A.n
        self._omega = omega
        self._diag = A.diagonal()
        for i, d in enumerate(self._diag):
            if d <= 0.0:
                raise NotPositiveDefiniteError(
                    "A[%d][%d] = %r <= 0，矩阵非正定，不能使用 SSOR 预条件"
                    % (i, i, d))
        self._lower = [[] for _ in range(n)]  # 行 i 的严格下三角 [(j, A[i][j])]
        self._upper = [[] for _ in range(n)]  # 行 i 的严格上三角 [(j, A[i][j])]
        for i in range(n):
            for j, v in A._rows[i].items():
                if j < i:
                    self._lower[i].append((j, v))
                elif j > i:
                    self._upper[i].append((j, v))

    def apply(self, r):
        n = len(self._diag)
        w = self._omega
        # 前代: (D + wL) y = r
        y = [0.0] * n
        for i in range(n):
            s = r[i] - w * math.fsum(v * y[j] for j, v in self._lower[i])
            y[i] = s / self._diag[i]
        # 回代: (D + wL)^T z = D y
        z = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = (self._diag[i] * y[i]
                 - w * math.fsum(v * z[j] for j, v in self._upper[i]))
            z[i] = s / self._diag[i]
        return z


class CGResult:
    """PCG 求解结果。"""

    def __init__(self, x, converged, iterations, residual_history, message):
        self.x = x
        self.converged = converged
        self.iterations = iterations
        self.residual_history = residual_history  # 每步 ||r||_2，含初始值
        self.message = message

    def __repr__(self):
        return ("CGResult(converged=%r, iterations=%d, final_residual=%.3e)"
                % (self.converged, self.iterations, self.residual_history[-1]))


def pcg(A, b, x0=None, precond=None, tol=1e-10, max_iter=None):
    """预条件共轭梯度法解 Ax = b。

    参数:
        A        SparseMatrix，必须对称正定（运行中强制检查）
        b        右端向量
        x0       初始猜测（默认零向量）
        precond  预条件子，默认 IdentityPreconditioner()
        tol      相对残差收敛阈值 ||r|| <= tol * ||b||
        max_iter 迭代上限，默认 20*n + 100

    返回 CGResult；迭代上限耗尽时 converged=False 并返回当前最优迭代点。
    前提不满足时抛出 CGError 子类，绝不返回错误结果。
    """
    n = A.n
    if len(b) != n:
        raise ValueError("右端向量维度与矩阵不匹配")
    A.check_symmetric()
    if max_iter is None:
        max_iter = 20 * n + 100
    M = precond if precond is not None else IdentityPreconditioner()

    x = list(x0) if x0 is not None else [0.0] * n
    Ax = A.matvec(x)
    r = [bi - ai for bi, ai in zip(b, Ax)]
    bnorm = max(_norm(b), 1e-300)
    history = [_norm(r)]
    if history[0] <= tol * bnorm:
        return CGResult(x, True, 0, history, "初始猜测已满足精度")

    z = M.apply(r)
    rz = _dot(r, z)
    if rz <= 0.0:
        raise NotPositiveDefiniteError(
            "r^T M^{-1} r = %r <= 0，预条件子非正定" % rz)
    p = z

    for k in range(1, max_iter + 1):
        Ap = A.matvec(p)
        pAp = _dot(p, Ap)
        if pAp <= 0.0:
            raise NotPositiveDefiniteError(
                "第 %d 次迭代: p^T A p = %r <= 0，系数矩阵非正定" % (k, pAp))
        alpha = rz / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        history.append(_norm(r))
        if history[-1] <= tol * bnorm:
            # 递推残差可能漂移（低估），用真实残差 b - A x 复核后才判收敛
            Ax = A.matvec(x)
            r = [bi - ai for bi, ai in zip(b, Ax)]
            history[-1] = _norm(r)
            if history[-1] <= tol * bnorm:
                return CGResult(x, True, k, history, "收敛")
            # 复核未通过：以真实残差重启搜索方向，继续迭代
            z = M.apply(r)
            rz = _dot(r, z)
            if rz <= 0.0:
                raise NotPositiveDefiniteError(
                    "r^T M^{-1} r = %r <= 0，预条件子非正定" % rz)
            p = z
            continue
        z = M.apply(r)
        rz_new = _dot(r, z)
        if rz_new <= 0.0:
            raise NotPositiveDefiniteError(
                "第 %d 次迭代: r^T M^{-1} r = %r <= 0，预条件子非正定"
                % (k, rz_new))
        beta = rz_new / rz
        for i in range(n):
            p[i] = z[i] + beta * p[i]
        rz = rz_new

    return CGResult(x, False, max_iter, history,
                    "达到迭代上限 %d 仍未收敛" % max_iter)
