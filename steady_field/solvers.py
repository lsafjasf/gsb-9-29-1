"""稳态场迭代求解器：二维泊松方程 -Δu = f（单位正方形，Dirichlet 边界）。

五点差分离散，纯标准库实现：
  - jacobi        基本迭代
  - gauss_seidel  基本迭代
  - sor           超松弛加速（含模型问题最优松弛因子）
  - multigrid     几何多重网格 V 循环加速

收敛判据：相对残差 ||r||_2 / ||b||_2 <= tol（b = 0 时退化为相对初始残差），
与迭代次数无关；每个求解器都返回逐轮残差历史，并检测发散。
"""

import math
import time
from dataclasses import dataclass

__all__ = [
    "SolveResult",
    "apply_boundary",
    "residual_norm",
    "optimal_omega",
    "jacobi",
    "gauss_seidel",
    "sor",
    "multigrid",
]

# 残差超过初始残差的该倍数即判定发散
DIVERGE_FACTOR = 1e6


@dataclass
class SolveResult:
    method: str          # 求解器名称
    converged: bool      # 是否收敛
    iterations: int      # 实际迭代轮数（多重网格为 V 循环数）
    residuals: list      # 每轮相对残差，residuals[0] 为初始残差
    reason: str          # "converged" / "max_iter" / "diverged"
    wall_time: float     # 秒
    u: list              # 解网格，(n+2)x(n+2)，含边界


def _zeros(n):
    return [[0.0] * (n + 2) for _ in range(n + 2)]


def apply_boundary(u, bc):
    """用边界函数 bc(x, y) 填充网格边界。"""
    n = len(u) - 2
    h = 1.0 / (n + 1)
    for i in range(n + 2):
        x = i * h
        u[i][0] = bc(x, 0.0)
        u[i][n + 1] = bc(x, 1.0)
        u[0][i] = bc(0.0, x)
        u[n + 1][i] = bc(1.0, x)
    return u


def residual_norm(u, f, n):
    """内部点上残差 r = f - A u 的 L2 范数（A 为五点离散 -Δ）。"""
    h2 = 1.0 / ((n + 1) * (n + 1))
    total = 0.0
    for i in range(1, n + 1):
        ui = u[i]
        up = u[i - 1]
        dn = u[i + 1]
        fi = f[i]
        for j in range(1, n + 1):
            r = fi[j] - (4.0 * ui[j] - ui[j - 1] - ui[j + 1] - up[j] - dn[j]) / h2
            total += r * r
    return math.sqrt(total)


def _residual_grid(u, f, n):
    h2 = 1.0 / ((n + 1) * (n + 1))
    r = _zeros(n)
    for i in range(1, n + 1):
        ui = u[i]
        up = u[i - 1]
        dn = u[i + 1]
        fi = f[i]
        ri = r[i]
        for j in range(1, n + 1):
            ri[j] = fi[j] - (4.0 * ui[j] - ui[j - 1] - ui[j + 1] - up[j] - dn[j]) / h2
    return r


def optimal_omega(n):
    """模型问题（五点拉普拉斯 + 字典序 GS）的最优 SOR 松弛因子。

    依据：GS 谱半径 ρ_GS = cos²(πh)，Young 理论给出
        ω_opt = 2 / (1 + sqrt(1 - ρ_GS)) = 2 / (1 + sin(πh))，
    其中 h = 1/(n+1)。ω >= 2 时迭代矩阵谱半径 >= 1，必然不收敛。
    """
    h = 1.0 / (n + 1)
    return 2.0 / (1.0 + math.sin(math.pi * h))


def _iterate(method, n, f, bc, tol, max_iter, sweep):
    """通用定常迭代驱动：每轮调用一次 sweep(u)，随后计算相对残差。"""
    t0 = time.perf_counter()
    u = apply_boundary(_zeros(n), bc)

    b_norm = 0.0
    for i in range(1, n + 1):
        fi = f[i]
        for j in range(1, n + 1):
            b_norm += fi[j] * fi[j]
    b_norm = math.sqrt(b_norm)

    r0 = residual_norm(u, f, n)
    ref = b_norm if b_norm > 0.0 else r0
    residuals = [r0 / ref if ref > 0.0 else 0.0]

    converged = residuals[0] <= tol
    reason = "converged" if converged else "max_iter"
    it = 0
    while not converged and it < max_iter:
        sweep(u)
        it += 1
        rel = residual_norm(u, f, n) / ref
        residuals.append(rel)
        if not math.isfinite(rel) or rel > DIVERGE_FACTOR * residuals[0]:
            reason = "diverged"
            break
        if rel <= tol:
            converged = True
            reason = "converged"

    return SolveResult(method, converged, it, residuals, reason,
                       time.perf_counter() - t0, u)


def jacobi(n, f, bc, tol=1e-6, max_iter=100000):
    """Jacobi 迭代：u <- (Σ四邻 + h²f) / 4。"""
    h2 = 1.0 / ((n + 1) * (n + 1))

    def sweep(u):
        old = [row[:] for row in u]
        for i in range(1, n + 1):
            oi = old[i]
            up = old[i - 1]
            dn = old[i + 1]
            fi = f[i]
            ui = u[i]
            for j in range(1, n + 1):
                ui[j] = 0.25 * (oi[j - 1] + oi[j + 1] + up[j] + dn[j] + h2 * fi[j])

    return _iterate("jacobi", n, f, bc, tol, max_iter, sweep)


def gauss_seidel(n, f, bc, tol=1e-6, max_iter=100000):
    """Gauss-Seidel 迭代（字典序，就地更新）。"""
    h2 = 1.0 / ((n + 1) * (n + 1))

    def sweep(u):
        for i in range(1, n + 1):
            ui = u[i]
            up = u[i - 1]
            dn = u[i + 1]
            fi = f[i]
            for j in range(1, n + 1):
                ui[j] = 0.25 * (ui[j - 1] + ui[j + 1] + up[j] + dn[j] + h2 * fi[j])

    return _iterate("gauss_seidel", n, f, bc, tol, max_iter, sweep)


def sor(n, f, bc, omega=None, tol=1e-6, max_iter=100000):
    """逐次超松弛（SOR）。omega 缺省时取模型问题最优值 optimal_omega(n)。

    收敛域 0 < ω < 2；ω >= 2 时谱半径 >= 1，会被发散检测截获。
    """
    if omega is None:
        omega = optimal_omega(n)
    h2 = 1.0 / ((n + 1) * (n + 1))
    w = omega
    w1 = 1.0 - omega

    def sweep(u):
        for i in range(1, n + 1):
            ui = u[i]
            up = u[i - 1]
            dn = u[i + 1]
            fi = f[i]
            for j in range(1, n + 1):
                gs = 0.25 * (ui[j - 1] + ui[j + 1] + up[j] + dn[j] + h2 * fi[j])
                ui[j] = w1 * ui[j] + w * gs

    return _iterate("sor(w=%.4f)" % w, n, f, bc, tol, max_iter, sweep)


# ---------------------------------------------------------------- 多重网格

def _weighted_jacobi(u, f, n, h2, w=0.8):
    """加权 Jacobi 光滑子。ω=4/5 对五点拉普拉斯的高频分量阻尼最优。"""
    old = [row[:] for row in u]
    w1 = 1.0 - w
    for i in range(1, n + 1):
        oi = old[i]
        up = old[i - 1]
        dn = old[i + 1]
        fi = f[i]
        ui = u[i]
        for j in range(1, n + 1):
            ui[j] = w1 * oi[j] + w * 0.25 * (oi[j - 1] + oi[j + 1] + up[j] + dn[j]
                                             + h2 * fi[j])


def _gs_sweep(u, f, n, h2):
    for i in range(1, n + 1):
        ui = u[i]
        up = u[i - 1]
        dn = u[i + 1]
        fi = f[i]
        for j in range(1, n + 1):
            ui[j] = 0.25 * (ui[j - 1] + ui[j + 1] + up[j] + dn[j] + h2 * fi[j])


def _restrict_full_weighting(r, n_f):
    """全加权限制：细网格残差 -> 粗网格右端（n_f = 2*n_c + 1）。"""
    n_c = (n_f - 1) // 2
    rc = _zeros(n_c)
    for ic in range(1, n_c + 1):
        i = 2 * ic
        rup = r[i - 1]
        rmid = r[i]
        rdn = r[i + 1]
        row = rc[ic]
        for jc in range(1, n_c + 1):
            j = 2 * jc
            row[jc] = (4.0 * rmid[j]
                       + 2.0 * (rmid[j - 1] + rmid[j + 1] + rup[j] + rdn[j])
                       + rup[j - 1] + rup[j + 1] + rdn[j - 1] + rdn[j + 1]) / 16.0
    return rc


def _prolong_add(ec, n_c, uf, n_f):
    """双线性延拓并把粗网格修正累加到细网格解上。"""
    for i in range(1, n_f + 1):
        ic, di = divmod(i, 2)
        row_f = uf[i]
        if di == 0:
            src = ec[ic]
            for j in range(1, n_f + 1):
                jc, dj = divmod(j, 2)
                row_f[j] += src[jc] if dj == 0 else 0.5 * (src[jc] + src[jc + 1])
        else:
            s0 = ec[ic]
            s1 = ec[ic + 1]
            for j in range(1, n_f + 1):
                jc, dj = divmod(j, 2)
                if dj == 0:
                    row_f[j] += 0.5 * (s0[jc] + s1[jc])
                else:
                    row_f[j] += 0.25 * (s0[jc] + s0[jc + 1] + s1[jc] + s1[jc + 1])


def _vcycle(u, f, n, nu1, nu2):
    h2 = 1.0 / ((n + 1) * (n + 1))
    for _ in range(nu1):
        _weighted_jacobi(u, f, n, h2)
    if n <= 3:
        # 最粗层直接（近似）精确求解
        for _ in range(32):
            _gs_sweep(u, f, n, h2)
        return
    r = _residual_grid(u, f, n)
    rc = _restrict_full_weighting(r, n)
    n_c = (n - 1) // 2
    ec = _zeros(n_c)
    _vcycle(ec, rc, n_c, nu1, nu2)
    _prolong_add(ec, n_c, u, n)
    for _ in range(nu2):
        _weighted_jacobi(u, f, n, h2)


def multigrid(n, f, bc, tol=1e-6, max_iter=100, nu1=2, nu2=2):
    """几何多重网格 V 循环。要求 n = 2^k - 1 以便逐层二分粗化。

    加速依据：光滑子只能快速消除高频误差，低频误差被限制到粗网格后
    相对网格步长变为高频而被消除；每层工作量 O(n²)，总复杂度 O(n²)，
    收敛因子与网格尺寸无关（迭代轮数不随加密增长）。
    """
    if n < 1 or (n + 1) & n != 0:
        raise ValueError("multigrid 要求 n = 2^k - 1， got n=%d" % n)

    t0 = time.perf_counter()
    u = apply_boundary(_zeros(n), bc)

    b_norm = 0.0
    for i in range(1, n + 1):
        fi = f[i]
        for j in range(1, n + 1):
            b_norm += fi[j] * fi[j]
    b_norm = math.sqrt(b_norm)

    r0 = residual_norm(u, f, n)
    ref = b_norm if b_norm > 0.0 else r0
    residuals = [r0 / ref if ref > 0.0 else 0.0]

    converged = residuals[0] <= tol
    reason = "converged" if converged else "max_iter"
    it = 0
    while not converged and it < max_iter:
        _vcycle(u, f, n, nu1, nu2)
        it += 1
        rel = residual_norm(u, f, n) / ref
        residuals.append(rel)
        if not math.isfinite(rel) or rel > DIVERGE_FACTOR * residuals[0]:
            reason = "diverged"
            break
        if rel <= tol:
            converged = True
            reason = "converged"

    return SolveResult("multigrid", converged, it, residuals, reason,
                       time.perf_counter() - t0, u)
