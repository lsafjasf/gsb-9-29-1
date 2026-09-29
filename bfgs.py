"""拟牛顿 BFGS 无约束优化库（仅使用 Python 标准库）。

核心思想：不显式求逆，而是用低秩（rank-2）更新直接维护 Hessian 的
*近似逆矩阵* H_k，每次更新代价 O(n^2)。

逆 BFGS 更新公式（记 s = x_{k+1} - x_k, y = g_{k+1} - g_k, rho = 1/(y^T s)）：

    H_{k+1} = (I - rho*s*y^T) H_k (I - rho*y*s^T) + rho*s*s^T

对称正定性（SPD）保持：
  * 若 H_k 对称正定且曲率条件 y^T s > 0 成立，则 H_{k+1} 必对称正定
    （标准结论，见 Nocedal & Wright, Thm 6.1）。
  * 实现上先做数值化曲率判定 y^T s > eps*|s|*|y|；不满足时采用
    对偶 Powell 阻尼：把 s 替换为 s~ = theta*s + (1-theta)*H*y，
    使 s~^T y >= 0.2 * y^T H y > 0，再代入同一公式，从而严格保持 SPD。
  * 阻尼也无法挽救的退化情形（如 y == 0）跳过更新并记录日志。

线搜索：先尝试满足强 Wolfe 条件的步长（含 zoom），失败则降级为
Armijo 回溯；再失败则重置 H = I 用最速下降方向重试一次；仍失败则
终止并报告原因。所有降级都会写入 result.fallback_log。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

Vector = List[float]
Matrix = List[List[float]]


# ---------------------------------------------------------------- 基础工具

def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Vector) -> float:
    return math.sqrt(_dot(a, a))


def _norm_inf(a: Vector) -> float:
    return max(abs(x) for x in a)


def _eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matvec(H: Matrix, v: Vector) -> Vector:
    return [_dot(row, v) for row in H]


def _add_scaled(x: Vector, d: Vector, a: float) -> Vector:
    return [xi + a * di for xi, di in zip(x, d)]


# ---------------------------------------------------------------- 线搜索

@dataclass
class LineSearchResult:
    alpha: Optional[float]
    phi: Optional[float]
    n_fev: int
    n_gev: int
    status: str            # "wolfe" | "armijo_fallback" | "failed"
    reason: str = ""


def _zoom(phi, dphi, alo, ahi, phi_lo, phi0, der0, c1, c2, max_zoom):
    """强 Wolfe 的 zoom 阶段（Nocedal & Wright, Alg 3.6）。"""
    for _ in range(max_zoom):
        aj = 0.5 * (alo + ahi)
        phi_j = phi(aj)
        if phi_j > phi0 + c1 * aj * der0 or phi_j >= phi_lo:
            ahi = aj
        else:
            dphi_j = dphi(aj)
            if abs(dphi_j) <= -c2 * der0:
                return aj, phi_j, "wolfe", ""
            if dphi_j * (ahi - alo) >= 0.0:
                ahi = alo
            alo, phi_lo = aj, phi_j
        if abs(ahi - alo) <= 1e-14 * max(1.0, abs(alo), abs(ahi)):
            return None, None, "failed", "zoom_interval_collapsed"
    return None, None, "failed", "zoom_max_iter"


def _strong_wolfe(f, grad, x, fx, gx, d, c1, c2,
                  alpha_max=50.0, max_iter=25, max_zoom=25):
    """强 Wolfe 线搜索：Armijo 下降条件 + 曲率条件 |phi'(a)| <= c2*|phi'(0)|。"""
    n = len(x)
    cnt = {"fev": 0, "gev": 0}

    def phi(a):
        cnt["fev"] += 1
        return f(_add_scaled(x, d, a))

    def dphi(a):
        cnt["gev"] += 1
        return _dot(grad(_add_scaled(x, d, a)), d)

    der0 = _dot(gx, d)
    a_prev, phi_prev = 0.0, fx
    a = 1.0
    for it in range(max_iter):
        phi_a = phi(a)
        if phi_a > fx + c1 * a * der0 or (it > 0 and phi_a >= phi_prev):
            aj, phi_j, st, rs = _zoom(phi, dphi, a_prev, a, phi_prev,
                                      fx, der0, c1, c2, max_zoom)
            return aj, phi_j, cnt, st, rs
        dphi_a = dphi(a)
        if abs(dphi_a) <= -c2 * der0:
            return a, phi_a, cnt, "wolfe", ""
        if dphi_a >= 0.0:
            aj, phi_j, st, rs = _zoom(phi, dphi, a, a_prev, phi_a,
                                      fx, der0, c1, c2, max_zoom)
            return aj, phi_j, cnt, st, rs
        a_prev, phi_prev = a, phi_a
        a = min(a * 2.0, alpha_max)
    return None, None, cnt, "failed", "wolfe_max_iter"


def _armijo(f, x, fx, gx, d, c1=1e-4, shrink=0.5,
            alpha0=1.0, min_alpha=1e-16, max_iter=60):
    """Armijo 回溯线搜索：phi(a) <= phi(0) + c1*a*phi'(0)（充分下降）。"""
    der0 = _dot(gx, d)
    a = alpha0
    fev = 0
    for _ in range(max_iter):
        if a < min_alpha:
            break
        fev += 1
        phi_a = f(_add_scaled(x, d, a))
        if phi_a <= fx + c1 * a * der0:
            return a, phi_a, fev, "armijo_fallback", ""
        a *= shrink
    return None, None, fev, "failed", "armijo_min_alpha_underflow"


def _line_search(f, grad, x, fx, gx, d, c1, c2, log, tag):
    """强 Wolfe -> Armijo 回溯 的两级降级线搜索。"""
    a, phi_a, cnt, st, rs = _strong_wolfe(f, grad, x, fx, gx, d, c1, c2)
    res = LineSearchResult(a, phi_a, cnt["fev"], cnt["gev"], st, rs)
    if st != "failed":
        if st != "wolfe":
            log.append(f"iter {tag}: wolfe degraded ({st}: {rs})")
        return res
    log.append(f"iter {tag}: strong Wolfe failed ({rs}); fallback to Armijo")
    a2, phi2, fev2, st2, rs2 = _armijo(f, x, fx, gx, d, c1=c1)
    res.n_fev += fev2
    res.alpha, res.phi, res.status, res.reason = a2, phi2, st2, rs2
    if st2 == "failed":
        log.append(f"iter {tag}: Armijo fallback failed ({rs2})")
    return res


def _bfgs_update(H: Matrix, s: Vector, y: Vector,
                 damping: bool, log: List[str], tag) -> None:
    """原地执行逆 BFGS 低秩更新，保持 H 对称正定。

    曲率条件 y^T s > 0 满足时直接更新；不满足时用对偶 Powell 阻尼
    （把 s 替换为 theta*s + (1-theta)*H*y，使 s~^T y >= 0.2*y^T*H*y）；
    完全退化（y == 0）时跳过更新。
    """
    n = len(s)
    sy = _dot(s, y)
    Hy = _matvec(H, y)
    yHy = _dot(y, Hy)
    if sy > 1e-10 * _norm(s) * _norm(y):
        s_upd = s
    elif damping and yHy > 0.0:
        theta = 0.8 * yHy / (yHy - sy)
        s_upd = [theta * si + (1.0 - theta) * hyi for si, hyi in zip(s, Hy)]
        sy = _dot(s_upd, y)
        log.append(f"iter {tag}: curvature y^T s too small; "
                   f"damped update (theta = {theta:.4f})")
    else:
        log.append(f"iter {tag}: degenerate pair (y^T s = {sy:.3e}, "
                   f"y^T H y = {yHy:.3e}); skip update")
        return
    rho = 1.0 / sy
    Hy = _matvec(H, y)
    yHy = _dot(y, Hy)
    coef = rho * (1.0 + rho * yHy)
    # H += coef*s*s^T - rho*(s*(Hy)^T + (Hy)*s^T)
    for i in range(n):
        si = s_upd[i]
        hyi = Hy[i]
        row = H[i]
        for j in range(n):
            row[j] += coef * si * s_upd[j] - rho * (si * Hy[j] + hyi * s_upd[j])


# ---------------------------------------------------------------- 结果类型

@dataclass
class OptimizeResult:
    x: Vector
    fun: float
    grad: Vector
    n_iter: int
    n_fev: int
    n_gev: int
    status: str                     # converged | stalled | max_iter | line_search_failed
    message: str
    hess_inv: Optional[Matrix] = None
    fallback_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- BFGS 主流程

def minimize_bfgs(f: Callable[[Vector], float],
                  grad: Callable[[Vector], Vector],
                  x0: Vector,
                  tol: float = 1e-8,
                  max_iter: int = 1000,
                  c1: float = 1e-4,
                  c2: float = 0.9,
                  damping: bool = True) -> OptimizeResult:
    """BFGS 拟牛顿法，维护近似逆 Hessian，O(n^2)/次低秩更新。"""
    n = len(x0)
    log: List[str] = []
    H = _eye(n)
    x = list(x0)
    fx = f(x)
    g = list(grad(x))
    n_fev, n_gev = 1, 1

    if _norm_inf(g) <= tol:
        return OptimizeResult(x, fx, g, 0, n_fev, n_gev, "converged",
                              "initial point already satisfies |g|_inf <= tol",
                              H, log)

    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        # 搜索方向 p = -H g；若非下降方向则重置 H = I
        Hg = _matvec(H, g)
        p = [-v for v in Hg]
        gp = _dot(g, p)
        if gp >= -1e-12 * _norm(g) * max(_norm(p), 1e-300):
            log.append(f"iter {k}: non-descent direction (g^T p = {gp:.3e}); "
                       f"reset H = I, use steepest descent")
            H = _eye(n)
            p = [-v for v in g]

        ls = _line_search(f, grad, x, fx, g, p, c1, c2, log, k)
        n_fev += ls.n_fev
        n_gev += ls.n_gev
        if ls.status == "failed":
            # 最后一级降级：重置 H，用最速下降方向再试一次
            log.append(f"iter {k}: line search failed; retry with H = I")
            H = _eye(n)
            p = [-v for v in g]
            ls = _line_search(f, grad, x, fx, g, p, c1, c2, log, f"{k}r")
            n_fev += ls.n_fev
            n_gev += ls.n_gev
            if ls.status == "failed":
                return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev,
                                      "line_search_failed",
                                      "line search failed even with steepest "
                                      "descent direction; see fallback_log",
                                      H, log)

        alpha = ls.alpha
        s = [alpha * pi for pi in p]
        x_new = _add_scaled(x, p, alpha)
        f_new = ls.phi
        g_new = list(grad(x_new))
        n_gev += 1
        y = [gn - go for gn, go in zip(g_new, g)]

        # ---- 逆 BFGS 低秩更新（含 SPD 保持处理）----
        _bfgs_update(H, s, y, damping, log, k)

        step_norm = _norm(s)
        x, fx, g = x_new, f_new, g_new

        if _norm_inf(g) <= tol:
            return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev, "converged",
                                  f"|g|_inf <= {tol:g}", H, log)
        if step_norm <= 1e-14 * (1.0 + _norm(x)):
            return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev, "stalled",
                                  "step norm below machine resolution", H, log)

    return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev, "max_iter",
                          f"reached max_iter = {max_iter}", H, log)


# ---------------------------------------------------------------- 梯度下降基线

def minimize_gd(f: Callable[[Vector], float],
                grad: Callable[[Vector], Vector],
                x0: Vector,
                tol: float = 1e-8,
                max_iter: int = 50000,
                c1: float = 1e-4) -> OptimizeResult:
    """最速下降 + Armijo 回溯（与 BFGS 共用同一下降条件，便于公平对拍）。"""
    log: List[str] = []
    x = list(x0)
    fx = f(x)
    g = list(grad(x))
    n_fev, n_gev = 1, 1

    if _norm_inf(g) <= tol:
        return OptimizeResult(x, fx, g, 0, n_fev, n_gev, "converged",
                              "initial point already satisfies |g|_inf <= tol",
                              None, log)

    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        p = [-v for v in g]
        a, phi_a, fev, st, rs = _armijo(f, x, fx, g, p, c1=c1)
        n_fev += fev
        if st == "failed":
            log.append(f"iter {k}: Armijo failed ({rs})")
            return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev,
                                  "line_search_failed",
                                  "Armijo line search failed; see fallback_log",
                                  None, log)
        x = _add_scaled(x, p, a)
        fx = phi_a
        g = list(grad(x))
        n_gev += 1
        if _norm_inf(g) <= tol:
            return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev, "converged",
                                  f"|g|_inf <= {tol:g}", None, log)
    return OptimizeResult(x, fx, g, n_iter, n_fev, n_gev, "max_iter",
                          f"reached max_iter = {max_iter}", None, log)
