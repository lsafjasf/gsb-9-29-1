"""L-BFGS 风格的稠密 BFGS：用低秩更新维护逆 Hessian 近似 H_k。

更新公式（H = H_k, s = x_{k+1}-x_k, y = g_{k+1}-g_k, rho = 1/(y^T s)）：

    H_{k+1} = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

每个 H 是 n x n 稠密阵，但更新用秩 2 修正完成，单次迭代 O(n^2)，
不显式求任何矩阵的逆。

对称正定（SPD）保持：
- 数学上，只要 H SPD 且曲率 y^T s > 0，上式给出的 H_{k+1} 仍 SPD
  （对任意 z != 0 展开 z^T H_{k+1} z 即可见非负项之和）。
- 数值处理：
  1. 曲率门限 y^T s <= eps * ||s|| ||y|| 时跳过本次更新，事件记
     "skip_curvature"，H 原样保留（SPD 不被破坏）。
  2. 更新后做对称化 (H + H^T)/2，消去舍入产生的非对称。
  3. 若方向 p = -H g 不再是严格下降方向（g^T p < 0 被破坏，含 NaN），
     丢弃当前近似，重置 H = I（Shanno 缩放后重置），事件记
     "reset_hessian"。
- 初始 H0 = I；第二次迭代起采用 Shanno/Barzilai-Borwein 缩放
  H0 <- (s^T y)/(y^T y) * I，只在该因子为有限正数时启用。
"""

import math

from .linesearch import line_search


class OptimizeResult:
    def __init__(self):
        self.success = False
        self.status = None
        self.message = ""
        self.x = None
        self.fun = None
        self.jac = None
        self.nit = 0
        self.nfev = 0
        self.ngev = 0
        self.events = []
        self.history = []  # (nit, f, ||g||inf, alpha, method)

    @property
    def grad_norm(self):
        if self.jac is None:
            return math.inf
        return max(abs(v) for v in self.jac)


def _mat_vec(H, v):
    return [sum(H[i][j] * v[j] for j in range(len(v))) for i in range(len(v))]


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _axpy(y, a, x):
    return [yi + a * xi for yi, xi in zip(y, x)]


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def _bfgs_inverse_update(H, s, y, eps):
    """原位秩 2 更新；返回 (ok, sy)。ok=False 表示曲率不足需跳过。"""
    n = len(s)
    sy = _dot(s, y)
    yy = _dot(y, y)
    ss = _dot(s, s)
    if sy <= eps * math.sqrt(ss * yy) or sy <= 0.0 or yy <= 0.0:
        return False, sy
    rho = 1.0 / sy
    # H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T
    # 先算 H y 与 H s 相关项，按矩阵元 O(n^2) 实现
    Hy = _mat_vec(H, y)
    # A = H - rho (Hy s^T + s (H y)^T) + rho*(rho*s^T Hy - 1)* s s^T 等
    # 直接按两遍外积实现更稳妥：
    # H_new[i][j] = H[i][j]
    #   - rho * (s_i * Hy_j + Hy_i * s_j)
    # 标准展开: (I - rho s y^T) H (I - rho y s^T)
    #   = H - rho s (y^T H) - rho (H y) s^T + rho^2 s (y^T H y) s^T
    # 再 + rho s s^T。y^T H = (H y)^T（更新前对称化过）。
    yHy = rho * _dot(y, Hy)  # rho * y^T H y
    for i in range(n):
        Hsi = rho * s[i]
        for j in range(n):
            val = H[i][j]
            val -= rho * s[i] * Hy[j]
            val -= rho * Hy[i] * s[j]
            val += rho * yHy * s[i] * s[j]
            val += Hsi * s[j]
            H[i][j] = val
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (H[i][j] + H[j][i])
            H[i][j] = H[j][i] = avg
    ok = all(math.isfinite(v) for row in H for v in row)
    return ok, sy


def _check_stop(g, f, f_prev, gtol, ftol):
    ginf = max(abs(v) for v in g)
    if ginf <= gtol:
        return "converged_grad"
    if f_prev is not None and abs(f_prev) > 0.0 \
            and abs(f_prev - f) <= ftol * abs(f_prev) and ginf <= 1e-6:
        return "converged_f"
    return None


def minimize_bfgs(fun, x0, jac=None, *,
                  gtol=1e-8, ftol=1e-12, maxiter=500,
                  c1=1e-4, c2=0.9, alpha_min=1e-16,
                  curvature_eps=1e-10,
                  shanno_scaling=True,
                  allow_wolfe=True,
                  events=None, record_history=True):
    """BFGS 极小化。fun(x)->f；jac(x)->梯度列表。"""
    if events is None:
        events = []
    n = len(x0)
    x = [float(v) for v in x0]
    fval = float(fun(x))
    grad = [float(v) for v in jac(x)]
    nfev = ngev = 1
    gnorm0 = max((abs(v) for v in grad), default=0.0)
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    res = OptimizeResult()
    res.events = events
    if not _finite(fval) or any(not math.isfinite(v) for v in grad):
        res.status, res.message = "bad_start", "起始点函数值或梯度非有限值"
        res.x, res.fun, res.jac = x, fval, grad
        res.nfev, res.ngev = nfev, ngev
        return res
    if gnorm0 <= gtol:
        res.success = True
        res.status, res.message = "converged_grad", "起始点即驻点（零迭代）"
        res.x, res.fun, res.jac = x, fval, grad
        res.nfev, res.ngev = nfev, ngev
        events.append(("optimal_start", "||g||=%r" % gnorm0))
        if record_history:
            res.history.append((0, fval, gnorm0, 0.0, "none"))
        return res

    prev_s = prev_y = None
    f_prev = None
    for k in range(1, maxiter + 1):
        if k == 2 and shanno_scaling and prev_s is not None:
            sy = _dot(prev_s, prev_y)
            yy = _dot(prev_y, prev_y)
            if yy > 0.0 and sy > 0.0:
                gamma = sy / yy
                if math.isfinite(gamma) and gamma > 0.0:
                    H = [[gamma if i == j else 0.0 for j in range(n)]
                         for i in range(n)]
                    events.append(("shanno_scale", "gamma=%.6e" % gamma))
        p = _mat_vec(H, [-v for v in grad])
        gp = _dot(grad, p)
        if not math.isfinite(gp) or gp >= 0.0:
            events.append(("reset_hessian", "g^T p=%r 非下降" % gp))
            H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            p = [-v for v in grad]
            gp = _dot(grad, p)
            if not math.isfinite(gp) or gp >= 0.0:
                res.status = "no_descent_direction"
                res.message = "重置后仍无下降方向"
                res.x, res.fun, res.jac = x, fval, grad
                res.nfev, res.ngev, res.nit = nfev, ngev, k - 1
                if record_history:
                    res.history.append((k - 1, fval, gnorm0, 0.0, "abort"))
                return res

        alpha_init = 1.0
        ls_events = []
        ls = line_search(fun, jac, x, p, fval, grad,
                         c1=c1, c2=c2, alpha_init=alpha_init,
                         alpha_min=alpha_min,
                         allow_wolfe=allow_wolfe, events=ls_events)
        nfev += ls.nfev
        ngev += ls.ngev
        events.extend(ls_events)
        if not ls.success:
            ginf = max(abs(v) for v in grad)
            if ginf <= max(gtol * 100.0, 1e-7):
                res.success = True
                res.status = "converged_grad"
                res.message = ("线搜索在浮点极限处失败，但 ||g||inf=%.3e "
                               "已接近容差，按收敛处理" % ginf)
                res.x, res.fun, res.jac = x, fval, grad
                res.nfev, res.ngev, res.nit = nfev, ngev, len(res.history)
                events.append(("graceful_stop", res.message))
                return res
            res.status = "line_search_failed"
            res.message = ls.reason
            res.x, res.fun, res.jac = x, fval, grad
            res.nfev, res.ngev, res.nit = nfev, ngev, k - 1
            if record_history:
                res.history.append((k - 1, fval,
                                    max(abs(v) for v in grad), 0.0, "failed"))
            return res

        p_actual = ls.p_used if ls.p_used is not None else p
        x_new = _axpy(x, ls.alpha, p_actual)
        f_new, g_new = ls.f_new, ls.g_new
        if not _finite(f_new) or any(not math.isfinite(v) for v in g_new):
            res.status = "nonfinite_step"
            res.message = "线搜索返回非有限函数值/梯度"
            res.x, res.fun, res.jac = x, fval, grad
            res.nfev, res.ngev, res.nit = nfev, ngev, k
            return res

        s = [ls.alpha * pi for pi in p_actual]
        y = [gn - go for gn, go in zip(g_new, grad)]
        ok, sy = _bfgs_inverse_update(H, s, y, curvature_eps)
        if not ok:
            if sy <= curvature_eps * math.sqrt(_dot(s, s) * _dot(y, y)):
                events.append(("skip_curvature", "y^T s=%r" % sy))
            else:
                events.append(("reset_hessian", "更新产生非有限值"))
                H = [[1.0 if i == j else 0.0 for j in range(n)]
                     for i in range(n)]
        prev_s, prev_y = s, y

        x, f_prev, fval, grad = x_new, fval, f_new, g_new
        if record_history:
            res.history.append((k, fval,
                                max(abs(v) for v in grad),
                                ls.alpha, ls.method))
        stop = _check_stop(grad, fval, f_prev, gtol, ftol)
        if stop is not None:
            res.success = True
            res.status = stop
            res.message = "梯度/函数值满足停机条件"
            break
    else:
        res.status = "max_iter"
        res.message = "达到最大迭代次数"

    res.x, res.fun, res.jac = x, fval, grad
    res.nfev, res.ngev, res.nit = nfev, ngev, len(res.history)
    return res


def minimize_gd(fun, x0, jac=None, *,
                gtol=1e-8, ftol=1e-12, maxiter=5000,
                c1=1e-4, c2=0.9, alpha_min=1e-16,
                allow_wolfe=True, events=None, record_history=True):
    """梯度下降对拍：方向 -g，复用同一套线搜索（强 Wolfe + 降级）。"""
    if events is None:
        events = []
    n = len(x0)
    x = [float(v) for v in x0]
    fval = float(fun(x))
    grad = [float(v) for v in jac(x)]
    nfev = ngev = 1
    gnorm0 = max((abs(v) for v in grad), default=0.0)
    res = OptimizeResult()
    res.events = events
    if gnorm0 <= gtol:
        res.success = True
        res.status, res.message = "converged_grad", "起始点即驻点（零迭代）"
        res.x, res.fun, res.jac = x, fval, grad
        res.nfev, res.ngev = nfev, ngev
        events.append(("optimal_start", "||g||=%r" % gnorm0))
        if record_history:
            res.history.append((0, fval, gnorm0, 0.0, "none"))
        return res

    f_prev = None
    alpha_init = 1.0
    for k in range(1, maxiter + 1):
        p = [-v for v in grad]
        ls_events = []
        ls = line_search(fun, jac, x, p, fval, grad,
                         c1=c1, c2=c2, alpha_init=min(1.0, alpha_init),
                         alpha_min=alpha_min,
                         allow_wolfe=allow_wolfe, events=ls_events)
        nfev += ls.nfev
        ngev += ls.ngev
        events.extend(ls_events)
        if not ls.success:
            ginf = max(abs(v) for v in grad)
            if ginf <= max(gtol * 100.0, 1e-7):
                res.success = True
                res.status = "converged_grad"
                res.message = ("线搜索在浮点极限处失败，但 ||g||inf=%.3e "
                               "已接近容差，按收敛处理" % ginf)
                res.x, res.fun, res.jac = x, fval, grad
                res.nfev, res.ngev, res.nit = nfev, ngev, len(res.history)
                events.append(("graceful_stop", res.message))
                return res
            res.status = "line_search_failed"
            res.message = ls.reason
            break
        x_new = _axpy(x, ls.alpha, p)
        f_new, g_new = ls.f_new, ls.g_new
        x, f_prev, fval, grad = x_new, fval, f_new, g_new
        alpha_init = ls.alpha * 2.0  # 下次试探步放宽
        if record_history:
            res.history.append((k, fval,
                                max(abs(v) for v in grad),
                                ls.alpha, ls.method))
        stop = _check_stop(grad, fval, f_prev, gtol, ftol)
        if stop is not None:
            res.success = True
            res.status = stop
            res.message = "梯度/函数值满足停机条件"
            break
    else:
        res.status = "max_iter"
        res.message = "达到最大迭代次数"

    res.x, res.fun, res.jac = x, fval, grad
    res.nfev, res.ngev, res.nit = nfev, ngev, len(res.history)
    return res
