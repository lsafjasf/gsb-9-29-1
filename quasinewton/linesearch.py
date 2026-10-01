"""线搜索：强 Wolfe（bracket + zoom），失败时逐级降级。

仅依赖标准库。所有求值次数都在返回结果中统计；每次降级/失败都会把
结构化原因写入 events（若提供）。
"""

import math


class LineSearchResult:
    def __init__(self):
        self.success = False
        self.alpha = 0.0
        self.f_new = None
        self.g_new = None
        self.nfev = 0
        self.ngev = 0
        self.method = None        # strong_wolfe / armijo / armijo_neggrad
        self.reason = None        # 失败或降级原因（成功时也可能记录降级原因）
        self.p_used = None        # 降级为 -grad 时实际使用的方向
        self.trials = []          # (method_stage, alpha, f_value)

    def as_tuple(self):
        return (self.success, self.alpha, self.f_new, self.g_new)


def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def _axpy(y, a, x):
    return [yi + a * xi for yi, xi in zip(y, x)]


def _norm(a):
    return math.sqrt(sum(vi * vi for vi in a))


def _cubic_min(a, fa, da, b, fb, db):
    """已知两点函数值与导数的三次插值极小点；失败返回 None。"""
    try:
        gap = b - a
        d1 = da + db - 3.0 * (fa - fb) / gap
        rad = d1 * d1 - da * db
        if rad < 0.0:
            return None
        d2 = math.copysign(math.sqrt(rad), gap)
        denom = db - da + 2.0 * d2
        if denom == 0.0:
            return None
        t = b - gap * (db + d2 - d1) / denom
        if not math.isfinite(t):
            return None
        return t
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _safeguard(trial, lo, hi, shrink=0.1):
    """把试探点夹逼到区间内部，防止端点粘连。"""
    left = lo + shrink * (hi - lo)
    right = hi - shrink * (hi - lo)
    if left > right:
        left, right = right, left
    if trial is None or not (left <= trial <= right):
        trial = 0.5 * (lo + hi)
    return trial


def _zoom(f, g, x, p, f0, d0, lo, hi,
          c1, c2, alpha_min, max_iter, stage_events):
    """Nocedal & Wright 的 zoom 过程。lo/hi 均为 (alpha, f, dphi|None)。"""
    nfev = ngev = 0
    for _ in range(max_iter):
        if abs(hi[0] - lo[0]) <= alpha_min:
            return None, nfev, ngev, "wolfe_zoom_interval_too_small"
        trial = None
        if lo[2] is not None and hi[2] is not None:
            trial = _cubic_min(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        trial = _safeguard(trial, lo[0], hi[0])
        xt = _axpy(x, trial, p)
        try:
            ft = f(xt)
        except Exception as exc:  # 目标函数异常按不可行点处理
            ft = math.inf
            stage_events.append(("objective_exception", str(exc)))
        nfev += 1
        armijo = ft <= f0 + c1 * trial * d0
        if (not math.isfinite(ft)) or (not armijo) or ft >= lo[1]:
            hi = (trial, ft, None)
            continue
        gt = g(xt)
        ngev += 1
        dt = _dot(gt, p)
        if abs(dt) <= -c2 * d0:  # d0 < 0，强 Wolfe 曲率条件
            res = trial, ft, gt
            return res, nfev, ngev, None
        if dt * (hi[0] - lo[0]) >= 0.0:
            hi = lo
        lo = (trial, ft, dt)
    return None, nfev, ngev, "wolfe_zoom_max_iter"


def _strong_wolfe(f, g, x, p, f0, d0, alpha_init, c1, c2,
                  alpha_min, max_bracket, max_zoom, events):
    nfev = ngev = 0
    a_prev, f_prev, d_prev = 0.0, f0, d0
    alpha = alpha_init
    for i in range(max_bracket):
        xt = _axpy(x, alpha, p)
        try:
            fa = f(xt)
        except Exception as exc:
            fa = math.inf
            events.append(("objective_exception", str(exc)))
        nfev += 1
        armijo = fa <= f0 + c1 * alpha * d0
        if (not math.isfinite(fa)) or (not armijo) or (i > 0 and fa >= f_prev):
            z, n1, n2, why = _zoom(
                f, g, x, p, f0, d0,
                (a_prev, f_prev, d_prev), (alpha, fa, None),
                c1, c2, alpha_min, max_zoom, events)
            nfev += n1
            ngev += n2
            if z is None:
                return None, nfev, ngev, why
            return (z[0], z[1], z[2]), nfev, ngev, None
        ga = g(xt)
        ngev += 1
        da = _dot(ga, p)
        if abs(da) <= -c2 * d0:
            return (alpha, fa, ga), nfev, ngev, None
        if da >= 0.0:
            z, n1, n2, why = _zoom(
                f, g, x, p, f0, d0,
                (alpha, fa, da), (a_prev, f_prev, d_prev),
                c1, c2, alpha_min, max_zoom, events)
            nfev += n1
            ngev += n2
            if z is None:
                return None, nfev, ngev, why
            return (z[0], z[1], z[2]), nfev, ngev, None
        a_prev, f_prev, d_prev = alpha, fa, da
        alpha *= 2.0
        if not math.isfinite(alpha):
            return None, nfev, ngev, "wolfe_expansion_overflow"
    return None, nfev, ngev, "wolfe_bracket_max_iter"


def _armijo(f, x, p, f0, d0, alpha_init, c1, rho, alpha_min, max_iter):
    """纯 Armijo 回溯（不需要梯度求值，除调用方已有的 g0 外）。"""
    nfev = 0
    alpha = alpha_init
    for _ in range(max_iter):
        xt = _axpy(x, alpha, p)
        try:
            fa = f(xt)
        except Exception:
            fa = math.inf
        nfev += 1
        if math.isfinite(fa) and fa <= f0 + c1 * alpha * d0:
            return alpha, fa, nfev, None
        alpha *= rho
        if alpha < alpha_min:
            return None, None, nfev, "armijo_step_below_alpha_min"
    return None, None, nfev, "armijo_max_iter"


def line_search(f, g, x, p, f0, g0, *,
                c1=1e-4, c2=0.9, alpha_init=1.0, rho=0.5,
                alpha_min=1e-16, max_bracket=20, max_zoom=25,
                max_armijo=60, allow_wolfe=True, events=None):
    """三级线搜索。

    1. 强 Wolfe（bracket/zoom，c1 下降条件 + c2 曲率条件）；
    2. 失败 -> 同方向 Armijo 回溯（只保下降条件）；
    3. 再失败 -> 沿 -g0 方向 Armijo（方向被怀疑时的救援）；
    4. 全部失败 -> success=False，reason 记录完整原因链。
    """
    if events is None:
        events = []
    res = LineSearchResult()
    d0 = _dot(g0, p)
    if not math.isfinite(d0) or d0 >= 0.0:
        res.reason = "not_descent_direction:dphi0=%r" % d0
        events.append(("line_search_abort", res.reason))
        return res

    if allow_wolfe:
        z, nf, ng, why = _strong_wolfe(
            f, g, x, p, f0, d0, alpha_init, c1, c2,
            alpha_min, max_bracket, max_zoom, events)
        res.nfev += nf
        res.ngev += ng
        if z is not None:
            res.success, res.alpha, res.f_new, res.g_new = True, z[0], z[1], z[2]
            res.method = "strong_wolfe"
            res.trials.append(("strong_wolfe", res.alpha, res.f_new))
            return res
        events.append(("wolfe_failed", why))
        res.reason = "strong_wolfe:" + why

    arm = _armijo(f, x, p, f0, d0, alpha_init, c1, rho,
                  alpha_min, max_armijo)
    res.nfev += arm[2]
    if arm[0] is not None:
        alpha, fa = arm[0], arm[1]
        ga = g(_axpy(x, alpha, p))
        res.ngev += 1
        res.success, res.alpha, res.f_new, res.g_new = True, alpha, fa, ga
        res.method = "armijo_fallback"
        if res.reason:
            events.append(("armijo_fallback", res.reason))
        res.trials.append(("armijo_fallback", alpha, fa))
        return res
    why2 = arm[3]
    events.append(("armijo_failed", why2))
    res.reason = (res.reason or "strong_wolfe_disabled") + "|armijo:" + why2

    gnorm = _norm(g0)
    if gnorm == 0.0:
        res.reason += "|neggrad:zero_gradient"
        events.append(("line_search_failed", res.reason))
        return res
    p2 = [-gi for gi in g0]
    d2 = -gnorm * gnorm
    pnorm = _norm(p)
    a0_2 = alpha_init * pnorm / gnorm  # 使物理步长与原方向首次试探一致
    arm2 = _armijo(f, x, p2, f0, d2, a0_2, c1, rho, alpha_min, max_armijo)
    res.nfev += arm2[2]
    if arm2[0] is not None:
        alpha, fa = arm2[0], arm2[1]
        ga = g(_axpy(x, alpha, p2))
        res.ngev += 1
        res.success, res.alpha, res.f_new, res.g_new = True, alpha, fa, ga
        res.method = "armijo_neggrad"
        res.p_used = p2
        events.append(("neggrad_rescue", res.reason))
        res.trials.append(("armijo_neggrad", alpha, fa))
        return res
    res.reason += "|neggrad:" + arm2[3]
    events.append(("line_search_failed", res.reason))
    return res
