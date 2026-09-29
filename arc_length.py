"""弧长参数化库（纯标准库）。

核心思路
--------
1. 用自适应 Simpson 积分对速度函数 |C'(t)| 做数值积分，建立
   参数 t -> 累计弧长 s 的映射表 (t_i, s_i)。
2. 反查 s -> t：在表中二分定位区间，再在区间内对积分方程
   ∫_{t_i}^{t} |C'(u)| du = s - s_i 做二分求解。

精度控制与误差上界
------------------
自适应 Simpson：设 S(a,b) 为整段 Simpson 估计，S2 = S(a,m)+S(m,b)
为对半细分后的估计，则局部误差估计为

    E ≈ |S2 - S| / 15          （Richardson 外推，Simpson 为 4 阶）

当 E <= tol * (b-a) 时接受该区间（容差按区间长度分配），因此
全区间积分误差上界为

    |I_num - I_exact| <= tol = abs_tol + rel_tol * L_est

其中 L_est 是弧长的粗估计。返回值使用 Richardson 修正
S2 + (S2 - S1)/15，实际误差通常远小于该上界（约 1/16 量级再低一阶）。

反查的往返一致性：param_at_length 在小区间内用同一积分器做二分，
故 |length_at(param_at_length(s)) - s| <= tol + 二分终止容差折算的弧长。
"""

from __future__ import annotations

import bisect
import math
from typing import List, Sequence, Tuple

Vec = Tuple[float, ...]


# ---------------------------------------------------------------------------
# 曲线定义
# ---------------------------------------------------------------------------

class BezierCurve:
    """任意次 Bezier 曲线（de Casteljau 求值，支持任意维数）。"""

    def __init__(self, points: Sequence[Sequence[float]]):
        if len(points) < 2:
            raise ValueError("Bezier 曲线至少需要 2 个控制点")
        self.points: List[Vec] = [tuple(float(c) for c in p) for p in points]
        self.dim = len(self.points[0])
        if any(len(p) != self.dim for p in self.points):
            raise ValueError("控制点维数不一致")
        # 导数曲线的控制点：n * (P_{i+1} - P_i)
        n = len(self.points) - 1
        self._deriv = None
        if n >= 1:
            dpts = [
                tuple(n * (self.points[i + 1][k] - self.points[i][k])
                      for k in range(self.dim))
                for i in range(n)
            ]
            self._deriv = dpts

    def eval(self, t: float) -> Vec:
        pts = [list(p) for p in self.points]
        for r in range(1, len(pts)):
            for i in range(len(pts) - r):
                a, b = pts[i], pts[i + 1]
                pts[i] = [(1.0 - t) * a[k] + t * b[k] for k in range(self.dim)]
        return tuple(pts[0])

    def speed(self, t: float) -> float:
        if self._deriv is None:
            return 0.0
        pts = [list(p) for p in self._deriv]
        for r in range(1, len(pts)):
            for i in range(len(pts) - r):
                a, b = pts[i], pts[i + 1]
                pts[i] = [(1.0 - t) * a[k] + t * b[k] for k in range(self.dim)]
        return math.sqrt(sum(c * c for c in pts[0]))


class PolylineCurve:
    """分段线性折线（用于超长/多段曲线测试）。"""

    def __init__(self, points: Sequence[Sequence[float]]):
        if len(points) < 2:
            raise ValueError("折线至少需要 2 个点")
        self.points: List[Vec] = [tuple(float(c) for c in p) for p in points]
        self._cum = [0.0]
        for i in range(len(self.points) - 1):
            d = math.dist(self.points[i], self.points[i + 1])
            self._cum.append(self._cum[-1] + d)
        self._total = self._cum[-1]

    def _locate(self, t: float) -> Tuple[int, float]:
        seg = min(int(t * (len(self.points) - 1)), len(self.points) - 2)
        u = t * (len(self.points) - 1) - seg
        return seg, u

    def eval(self, t: float) -> Vec:
        seg, u = self._locate(t)
        a, b = self.points[seg], self.points[seg + 1]
        return tuple((1.0 - u) * a[k] + u * b[k] for k in range(len(a)))

    def speed(self, t: float) -> float:
        seg, _ = self._locate(t)
        d = self._cum[seg + 1] - self._cum[seg]
        return d * (len(self.points) - 1)


# ---------------------------------------------------------------------------
# 自适应 Simpson 积分
# ---------------------------------------------------------------------------

class AdaptiveSimpson:
    """自适应 Simpson 积分器，误差上界 = 构造时给定的 tol。"""

    def __init__(self, f, tol: float, max_depth: int = 50):
        self.f = f
        self.tol = tol
        self.max_depth = max_depth
        self.eval_count = 0

    @staticmethod
    def _simpson(a: float, b: float, fa: float, fm: float, fb: float) -> float:
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    def integrate(self, a: float, b: float) -> float:
        """返回 ∫_a^b f，并保证 |误差| <= tol（在估计器可靠的前提下）。"""
        if not (b > a):
            return 0.0
        fa = self._eval(a)
        fb = self._eval(b)
        m = 0.5 * (a + b)
        fm = self._eval(m)
        whole = self._simpson(a, b, fa, fm, fb)
        value, _err = self._recurse(a, b, fa, fm, fb, whole, self.tol, 0)
        return value

    def integrate_with_intervals(self, a: float, b: float):
        """积分并返回 (总值, [(x0, x1, 该子区间积分值), ...])，用于建表。"""
        if not (b > a):
            return 0.0, []
        fa = self._eval(a)
        fb = self._eval(b)
        m = 0.5 * (a + b)
        fm = self._eval(m)
        whole = self._simpson(a, b, fa, fm, fb)
        self._intervals: List[Tuple[float, float, float]] = []
        value, _err = self._recurse(a, b, fa, fm, fb, whole, self.tol, 0)
        return value, self._intervals

    def _eval(self, x: float) -> float:
        self.eval_count += 1
        return self.f(x)

    def _recurse(self, a, b, fa, fm, fb, whole, tol, depth):
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = self._eval(lm)
        frm = self._eval(rm)
        left = self._simpson(a, m, fa, flm, fm)
        right = self._simpson(m, b, fm, frm, fb)
        delta = left + right - whole
        # 误差估计 |delta|/15 <= tol 时接受；Richardson 修正后返回
        if depth >= self.max_depth or abs(delta) <= 15.0 * tol:
            value = left + right + delta / 15.0
            if hasattr(self, "_intervals"):
                self._intervals.append((a, b, value))
            return value, abs(delta) / 15.0
        lv, le = self._recurse(a, m, fa, flm, fm, left, tol / 2.0, depth + 1)
        rv, re = self._recurse(m, b, fm, frm, fb, right, tol / 2.0, depth + 1)
        return lv + rv, le + re


# ---------------------------------------------------------------------------
# 弧长参数化
# ---------------------------------------------------------------------------

class ArcLengthParameterization:
    """曲线的弧长参数化。

    curve      : 具有 eval(t) 与 speed(t) 方法的对象，t ∈ [0, 1]
    rel_tol    : 相对容差（相对弧长粗估计）
    abs_tol    : 绝对容差下限（保护零长度曲线）
    """

    def __init__(self, curve, rel_tol: float = 1e-10, abs_tol: float = 1e-12):
        self.curve = curve
        # 1) 粗估计弧长，用于把相对容差折算成绝对容差
        coarse = AdaptiveSimpson(curve.speed, tol=1e-6)
        est = coarse.integrate(0.0, 1.0)
        self._tol = abs_tol + rel_tol * max(est, 1.0)
        # 2) 用正式容差建表：自适应细分的断点即映射表节点
        integ = AdaptiveSimpson(curve.speed, tol=self._tol)
        total, intervals = integ.integrate_with_intervals(0.0, 1.0)
        intervals.sort(key=lambda iv: iv[0])
        self._ts: List[float] = [0.0]
        self._ss: List[float] = [0.0]
        s = 0.0
        for x0, x1, v in intervals:
            s += max(v, 0.0)
            self._ts.append(x1)
            self._ss.append(s)
        self._total = s
        self._integ_tol = self._tol

    # -- 基本量 ------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """总弧长，误差 <= abs_tol + rel_tol * L。"""
        return self._total

    @property
    def tolerance(self) -> float:
        """积分容差（弧长误差上界）。"""
        return self._integ_tol

    # -- 正查：t -> s -------------------------------------------------------

    def length_at(self, t: float) -> float:
        """参数 t 处的累计弧长 s(t) = ∫_0^t |C'(u)| du。"""
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return self._total
        i = bisect.bisect_right(self._ts, t) - 1
        t0, s0 = self._ts[i], self._ss[i]
        if t <= t0:
            return s0
        local = AdaptiveSimpson(self.curve.speed, tol=self._integ_tol)
        return s0 + local.integrate(t0, t)

    # -- 反查：s -> t -------------------------------------------------------

    def param_at_length(self, s: float) -> float:
        """给定弧长 s 求参数 t，使 s(t) = s。"""
        if self._total == 0.0:
            if abs(s) <= self._integ_tol:
                return 0.0
            raise ValueError("零长度曲线只有弧长 0 合法")
        if s <= 0.0:
            return 0.0
        if s >= self._total:
            if s <= self._total + self._integ_tol:
                return 1.0
            raise ValueError(f"弧长 {s} 超出总长 {self._total}")
        # 表中二分定位：找到 s_i <= s < s_{i+1} 且区间非零长
        i = bisect.bisect_right(self._ss, s) - 1
        while i < len(self._ss) - 2 and self._ss[i + 1] <= s:
            i += 1
        t0, t1 = self._ts[i], self._ts[i + 1]
        s0 = self._ss[i]
        if self._ss[i + 1] - s0 <= 0.0:
            return t0  # 零长度退化区间（速度为 0 的区段）
        # 在 [t0, t1] 内二分求解 ∫_{t0}^{t} speed = s - s0
        target = s - s0
        lo, hi = t0, t1
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            local = AdaptiveSimpson(self.curve.speed, tol=self._integ_tol)
            val = local.integrate(t0, mid)
            if val < target:
                lo = mid
            else:
                hi = mid
            if hi - lo <= 1e-13:
                break
        return 0.5 * (lo + hi)

    def point_at_length(self, s: float) -> Vec:
        """给定弧长，返回曲线上的点。"""
        return self.curve.eval(self.param_at_length(s))

    # -- 等弧长采样 ----------------------------------------------------------

    def sample_even(self, n: int) -> List[Tuple[float, float, Vec]]:
        """返回 n+1 个等弧长采样点 [(s_k, t_k, point_k)]。"""
        if n < 1:
            raise ValueError("n 必须 >= 1")
        out = []
        for k in range(n + 1):
            s = self._total * k / n
            t = self.param_at_length(s)
            out.append((s, t, self.curve.eval(t)))
        return out
