"""弧长参数化库（仅标准库）。

对任意参数曲线 curve(t) -> (x, y[, z, ...]) 建立 参数t <-> 弧长s 的双向映射。

精度控制方式
------------
采用自适应 Simpson 积分对速度 |curve'(t)| 积分：
  - 对每个区间比较整段 Simpson 值 S 与两半段之和 S1+S2，
    误差估计 err = |S1 + S2 - S| / 15（Simpson 公式的标准后验估计）。
  - 若 err 超过该区间分摊的容差则递归二分，容差对半分配。
  - 接受时返回 Richardson 外推值 S1 + S2 + (S1 + S2 - S)/15。
误差上界：设总容差 eps_abs = tol * max(L0, L_MIN)（L0 为弧长粗估计），
则整段积分总误差 <= ~eps_abs（各叶子区间误差之和，每片 <= 其分摊容差）。
反查（弧长 -> 参数）用二分定位 + safeguarded Newton 细化，
保证 |S(t) - s| <= eps_abs 量级，从而往返一致。
"""

import bisect
import math

L_MIN = 1e-12  # 相对容差的最小参照长度，防止零长度曲线容差为 0


def cubic_bezier(p0, p1, p2, p3):
    """返回 (curve, deriv) 两个函数，参数 t in [0, 1]。点可为任意维元组。"""

    def curve(t):
        u = 1.0 - t
        return tuple(
            u * u * u * p0[i]
            + 3.0 * u * u * t * p1[i]
            + 3.0 * u * t * t * p2[i]
            + t * t * t * p3[i]
            for i in range(len(p0))
        )

    def deriv(t):
        u = 1.0 - t
        return tuple(
            3.0 * u * u * (p1[i] - p0[i])
            + 6.0 * u * t * (p2[i] - p1[i])
            + 3.0 * t * t * (p3[i] - p2[i])
            for i in range(len(p0))
        )

    return curve, deriv


class ArcLengthMap:
    """curve: t -> 坐标元组；deriv（可选）: t -> 导数元组。

    tol 为相对容差（相对弧长粗估计），默认 1e-9。
    """

    def __init__(self, curve, t0=0.0, t1=1.0, deriv=None, tol=1e-9, max_depth=40):
        if not t1 > t0:
            raise ValueError("需要 t1 > t0")
        self._curve = curve
        self._t0 = float(t0)
        self._t1 = float(t1)
        self._tol = float(tol)
        self._max_depth = max_depth
        span = self._t1 - self._t0
        self._fd_h = max(span * 1e-6, 1e-12)  # 无解析导数时的差分步长
        self._deriv = deriv

        # 先用非自适应 Simpson 粗估总长，确定绝对容差
        coarse = self._simpson(self._t0, self._t1)
        for _ in range(4):  # 细分几层让粗估计更可靠
            m = 0.5 * (self._t0 + self._t1)
            coarse = self._simpson(self._t0, m) + self._simpson(m, self._t1)
        self._eps = self._tol * max(abs(coarse), L_MIN)

        # 自适应积分，同时记录叶子区间端点，构建 (t, s) 单调表
        leaves = []  # [(a, b, value)]，按 a 升序（递归为中序）
        self._length = self._adaptive(self._t0, self._t1, self._eps, max_depth, leaves)
        self._ts = [self._t0]
        self._ss = [0.0]
        acc = 0.0
        for a, b, v in leaves:
            acc += v
            self._ts.append(b)
            self._ss.append(acc)
        self._ss[-1] = self._length  # 消除累加舍入

    # ---- 基本量 ----

    @property
    def total_length(self):
        return self._length

    @property
    def abs_tol(self):
        """弧长绝对误差上界（近似）。"""
        return self._eps

    def _speed(self, t):
        if self._deriv is not None:
            d = self._deriv(t)
        else:
            h = self._fd_h
            c1 = self._curve(t + h)
            c0 = self._curve(t - h)
            d = tuple((a - b) / (2.0 * h) for a, b in zip(c1, c0))
        return math.sqrt(sum(x * x for x in d))

    def _simpson(self, a, b):
        m = 0.5 * (a + b)
        return (b - a) / 6.0 * (self._speed(a) + 4.0 * self._speed(m) + self._speed(b))

    def _adaptive(self, a, b, eps, depth, leaves):
        m = 0.5 * (a + b)
        whole = self._simpson(a, b)
        left = self._simpson(a, m)
        right = self._simpson(m, b)
        err = abs(left + right - whole) / 15.0
        if depth <= 0 or err <= eps:
            value = left + right + (left + right - whole) / 15.0
            if leaves is not None:
                leaves.append((a, b, value))
            return value
        half = 0.5 * eps
        return (
            self._adaptive(a, m, half, depth - 1, leaves)
            + self._adaptive(m, b, half, depth - 1, leaves)
        )

    # ---- 正查：参数 -> 弧长 ----

    def length(self, a=None, b=None):
        """曲线在参数区间 [a, b] 上的弧长（默认全程）。"""
        a = self._t0 if a is None else float(a)
        b = self._t1 if b is None else float(b)
        if b < a:
            return -self.length(b, a)
        return self._adaptive(a, b, self._eps, self._max_depth, None)

    # ---- 反查：弧长 -> 参数 ----

    def parameter_at(self, s):
        """给定弧长 s in [0, L]，返回对应参数 t。零长度曲线仅接受 s=0。"""
        if self._length <= 0.0:
            if abs(s) <= self._eps:
                return self._t0
            raise ValueError("零长度曲线只有 s=0 可达")
        if s < -self._eps or s > self._length + self._eps:
            raise ValueError("s 超出 [0, total_length]")
        s = min(max(s, 0.0), self._length)
        if s == 0.0:
            return self._t0
        if s == self._length:
            return self._t1

        i = max(bisect.bisect_right(self._ss, s) - 1, 0)
        base = self._ts[i]          # 积分基点，固定不变
        s_base = self._ss[i]
        lo, hi = base, self._ts[i + 1]
        # 线性插值初值
        seg = self._ss[i + 1] - s_base
        t = lo + (hi - lo) * (s - s_base) / seg if seg > 0.0 else 0.5 * (lo + hi)

        local_tol = self._eps * 0.01
        for _ in range(60):
            st = self._adaptive(base, t, local_tol, self._max_depth, None)
            f = s_base + st - s
            if abs(f) <= self._eps * 0.1:
                return t
            if f > 0.0:
                hi = t
            else:
                lo = t
            v = self._speed(t)
            if v > 0.0:
                tn = t - f / v
                if lo < tn < hi:
                    t = tn
                    continue
            t = 0.5 * (lo + hi)  # Newton 出界或速度为 0 时退化为二分
        return t

    def point_at(self, s):
        """给定弧长，返回曲线上的点。"""
        return self._curve(self.parameter_at(s))

    # ---- 等弧长取点 ----

    def uniform_parameters(self, n):
        """返回 n+1 个参数（含端点），把曲线按弧长 n 等分。"""
        if n < 1:
            raise ValueError("n >= 1")
        if self._length <= 0.0:
            return [self._t0] * (n + 1)
        step = self._length / n
        return [self.parameter_at(k * step) for k in range(n + 1)]

    def uniform_points(self, n):
        """返回 n+1 个等弧长分布的点（含端点）。"""
        return [self._curve(t) for t in self.uniform_parameters(n)]
