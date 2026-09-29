"""binning.py — 风控特征分箱库（仅依赖 Python 标准库）

功能：
- 分位点分箱（quantile）与最优分割分箱（optimal，ChiMerge 自底向上合并）
- 强制单调约束：基于 PAVA（Pool Adjacent Violators Algorithm）合并，
  不满足单调时给出违约点对与合并建议
- 边界稳定：同一样本重复 fit 结果一致；transform 只读边界，绝不修改
- 空值（None/NaN）独立分箱，异常值（分位点围栏外）截断并单独计数
"""

from __future__ import annotations

import bisect
import math

MISSING_BIN = -1
_EPS = 1e-12


def is_missing(value) -> bool:
    """空值判定：None 或 NaN。"""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _quantile(sorted_vals, q):
    """确定性分位点（线性插值，与 numpy 默认 type-7 一致）。输入必须已排序。"""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _quantile_edges(sorted_vals, n_bins):
    """由排序样本生成分箱边界，重复边界去重（长尾数据下边界数可能少于 n_bins-1）。"""
    edges = []
    for i in range(1, n_bins):
        e = _quantile(sorted_vals, i / n_bins)
        if not edges or e > edges[-1] + _EPS:
            edges.append(e)
    return edges


def _chi2_pair(a, b):
    """相邻两箱 (good, bad) 的卡方统计量（2x2 列联表）。"""
    good_total = a[0] + b[0]
    bad_total = a[1] + b[1]
    total = good_total + bad_total
    if total <= 0 or good_total <= 0 or bad_total <= 0:
        return 0.0
    chi = 0.0
    for good, bad in (a, b):
        n = good + bad
        exp_good = n * good_total / total
        exp_bad = n * bad_total / total
        if exp_good > 0:
            chi += (good - exp_good) ** 2 / exp_good
        if exp_bad > 0:
            chi += (bad - exp_bad) ** 2 / exp_bad
    return chi


def _pava_groups(counts, direction):
    """PAVA 相邻违约者合并。

    counts: [(good, bad), ...] 按箱顺序。
    返回合并后的块列表，每块 [start_idx, end_idx, good, bad]，
    块的风险率在 direction 方向上单调（不增/不减）。
    """
    def rate(block):
        total = block[2] + block[3]
        return block[3] / total if total > 0 else 0.0

    def violates(r_left, r_right):
        if direction == "increasing":
            return r_left > r_right + _EPS
        return r_left < r_right - _EPS

    stack = []
    for i, (good, bad) in enumerate(counts):
        stack.append([i, i, good, bad])
        while len(stack) >= 2 and violates(rate(stack[-2]), rate(stack[-1])):
            right = stack.pop()
            left = stack.pop()
            stack.append([left[0], right[1],
                          left[2] + right[2], left[3] + right[3]])
    return stack


def check_monotonicity(event_rates, direction="auto"):
    """单调性检测。

    返回 dict：
      direction   实际检测方向（auto 时取违约较少的一侧，平局取 increasing）
      is_monotonic
      violations  [(i, i+1, rate_i, rate_i+1), ...] 违约相邻对
      merge_suggestions  PAVA 合并建议：[[bin_idx, ...], ...] 每组应并为一箱
    """
    def violations_for(d):
        out = []
        for i in range(len(event_rates) - 1):
            r1, r2 = event_rates[i], event_rates[i + 1]
            if d == "increasing" and r1 > r2 + _EPS:
                out.append((i, i + 1, r1, r2))
            elif d == "decreasing" and r1 < r2 - _EPS:
                out.append((i, i + 1, r1, r2))
        return out

    if direction == "auto":
        inc = violations_for("increasing")
        dec = violations_for("decreasing")
        direction = "increasing" if len(inc) <= len(dec) else "decreasing"
    violations = violations_for(direction)

    merge_suggestions = []
    if violations:
        counts = [(1.0, r) for r in event_rates]  # 权重均等，仅用于分组
        groups = _pava_groups(counts, direction)
        merge_suggestions = [list(range(g[0], g[1] + 1)) for g in groups]

    return {
        "direction": direction,
        "is_monotonic": not violations,
        "violations": violations,
        "merge_suggestions": merge_suggestions,
    }


class Binner:
    """特征分箱器。

    参数：
      method            "quantile"（分位点）或 "optimal"（卡方合并最优分割，需要 y）
      n_bins            目标箱数（去重/合并后可能更少）
      monotonic         None | "increasing" | "decreasing" | "auto"，需要 y
      min_bin_pct       每箱最小样本占比（仅 optimal 合并阶段生效）
      outlier_quantile  异常值围栏分位点，<=0 关闭；围栏外取值截断并计数
      max_prebins       optimal 方法的预分箱数
    """

    def __init__(self, n_bins=5, method="quantile", monotonic=None,
                 min_bin_pct=0.05, outlier_quantile=0.01, max_prebins=20):
        if method not in ("quantile", "optimal"):
            raise ValueError("method must be 'quantile' or 'optimal'")
        if monotonic not in (None, "increasing", "decreasing", "auto"):
            raise ValueError("invalid monotonic")
        self.n_bins = n_bins
        self.method = method
        self.monotonic = monotonic
        self.min_bin_pct = min_bin_pct
        self.outlier_quantile = outlier_quantile
        self.max_prebins = max_prebins
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, x, y=None):
        x = list(x)
        if y is not None:
            y = list(y)
            if len(y) != len(x):
                raise ValueError("x and y length mismatch")
        if self.method == "optimal" and y is None:
            raise ValueError("optimal binning requires y")
        if self.monotonic is not None and y is None:
            raise ValueError("monotonic constraint requires y")

        clean = sorted(v for v in x if not is_missing(v))
        self.n_total_ = len(x)
        self.missing_count_ = self.n_total_ - len(clean)

        # 异常值围栏：截断并计数，不进入边界估计
        self.fence_lo_ = None
        self.fence_hi_ = None
        self.outlier_count_ = 0
        if clean and self.outlier_quantile and self.outlier_quantile > 0:
            q = min(self.outlier_quantile, 0.5)
            self.fence_lo_ = _quantile(clean, q)
            self.fence_hi_ = _quantile(clean, 1.0 - q)
            clipped = []
            for v in clean:
                if v < self.fence_lo_:
                    self.outlier_count_ += 1
                    clipped.append(self.fence_lo_)
                elif v > self.fence_hi_:
                    self.outlier_count_ += 1
                    clipped.append(self.fence_hi_)
                else:
                    clipped.append(v)
            clean = clipped

        # 边界计算
        if len(clean) < 2 or clean[0] == clean[-1]:
            self.edges_ = []
        elif self.method == "quantile":
            n_distinct = len(set(clean))
            self.edges_ = _quantile_edges(clean, min(self.n_bins, n_distinct))
        else:
            self.edges_ = self._optimal_edges(clean, x, y)

        # 单调约束：对当前箱做 PAVA 合并
        if self.monotonic is not None and y is not None and len(self.edges_) > 0:
            self._enforce_monotonic(x, y)

        self._compute_stats(x, y)
        self._fitted = True
        return self

    def _assign_clean(self, v):
        return bisect.bisect_right(self.edges_, v)

    def _clip(self, v):
        if self.fence_lo_ is not None and v < self.fence_lo_:
            return self.fence_lo_
        if self.fence_hi_ is not None and v > self.fence_hi_:
            return self.fence_hi_
        return v

    def _optimal_edges(self, clean, x, y):
        """ChiMerge：细粒度预分箱 -> 小箱合并 -> 卡方最小相邻合并至 n_bins。"""
        n_distinct = len(set(clean))
        pre_edges = _quantile_edges(clean, min(self.max_prebins, n_distinct))
        n_pre = len(pre_edges) + 1
        good = [0.0] * n_pre
        bad = [0.0] * n_pre
        for xi, yi in zip(x, y):
            if is_missing(xi):
                continue
            idx = bisect.bisect_right(pre_edges, self._clip(xi))
            if yi == 1:
                bad[idx] += 1.0
            else:
                good[idx] += 1.0

        bins = [[i, i, good[i], bad[i]] for i in range(n_pre)]  # [start, end, good, bad]

        def merge_at(blocks, k):
            left, right = blocks[k], blocks[k + 1]
            merged = [left[0], right[1],
                      left[2] + right[2], left[3] + right[3]]
            return blocks[:k] + [merged] + blocks[k + 2:]

        # 1) 合并小于 min_bin_size 的箱
        min_size = max(1.0, self.min_bin_pct * len(clean))
        while len(bins) > 1:
            small = [i for i, b in enumerate(bins) if b[2] + b[3] < min_size]
            if not small:
                break
            i = small[0]
            if len(bins) == 2:
                k = 0
            elif i == 0:
                k = 0
            elif i == len(bins) - 1:
                k = i - 1
            else:
                chi_l = _chi2_pair((bins[i][2], bins[i][3]),
                                   (bins[i - 1][2], bins[i - 1][3]))
                chi_r = _chi2_pair((bins[i][2], bins[i][3]),
                                   (bins[i + 1][2], bins[i + 1][3]))
                k = i - 1 if chi_l <= chi_r else i
            bins = merge_at(bins, k)

        # 2) 卡方最小相邻合并（平局取最左，保证确定性）
        target = max(2, self.n_bins)
        while len(bins) > target:
            best_k, best_chi = 0, None
            for k in range(len(bins) - 1):
                chi = _chi2_pair((bins[k][2], bins[k][3]),
                                 (bins[k + 1][2], bins[k + 1][3]))
                if best_chi is None or chi < best_chi - _EPS:
                    best_chi, best_k = chi, k
            bins = merge_at(bins, best_k)

        return [pre_edges[b[1]] for b in bins[:-1]]

    def _enforce_monotonic(self, x, y):
        """对当前边界按 PAVA 合并，使箱风险率单调。"""
        n_bins_now = len(self.edges_) + 1
        good = [0.0] * n_bins_now
        bad = [0.0] * n_bins_now
        for xi, yi in zip(x, y):
            if is_missing(xi):
                continue
            idx = self._assign_clean(self._clip(xi))
            if yi == 1:
                bad[idx] += 1.0
            else:
                good[idx] += 1.0

        direction = self.monotonic
        if direction == "auto":
            rates = [bad[i] / (good[i] + bad[i]) if good[i] + bad[i] > 0 else 0.0
                     for i in range(n_bins_now)]
            direction = check_monotonicity(rates, "auto")["direction"]

        groups = _pava_groups(list(zip(good, bad)), direction)
        self.monotonic_direction_ = direction
        self.edges_ = [self.edges_[g[1]] for g in groups[:-1]]

    # ---------------------------------------------------------------- stats
    def _compute_stats(self, x, y):
        n_bins = len(self.edges_) + 1
        counts = [0] * n_bins
        bads = [0.0] * n_bins
        miss_bad = 0.0
        for i, xi in enumerate(x):
            if is_missing(xi):
                if y is not None and y[i] == 1:
                    miss_bad += 1.0
                continue
            idx = self._assign_clean(self._clip(xi))
            counts[idx] += 1
            if y is not None and y[i] == 1:
                bads[idx] += 1.0

        self.bin_stats_ = []
        total_good = sum(c for c in counts) - sum(bads)
        total_bad = sum(bads)
        iv_total = 0.0
        for i in range(n_bins):
            good = counts[i] - bads[i]
            stat = {
                "bin": i,
                "count": counts[i],
                "bad": bads[i],
                "event_rate": (bads[i] / counts[i]) if counts[i] > 0 else None,
            }
            if y is not None and total_good > 0 and total_bad > 0:
                dg = (good + 0.5) / (total_good + 0.5 * n_bins)
                db = (bads[i] + 0.5) / (total_bad + 0.5 * n_bins)
                stat["woe"] = math.log(dg / db)
                stat["iv"] = (dg - db) * stat["woe"]
                iv_total += stat["iv"]
            self.bin_stats_.append(stat)

        self.missing_stats_ = {
            "bin": MISSING_BIN,
            "count": self.missing_count_,
            "bad": miss_bad,
            "event_rate": (miss_bad / self.missing_count_)
                          if self.missing_count_ > 0 else None,
        }
        self.iv_ = iv_total if y is not None else None

    # ------------------------------------------------------------- transform
    def transform(self, x):
        """打分：只读边界，绝不修改。空值 -> MISSING_BIN(-1)。"""
        if not self._fitted:
            raise RuntimeError("Binner is not fitted")
        out = []
        for v in x:
            if is_missing(v):
                out.append(MISSING_BIN)
            else:
                out.append(self._assign_clean(self._clip(v)))
        return out

    # -------------------------------------------------------------- report
    def monotonicity_report(self):
        """单调性检测数据：违约点对 + PAVA 合并建议。"""
        if not self._fitted:
            raise RuntimeError("Binner is not fitted")
        rates = [s["event_rate"] for s in self.bin_stats_]
        if any(r is None for r in rates):
            return {"available": False,
                    "reason": "y not provided or empty bin exists"}
        report = check_monotonicity(rates, self.monotonic or "auto")
        report["available"] = True
        report["event_rates"] = rates
        return report

    def summary(self):
        if not self._fitted:
            raise RuntimeError("Binner is not fitted")
        return {
            "method": self.method,
            "edges": list(self.edges_),
            "n_total": self.n_total_,
            "missing_count": self.missing_count_,
            "outlier_count": self.outlier_count_,
            "fence_lo": self.fence_lo_,
            "fence_hi": self.fence_hi_,
            "bins": self.bin_stats_,
            "missing_bin": self.missing_stats_,
            "iv": self.iv_,
        }
