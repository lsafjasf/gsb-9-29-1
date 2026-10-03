"""同一数据下三种 bootstrap 区间的数值对比。

用法：python3 -m bootstrap_ci.demo
"""

import math

from .bootstrap import bootstrap_ci
from .rngutil import make_rng
from .stats import stat_mean, stat_median, stat_variance, stat_quantile
from .distributions import TRUTHS


def _fmt(v):
    return "%+.4f" % v if math.isfinite(v) else str(v)


def main():
    # 固定种子：20 个 Exp(1) 样本（右偏，真实均值=1、中位数=ln2）
    rng = make_rng(4, 0)
    data = [rng.expovariate(1.0) for _ in range(20)]

    print("同一数据（n=%d, Exp(1) 右偏样本），B=9999, seed=42" % len(data))
    print("-" * 78)
    for name, stat, truth in (
        ("均值", stat_mean, TRUTHS["exponential"]["mean"]),
        ("中位数", stat_median, TRUTHS["exponential"]["median"]),
        ("方差", stat_variance, TRUTHS["exponential"]["variance"]),
        ("90%分位点", (lambda d: stat_quantile(d, 0.9)),
         TRUTHS["exponential"]["q90"]),
    ):
        res = bootstrap_ci(data, stat, B=9999, alpha=0.05, seed=42)
        print("%-7s 点估计 %.4f（真值 %.4f）  SE %.4f  z0=%+.3f  a=%+.4f"
              % (name, res["theta_hat"], truth, res["se"],
                 res["z0"], res["acceleration"]))
        for m in ("percentile", "bc", "bca"):
            lo, hi = res["intervals"][m]
            cover = "含真值" if lo <= truth <= hi else "不含真值"
            print("        %-11s [%s, %s]  宽度 %.4f  %s"
                  % (m, _fmt(lo), _fmt(hi), hi - lo, cover))
        print("-" * 78)

    print("说明：右偏小样本下方差等统计量 z0>0、a>0，BC/BCa 把区间整体")
    print("向上调整（如本例方差：上界 1.12 -> 1.29）；三者端点通常不同但都在")
    print("百分位区间的内插格点上（B+1 个分位点），差异只在分位点选择。")


if __name__ == "__main__":
    main()
