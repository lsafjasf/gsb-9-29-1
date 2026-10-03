"""覆盖率模拟用的已知分布。

每个分布提供：
- sample(rng, n)：从分布中抽 n 个 i.i.d. 样本（rng 为 random.Random）
- TRUTH：各统计量的真实值（解析或高精度数值解）

四个分布：
- normal：N(0,1)，对称，作为基准；
- exponential：Exp(1)，右偏，小样本下样本均值/方差明显偏斜；
- bimodal：0.5*N(-2,1) + 0.5*N(2,1)，双峰，小样本容易只采到一个峰；
- outlier：0.9*N(0,1) + 0.1*N(0,10^2)，重尾污染，离群点频发。

另有一个 ratio 场景（成对计数数据），用于比值统计量。
"""

import math
from functools import partial

from .bootstrap import norm_ppf, norm_cdf
from .stats import stat_mean, stat_median, stat_variance, stat_quantile, stat_ratio


def _solve_quantile(cdf, q, lo, hi, tol=1e-12):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- 采样器

def sample_normal(rng, n):
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def sample_exponential(rng, n):
    return [rng.expovariate(1.0) for _ in range(n)]


def sample_bimodal(rng, n):
    out = []
    for _ in range(n):
        mu = -2.0 if rng.random() < 0.5 else 2.0
        out.append(rng.gauss(mu, 1.0))
    return out


def sample_outlier(rng, n):
    out = []
    for _ in range(n):
        if rng.random() < 0.9:
            out.append(rng.gauss(0.0, 1.0))
        else:
            out.append(rng.gauss(0.0, 10.0))
    return out


def _poisson(rng, lam):
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


def sample_ratio_pairs(rng, n):
    """成对计数 (num, den)：den ~ 1+Poisson(5)，num|den ~ Binomial(den, 0.33)。

    总量比率 sum(num)/sum(den) 依概率收敛到 E[num]/E[den] = 0.33。
    """
    out = []
    for _ in range(n):
        den = 1 + _poisson(rng, 5.0)
        num = 0
        for _ in range(den):
            if rng.random() < 0.33:
                num += 1
        out.append((float(num), float(den)))
    return out


# ---------------------------------------------------------------- 真实值

_OUTLIER_CDF = lambda x: 0.9 * norm_cdf(x) + 0.1 * norm_cdf(x / 10.0)

TRUTHS = {
    "normal": {
        "mean": 0.0,
        "median": 0.0,
        "variance": 1.0,
        "q90": norm_ppf(0.9),
    },
    "exponential": {
        "mean": 1.0,
        "median": math.log(2.0),
        "variance": 1.0,
        "q90": math.log(10.0),
    },
    "bimodal": {
        "mean": 0.0,
        "median": 0.0,
        "variance": 5.0,  # 1 + 2^2
        "q90": 2.0 + norm_ppf(0.8),  # 0.5*Phi(q+2)+0.5*Phi(q-2)=0.9 的解
    },
    "outlier": {
        "mean": 0.0,
        "median": 0.0,
        "variance": 0.9 * 1.0 + 0.1 * 100.0,  # 10.9
        "q90": _solve_quantile(_OUTLIER_CDF, 0.9, 0.0, 10.0),
    },
    "ratio_pairs": {
        "ratio": 0.33,
    },
}

SAMPLERS = {
    "normal": sample_normal,
    "exponential": sample_exponential,
    "bimodal": sample_bimodal,
    "outlier": sample_outlier,
    "ratio_pairs": sample_ratio_pairs,
}

STATISTICS = {
    "mean": stat_mean,
    "median": stat_median,
    "variance": stat_variance,
    "q90": partial(stat_quantile, q=0.9),
    "ratio": stat_ratio,
}
