"""常用统计量。

所有统计量接受 float 序列，返回 float。
退化情形（空样本、零分母等）返回 NaN，由调用方（bootstrap 引擎）统一处理。
非有限输入（NaN/inf）默认会传播到结果中，即"garbage in, NaN out"，
是否允许由引擎的 allow_nonfinite 参数控制。
"""

import math

NAN = float("nan")


def stat_mean(data):
    n = len(data)
    if n == 0:
        return NAN
    return math.fsum(data) / n


def stat_median(data):
    return stat_quantile(data, 0.5)


def stat_variance(data, ddof=1):
    """方差，默认样本方差 (ddof=1)；ddof=0 为总体方差。"""
    n = len(data)
    if n - ddof <= 0:
        return NAN
    mean = math.fsum(data) / n
    ss = math.fsum((x - mean) ** 2 for x in data)
    return ss / (n - ddof)


def stat_quantile(data, q):
    """分位点，Hyndman-Fan type 7（线性插值，与 numpy 默认一致）。

    q 在 [0, 1] 内；q=0 为最小值，q=1 为最大值。
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("q 必须在 [0, 1] 内")
    n = len(data)
    if n == 0:
        return NAN
    xs = sorted(data)
    if n == 1:
        return xs[0]
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    frac = h - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def stat_ratio(data, zero_denom=NAN):
    """比值统计量：sum(num) / sum(den)。

    data 为 (num, den) 元组序列，例如 [(点击量, 曝光量), ...]，
    估计的是总量比率（如整体点击率），重采样时对整对 (num, den) 一起抽样。

    分母和为零时返回 zero_denom（默认 NaN）。
    """
    num_sum = 0.0
    den_sum = 0.0
    for num, den in data:
        num_sum += num
        den_sum += den
    if den_sum == 0.0:
        return zero_denom
    return num_sum / den_sum
