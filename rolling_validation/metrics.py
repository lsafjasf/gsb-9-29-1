"""折间指标、波动与置信区间。仅依赖标准库。

降级策略：
  - 单折：无法估计波动与区间 -> 对应字段为 None，并产生告警；
  - 某折验证值全部缺失 -> 该折指标为 None，聚合时剔除并告警；
  - 所有折均无法计算 -> 汇总为 None 并告警；
  - 指标在数学上无定义（如 MAPE 真值全为 0）-> None + 告警，不抛异常。
"""

import math
import statistics
import warnings
from dataclasses import dataclass, field


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _paired(y_true, y_pred):
    """剔除任一侧缺失的样本对，返回 (pairs, n_dropped)。"""
    pairs = [(t, p) for t, p in zip(y_true, y_pred)
             if not _is_missing(t) and not _is_missing(p)]
    return pairs, len(y_true) - len(pairs)


def mae(y_true, y_pred):
    pairs, _ = _paired(y_true, y_pred)
    if not pairs:
        return None
    return sum(abs(t - p) for t, p in pairs) / len(pairs)


def rmse(y_true, y_pred):
    pairs, _ = _paired(y_true, y_pred)
    if not pairs:
        return None
    return math.sqrt(sum((t - p) ** 2 for t, p in pairs) / len(pairs))


def mape(y_true, y_pred):
    """真值为 0 的点无法定义百分比误差，剔除；全部剔除则返回 None。"""
    pairs, _ = _paired(y_true, y_pred)
    pairs = [(t, p) for t, p in pairs if t != 0]
    if not pairs:
        return None
    return sum(abs((t - p) / t) for t, p in pairs) / len(pairs) * 100.0


def smape(y_true, y_pred):
    pairs, _ = _paired(y_true, y_pred)
    terms = []
    for t, p in pairs:
        denom = abs(t) + abs(p)
        if denom == 0:
            continue  # 0/0 无定义，剔除
        terms.append(2.0 * abs(t - p) / denom)
    if not terms:
        return None
    return sum(terms) / len(terms) * 100.0


METRICS = {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


# 双侧 t 临界值表（常用置信水平），df=1..30 及若干大自由度
_T_TABLE = {
    0.90: [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833,
           1.812, 1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734,
           1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703,
           1.701, 1.699, 1.697],
    0.95: [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
           2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
           2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
           2.048, 2.045, 2.042],
    0.99: [63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250,
           3.169, 3.106, 3.055, 3.012, 2.977, 2.947, 2.921, 2.898, 2.878,
           2.861, 2.845, 2.831, 2.819, 2.807, 2.797, 2.787, 2.779, 2.771,
           2.763, 2.756, 2.750],
}


def _t_critical(confidence, df):
    """t 临界值；表外自由度用正态近似（statistics.NormalDist）。"""
    if df >= 1:
        table = _T_TABLE.get(confidence)
        if table is not None:
            if df <= 30:
                return table[df - 1]
            # df>30 与正态已很接近，用正态分位近似
    return statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)


def summarize(values, confidence=0.95):
    """对跨折指标序列做汇总。

    返回 dict(mean, std, n, ci)；std 为样本标准差（ddof=1）。
    单点：std/ci 为 None；空序列：全 None。调用方负责告警。
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "ci": None,
                "confidence": confidence}
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return {"mean": mean, "std": None, "n": 1, "ci": None,
                "confidence": confidence}
    std = statistics.stdev(vals)
    half = _t_critical(confidence, len(vals) - 1) * std / math.sqrt(len(vals))
    return {"mean": mean, "std": std, "n": len(vals),
            "ci": (mean - half, mean + half), "confidence": confidence}


def warn(msg, sink):
    """既发 warnings 告警，也记录到结果对象的 warnings 列表。"""
    warnings.warn(msg, RuntimeWarning, stacklevel=2)
    sink.append(msg)
