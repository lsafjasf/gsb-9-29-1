"""引导法（bootstrap）置信区间核心引擎。

支持三种区间：
- percentile：百分位法。直接取重采样分布的 alpha/2 与 1-alpha/2 分位点。
  适用：统计量近似无偏、重采样分布近似对称。实现最简单，无额外假设计算。
- bc：偏差校正法。用重采样分布中小于原始估计的比例估计偏差 z0 并校正分位点。
  适用：统计量有偏但偏差不随参数值变化（无加速效应）。
- bca：偏差校正加速法。在 BC 基础上用 jackknife 估计加速常数 a（刻画标准误
  随参数值的变化率），校正分位点。适用：有偏且标准误依赖未知参数的情形，
  是三者中假设最弱、覆盖率通常最好的一阶+二阶校正方法。

实现差异：
- percentile 只用重采样分布本身；
- bc 额外需要 z0 = Phi^{-1}(#{theta* < theta_hat} / B)；
- bca 在 bc 之上额外需要 n 次 jackknife（留一法）统计量来估计加速常数 a，
  分位点水平调整为 Phi(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))。

可复现性：所有随机性来自整数种子。B 次重采样被划分为若干 chunk，
chunk c 的种子为 derive_seed(seed, c)，与调度方式无关；因此串行与
多进程并行（n_jobs>1）结果逐位一致。chunk_size 改变会改变随机流，
属于预期行为（相当于换了一种索引生成顺序）。
"""

import math
import multiprocessing
from array import array

from .rngutil import derive_seed
from .stats import stat_quantile

NAN = float("nan")
INF = float("inf")

_DEFAULT_CHUNKS = 32


# ---------------------------------------------------------------- 正态分布函数

def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p):
    """标准正态逆 CDF，Acklam 有理逼近（最大误差约 1e-9）。"""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -INF
        if p == 1.0:
            return INF
        raise ValueError("p 必须在 (0, 1) 内")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---------------------------------------------------------------- 重采样

def _chunk_bounds(B, chunk_size):
    bounds = []
    start = 0
    while start < B:
        end = min(start + chunk_size, B)
        bounds.append((start, end - start))
        start = end
    return bounds


def default_chunk_size(B):
    return max(1, (B + _DEFAULT_CHUNKS - 1) // _DEFAULT_CHUNKS)


def bootstrap_chunk(data, statistic, chunk_seed, B_chunk):
    """生成一个 chunk 的重采样统计量。

    索引由 random.Random(chunk_seed) 以 getrandbits(32) % n 批量生成，
    只依赖 (chunk_seed, n, B_chunk)，与执行进程/线程无关。
    """
    import random

    n = len(data)
    rng = random.Random(chunk_seed)
    idx = array("I", (rng.getrandbits(32) % n for _ in range(B_chunk * n)))
    getitem = data.__getitem__
    out = []
    append = out.append
    for r in range(B_chunk):
        base = r * n
        append(statistic([getitem(i) for i in idx[base:base + n]]))
    return out


def _run_chunk(args):
    data, statistic, chunk_seed, B_chunk = args
    return bootstrap_chunk(data, statistic, chunk_seed, B_chunk)


def bootstrap_replicates(data, statistic, B, seed=None, rng=None,
                         n_jobs=1, chunk_size=None):
    """生成 B 个 bootstrap 重采样统计量，按原始顺序返回。

    seed 模式（推荐）：结果完全可复现，且 n_jobs>1 并行与串行一致。
    rng 模式：从用户注入的 rng 顺序抽样（此时仅支持 n_jobs=1）。
    """
    if B <= 0:
        raise ValueError("B 必须为正")
    if rng is not None:
        if n_jobs != 1:
            raise ValueError("注入 rng 时不支持并行；请改用 seed")
        from .rngutil import resample_indices
        n = len(data)
        getitem = data.__getitem__
        return [statistic([getitem(i) for i in resample_indices(n, n, rng)])
                for _ in range(B)]
    if seed is None:
        raise ValueError("必须提供 seed 或 rng")

    if chunk_size is None:
        chunk_size = default_chunk_size(B)
    bounds = _chunk_bounds(B, chunk_size)
    tasks = [(data, statistic, derive_seed(seed, c), size)
             for c, (_, size) in enumerate(bounds)]
    if n_jobs == 1 or len(tasks) == 1:
        chunks = [_run_chunk(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=min(n_jobs, len(tasks))) as pool:
            chunks = pool.map(_run_chunk, tasks)
    return merge_chunks(chunks)


def merge_chunks(chunks):
    """按 chunk 顺序拼接重采样结果。与串行执行逐位一致。"""
    out = []
    for chunk in chunks:
        out.extend(chunk)
    return out


# ---------------------------------------------------------------- jackknife

def jackknife(data, statistic):
    """留一法统计量，长度与 data 相同。"""
    n = len(data)
    if n < 2:
        return []
    return [statistic(data[:i] + data[i + 1:]) for i in range(n)]


def acceleration(jack):
    """由 jackknife 值估计 BCa 加速常数 a。

    退化处理：有效值不足 3 个或离差平方和为 0（如样本全部相同）时返回 0，
    此时 BCa 退化为 BC。
    """
    vals = [x for x in jack if math.isfinite(x)]
    m = len(vals)
    if m < 3:
        return 0.0
    jbar = math.fsum(vals) / m
    diffs = [jbar - x for x in vals]
    num = math.fsum(d ** 3 for d in diffs)
    den = math.fsum(d * d for d in diffs)
    if den == 0.0:
        return 0.0
    return num / (6.0 * den ** 1.5)


# ---------------------------------------------------------------- 区间

def _finite_sorted(values):
    return sorted(v for v in values if math.isfinite(v))


def _z0_from(reps_sorted, theta_hat):
    B = len(reps_sorted)
    less = 0
    for r in reps_sorted:
        if r < theta_hat:
            less += 1
        else:
            break
    prop = less / B
    if prop <= 0.0:
        return -INF
    if prop >= 1.0:
        return INF
    return norm_ppf(prop)


def _adjusted_level(z0, a, z_alpha):
    """BC/BCa 校正后的分位点水平，钳制到 [0, 1]。"""
    if z0 == -INF:
        return 0.0
    if z0 == INF:
        return 1.0
    denom = 1.0 - a * (z0 + z_alpha)
    if denom == 0.0:
        return 1.0 if (z0 + z_alpha) > 0 else 0.0
    p = norm_cdf(z0 + (z0 + z_alpha) / denom)
    return min(1.0, max(0.0, p))


def _intervals_from_reps(reps, theta_hat, alpha, jack, methods, warnings):
    reps_f = _finite_sorted(reps)
    B = len(reps_f)
    out = {}
    if B == 0:
        for m in methods:
            out[m] = (NAN, NAN)
        warnings.append("所有重采样统计量均非有限，区间不可用")
        return out, NAN, NAN

    z0 = _z0_from(reps_f, theta_hat) if math.isfinite(theta_hat) else NAN
    a = acceleration(jack) if "bca" in methods else 0.0

    z_lo = norm_ppf(alpha / 2.0)
    z_hi = norm_ppf(1.0 - alpha / 2.0)

    for m in methods:
        if m == "percentile":
            p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
        elif m == "bc":
            if not math.isfinite(z0):
                out[m] = (NAN, NAN)
                continue
            p_lo = _adjusted_level(z0, 0.0, z_lo)
            p_hi = _adjusted_level(z0, 0.0, z_hi)
        elif m == "bca":
            if not math.isfinite(z0):
                out[m] = (NAN, NAN)
                continue
            p_lo = _adjusted_level(z0, a, z_lo)
            p_hi = _adjusted_level(z0, a, z_hi)
        else:
            raise ValueError("未知方法: %r" % (m,))
        out[m] = (stat_quantile(reps_f, p_lo), stat_quantile(reps_f, p_hi))

    if z0 in (-INF, INF):
        warnings.append(
            "重采样统计量全部位于点估计一侧（z0 为无穷），BC/BCa 退化为端点；"
            "常见于统计量取到样本极值（如 max/min）或样本退化，"
            "此时百分位法同样不可信，建议换用参数化 bootstrap 或贝叶斯方法")
    return out, z0, a


def _check_data(data, allow_nonfinite):
    if len(data) == 0:
        raise ValueError("样本为空")
    if allow_nonfinite:
        return
    for x in data:
        if isinstance(x, (tuple, list)):
            vals = x
        else:
            vals = (x,)
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(
                    "数据含非有限值（NaN/inf）。默认拒绝；"
                    "如确需传播，请使用 allow_nonfinite=True")


def bootstrap_ci(data, statistic, B=1999, alpha=0.05, seed=None, rng=None,
                 methods=("percentile", "bc", "bca"), n_jobs=1, chunk_size=None,
                 allow_nonfinite=False, jackknife_values=None):
    """计算 bootstrap 置信区间。

    参数
    ----
    data : 序列。标量样本（如 [1.2, 3.4, ...]）或成对样本
        （如 [(num, den), ...]，配合 stat_ratio）。
    statistic : callable(data_subset) -> float。并行时必须是可 pickle 的
        模块级函数（可用 functools.partial 绑定参数）。
    B : 重采样次数，建议 >= 1999。
    alpha : 显著性水平，0.05 对应 95% 区间。
    seed / rng : 随机源，二选一。seed 保证完全可复现且支持并行。
    methods : "percentile" / "bc" / "bca" 的任意子集。
    n_jobs : 进程数；>1 时用多进程跑 chunk，结果与串行一致。
    allow_nonfinite : 默认 False，数据含 NaN/inf 直接报错；
        True 时放行，非有限的重采样统计量在分位数计算前被剔除并计数。
    jackknife_values : 可选，预先算好的 jackknife 值（模拟中可复用以省时）。

    返回 dict，含 intervals、theta_hat、se、z0、acceleration、warnings 等。
    """
    data = list(data)
    _check_data(data, allow_nonfinite)
    n = len(data)
    warnings = []

    theta_hat = statistic(data)
    reps = bootstrap_replicates(data, statistic, B, seed=seed, rng=rng,
                                n_jobs=n_jobs, chunk_size=chunk_size)

    n_nonfinite = sum(1 for r in reps if not math.isfinite(r))
    if n_nonfinite:
        warnings.append(
            "%d/%d 个重采样统计量非有限（如零分母），已在分位数计算中剔除"
            % (n_nonfinite, B))

    if jackknife_values is None:
        jackknife_values = jackknife(data, statistic) if "bca" in methods else []

    intervals, z0, a = _intervals_from_reps(reps, theta_hat, alpha,
                                            jackknife_values, methods, warnings)

    reps_f = _finite_sorted(reps)
    if len(reps_f) >= 2:
        m = math.fsum(reps_f) / len(reps_f)
        se = math.sqrt(math.fsum((r - m) ** 2 for r in reps_f) / (len(reps_f) - 1))
    else:
        se = NAN

    if n == 1:
        warnings.append("n=1：重采样无变异，所有区间退化为点估计，不具推断意义")

    return {
        "n": n,
        "B": B,
        "alpha": alpha,
        "theta_hat": theta_hat,
        "se": se,
        "intervals": intervals,
        "z0": z0,
        "acceleration": a,
        "n_nonfinite_replicates": n_nonfinite,
        "warnings": warnings,
        "replicates": reps,
    }
