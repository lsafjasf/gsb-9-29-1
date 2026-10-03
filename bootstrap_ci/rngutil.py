"""可复现随机源工具。

设计要点：
- 所有随机性都来自一个整数种子。seed -> (chunk_seed, replicate_seed) 的派生
  使用 splitmix64，与平台无关、与 Python 版本无关，保证完全可复现。
- 分块并行时，每个 chunk 用独立派生种子生成"索引矩阵"，worker 之间无共享
  状态，合并时按 chunk 顺序拼接，因此与串行结果逐位一致。
- 索引用 getrandbits(32) % n 生成。模偏差存在但可忽略：2^32 是 n 的倍数时
  无偏差；一般 n 下每个索引的概率偏差 < n / 2^32（n=100 时约 2e-8 量级），
  对 bootstrap 推断无实际影响。文档中明确说明这一点。
"""

import random

MASK64 = (1 << 64) - 1


def splitmix64(x):
    """splitmix64 混合函数：确定性、雪崩效应好，用于种子派生。"""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def derive_seed(seed, *labels):
    """从 (seed, labels...) 确定性地派生一个 64 位种子。"""
    state = splitmix64(seed & MASK64)
    for label in labels:
        state = splitmix64(state ^ (int(label) & MASK64))
    return state


def make_rng(seed, *labels):
    """构造一个 random.Random，种子由 (seed, labels...) 派生。"""
    return random.Random(derive_seed(seed, *labels))


def resample_indices(n, k, rng):
    """用 rng 生成 k 个 [0, n) 的重采样索引（有放回）。

    rng 需是 random.Random 兼容对象（提供 getrandbits 或 randrange）。
    """
    if n <= 0:
        raise ValueError("n 必须为正")
    getbits = getattr(rng, "getrandbits", None)
    if getbits is not None:
        return [getbits(32) % n for _ in range(k)]
    return [rng.randrange(n) for _ in range(k)]


def resample(data, rng):
    """返回 data 的一个 bootstrap 重采样副本（与 data 等长、有放回）。"""
    n = len(data)
    return [data[i] for i in resample_indices(n, n, rng)]
