"""
prime_tools.py —— 区间筛（Segmented Sieve）+ 轮式试除（Wheel Factorization）

只使用 Python 3 标准库。

公开接口
--------
primes_up_to(n, chunk=1_000_000)        -> 生成器，按序产出 2..n 的全部素数
primes_in_range(low, high, chunk=...)    -> 生成器，按序产出 [low, high) 的全部素数
factorize(n)                             -> 有序的 [(p, e), ...]，满足 ∏ p**e == n
factors_product(factors)                 -> 由 [(p, e), ...] 还原整数

内存与范围上界
--------------
区间筛不在内存里保存 0..n 的整张表，而是每次只处理长度 chunk 的一段。
段内只表示奇数，单段缓冲区约为 chunk/2 字节（bytearray，每奇数 1 字节）。
另外需要保存 sqrt(high) 以内的小素数，约 2*sqrt(high)/ln(sqrt(high)) 个
Python int（64 位 CPython 上每个约 28 字节 + 列表指针 8 字节）。

因此：
    工作内存 ≈ chunk/2 字节 + O(pi(sqrt(high)))
与 high 基本无关，只与 chunk 与 sqrt(high) 有关。
    例：chunk = 1_000_000 -> 段缓冲约 0.5 MiB；high = 10^8 时小素数仅 1229 个。
真正与 high 成比例的是“时间”（约 O(n log log n)）和“若把结果存成 list”
的输出空间（约 n/ln n 个素数对象）；生成器接口（primes_up_to/primes_in_range）
是流式的，不随 n 线性增长内存，因此没有固定的范围上界，瓶颈是运行时间。
"""

from __future__ import annotations

from math import isqrt
from typing import Iterator, List, Tuple

Factor = Tuple[int, int]  # (素数, 指数)

__all__ = [
    "primes_up_to",
    "primes_in_range",
    "factorize",
    "factors_product",
]


def _base_primes(limit: int) -> List[int]:
    """返回所有 <= limit 的素数。对 sqrt(high) 这种小规模量直接做普通筛。

    只表示奇数：seg[i] 对应奇数 2*i+3；仅当 limit >= 3 时分配。
    标记循环使用切片批量赋值（CPython 在 C 层执行），避免逐格 Python 循环。
    """
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * ((limit - 1) // 2)  # 奇数 3,5,7,...,<=limit
    mid = isqrt(limit)
    for i, flag in enumerate(sieve[: (mid - 1) // 2]):
        if flag:
            p = 2 * i + 3
            start = (p * p - 3) // 2
            sieve[start::p] = b"\x00" * (((len(sieve) - 1 - start) // p) + 1)
    return [2] + [2 * i + 3 for i, f in enumerate(sieve) if f]


def _segmented_odd_primes(low: int, high: int, chunk: int) -> Iterator[int]:
    """分段筛的内部实现：按序产出 [low, high) 内所有奇素数（不含 2）。

    每个段对齐到奇数开始，段内 seg[i] 表示奇数 seg_start + 2*i。
    """
    if high <= 3:
        return
    base = _base_primes(isqrt(high - 1))
    base = [p for p in base if p >= 3]  # 2 已被排除（只表示奇数）

    seg_start = max(low, 3)
    if seg_start % 2 == 0:
        seg_start += 1
    while seg_start < high:
        seg_end = min(seg_start + 2 * chunk, high + 1)
        # 保证段长也是偶数，下一段起点仍为奇数
        if (seg_end - seg_start) % 2:
            seg_end += 1
        seg_end = min(seg_end, high + 1)

        seg = bytearray(b"\x01") * ((seg_end - seg_start) // 2)
        for p in base:
            start = max(p * p, ((seg_start + p - 1) // p) * p)
            if start % 2 == 0:
                start += p  # 偶数倍 -> 下一个奇数倍
            i = (start - seg_start) // 2
            count = (len(seg) - 1 - i) // p + 1
            seg[i::p] = b"\x00" * count

        for i, flag in enumerate(seg):
            if flag:
                yield seg_start + 2 * i
        seg_start = seg_end


def primes_in_range(low: int, high: int, chunk: int = 1_000_000) -> Iterator[int]:
    """按序产出半开区间 [low, high) 内的全部素数。

    low/high 可为任意非负整数；high <= 2 时不产出任何素数。
    chunk 为单段容纳的奇数个数，决定段缓冲大小（约 chunk 字节级）。
    """
    if not isinstance(low, int) or not isinstance(high, int):
        raise TypeError("low/high 必须是 int")
    if chunk < 1:
        raise ValueError("chunk 必须 >= 1")
    if low <= 2 < high:
        yield 2
    lo = max(low, 3)
    if lo < high:
        yield from _segmented_odd_primes(lo, high, chunk)


def primes_up_to(n: int, chunk: int = 1_000_000) -> Iterator[int]:
    """按序产出所有 <= n 的素数。"""
    if n < 2:
        return
    yield from primes_in_range(2, n + 1, chunk)


# 2,3,5 轮式分解：一轮跳过的余数（模 30 中与 30 互质的 8 个余数）
_WHEEL_REMAINDERS = (1, 7, 11, 13, 17, 19, 23, 29)
# 由余数给出下一个候选的“增量”（环形）
_WHEEL_STEPS = (
    6, 4, 2, 4, 2, 4, 6, 2,  # 1->7->11->13->17->19->23->29->31(=1 mod 30)
)


def factorize(n: int, *, trial_limit: int = 100_000) -> List[Factor]:
    """返回 n 的完整质因数分解 [(p, e), ...]，按 p 升序。

    采用 2,3,5（mod 30）轮式试除：先剥离 2/3/5，之后只试与 30 互质的
    候选（7,11,13,17,19,23,29,31,...，每 30 个整数只试 8 个，少做约 73%
    的取模）。试除到 sqrt(n) 为止（剩余值随剥离不断缩小）。

    约定：factorize(0) == []，factorize(1) == []（无质因数）。
    负数按 |n| 分解；调用方若只接受正整数，可自行先做断言。

    trial_limit 仅作为“小因子快速剥离”与大素数判定的语义说明：本实现始终
    试除到当前剩余值的平方根，因此结果必然完整。对大素数（无因子）成本为
    O(sqrt(n)/轮稀疏度) 次取模。
    """
    if not isinstance(n, int):
        raise TypeError("n 必须是 int")
    if n < 0:
        n = -n
    factors: List[Factor] = []
    if n < 2:  # 0 和 1：没有质因数
        return factors

    for p in (2, 3, 5):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors.append((p, e))

    # 从 7 开始走 mod-30 轮（7 对应余数下标 1）
    candidate = 7
    step_index = 1
    while candidate * candidate <= n:
        if n % candidate == 0:
            e = 0
            while n % candidate == 0:
                n //= candidate
                e += 1
            factors.append((candidate, e))
        candidate += _WHEEL_STEPS[step_index]
        step_index = (step_index + 1) % 8

    if n > 1:
        factors.append((n, 1))
    return factors


def factors_product(factors: List[Factor]) -> int:
    """由 [(p, e), ...] 还原整数，用于断言“乘积等于原数”。"""
    product = 1
    for p, e in factors:
        product *= p ** e
    return product
