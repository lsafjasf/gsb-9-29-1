"""primekit —— 区间筛 + 轮式试除分解（仅标准库）。

内存与范围上界的关系：
- simple_sieve(n)：bytearray 只存奇数，内存 = n/2 字节，筛到 1e8 约 50MB。
- primes_in_range(lo, hi)：分段筛，内存 = O(SEG_ODDS + sqrt(hi))，
  与区间长度无关：每段固定 SEG_ODDS 个奇数（默认 50 万，约 0.5MB），
  基素数表到 sqrt(hi)（hi=1e12 时 sqrt=1e6，约 7.8 万个素数，<1MB）。
  因此上界几乎只受时间限制，不受内存限制。
- factor(n)：30 轮式试除，O(1) 额外内存，单数分解实用范围到 ~1e12；
  factor_with_primes 配合预算素数表可批量分解到 ~1e14。
"""

from math import isqrt, prod

__all__ = ["simple_sieve", "primes_in_range", "factor", "factor_with_primes", "product_of"]

SEG_ODDS = 500_000  # 每段覆盖的奇数个数（= 1e6 的数值跨度，约 0.5MB）


def simple_sieve(n):
    """返回所有 <= n 的素数列表。只存奇数，内存 n/2 字节。"""
    if n < 2:
        return []
    size = (n + 1) // 2  # 下标 i (i>=1) 对应奇数 2i+1，覆盖到 n 本身
    sieve = bytearray(b"\x01") * size
    sieve[0] = 0  # 1 不是素数
    for i in range(1, size):
        if sieve[i]:
            p = 2 * i + 1
            if p * p > n:
                break
            start = (p * p - 1) // 2
            sieve[start::p] = b"\x00" * ((size - 1 - start) // p + 1)
    primes = [2]
    primes.extend(2 * i + 1 for i in range(1, size) if sieve[i])
    return primes


def primes_in_range(lo, hi):
    """分段筛：返回 [lo, hi] 内全部素数（升序）。内存 O(SEG_ODDS + sqrt(hi))。"""
    lo = max(lo, 2)
    if hi < lo:
        return []
    result = []
    if lo <= 2 <= hi:
        result.append(2)
    seg_lo = max(lo, 3) | 1  # >= lo 的第一个奇数
    if seg_lo > hi:
        return result
    base_odd = [p for p in simple_sieve(isqrt(hi)) if p != 2]
    while seg_lo <= hi:
        seg_hi = min(seg_lo + 2 * SEG_ODDS - 2, hi)  # 闭区间
        size = (seg_hi - seg_lo) // 2 + 1
        sieve = bytearray(b"\x01") * size  # 下标 i 对应奇数 seg_lo + 2i
        for p in base_odd:
            if p * p > seg_hi:
                break
            start = max(p * p, -(-seg_lo // p) * p)  # >= seg_lo 的第一个 p 的倍数
            if start % 2 == 0:
                start += p
            j0 = (start - seg_lo) // 2
            sieve[j0::p] = b"\x00" * ((size - 1 - j0) // p + 1)
        result.extend(seg_lo + 2 * i for i in range(size) if sieve[i])
        seg_lo = seg_hi + 2
    return result


# 2*3*5 = 30 的轮：从 7 起，候选与 30 互素的步长循环
_WHEEL = (4, 2, 4, 2, 4, 6, 2, 6)  # 7,11,13,17,19,23,29,31,37,...


def product_of(factors):
    """由 [(p, e), ...] 还原原数。"""
    return prod(p ** e for p, e in factors)


def factor(n):
    """轮式试除分解 n，返回升序 [(质因数, 指数), ...]。

    n = 1 返回 []（空积定义为 1）；n <= 0 无素因子分解，抛 ValueError。
    轮上出现的合数候选（如 49）不会误判：更小的质因数已被除尽。
    """
    if n <= 0:
        raise ValueError(f"{n} 没有素因子分解（仅支持正整数）")
    if n == 1:
        return []
    original = n
    factors = []
    for p in (2, 3, 5):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors.append((p, e))
    i, k = 7, 0
    while i * i <= n:
        if n % i == 0:
            e = 0
            while n % i == 0:
                n //= i
                e += 1
            factors.append((i, e))
        i += _WHEEL[k]
        k = (k + 1) & 7
    if n > 1:
        factors.append((n, 1))
    assert product_of(factors) == original, f"分解校验失败: {original}"
    return factors


def factor_with_primes(n, primes):
    """用预算好的素数表（需覆盖到 isqrt(n)）试除分解，适合批量场景。"""
    if n <= 0:
        raise ValueError(f"{n} 没有素因子分解（仅支持正整数）")
    if n == 1:
        return []
    original = n
    factors = []
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors.append((p, e))
    if n > 1:
        factors.append((n, 1))
    assert product_of(factors) == original, f"分解校验失败: {original}"
    return factors
