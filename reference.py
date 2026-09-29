"""
reference.py —— 朴素参照实现（故意写得直白，无轮式、无分段）

- ref_primes(low, high)：用整数 sqrt 试除逐个判定素数（含偶数），O((hi-lo)*sqrt)
- ref_factorize(n)：从 2 开始逐个整数试除，不跳候选

仅用于与 prime_tools 对拍；效率低，不处理超大范围。
"""

from __future__ import annotations

from math import isqrt
from typing import List, Tuple


def ref_is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def ref_primes(low: int, high: int) -> List[int]:
    """[low, high) 内的全部素数，朴素逐个试除判定。"""
    return [n for n in range(max(low, 0), high) if ref_is_prime(n)]


def ref_factorize(n: int) -> List[Tuple[int, int]]:
    """从 2 起逐个整数试除的完整分解；0/1 返回 []。"""
    if n < 0:
        n = -n
    factors: List[Tuple[int, int]] = []
    if n < 2:
        return factors
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors
