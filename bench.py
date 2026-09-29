"""耗时基准：区间筛 vs 朴素逐个试除；批量分解 vs 朴素分解。

运行：python3 bench.py
"""

import random
import time
from math import isqrt

from primekit import factor, factor_with_primes, primes_in_range, simple_sieve


def ref_is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def ref_factor(n):
    out = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            out.append(i)
            n //= i
        i += 1
    if n > 1:
        out.append(n)
    return out


def timed(label, fn):
    t0 = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<52} {dt:>8.3f}s")
    return r


print("== 素数表：分段筛 vs 朴素逐个试除 ==")
timed("朴素逐个试除 [2, 1e6]", lambda: [n for n in range(2, 10**6) if ref_is_prime(n)])
timed("分段筛 primes_in_range(2, 1e6)", lambda: primes_in_range(2, 10**6))
timed("分段筛 primes_in_range(2, 1e7)", lambda: primes_in_range(2, 10**7))
timed("分段筛 primes_in_range(2, 1e8)", lambda: primes_in_range(2, 10**8))
timed("分段筛远端区间 [1e12, 1e12+1e6]", lambda: primes_in_range(10**12, 10**12 + 10**6))

print("== 整数分解：轮式/素数表试除 vs 朴素试除 ==")
rng = random.Random(2026)
small = [rng.randint(1, 10**6) for _ in range(20_000)]
timed("朴素试除 x20000 (n <= 1e6)", lambda: [ref_factor(n) for n in small])
timed("轮式 factor x20000 (n <= 1e6)", lambda: [factor(n) for n in small])

big = [rng.randint(1, 10**12) for _ in range(20_000)]
primes = simple_sieve(isqrt(10**12) + 1)
timed("朴素试除 x2000 (n <= 1e12)", lambda: [ref_factor(n) for n in big[:2000]])
timed("轮式 factor x20000 (n <= 1e12)", lambda: [factor(n) for n in big])
timed("素数表 factor_with_primes x20000 (n <= 1e12)",
      lambda: [factor_with_primes(n, primes) for n in big])

print("== 特殊形态单数 ==")
timed("大素数 999999999989", lambda: factor(999999999989))
timed("大素数平方 999983^2", lambda: factor(999983**2))
timed("半素数 999983 * 999979 (~1e12)", lambda: factor(999983 * 999979))
timed("高幂次 2^60 * 3^40", lambda: factor(2**60 * 3**40))
