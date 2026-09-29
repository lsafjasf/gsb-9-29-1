"""primekit 自测：朴素参照实现对拍 + 边界用例 + 断言。

运行：python3 test_primekit.py [-v]
"""

import random
import sys
import time
import unittest
from math import isqrt

from primekit import (
    factor,
    factor_with_primes,
    primes_in_range,
    product_of,
    simple_sieve,
)


# ---------- 朴素参照实现（刻意写得简单直白，作为对拍基准） ----------

def ref_is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def ref_primes(lo, hi):
    return [n for n in range(max(lo, 2), hi + 1) if ref_is_prime(n)]


def ref_factor(n):
    if n <= 0:
        raise ValueError("no factorization")
    out = []
    i = 2
    while i * i <= n:
        if n % i == 0:
            e = 0
            while n % i == 0:
                n //= i
                e += 1
            out.append((i, e))
        i += 1
    if n > 1:
        out.append((n, 1))
    return out


# ---------- 边界用例 ----------

class TestEdgeCases(unittest.TestCase):
    def test_zero_and_negative_raise(self):
        for n in (0, -1, -360):
            with self.assertRaises(ValueError):
                factor(n)
            with self.assertRaises(ValueError):
                factor_with_primes(n, [2, 3, 5])

    def test_one_is_empty_product(self):
        self.assertEqual(factor(1), [])
        self.assertEqual(product_of([]), 1)  # 空积定义为 1

    def test_primes(self):
        for p in (2, 3, 5, 97, 999983, 999999999989):  # 含 12 位大素数
            self.assertEqual(factor(p), [(p, 1)])

    def test_perfect_squares(self):
        self.assertEqual(factor(2 ** 10), [(2, 10)])
        self.assertEqual(factor(97 ** 2), [(97, 2)])
        self.assertEqual(factor((2 * 3 * 5 * 7) ** 2),
                         [(2, 2), (3, 2), (5, 2), (7, 2)])
        self.assertEqual(factor(999983 ** 2), [(999983, 2)])  # 大素数平方

    def test_large_prime_and_semiprime(self):
        big_p = 999999999989          # 大素数
        self.assertEqual(factor(big_p), [(big_p, 1)])
        semi = 999983 * 999979        # 两个大素数之积
        self.assertEqual(factor(semi), [(999979, 1), (999983, 1)])
        self.assertEqual(product_of(factor(semi)), semi)

    def test_output_sorted_and_product_verified(self):
        rng = random.Random(42)
        for _ in range(2000):
            n = rng.randint(1, 10 ** 9)
            f = factor(n)
            primes_only = [p for p, _ in f]
            self.assertEqual(primes_only, sorted(primes_only))       # 有序
            self.assertTrue(all(e >= 1 for _, e in f))               # 指数 >= 1
            self.assertTrue(all(ref_is_prime(p) for p in primes_only))
            self.assertEqual(product_of(f), n)                       # 乘积还原


# ---------- 对拍：与朴素参照实现比较 ----------

class TestDifferential(unittest.TestCase):
    def test_sieve_matches_reference_full_ranges(self):
        for lo, hi in ((0, 2000), (99990, 100010), (10 ** 6 - 500, 10 ** 6 + 500)):
            self.assertEqual(primes_in_range(lo, hi), ref_primes(lo, hi), (lo, hi))

    def test_sieve_boundary_alignment(self):
        # 段边界/奇偶边界对齐：跨段区间结果必须与整段一致
        full = primes_in_range(2, 200_000)
        self.assertEqual(full, ref_primes(2, 200_000))
        for lo, hi in ((2, 2), (3, 3), (0, 1), (4, 4), (2, 100), (99_999, 100_001)):
            self.assertEqual(primes_in_range(lo, hi),
                             [p for p in full if lo <= p <= hi])

    def test_simple_sieve_matches_reference(self):
        self.assertEqual(simple_sieve(50_000), ref_primes(0, 50_000))

    def test_factor_matches_reference_dense(self):
        for n in range(1, 5001):  # 稠密小区间全覆盖
            self.assertEqual(factor(n), ref_factor(n), n)

    def test_factor_matches_reference_random(self):
        rng = random.Random(7)
        primes = simple_sieve(isqrt(10 ** 12) + 1)  # 供 factor_with_primes 用
        for _ in range(300):
            n = rng.randint(1, 10 ** 12)
            expected = ref_factor(n)
            self.assertEqual(factor(n), expected, n)
            self.assertEqual(factor_with_primes(n, primes), expected, n)

    def test_factor_with_primes_matches_wheel(self):
        rng = random.Random(13)
        primes = simple_sieve(isqrt(10 ** 10) + 1)
        for _ in range(300):
            n = rng.randint(1, 10 ** 10)
            self.assertEqual(factor_with_primes(n, primes), factor(n), n)


# ---------- 耗时冒烟（精确基准见 bench.py） ----------

class TestTimingSmoke(unittest.TestCase):
    def test_timing_smoke(self):
        t0 = time.perf_counter()
        primes_in_range(2, 10 ** 7)
        t1 = time.perf_counter()
        for n in range(10 ** 6, 10 ** 6 + 10_000):
            factor(n)
        t2 = time.perf_counter()
        print(f"\n[timing] sieve(1e7)={t1 - t0:.3f}s  "
              f"factor x10000 around 1e6={t2 - t1:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=("-v" in sys.argv))
