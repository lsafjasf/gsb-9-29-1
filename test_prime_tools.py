"""
test_prime_tools.py —— 断言自测 + 与朴素参照实现对拍

运行：python3 -m unittest -v test_prime_tools
（或直接 python3 test_prime_tools.py）
"""

import random
import unittest
from math import isqrt

import prime_tools as pt
import reference as ref


def is_prime_simple(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True


class TestFactorize(unittest.TestCase):
    def assert_complete(self, n: int) -> None:
        """核心不变量：有序、底数为素数、指数为正、乘积严格等于原数。"""
        fac = pt.factorize(n)
        ps = [p for p, _ in fac]
        self.assertEqual(ps, sorted(ps), f"{n}: 因子未按升序排列")
        self.assertEqual(len(ps), len(set(ps)), f"{n}: 存在重复因子")
        for p, e in fac:
            self.assertGreaterEqual(p, 2)
            self.assertIsInstance(p, int)
            self.assertTrue(is_prime_simple(p), f"{n}: {p} 不是素数")
            self.assertGreaterEqual(e, 1)
        # 关键断言：分解乘积必须等于原数
        self.assertEqual(pt.factors_product(fac), n, f"{n}: 乘积不等于原数")

    def test_boundary_0_1(self) -> None:
        self.assertEqual(pt.factorize(0), [])
        self.assertEqual(pt.factorize(1), [])
        self.assertEqual(pt.factors_product([]), 1)
        self.assertEqual(pt.factorize(-12), pt.factorize(12))  # 负数按绝对值

    def test_small_exhaustive(self) -> None:
        for n in range(2, 20_000):
            self.assert_complete(n)
            self.assertEqual(pt.factorize(n), ref.ref_factorize(n), n)

    def test_primes(self) -> None:
        for p in (2, 3, 5, 7, 13, 97, 999983):
            self.assertEqual(pt.factorize(p), [(p, 1)])

    def test_perfect_squares(self) -> None:
        # 完全平方：指数都应为偶数
        for p, e in ((2, 10), (3, 4), (7, 2), (13, 2)):
            n = p ** e
            self.assertEqual(pt.factorize(n), [(p, e)])
        self.assertEqual(pt.factorize(12 ** 2), pt.factorize(144))
        self.assert_complete(999_983 ** 2)      # 大素数(≈10^6)的完全平方 ≈10^12
        # 30^2 = 2^2 * 3^2 * 5^2，覆盖轮式的三个固定因子
        self.assertEqual(pt.factorize(900), [(2, 2), (3, 2), (5, 2)])

    def test_carmichael_and_powers(self) -> None:
        self.assertEqual(pt.factorize(561), [(3, 1), (11, 1), (17, 1)])   # 卡迈克尔数
        self.assertEqual(pt.factorize(2 ** 40), [(2, 40)])
        self.assertEqual(pt.factorize(2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29),
                         [(2, 1), (3, 1), (5, 1), (7, 1), (11, 1), (13, 1),
                          (17, 1), (19, 1), (23, 1), (29, 1)])

    def test_large_numbers(self) -> None:
        rng = random.Random(20260930)
        # 取若干 1e6 附近的素数，拼成最多约 1e12 的合数
        small_primes = list(pt.primes_in_range(100_000, 1_000_000))
        for _ in range(25):
            p, q = rng.sample(small_primes, 2)
            self.assert_complete(p * q)
            self.assertEqual(pt.factorize(p * q), ref.ref_factorize(p * q))
        # 大素数本身（无因子，需试除到 sqrt）
        big = list(pt.primes_in_range(10 ** 11 - 200, 10 ** 11))[-1]
        self.assertTrue(is_prime_simple(big))
        self.assertEqual(pt.factorize(big), [(big, 1)])
        # 大素数 * 小素数
        self.assert_complete(big * 1_000_003)
        # 随机 12 位以内整数对拍
        for _ in range(25):
            n = rng.randrange(2, 10 ** 12)
            self.assert_complete(n)
            self.assertEqual(pt.factorize(n), ref.ref_factorize(n))


class TestSieve(unittest.TestCase):
    def test_primes_up_to_known(self) -> None:
        self.assertEqual(list(pt.primes_up_to(0)), [])
        self.assertEqual(list(pt.primes_up_to(1)), [])
        self.assertEqual(list(pt.primes_up_to(2)), [2])
        self.assertEqual(list(pt.primes_up_to(10)), [2, 3, 5, 7])
        self.assertEqual(list(pt.primes_up_to(19)), [2, 3, 5, 7, 11, 13, 17, 19])

    def test_against_reference_small_ranges(self) -> None:
        for high in (2, 3, 4, 10, 31, 100, 997, 5000):
            self.assertEqual(list(pt.primes_up_to(high)), ref.ref_primes(0, high + 1))

    def test_arbitrary_ranges(self) -> None:
        # 各种不贴齐边界的区间，且用很小的 chunk 强制多分几段
        bounds = [
            (0, 2), (1, 3), (2, 4), (8, 30), (10, 20), (11, 19),
            (1_000_000 - 100, 1_000_000 + 100),
            (999_983, 1_000_033),
            (2, 10_000),
        ]
        for low, high in bounds:
            got = list(pt.primes_in_range(low, high, chunk=37))
            want = ref.ref_primes(low, high)
            self.assertEqual(got, want, f"range [{low},{high}) mismatch")

    def test_range_consistency_with_full(self) -> None:
        # 区间拼接结果 == 整表结果；并测试多个 chunk 取值
        n = 20_000
        full = list(pt.primes_up_to(n))
        for chunk in (1, 2, 7, 1000, 1_000_000):
            merged: list[int] = []
            cuts = list(range(0, n + 1, 333)) + [n + 1]
            for a, b in zip(cuts, cuts[1:]):
                merged.extend(pt.primes_in_range(a, b, chunk=chunk))
            self.assertEqual(merged, full, f"chunk={chunk}")

    def test_large_range_crosscheck(self) -> None:
        # 大区间不与朴素 O(n sqrt) 对拍（太慢），改为逐数确定性 Miller 之外的
        # 充分性校验：筛出的每个数都通过朴素素性判定（抽样），且计数等于
        # 已知的 pi(10^6)=78498。
        primes = list(pt.primes_up_to(1_000_000))
        self.assertEqual(len(primes), 78_498)
        self.assertEqual(primes[0], 2)
        self.assertEqual(primes[-1], 999_983)
        rng = random.Random(7)
        for p in rng.sample(primes, 200):
            self.assertTrue(is_prime_simple(p))
        # 区间版与整表版在重叠范围上完全一致
        a, b = 500_000, 520_000
        self.assertEqual(
            list(pt.primes_in_range(a, b)),
            [p for p in primes if a <= p < b],
        )

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            list(pt.primes_in_range(0, 10, chunk=0))
        with self.assertRaises(TypeError):
            pt.factorize("12")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
