"""数论库单元测试 / 边界用例（python3 -m unittest test_numtheory -v）。"""

import random
import unittest

from numtheory_pkg import (
    egcd,
    egcd_count,
    gcd,
    lcm,
    modinv,
    solve_linear_congruence,
    linear_congruence_solutions,
    crt,
    crt_representatives,
    InverseCache,
    NoInverseError,
    NoSolutionError,
    CongruenceConflictError,
)


class TestEGCD(unittest.TestCase):
    def test_basic_and_negative(self):
        for a, b in [(240, 46), (0, 7), (7, 0), (0, 0), (-240, 46),
                     (240, -46), (-240, -46), (1, 1), (-3, -5)]:
            g, x, y = egcd(a, b)
            self.assertEqual(g, gcd(a, b))
            self.assertGreaterEqual(g, 0)
            self.assertEqual(a * x + b * y, g)

    def test_iteration_count(self):
        # 相邻斐波那契数是欧几里得算法的最坏情形
        f0, f1 = 0, 1
        for _ in range(2000):
            f0, f1 = f1, f0 + f1
        g, x, y, n = egcd_count(f1, f0)
        self.assertEqual(g, 1)
        self.assertEqual(f1 * x + f0 * y, 1)
        self.assertGreater(n, 1500)
        self.assertLess(n, 2500)


class TestModInv(unittest.TestCase):
    def test_inverse_and_resubstitution(self):
        for a, m in [(3, 11), (-3, 11), (3, -11), (1, 1), (0, 1), (2, 9)]:
            inv = modinv(a, m)
            if abs(m) != 1:
                self.assertEqual((a * inv) % abs(m), 1)
            self.assertTrue(0 <= inv < abs(m))

    def test_modulus_zero(self):
        with self.assertRaises(NoInverseError):
            modinv(5, 0)

    def test_modulus_one(self):
        self.assertEqual(modinv(12345, 1), 0)
        self.assertEqual(modinv(12345, -1), 0)

    def test_no_inverse_raises_with_criterion(self):
        for a, m in [(6, 9), (0, 5), (-6, 9), (15, 5)]:
            with self.assertRaises(NoInverseError) as ctx:
                modinv(a, m)
            self.assertIn("gcd", str(ctx.exception))
            self.assertNotEqual(gcd(a, abs(m)), 1)

    def test_huge_integers(self):
        p = 10 ** 100 + 267  # 101 位大整数
        a = 3 ** 200
        inv = modinv(a, p)
        self.assertEqual((a * inv) % p, 1)


class TestLinearCongruence(unittest.TestCase):
    def test_multiple_solutions(self):
        # 6x ≡ 12 (mod 18)：g=6，应有 6 个模 18 的解
        sols = linear_congruence_solutions(6, 12, 18)
        self.assertEqual(len(sols), 6)
        self.assertEqual(sorted(sols), sols)
        for x in sols:
            self.assertEqual((6 * x - 12) % 18, 0)
        sol = solve_linear_congruence(6, 12, 18)
        self.assertEqual(sol.x0, 2)
        self.assertEqual(sol.modulus, 3)  # 通解 x ≡ 2 (mod 3)

    def test_no_solution(self):
        with self.assertRaises(NoSolutionError):
            solve_linear_congruence(6, 1, 18)  # gcd(6,18)=6 ∤ 1

    def test_negative_inputs(self):
        sols = linear_congruence_solutions(-6, -12, -18)
        self.assertEqual(sols, [2, 5, 8, 11, 14, 17])
        for x in sols:
            self.assertEqual((-6 * x - (-12)) % 18, 0)

    def test_modulus_one(self):
        # 任意 x 都满足；[0,1) 内唯一代表 0
        self.assertEqual(linear_congruence_solutions(7, 3, 1), [0])

    def test_modulus_zero_equality(self):
        self.assertEqual(solve_linear_congruence(2, 10, 0).x0, 5)
        with self.assertRaises(NoSolutionError):
            solve_linear_congruence(2, 9, 0)
        with self.assertRaises(NoSolutionError):
            solve_linear_congruence(0, 1, 0)
        sol = solve_linear_congruence(0, 0, 0)  # 任意整数
        self.assertTrue(sol.contains(123))

    def test_huge(self):
        a, b, m = 2 ** 512 + 3, 2 ** 256 - 1, 2 ** 1024 + 643
        sol = solve_linear_congruence(a, b, m)
        self.assertEqual((a * sol.x0 - b) % m, 0)


class TestCRT(unittest.TestCase):
    def test_coprime_classic(self):
        sol = crt([(2, 3), (3, 5), (2, 7)])
        self.assertEqual(sol.x0, 23)
        self.assertEqual(sol.modulus, 105)

    def test_non_coprime_solvable(self):
        # x ≡ 2 (mod 6), x ≡ 8 (mod 12)：相容，lcm=12，x ≡ 8 (mod 12)
        sol = crt([(2, 6), (8, 12)])
        self.assertEqual((sol.x0 - 2) % 6, 0)
        self.assertEqual((sol.x0 - 8) % 12, 0)
        self.assertEqual(sol.modulus, 12)
        self.assertEqual(sol.x0, 8)

    def test_non_coprime_conflict_reports_pair(self):
        # x ≡ 1 (mod 6) 与 x ≡ 0 (mod 4) 冲突：gcd(6,4)=2 ∤ (0-1)
        try:
            crt([(1, 6), (0, 4)])
        except CongruenceConflictError as e:
            self.assertEqual((e.index1, e.index2), (0, 1))
            self.assertEqual((e.modulus1, e.modulus2), (6, 4))
            self.assertEqual((e.residue1, e.residue2), (1, 0))
            self.assertEqual(e.common_divisor, 2)
        else:
            self.fail("应当抛出 CongruenceConflictError")

    def test_conflict_between_non_adjacent(self):
        # 前两条相容，第三条与第 0 条冲突，必须报出 (0, 2)
        try:
            crt([(2, 3), (3, 5), (4, 6)])
        except CongruenceConflictError as e:
            self.assertEqual((e.index1, e.index2), (0, 2))
            self.assertEqual(e.common_divisor, 3)
        else:
            self.fail("应当抛出 CongruenceConflictError")

    def test_negative_moduli_and_residues(self):
        sol = crt([(-2, -3), (3, -5)])  # 即 x ≡ 1 (mod 3), x ≡ 3 (mod 5)
        self.assertEqual(sol.x0, 13)

    def test_modulus_one_is_trivial(self):
        sol = crt([(0, 1), (3, 7), (99, -1)])
        self.assertEqual(sol.x0, 3)
        self.assertEqual(sol.modulus, 7)

    def test_modulus_zero_exact_value(self):
        sol = crt([(11, 0), (2, 3)])  # x = 11 且 x ≡ 2 (mod 3)
        self.assertEqual(sol.modulus, 0)
        self.assertEqual(sol.x0, 11)
        with self.assertRaises(CongruenceConflictError):
            crt([(11, 0), (0, 3)])  # 11 ≢ 0 (mod 3)
        with self.assertRaises(CongruenceConflictError):
            crt([(11, 0), (12, 0)])  # 两个不同精确值

    def test_representatives_inside_lcm(self):
        reps = crt_representatives([(2, 4), (0, 6)])  # lcm=12, x ≡ 6
        self.assertEqual(reps, [6])
        multi = crt_representatives([(2, 4), (0, 6)], copies=3)
        self.assertEqual(multi, [6, 18, 30])

    def test_huge_non_coprime(self):
        m1, m2 = 2 ** 512 * 3, 2 ** 512 * 5
        sol = crt([(2 ** 512 + 1, m1), (1, m2)])
        self.assertEqual((sol.x0 - (2 ** 512 + 1)) % m1, 0)
        self.assertEqual((sol.x0 - 1) % m2, 0)


class TestInverseCache(unittest.TestCase):
    def test_batch_matches_individual(self):
        m = 2 ** 61 - 1
        values = [2, 3, 5, 7, 11, 13, 2 ** 40, -17]
        cache = InverseCache(m)
        batch_res = cache.batch(values)
        for v, inv in zip(values, batch_res):
            self.assertEqual((v * inv) % m, 1)
            self.assertEqual(cache.inverse(v), inv)  # 缓存命中且一致

    def test_batch_noninvertible_element(self):
        cache = InverseCache(15)
        with self.assertRaises(NoInverseError) as ctx:
            cache.batch([2, 6, 7])  # gcd(6,15)=3，第 1 个
        self.assertIn("第 1", str(ctx.exception))

    def test_modulus_one(self):
        cache = InverseCache(1)
        self.assertEqual(cache.batch([0, 1, 2]), [0, 0, 0])

    def test_random_batch_roundtrip(self):
        random.seed(42)
        m = 10 ** 30 + 57
        values = [random.randrange(1, m) for _ in range(300)]
        cache = InverseCache(m)
        res = cache.batch(values)
        for v, inv in zip(values, res):
            self.assertEqual((v * inv) % m, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
