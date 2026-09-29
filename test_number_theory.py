"""Self-tests for number_theory.py.

Run with either:
    python3 -m unittest -v test_number_theory.py
    python3 test_number_theory.py
"""

from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from math import gcd

from number_theory import (
    LinearCongruenceSolution,
    NoInverseError,
    NoSolutionError,
    NegativeModulusError,
    NonIntegerInputError,
    ZeroModulusError,
    extended_gcd,
    mod_inverse,
    solve_linear_congruence,
)


CASE_FILE = Path(__file__).with_name("boundary_cases.json")


def load_cases() -> dict:
    with CASE_FILE.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    return {
        section: [
            {key: (int(value) if isinstance(value, str) and value.lstrip("-").isdigit() else value)
             for key, value in case.items()}
            for case in cases
        ]
        for section, cases in raw.items()
    }


CASES = load_cases()


class ExtendedGcdTests(unittest.TestCase):
    def test_boundary_cases(self) -> None:
        for case in CASES["extended_gcd"]:
            with self.subTest(case=case["name"]):
                g, x, y = extended_gcd(case["a"], case["b"])
                self.assertEqual(g, case["g"])
                self.assertEqual(x, case["x"])
                self.assertEqual(y, case["y"])
                self.assertGreaterEqual(g, 0)
                self.assertEqual(case["a"] * x + case["b"] * y, g)

    def test_random_bezout_identity(self) -> None:
        rng = random.Random(20260930)
        for _ in range(500):
            a = rng.randrange(-10_000, 10_000)
            b = rng.randrange(-10_000, 10_000)
            g, x, y = extended_gcd(a, b)
            self.assertEqual(a * x + b * y, g)
            self.assertGreaterEqual(g, 0)


class ModInverseTests(unittest.TestCase):
    def test_boundary_cases(self) -> None:
        for case in CASES["mod_inverse"]:
            with self.subTest(case=case["name"]):
                if "error" in case:
                    exception_type = {
                        "ZeroModulusError": ZeroModulusError,
                        "NoInverseError": NoInverseError,
                    }[case["error"]]
                    with self.assertRaises(exception_type):
                        mod_inverse(case["a"], case["m"])
                else:
                    inverse = mod_inverse(case["a"], case["m"])
                    self.assertEqual(inverse, case["inverse"])
                    modulus = abs(case["m"])
                    self.assertTrue(0 <= inverse < modulus)
                    self.assertEqual((case["a"] * inverse) % modulus, 1 % modulus)

    def test_no_inverse_reports_gcd_criterion(self) -> None:
        with self.assertRaisesRegex(NoInverseError, r"gcd\(6, 9\) = 3, not 1"):
            mod_inverse(6, 9)

    def test_negative_modulus_can_be_rejected(self) -> None:
        with self.assertRaises(NegativeModulusError):
            mod_inverse(2, -5, allow_negative_modulus=False)

    def test_random_inverses(self) -> None:
        rng = random.Random(918273)
        for _ in range(500):
            m = rng.randrange(1, 10_000)
            a = rng.randrange(-10_000, 10_000)
            if gcd(a, m) != 1:
                with self.assertRaises(NoInverseError):
                    mod_inverse(a, m)
                continue
            if m == 1:
                self.assertEqual(mod_inverse(a, m), 0)
                continue
            inverse = mod_inverse(a, m)
            self.assertTrue(0 <= inverse < m)
            self.assertEqual((a * inverse) % m, 1)


class LinearCongruenceTests(unittest.TestCase):
    def test_boundary_cases(self) -> None:
        for case in CASES["linear_congruence"]:
            with self.subTest(case=case["name"]):
                if "error" in case:
                    exception_type = {
                        "ZeroModulusError": ZeroModulusError,
                        "NoSolutionError": NoSolutionError,
                    }[case["error"]]
                    with self.assertRaises(exception_type):
                        solve_linear_congruence(case["a"], case["b"], case["m"])
                else:
                    result = solve_linear_congruence(case["a"], case["b"], case["m"])
                    self.assertEqual(
                        result,
                        LinearCongruenceSolution(
                            x0=case["x0"], step=case["step"], divisor=case["divisor"]
                        ),
                    )
                    m = abs(case["m"])
                    sample_count = min(result.divisor, 3)
                    for representative in result.representatives(sample_count):
                        self.assertTrue(0 <= representative < m)
                        self.assertEqual(
                            (case["a"] * representative - case["b"]) % m, 0
                        )
                    self.assertEqual(result.representative(0), result.x0)
                    if result.divisor > 1:
                        self.assertEqual(
                            result.representative(result.divisor - 1),
                            result.x0 + result.step * (result.divisor - 1),
                        )

    def test_no_solution_reports_divisibility_criterion(self) -> None:
        with self.assertRaisesRegex(
            NoSolutionError,
            r"gcd\(6, 15\) = 3 does not divide 10",
        ):
            solve_linear_congruence(6, 10, 15)

    def test_general_solution_text(self) -> None:
        partial = solve_linear_congruence(2, 4, 10)
        self.assertEqual(partial.general_solution(), "x ≡ 2 (mod 5); x = 2 + 5k, k ∈ Z")
        identity = solve_linear_congruence(7, 14, 7)
        self.assertEqual(
            identity.general_solution(),
            "x ≡ 0 (mod 1); every integer x is a solution",
        )

    def test_negative_modulus_equivalence_and_optional_rejection(self) -> None:
        positive = solve_linear_congruence(-3, 6, 9)
        negative = solve_linear_congruence(-3, 6, -9)
        self.assertEqual(positive, negative)
        with self.assertRaises(NegativeModulusError):
            solve_linear_congruence(-3, 6, -9, allow_negative_modulus=False)

    def test_random_solutions_by_enumeration(self) -> None:
        rng = random.Random(564738)
        for _ in range(500):
            m = rng.randrange(1, 80)
            a = rng.randrange(-80, 80)
            b = rng.randrange(-80, 80)
            expected = [x for x in range(m) if (a * x - b) % m == 0]
            if not expected:
                with self.assertRaises(NoSolutionError):
                    solve_linear_congruence(a, b, m)
                continue
            result = solve_linear_congruence(a, b, m)
            self.assertEqual(list(result.representatives(result.divisor)), expected)


class InputValidationTests(unittest.TestCase):
    def test_non_integers_are_rejected(self) -> None:
        invalid_values = ["1", 1.0, 1 + 0j, None]
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(NonIntegerInputError):
                    extended_gcd(value, 2)
                with self.assertRaises(NonIntegerInputError):
                    mod_inverse(value, 5)
                with self.assertRaises(NonIntegerInputError):
                    solve_linear_congruence(value, 1, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
