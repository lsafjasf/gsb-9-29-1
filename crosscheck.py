"""Differential tests against reference_implementation.py.

The reference implementation intentionally uses a different extended GCD
formulation and Python's built-in ``pow(a, -1, m)``. For small moduli this
script also enumerates every residue to verify the complete solution set.

Usage:
    python3 crosscheck.py
    python3 crosscheck.py --iterations 5000 --seed 20260930 --max-bits 4096
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import number_theory as candidate
import reference_implementation as reference


CASE_FILE = Path(__file__).with_name("boundary_cases.json")


def random_integer(rng: random.Random, max_bits: int) -> int:
    kind = rng.randrange(10)
    if kind == 0:
        return rng.choice([-2, -1, 0, 1, 2])
    value = rng.getrandbits(rng.randrange(1, max_bits + 1))
    if kind % 2 == 0:
        value = -value
    return value


def random_modulus(rng: random.Random, max_bits: int) -> int:
    kind = rng.randrange(12)
    if kind == 0:
        return rng.choice([0, 1, -1])
    value = rng.getrandbits(rng.randrange(1, max_bits + 1))
    if value == 0:
        value = 1
    return -value if kind % 2 == 0 else value


def error_key(exc: Exception) -> str:
    if isinstance(exc, candidate.ZeroModulusError):
        return "ZERO_MODULUS"
    if isinstance(exc, candidate.NoInverseError):
        return "NO_INVERSE"
    if isinstance(exc, candidate.NoSolutionError):
        return "NO_SOLUTION"
    return "OTHER_ERROR"


def reference_error_key(exc: Exception) -> str:
    text = str(exc)
    if text in {"ZERO_MODULUS", "NO_INVERSE", "NO_SOLUTION"}:
        return text
    return "OTHER_ERROR"


def assert_inverse_identity(a: int, m: int, inverse: int) -> None:
    modulus = abs(m)
    if not 0 <= inverse < modulus:
        raise AssertionError(f"inverse {inverse} not in [0, {modulus - 1})")
    if (a * inverse) % modulus != 1 % modulus:
        raise AssertionError(f"{a} * {inverse} != 1 (mod {modulus})")


def assert_complete_solution(a: int, b: int, m: int, result: object) -> None:
    modulus = abs(m)
    divisor = candidate.extended_gcd(a, modulus)[0]
    if divisor == 0:
        raise AssertionError("internal test error: modulus must not be zero")
    if result.divisor != divisor:
        raise AssertionError(f"divisor {result.divisor} != gcd {divisor}")
    if result.step != modulus // divisor:
        raise AssertionError("step is not modulus / gcd")
    if not 0 <= result.x0 < result.step:
        raise AssertionError("x0 is not the least non-negative reduced solution")
    for offset in (-2, -1, 0, 1, 2):
        x = result.x0 + result.step * offset
        if (a * x - b) % modulus != 0:
            raise AssertionError(f"general solution failed at x={x}")
    if result.step > 1 and (a * result.x0 - b) % modulus != 0:
        raise AssertionError("x0 does not satisfy the congruence")


def assert_small_enumeration(a: int, b: int, m: int) -> None:
    modulus = abs(m)
    if modulus > 300:
        return
    expected = [x for x in range(modulus) if (a * x - b) % modulus == 0]
    if not expected:
        try:
            candidate.solve_linear_congruence(a, b, m)
        except candidate.NoSolutionError:
            return
        raise AssertionError("candidate returned a solution despite empty residue set")
    result = candidate.solve_linear_congruence(a, b, m)
    actual = list(result.representatives(result.divisor))
    if actual != expected:
        raise AssertionError(f"{actual} != enumerated {expected}")


def compare_extended_gcd(a: int, b: int) -> None:
    actual = candidate.extended_gcd(a, b)
    expected = reference.extended_gcd(a, b)
    if actual != expected:
        raise AssertionError(f"extended_gcd({a}, {b}): {actual} != {expected}")
    g, x, y = actual
    if g < 0 or a * x + b * y != g:
        raise AssertionError(f"invalid Bézout identity for ({a}, {b})")


def compare_inverse(a: int, m: int) -> None:
    try:
        actual = candidate.mod_inverse(a, m)
        actual_key = "OK"
    except Exception as exc:
        actual_error = error_key(exc)
        actual_key = actual_error

    try:
        expected = reference.mod_inverse(a, m)
        expected_key = "OK"
    except Exception as exc:
        expected_error = reference_error_key(exc)
        expected_key = expected_error

    if actual_key != expected_key:
        raise AssertionError(
            f"mod_inverse({a}, {m}): candidate={actual_key}, reference={expected_key}"
        )
    if actual_key == "OK":
        if actual != expected:
            raise AssertionError(f"mod_inverse({a}, {m}): {actual} != {expected}")
        assert_inverse_identity(a, m, actual)


def compare_congruence(a: int, b: int, m: int) -> None:
    try:
        actual = candidate.solve_linear_congruence(a, b, m)
        actual_tuple = (actual.x0, actual.step, actual.divisor)
        actual_key = "OK"
    except Exception as exc:
        actual_error = error_key(exc)
        actual_tuple = None
        actual_key = actual_error

    try:
        expected = reference.solve_linear_congruence(a, b, m)
        expected_tuple = (expected.x0, expected.step, expected.divisor)
        expected_key = "OK"
    except Exception as exc:
        expected_error = reference_error_key(exc)
        expected_tuple = None
        expected_key = expected_error

    if actual_key != expected_key or actual_tuple != expected_tuple:
        raise AssertionError(
            f"solve({a}, {b}, {m}): "
            f"candidate=({actual_key}, {actual_tuple}), "
            f"reference=({expected_key}, {expected_tuple})"
        )
    if actual_key == "OK":
        assert_complete_solution(a, b, m, actual)
        if m < 0:
            positive = candidate.solve_linear_congruence(a, b, -m)
            if (positive.x0, positive.step, positive.divisor) != actual_tuple:
                raise AssertionError("negative modulus differs from positive modulus")
        assert_small_enumeration(a, b, m)


def decode_boundary_cases() -> list[dict]:
    with CASE_FILE.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    decoded: list[dict] = []
    for section, cases in raw.items():
        for case in cases:
            item = {"section": section}
            for key, value in case.items():
                if isinstance(value, str) and value.lstrip("-").isdigit():
                    item[key] = int(value)
                else:
                    item[key] = value
            decoded.append(item)
    return decoded


def run_boundary_cases() -> int:
    count = 0
    for case in decode_boundary_cases():
        if case["section"] == "extended_gcd":
            compare_extended_gcd(case["a"], case["b"])
        elif case["section"] == "mod_inverse":
            compare_inverse(case["a"], case["m"])
        else:
            compare_congruence(case["a"], case["b"], case["m"])
        count += 1
    return count


def run(iterations: int, seed: int, max_bits: int) -> None:
    sys.setrecursionlimit(max(100_000, max_bits * 10))
    rng = random.Random(seed)
    checked = run_boundary_cases()

    fixed_triples = [
        (0, 0, 1),
        (0, 0, -1),
        (0, 1, 1),
        (1, 0, 1),
        (-1, -1, -1),
        (6, 10, 15),
        (-6, -10, -15),
    ]
    for a, b, m in fixed_triples:
        compare_extended_gcd(a, b)
        compare_inverse(a, m)
        compare_congruence(a, b, m)
        checked += 1

    for index in range(iterations):
        a = random_integer(rng, max_bits)
        b = random_integer(rng, max_bits)
        m = random_modulus(rng, max_bits)
        compare_extended_gcd(a, b)
        compare_inverse(a, m)
        compare_congruence(a, b, m)
        if index % 1000 == 0:
            print(f"checked {index + 1}/{iterations} random cases")
        checked += 1

    print(f"OK: {checked} boundary/fixed cases plus {iterations} random cases matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-check number theory implementations")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-bits", type=int, default=512)
    args = parser.parse_args()
    if args.iterations < 0 or args.max_bits < 1:
        parser.error("iterations must be >= 0 and max-bits must be >= 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.iterations, arguments.seed, arguments.max_bits)
