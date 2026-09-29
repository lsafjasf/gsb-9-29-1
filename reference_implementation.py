"""Independent reference implementation used only for differential tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        value = abs(a)
        if a < 0:
            return value, -1, 0
        return value, 1, 0
    quotient, remainder = divmod(a, b)
    common, coefficient_b, coefficient_r = extended_gcd(b, remainder)
    return common, coefficient_r, coefficient_b - quotient * coefficient_r


def mod_inverse(a: int, m: int) -> int:
    if m == 0:
        raise ValueError("ZERO_MODULUS")
    m = abs(m)
    try:
        return pow(a, -1, m)
    except ValueError as exc:
        raise ValueError("NO_INVERSE") from exc


@dataclass(frozen=True)
class ReferenceSolution:
    x0: int
    step: int
    divisor: int


def solve_linear_congruence(a: int, b: int, m: int) -> ReferenceSolution:
    if m == 0:
        raise ValueError("ZERO_MODULUS")
    m = abs(m)
    divisor = gcd(a, m)
    if b % divisor != 0:
        raise ValueError("NO_SOLUTION")

    reduced_a = a // divisor
    reduced_b = b // divisor
    reduced_m = m // divisor
    if reduced_m == 1:
        x0 = 0
    else:
        x0 = (pow(reduced_a, -1, reduced_m) * reduced_b) % reduced_m
    return ReferenceSolution(x0=x0, step=reduced_m, divisor=divisor)
