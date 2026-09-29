"""Number-theory helpers for modular inverses and linear congruences.

Only Python's standard library is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


class NumberTheoryError(ValueError):
    """Base class for invalid number-theory inputs."""


class NonIntegerInputError(NumberTheoryError):
    """Raised when an argument is not an integer."""


class ZeroModulusError(NumberTheoryError):
    """Raised when a modular operation receives modulus zero."""


class NegativeModulusError(NumberTheoryError):
    """Raised when negative moduli are explicitly disabled."""


class NoInverseError(NumberTheoryError):
    """Raised when gcd(a, m) is not one."""


class NoSolutionError(NumberTheoryError):
    """Raised when gcd(a, m) does not divide the right-hand side."""


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NonIntegerInputError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def canonical_modulus(m: object) -> int:
    """Return the accepted non-negative modulus.

    Negative moduli are converted to ``abs(m)``. Zero is rejected because
    residues are not defined modulo zero.
    """

    value = _require_integer(m, "m")
    if value == 0:
        raise ZeroModulusError("modulus m must not be zero")
    return abs(value)


def extended_gcd(a: object, b: object) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``g = gcd(a, b) = a*x + b*y``.

    ``g`` is always non-negative. With one input zero and the other negative,
    the returned Bézout coefficient carries the sign, for example
    ``extended_gcd(0, -3) == (3, 0, -1)``.
    """

    left = _require_integer(a, "a")
    right = _require_integer(b, "b")

    old_r, r = left, right
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def mod_inverse(a: object, m: object, *, allow_negative_modulus: bool = True) -> int:
    """Return the least non-negative inverse of ``a`` modulo ``m``.

    Raises ``NoInverseError`` exactly when ``gcd(a, m) != 1``. By convention,
    the inverse modulo ``1`` is ``0``; it is the unique residue and
    ``a * 0 ≡ 1 (mod 1)``. Negative moduli are treated as ``abs(m)``.
    """

    value = _require_integer(a, "a")
    modulus = _require_integer(m, "m")
    if modulus == 0:
        raise ZeroModulusError("modulus m must not be zero")
    if modulus < 0 and not allow_negative_modulus:
        raise NegativeModulusError("negative modulus is disabled")
    modulus = abs(modulus)

    gcd, coefficient, _ = extended_gcd(value, modulus)
    if gcd != 1:
        raise NoInverseError(
            f"inverse does not exist because gcd({value}, {modulus}) = {gcd}, not 1"
        )
    return coefficient % modulus


@dataclass(frozen=True)
class LinearCongruenceSolution:
    """All integer solutions of ``a*x ≡ b (mod m)``.

    The solutions are ``x = x0 + step*k`` for every integer ``k``. If
    ``step == 1``, every integer is a solution. There are ``divisor`` distinct
    residue classes modulo the original modulus.
    """

    x0: int
    step: int
    divisor: int

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.divisor <= 0:
            raise ValueError("divisor must be positive")
        if not (0 <= self.x0 < self.step):
            raise ValueError("x0 must be the least non-negative representative")

    @property
    def all_integers(self) -> bool:
        """True when the congruence is an identity."""

        return self.step == 1

    @property
    def residue_count(self) -> int:
        """Number of distinct residue classes modulo the original modulus."""

        return self.divisor

    def general_solution(self) -> str:
        """Return the parametric form as a readable string."""

        if self.all_integers:
            return "x ≡ 0 (mod 1); every integer x is a solution"
        return f"x ≡ {self.x0} (mod {self.step}); x = {self.x0} + {self.step}k, k ∈ Z"

    def representative(self, index: object) -> int:
        """Return one of the least non-negative residue representatives.

        Valid indices are ``0`` through ``divisor - 1``.
        """

        index_value = _require_integer(index, "index")
        if not 0 <= index_value < self.divisor:
            raise ValueError("index must satisfy 0 <= index < divisor")
        return self.x0 + self.step * index_value

    def representatives(self, count: object) -> Iterator[int]:
        """Yield up to ``count`` least non-negative representatives."""

        count_value = _require_integer(count, "count")
        if count_value < 0:
            raise ValueError("count must be non-negative")
        for index in range(min(count_value, self.divisor)):
            yield self.representative(index)


def solve_linear_congruence(
    a: object,
    b: object,
    m: object,
    *,
    allow_negative_modulus: bool = True,
) -> LinearCongruenceSolution:
    """Solve ``a*x ≡ b (mod m)``.

    Let ``d = gcd(a, m)``. A solution exists iff ``d`` divides ``b``.

    On success, ``x0`` is the least non-negative solution of the reduced
    congruence and ``step`` is ``abs(m)/d``. The complete solution is
    ``x = x0 + step*k`` for all integer ``k``. There are ``d`` residue
    classes modulo ``abs(m)``. Negative moduli are treated as ``abs(m)``;
    modulus zero is rejected.
    """

    coefficient = _require_integer(a, "a")
    rhs = _require_integer(b, "b")
    modulus = _require_integer(m, "m")
    if modulus == 0:
        raise ZeroModulusError("modulus m must not be zero")
    if modulus < 0 and not allow_negative_modulus:
        raise NegativeModulusError("negative modulus is disabled")
    modulus = abs(modulus)

    divisor, a_factor, _ = extended_gcd(coefficient, modulus)
    if rhs % divisor != 0:
        raise NoSolutionError(
            f"no solution because gcd({coefficient}, {modulus}) = {divisor} "
            f"does not divide {rhs}"
        )

    reduced_m = modulus // divisor
    reduced_b = rhs // divisor
    if reduced_m == 1:
        x0 = 0
    else:
        x0 = (a_factor * reduced_b) % reduced_m
    return LinearCongruenceSolution(x0=x0, step=reduced_m, divisor=divisor)
