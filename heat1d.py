"""One-dimensional heat equation solvers using only the Python standard library.

The equation is ``u_t = alpha * u_xx`` on ``[0, length]``.

Each boundary can be:
* ``"fixed"``: Dirichlet boundary temperature;
* ``"insulated"``: zero-flux Neumann boundary.

The grid has ``nx + 1`` nodes, ``dx = length / nx``.  The discrete energy
``dx * (u[0]/2 + sum(u[1:-1]) + u[-1]/2)`` is exactly conserved when both
boundaries are insulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, isfinite, log, pi, sin
from typing import Callable, Sequence, Union


Number = Union[int, float]
BoundaryValue = Union[Number, Callable[[float], float]]


@dataclass(frozen=True)
class HeatProblem:
    """Discrete heat-equation problem and its result after integration."""

    x: list[float]
    u: list[float]
    dx: float
    dt: float
    alpha: float
    time: float
    steps: int
    scheme: str
    boundaries: tuple[str, str]


def _boundary_value(value: BoundaryValue, time: float) -> float:
    if callable(value):
        return float(value(time))
    return float(value)


def _normalise_boundaries(
    boundaries: Sequence[str], fixed_temps: Sequence[BoundaryValue]
) -> tuple[tuple[str, str], tuple[BoundaryValue, BoundaryValue]]:
    if len(boundaries) != 2:
        raise ValueError("boundaries must contain exactly two entries")
    if len(fixed_temps) != 2:
        raise ValueError("fixed_temps must contain exactly two entries")

    normalised = []
    for boundary in boundaries:
        name = boundary.lower()
        if name in ("fixed", "dirichlet", "d"):
            normalised.append("fixed")
        elif name in ("insulated", "neumann", "n"):
            normalised.append("insulated")
        else:
            raise ValueError(f"unknown boundary type: {boundary!r}")
    return tuple(normalised), tuple(fixed_temps)  # type: ignore[return-value]


def make_grid(length: Number, nx: int) -> tuple[list[float], float]:
    """Return node coordinates and spacing."""
    if length <= 0:
        raise ValueError("length must be positive")
    if nx < 2:
        raise ValueError("nx must be at least 2")
    dx = float(length) / nx
    return [i * dx for i in range(nx + 1)], dx


def initial_from_function(
    length: Number, nx: int, func: Callable[[float], float]
) -> tuple[list[float], list[float], float]:
    x, dx = make_grid(length, nx)
    return x, [float(func(value)) for value in x], dx


def energy(u: Sequence[float], dx: Number) -> float:
    """Trapezoidal discrete heat content on a node grid."""
    if len(u) < 2:
        raise ValueError("at least two values are required")
    return float(dx) * (0.5 * u[0] + sum(u[1:-1]) + 0.5 * u[-1])


def assert_energy_conserved(
    initial: Sequence[float],
    final: Sequence[float],
    dx: Number,
    label: str = "",
    tolerance: float = 1e-12,
) -> tuple[float, float, float]:
    """Assert equality of trapezoidal heat content before and after evolution."""
    initial_energy = energy(initial, dx)
    final_energy = energy(final, dx)
    difference = final_energy - initial_energy
    scale = max(1.0, abs(initial_energy), abs(final_energy))
    if abs(difference) > tolerance * scale:
        raise AssertionError(
            f"{label + ': ' if label else ''}discrete energy changed by {difference!r}: "
            f"{initial_energy!r} -> {final_energy!r}"
        )
    return initial_energy, final_energy, difference


def solve_heat1d(
    initial: Sequence[float],
    dx: Number,
    dt: Number,
    alpha: Number,
    steps: int,
    scheme: str = "explicit",
    boundaries: Sequence[str] = ("insulated", "insulated"),
    fixed_temps: Sequence[BoundaryValue] = (0.0, 0.0),
) -> HeatProblem:
    """Integrate the heat equation.

    ``explicit`` is forward-time/centred-space.  It refuses to execute when
    ``alpha * dt / dx**2 > 0.5``.  ``implicit`` is backward Euler/BTCS and is
    unconditionally stable for nonnegative ``alpha``.
    """
    dx = float(dx)
    dt = float(dt)
    alpha = float(alpha)
    if dx <= 0:
        raise ValueError("dx must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if len(initial) < 3:
        raise ValueError("at least three grid nodes are required")

    kind = scheme.lower()
    if kind not in ("explicit", "implicit"):
        raise ValueError("scheme must be 'explicit' or 'implicit'")
    boundary_pair, temp_pair = _normalise_boundaries(boundaries, fixed_temps)

    r = alpha * dt / (dx * dx)
    if kind == "explicit" and r > 0.5:
        raise StabilityError(
            f"explicit FTCS is unstable: r={r:.12g} > 1/2; "
            "reduce dt or dx, or use the implicit scheme"
        )

    u = [float(value) for value in initial]
    x = [i * dx for i in range(len(u))]
    if kind == "explicit":
        current_time = _advance_explicit(u, dx, dt, r, steps, boundary_pair, temp_pair)
    else:
        current_time = _advance_implicit(u, dx, dt, r, steps, boundary_pair, temp_pair)

    return HeatProblem(
        x=x,
        u=u,
        dx=dx,
        dt=dt,
        alpha=alpha,
        time=current_time,
        steps=steps,
        scheme=kind,
        boundaries=boundary_pair,
    )


class StabilityError(ValueError):
    """Raised when the explicit scheme violates its FTCS stability bound."""


def _apply_fixed_boundaries(
    u: list[float], time: float, boundaries: tuple[str, str], temps: tuple
) -> None:
    if boundaries[0] == "fixed":
        u[0] = _boundary_value(temps[0], time)
    if boundaries[1] == "fixed":
        u[-1] = _boundary_value(temps[1], time)


def _advance_explicit(
    u: list[float],
    dx: float,
    dt: float,
    r: float,
    steps: int,
    boundaries: tuple[str, str],
    temps: tuple,
) -> float:
    current_time = 0.0
    _apply_fixed_boundaries(u, current_time, boundaries, temps)
    for _ in range(steps):
        old = u[:]
        for i in range(1, len(u) - 1):
            u[i] = old[i] + r * (old[i - 1] - 2.0 * old[i] + old[i + 1])

        if boundaries[0] == "insulated":
            u[0] = old[0] + 2.0 * r * (old[1] - old[0])
        if boundaries[1] == "insulated":
            u[-1] = old[-1] + 2.0 * r * (old[-2] - old[-1])

        current_time += dt
        _apply_fixed_boundaries(u, current_time, boundaries, temps)
    return current_time


def _advance_implicit(
    u: list[float],
    dx: float,
    dt: float,
    r: float,
    steps: int,
    boundaries: tuple[str, str],
    temps: tuple,
) -> float:
    n = len(u)
    lower = [0.0] * n
    diag = [1.0] * n
    upper = [0.0] * n

    for i in range(1, n - 1):
        lower[i] = -r
        diag[i] = 1.0 + 2.0 * r
        upper[i] = -r

    if boundaries[0] == "insulated":
        diag[0] = 1.0 + 2.0 * r
        upper[0] = -2.0 * r
    if boundaries[1] == "insulated":
        diag[-1] = 1.0 + 2.0 * r
        lower[-1] = -2.0 * r

    current_time = 0.0
    _apply_fixed_boundaries(u, current_time, boundaries, temps)
    for _ in range(steps):
        current_time += dt
        rhs = u[:]

        if boundaries[0] == "fixed":
            left_temp = _boundary_value(temps[0], current_time)
            diag[0] = 1.0
            upper[0] = 0.0
            lower[0] = 0.0
            rhs[0] = left_temp
        if boundaries[1] == "fixed":
            right_temp = _boundary_value(temps[1], current_time)
            diag[-1] = 1.0
            upper[-1] = 0.0
            lower[-1] = 0.0
            rhs[-1] = right_temp

        u[:] = solve_tridiagonal(lower, diag, upper, rhs)
    return current_time


def solve_tridiagonal(
    lower: Sequence[float],
    diag: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Thomas algorithm for a general tridiagonal linear system."""
    n = len(rhs)
    if not (len(lower) == len(diag) == len(upper) == n):
        raise ValueError("tridiagonal arrays must have the same length")

    cp = [0.0] * n
    dp = [0.0] * n
    if abs(diag[0]) <= 0.0:
        raise ValueError("singular tridiagonal system")
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denominator = diag[i] - lower[i] * cp[i - 1]
        if abs(denominator) <= 1e-300:
            raise ValueError("singular tridiagonal system")
        cp[i] = upper[i] / denominator if i < n - 1 else 0.0
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denominator

    result = [0.0] * n
    result[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        result[i] = dp[i] - cp[i] * result[i + 1]
    return result


def sine_dirichlet_solution(
    x: Sequence[float], time: Number, alpha: Number, length: Number
) -> list[float]:
    """Analytic solution for u0=sin(pi*x/L), u(0)=u(L)=0."""
    decay = exp(-alpha * (pi / length) ** 2 * time)
    return [decay * sin(pi * value / length) for value in x]


def cosine_neumann_solution(
    x: Sequence[float], time: Number, alpha: Number, length: Number
) -> list[float]:
    """Analytic solution for u0=cos(pi*x/L), u_x(0)=u_x(L)=0."""
    decay = exp(-alpha * (pi / length) ** 2 * time)
    return [decay * cos(pi * value / length) for value in x]


def mixed_cosine_solution(
    x: Sequence[float], time: Number, alpha: Number, length: Number
) -> list[float]:
    """Analytic solution for u_x(0)=0 and u(L)=0, with u0=cos(pi*x/(2L))."""
    decay = exp(-alpha * (pi / (2.0 * length)) ** 2 * time)
    return [decay * cos(pi * value / (2.0 * length)) for value in x]


def max_error(actual: Sequence[float], expected: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(actual, expected))


def rms_error(actual: Sequence[float], expected: Sequence[float], dx: Number) -> float:
    squared = [(a - b) ** 2 for a, b in zip(actual, expected)]
    return (energy(squared, dx) / (dx * (len(actual) - 1))) ** 0.5


def observed_order(errors: Sequence[float], refinements: int = 2) -> list[float]:
    """Log2 error ratios when each refinement halves dx and dt scales by 4."""
    orders = []
    for coarse, fine in zip(errors, errors[1:]):
        if coarse <= 0 or fine <= 0:
            orders.append(float("nan"))
        else:
            orders.append(log(coarse / fine) / log(refinements))
    return orders


def assert_finite(result: HeatProblem) -> None:
    if not all(isfinite(value) for value in result.u):
        raise AssertionError("solution contains a non-finite value")
