"""数论运算库：扩展欧几里得、模逆、线性同余、非互素同余组合并求解。"""

from .numtheory import (
    egcd,
    egcd_count,
    gcd,
    lcm,
    modinv,
    solve_linear_congruence,
    linear_congruence_solutions,
    crt,
    crt_representatives,
    CongruenceSolution,
    InverseCache,
    NoInverseError,
    NoSolutionError,
    CongruenceConflictError,
)

__all__ = [
    "egcd",
    "egcd_count",
    "gcd",
    "lcm",
    "modinv",
    "solve_linear_congruence",
    "linear_congruence_solutions",
    "crt",
    "crt_representatives",
    "CongruenceSolution",
    "InverseCache",
    "NoInverseError",
    "NoSolutionError",
    "CongruenceConflictError",
]
