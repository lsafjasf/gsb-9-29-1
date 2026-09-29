"""稳态场迭代求解库（纯标准库）。"""

from .solvers import (
    SolveResult,
    apply_boundary,
    residual_norm,
    optimal_omega,
    jacobi,
    gauss_seidel,
    sor,
    multigrid,
)
from .problems import (
    build_f,
    max_error,
    max_diff,
    sine_problem,
    uniform_boundary_problem,
    point_source_problem,
)
