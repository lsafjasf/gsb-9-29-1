"""Iterative solvers for 2D steady-state fields (Poisson equation).

Solves  -laplacian(u) = f  on the unit square (0, 1)^2 with Dirichlet
boundary conditions u = g, discretised with the standard 5-point
finite-difference stencil on a uniform grid with ``n`` interior nodes
per direction (mesh size ``h = 1 / (n + 1)``).

Provided methods
----------------
``jacobi``        basic Jacobi iteration
``gauss_seidel``  basic Gauss-Seidel iteration
``sor``           successive over-relaxation (acceleration method 1)
``multigrid``     geometric multigrid V-cycle (acceleration method 2)

Convergence is decided purely by the residual: the iteration stops when
``||r||_2 <= tol * ||r0||_2`` with ``r = f - A u`` and ``r0`` the initial
residual.  Divergence (e.g. SOR with ``omega >= 2``) is detected and
reported instead of iterating forever.  The relative residual of every
iteration is recorded in the returned :class:`SolveResult`.

Only the Python 3 standard library is used.
"""

import math
import time

__all__ = [
    "CONVERGED", "DIVERGED", "MAX_ITER_REACHED",
    "SolveResult", "new_problem", "residual_norms", "optimal_sor_omega",
    "max_error", "solve",
]

CONVERGED = "converged"
DIVERGED = "diverged"
MAX_ITER_REACHED = "max_iter_reached"

#: residual growth factor taken as proof of divergence
DIVERGENCE_FACTOR = 1e10


class SolveResult:
    """Outcome of :func:`solve`."""

    __slots__ = ("status", "iterations", "rel_residuals",
                 "initial_residual", "wall_time")

    def __init__(self, status, iterations, rel_residuals,
                 initial_residual, wall_time):
        self.status = status                      # CONVERGED / DIVERGED / MAX_ITER_REACHED
        self.iterations = iterations              # number of sweeps / V-cycles performed
        self.rel_residuals = rel_residuals        # ||r_k|| / ||r_0|| after every iteration
        self.initial_residual = initial_residual  # absolute L2 norm of the initial residual
        self.wall_time = wall_time                # seconds

    @property
    def final_relative_residual(self):
        return self.rel_residuals[-1]

    @property
    def final_residual(self):
        """Absolute L2 norm of the final residual."""
        return self.rel_residuals[-1] * self.initial_residual

    def __repr__(self):
        return ("SolveResult(status=%r, iterations=%d, final_rel_residual=%.3e)"
                % (self.status, self.iterations, self.final_relative_residual))


def new_problem(n, f=0.0, g=0.0):
    """Allocate the unknown vector ``u`` and right-hand side ``fv``.

    Both are flat row-major lists over the full ``(n+2) x (n+2)`` grid
    (boundary nodes included); node ``(i, j)`` sits at index ``i*(n+2)+j``
    and at physical coordinates ``(x, y) = (j*h, i*h)``.  ``u`` is
    initialised to zero in the interior and to ``g`` on the boundary.
    ``f`` and ``g`` may be scalars or callables ``f(x, y)`` / ``g(x, y)``.

    Returns ``(u, fv, h)``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    h = 1.0 / (n + 1)
    w = n + 2
    u = [0.0] * (w * w)
    fv = [0.0] * (w * w)
    f_is_fn = callable(f)
    g_is_fn = callable(g)
    for i in range(1, n + 1):
        y = i * h
        base = i * w
        for j in range(1, n + 1):
            fv[base + j] = f(j * h, y) if f_is_fn else f
    for k in range(n + 2):
        t = k * h
        u[k] = g(t, 0.0) if g_is_fn else g              # bottom, y = 0
        u[(n + 1) * w + k] = g(t, 1.0) if g_is_fn else g  # top, y = 1
        u[k * w] = g(0.0, t) if g_is_fn else g          # left, x = 0
        u[k * w + n + 1] = g(1.0, t) if g_is_fn else g  # right, x = 1
    return u, fv, h


def residual_norms(u, fv, n, h):
    """Return ``(L2, Linf)`` norms of the residual ``r = f - A u``.

    The discrete operator is ``(A u)_ij = (4 u_ij - sum of 4 neighbours)
    / h^2``; boundary nodes of ``u`` contribute to the residual of their
    interior neighbours, so Dirichlet data is accounted for exactly.
    """
    w = n + 2
    inv_h2 = 1.0 / (h * h)
    sum2 = 0.0
    rmax = 0.0
    for i in range(1, n + 1):
        base = i * w
        for j in range(1, n + 1):
            k = base + j
            r = fv[k] - (4.0 * u[k] - u[k - 1] - u[k + 1]
                         - u[k - w] - u[k + w]) * inv_h2
            sum2 += r * r
            a = r if r >= 0.0 else -r
            if a > rmax:
                rmax = a
    return math.sqrt(sum2), rmax


def optimal_sor_omega(n):
    """Theoretically optimal SOR relaxation factor for the model problem.

    For the 5-point Laplacian on a square with ``n`` interior nodes per
    direction, the Jacobi spectral radius is ``rho = cos(pi*h)`` with
    ``h = 1/(n+1)``, and the optimal SOR parameter is

        omega* = 2 / (1 + sqrt(1 - rho^2)) = 2 / (1 + sin(pi*h)).

    (Young's theorem; see e.g. Briggs, "A Multigrid Tutorial".)  It lies
    in ``(1, 2)`` and approaches 2 as the grid is refined.
    """
    h = 1.0 / (n + 1)
    return 2.0 / (1.0 + math.sin(math.pi * h))


def max_error(u, n, exact):
    """Max-norm error against an exact solution ``exact(x, y)``."""
    h = 1.0 / (n + 1)
    w = n + 2
    err = 0.0
    for i in range(1, n + 1):
        y = i * h
        base = i * w
        for j in range(1, n + 1):
            e = abs(u[base + j] - exact(j * h, y))
            if e > err:
                err = e
    return err


# ---------------------------------------------------------------- sweeps

def _sor_sweep(u, fv, n, h2, omega):
    """One in-place Gauss-Seidel (omega=1) / SOR sweep over the interior."""
    w = n + 2
    for i in range(1, n + 1):
        base = i * w
        for j in range(1, n + 1):
            k = base + j
            u_gs = 0.25 * (u[k - 1] + u[k + 1] + u[k - w] + u[k + w]
                           + h2 * fv[k])
            u[k] += omega * (u_gs - u[k])


def _jacobi_sweep(u, fv, n, h2, omega=1.0):
    """One (optionally weighted) Jacobi sweep over the interior."""
    w = n + 2
    old = u[:]
    for i in range(1, n + 1):
        base = i * w
        for j in range(1, n + 1):
            k = base + j
            u_gs = 0.25 * (old[k - 1] + old[k + 1] + old[k - w] + old[k + w]
                           + h2 * fv[k])
            u[k] += omega * (u_gs - old[k])


# ------------------------------------------------------------- multigrid

def _restrict_full_weighting(r, n, nc):
    """Full-weighting restriction of the residual ``r = f - A u``.

    Returns the coarse right-hand side ``fv_c = R r`` of the coarse
    correction equation ``A_2h e = R (f - A u)``.
    """
    w = n + 2
    wc = nc + 2
    fc = [0.0] * (wc * wc)
    for ic in range(1, nc + 1):
        fi = 2 * ic
        base_c = ic * wc
        base_f = fi * w
        for jc in range(1, nc + 1):
            k = base_f + 2 * jc
            fc[base_c + jc] = 0.0625 * (
                4.0 * r[k]
                + 2.0 * (r[k - 1] + r[k + 1] + r[k - w] + r[k + w])
                + r[k - w - 1] + r[k - w + 1]
                + r[k + w - 1] + r[k + w + 1])
    return fc


def _prolongate_and_add(u, ec, n, nc):
    """Bilinear prolongation of the coarse correction, added into ``u``."""
    w = n + 2
    wc = nc + 2
    for i in range(1, n + 1):
        ic = i >> 1
        i_even = (i & 1) == 0
        base = i * w
        row0 = ic * wc
        row1 = (ic + 1) * wc
        for j in range(1, n + 1):
            jc = j >> 1
            if i_even:
                if (j & 1) == 0:
                    v = ec[row0 + jc]
                else:
                    v = 0.5 * (ec[row0 + jc] + ec[row0 + jc + 1])
            else:
                if (j & 1) == 0:
                    v = 0.5 * (ec[row0 + jc] + ec[row1 + jc])
                else:
                    v = 0.25 * (ec[row0 + jc] + ec[row0 + jc + 1]
                                + ec[row1 + jc] + ec[row1 + jc + 1])
            u[base + j] += v


def _v_cycle(u, fv, n, h, nu1, nu2, smooth_weight):
    """One recursive V-cycle.  ``n`` must be ``2**k - 1``."""
    if n <= 3:
        # coarsest grid: solve (nearly) exactly with many GS sweeps
        for _ in range(50):
            _sor_sweep(u, fv, n, h * h, 1.0)
        return
    h2 = h * h
    w = n + 2
    inv_h2 = 1.0 / h2
    for _ in range(nu1):
        _jacobi_sweep(u, fv, n, h2, smooth_weight)
    # residual r = f - A u
    r = [0.0] * (w * w)
    for i in range(1, n + 1):
        base = i * w
        for j in range(1, n + 1):
            k = base + j
            r[k] = fv[k] - (4.0 * u[k] - u[k - 1] - u[k + 1]
                            - u[k - w] - u[k + w]) * inv_h2
    nc = (n - 1) // 2
    fc = _restrict_full_weighting(r, n, nc)
    ec = [0.0] * ((nc + 2) * (nc + 2))
    _v_cycle(ec, fc, nc, 2.0 * h, nu1, nu2, smooth_weight)
    _prolongate_and_add(u, ec, n, nc)
    for _ in range(nu2):
        _jacobi_sweep(u, fv, n, h2, smooth_weight)


# ---------------------------------------------------------------- driver

def solve(u, fv, n, method="sor", omega=None, tol=1e-8, max_iter=100000,
          nu1=2, nu2=2, smooth_weight=0.8):
    """Iterate ``method`` until the residual-based criterion is met.

    ``u`` (modified in place) and ``fv`` come from :func:`new_problem`.
    Stops when ``||r||_2 <= tol * ||r0||_2`` (status ``CONVERGED``), when
    the residual grows by :data:`DIVERGENCE_FACTOR` or becomes non-finite
    (status ``DIVERGED``), or when ``max_iter`` is exhausted (status
    ``MAX_ITER_REACHED``).  Never uses the iteration count as a success
    criterion.  Returns a :class:`SolveResult`.

    ``omega`` defaults to :func:`optimal_sor_omega` for ``method="sor"``.
    ``multigrid`` requires ``n = 2**k - 1`` so grids can be coarsened
    recursively; ``nu1``/``nu2`` are pre/post smoothing steps of weighted
    Jacobi with weight ``smooth_weight`` (0.8 is the standard choice for
    the 2D 5-point Laplacian).
    """
    if method not in ("jacobi", "gauss_seidel", "sor", "multigrid"):
        raise ValueError("unknown method: %r" % method)
    if method == "multigrid" and (n + 1) & n:
        raise ValueError("multigrid requires n = 2**k - 1, got n=%d" % n)
    if method == "sor" and omega is None:
        omega = optimal_sor_omega(n)

    h = 1.0 / (n + 1)
    h2 = h * h
    r0, _ = residual_norms(u, fv, n, h)
    if r0 == 0.0:
        return SolveResult(CONVERGED, 0, [0.0], 0.0, 0.0)

    rel_residuals = []
    status = MAX_ITER_REACHED
    iterations = 0
    t0 = time.perf_counter()
    while iterations < max_iter:
        iterations += 1
        if method == "jacobi":
            _jacobi_sweep(u, fv, n, h2, 1.0)
        elif method == "gauss_seidel":
            _sor_sweep(u, fv, n, h2, 1.0)
        elif method == "sor":
            _sor_sweep(u, fv, n, h2, omega)
        else:
            _v_cycle(u, fv, n, h, nu1, nu2, smooth_weight)
        r, _ = residual_norms(u, fv, n, h)
        rel = r / r0
        rel_residuals.append(rel)
        if not math.isfinite(r) or r > DIVERGENCE_FACTOR * r0:
            status = DIVERGED
            break
        if rel <= tol:
            status = CONVERGED
            break
    wall = time.perf_counter() - t0
    return SolveResult(status, iterations, rel_residuals, r0, wall)
