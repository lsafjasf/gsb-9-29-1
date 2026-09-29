"""Symmetric eigenvalue solver using only the Python standard library.

Algorithm
---------
1. Householder reduction of the real symmetric matrix A to symmetric
   tridiagonal form T = Q^T A Q (orthogonal similarity, exactly backward
   stable).
2. Implicit-shift QL iteration on T (orthogonal iteration with a
   Wilkinson-type shift; mathematically equivalent to shifted QR).
   Subdiagonal elements are driven to zero and the matrix is deflated.
3. Eigenvectors of A are Q * (eigenvectors of T); both factors are
   orthogonal, so the computed eigenvectors stay orthonormal.

Multiple / clustered eigenvalues
--------------------------------
Orthogonal iteration cannot (and need not) distinguish basis vectors
inside a degenerate eigenspace: for an eigenvalue of multiplicity k the
iteration deflates the cluster one eigenvalue at a time and returns an
*arbitrary orthonormal basis* of the k-dimensional invariant subspace.
Individual eigenvectors of a cluster are therefore not comparable to a
reference basis vector-by-vector; correctness inside a cluster must be
measured by the subspace projection error
    ||v - P_ref v||_2,   P_ref = projector onto the reference subspace.
Eigenvalues of a cluster still converge to the (multiple) eigenvalue
itself, at the same rate as simple eigenvalues.

Convergence criterion
---------------------
During QL iteration the subdiagonal element e[m] is treated as
negligible (and the problem is split / deflated) when

    |e[m]| <= tol * (|d[m]| + |d[m+1]|)

i.e. relative to the magnitude of the neighbouring diagonal elements.
With tol = 1e-14 this gives residuals at the level of a few units of
machine epsilon times ||A||.  A fallback guard (|e| below
machine-epsilon * ||T||_F) prevents infinite loops on
pathological inputs.

A cyclic Jacobi eigensolver is included as an *independent reference
implementation* for cross-checking (对拍) in the test-suite.
"""

import math

__all__ = [
    "eigen",
    "jacobi_eigen",
    "residuals",
    "orthogonality_error",
    "identity",
    "matmul",
    "matvec",
    "transpose",
    "frobenius",
]

_EPS = 2.220446049250313e-16  # machine epsilon for IEEE-754 double


# --------------------------------------------------------------------------
# small dense-matrix helpers (lists of lists)
# --------------------------------------------------------------------------

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(ra, cb)) for cb in bt] for ra in a]


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def frobenius(a):
    return math.sqrt(sum(v * v for row in a for v in row))


def _dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def _norm(x):
    return math.sqrt(_dot(x, x))


def _check_symmetric(a):
    n = len(a)
    if n == 0 or any(len(row) != n for row in a):
        raise ValueError("matrix must be non-empty and square")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-10 * max(1.0, abs(a[i][j]), abs(a[j][i])):
                raise ValueError("matrix is not symmetric (A[%d][%d] != A[%d][%d])"
                                 % (i, j, j, i))


# --------------------------------------------------------------------------
# Householder tridiagonalization
# --------------------------------------------------------------------------

def _tridiagonalize(a):
    """Reduce symmetric A to tridiagonal T = Q^T A Q.  Returns (T, Q)."""
    n = len(a)
    t = [row[:] for row in a]
    q = identity(n)
    for k in range(n - 2):
        m = n - k - 1
        x = [t[k + 1 + i][k] for i in range(m)]
        normx = _norm(x)
        if normx == 0.0:
            continue
        # v = x + sign(x0)*||x|| e1  (sign chosen to avoid cancellation)
        v = x[:]
        v[0] += math.copysign(normx, x[0])
        tau = 2.0 / _dot(v, v)
        # symmetric rank-2 update of the trailing block:
        # T' = (I - tau v v^T) T (I - tau v v^T)
        #    = T - v w^T - w v^T,  w = tau*T*v - (tau^2/2)(v^T T v) v
        tv = [sum(t[k + 1 + i][k + 1 + j] * v[j] for j in range(m))
              for i in range(m)]
        vtv = _dot(v, tv)
        w = [tau * tv[i] - 0.5 * tau * tau * vtv * v[i] for i in range(m)]
        for i in range(m):
            vi, wi = v[i], w[i]
            row = t[k + 1 + i]
            for j in range(m):
                row[k + 1 + j] -= vi * w[j] + wi * v[j]
        # exact zeros in the annihilated column/row
        t[k + 1][k] = t[k][k + 1] = -math.copysign(normx, x[0])
        for i in range(k + 2, n):
            t[i][k] = t[k][i] = 0.0
        # accumulate Q <- Q (I - tau v v^T)
        for i in range(n):
            qi = q[i]
            f = tau * sum(qi[k + 1 + j] * v[j] for j in range(m))
            for j in range(m):
                qi[k + 1 + j] -= f * v[j]
    return t, q


# --------------------------------------------------------------------------
# implicit-shift QL iteration on a symmetric tridiagonal matrix
# --------------------------------------------------------------------------

def _ql_tridiagonal(d, e, z, tol, max_iter):
    """Implicit-shift QL on tridiagonal (d, e); eigenvectors acculated in z.

    d : diagonal, length n (modified in place -> eigenvalues)
    e : subdiagonal, length n-1 (modified)
    z : orthogonal matrix; columns accumulate the eigenvectors
    """
    n = len(d)
    e = e[:] + [0.0]
    guard = _EPS * (math.sqrt(sum(v * v for v in d)
                            + 2.0 * sum(v * v for v in e)) or 1.0)
    for l in range(n):
        iteration = 0
        while True:
            # look for a single small subdiagonal element to split the matrix
            m = l
            while m < n - 1:
                scale = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= tol * scale or abs(e[m]) <= guard:
                    break
                m += 1
            if m == l:
                break  # eigenvalue d[l] has converged
            iteration += 1
            if iteration > max_iter:
                raise ArithmeticError(
                    "QL iteration failed to converge after %d iterations "
                    "(deflating row %d)" % (max_iter, l))
            # form the shift (Wilkinson-type, as in classic tqli)
            g = (d[l + 1] - d[l]) / (2.0 * e[l])
            r = math.hypot(g, 1.0)
            g = d[m] - d[l] + e[l] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0
            recovered = False
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                r = math.hypot(f, g)
                e[i + 1] = r
                if r == 0.0:
                    # recover from underflow
                    d[i + 1] -= p
                    e[m] = 0.0
                    recovered = True
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                # accumulate the plane rotation into the eigenvector matrix
                for k in range(n):
                    f = z[k][i + 1]
                    z[k][i + 1] = s * z[k][i] + c * f
                    z[k][i] = c * z[k][i] - s * f
            if recovered:
                continue
            d[l] -= p
            e[l] = g
            e[m] = 0.0
    return d


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def eigen(a, tol=1e-14, max_iter=200):
    """Compute all eigenvalues and orthonormal eigenvectors of a real
    symmetric matrix.

    Returns (values, vectors):
      values  : list of eigenvalues sorted in ascending order
      vectors : list of eigenvectors, vectors[i] belongs to values[i];
                the vectors are orthonormal (V^T V = I)
    """
    _check_symmetric(a)
    n = len(a)
    if n == 1:
        return [a[0][0]], [[1.0]]
    t, q = _tridiagonalize(a)
    d = [t[i][i] for i in range(n)]
    e = [t[i][i + 1] for i in range(n - 1)]
    _ql_tridiagonal(d, e, q, tol, max_iter)
    order = sorted(range(n), key=lambda i: d[i])
    values = [d[i] for i in order]
    vectors = [[q[row][i] for row in range(n)] for i in order]
    return values, vectors


def jacobi_eigen(a, tol=1e-14, max_sweeps=100):
    """Independent reference implementation: cyclic Jacobi rotations.

    Same return convention as :func:`eigen`.  Used only for
    cross-checking (对拍) in the test-suite.
    """
    _check_symmetric(a)
    n = len(a)
    if n == 1:
        return [a[0][0]], [[1.0]]
    t = [row[:] for row in a]
    v = identity(n)
    norm_a = frobenius(a) or 1.0
    for _sweep in range(max_sweeps):
        off = math.sqrt(sum(t[i][j] ** 2 for i in range(n)
                            for j in range(n) if i != j))
        if off <= tol * norm_a:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = t[p][q]
                if abs(apq) <= 1e-300 * norm_a:
                    continue
                theta = (t[q][q] - t[p][p]) / (2.0 * apq)
                if theta == 0.0:
                    tt = 1.0
                else:
                    tt = math.copysign(1.0, theta) / (abs(theta)
                                                      + math.hypot(theta, 1.0))
                c = 1.0 / math.sqrt(1.0 + tt * tt)
                s = tt * c
                for k in range(n):
                    tkp, tkq = t[k][p], t[k][q]
                    t[k][p] = c * tkp - s * tkq
                    t[k][q] = s * tkp + c * tkq
                for k in range(n):
                    tpk, tqk = t[p][k], t[q][k]
                    t[p][k] = c * tpk - s * tqk
                    t[q][k] = s * tpk + c * tqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    else:
        raise ArithmeticError("Jacobi iteration failed to converge")
    order = sorted(range(n), key=lambda i: t[i][i])
    values = [t[i][i] for i in order]
    vectors = [[v[row][i] for row in range(n)] for i in order]
    return values, vectors


# --------------------------------------------------------------------------
# quality metrics
# --------------------------------------------------------------------------

def residuals(a, values, vectors):
    """Return list of ||A v_i - lambda_i v_i||_2 for each eigenpair."""
    out = []
    for lam, vec in zip(values, vectors):
        av = matvec(a, vec)
        out.append(_norm([x - lam * y for x, y in zip(av, vec)]))
    return out


def orthogonality_error(vectors):
    """Return max |(V^T V - I)_ij| over all i, j."""
    n = len(vectors)
    worst = 0.0
    for i in range(n):
        for j in range(n):
            target = 1.0 if i == j else 0.0
            worst = max(worst, abs(_dot(vectors[i], vectors[j]) - target))
    return worst
