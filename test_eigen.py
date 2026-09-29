"""Self-test / cross-check (对拍) harness for eigen.py.

For every test case we
  * solve with the QL orthogonal-iteration solver (eigen.eigen),
  * solve independently with cyclic Jacobi (eigen.jacobi_eigen),
  * compare eigenvalues against the analytic ground truth (when known)
    and against the Jacobi reference,
  * measure eigenvector angles vs. the reference for *separated*
    eigenvalues, and subspace projection errors for *clusters*
    (multiple / nearly-multiple eigenvalues),
  * assert orthonormality of the eigenvector basis and small residuals.

Run:  python3 test_eigen.py
"""

import math
import random

import eigen
from eigen import (eigen as qr_eigen, jacobi_eigen, residuals,
                   orthogonality_error, matmul, transpose, identity,
                   frobenius)

CLUSTER_RTOL = 1e-8   # eigenvalues closer than this form a cluster
ASSERT_TOL = {
    "eig_scaled": 1e-10,   # max |d_lambda| / max(1, ||A||_F)
    "residual_scaled": 1e-10,  # max ||Av-lv|| / max(1, ||A||_F)
    "orthogonality": 1e-10,    # max |V^T V - I|
    # eigenvector checks use condition-aware tolerances:
    #   angle/subspace error <= 200 * eps * ||A||_F / spectral_gap
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def norm(x):
    return math.sqrt(dot(x, x))


def random_orthogonal(n, seed):
    """Random orthogonal matrix via modified Gram-Schmidt (seeded)."""
    rng = random.Random(seed)
    cols = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    basis = []
    for j in range(n):
        v = [cols[i][j] for i in range(n)]
        for u in basis:
            r = dot(v, u)
            v = [a - r * b for a, b in zip(v, u)]
        nv = norm(v)
        basis.append([a / nv for a in v])
    return [[basis[j][i] for j in range(n)] for i in range(n)]


def from_spectrum(diag, seed):
    """Build A = Q diag(diag) Q^T with random orthogonal Q.

    Returns (A, sorted_exact_eigenvalues, exact_eigenvectors_sorted).
    """
    n = len(diag)
    q = random_orthogonal(n, seed)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = float(diag[i])
    a = matmul(matmul(q, d), transpose(q))
    order = sorted(range(n), key=lambda i: diag[i])
    vals = [float(diag[i]) for i in order]
    vecs = [[q[row][i] for row in range(n)] for i in order]
    return a, vals, vecs


def angle_deg(u, v):
    """Angle between u and v in degrees, insensitive to the sign of the
    vectors.  Uses atan2(||v - c u||, |c|) with c = <u,v> so that very
    small angles (~1e-11 deg) stay resolvable in double precision
    (acos(1 - eps) would round to exactly 0)."""
    un, vn = norm(u), norm(v)
    c = dot(u, v) / (un * vn)
    s = norm([(vi - c * ui) / vn for ui, vi in zip(u, v)])
    return math.degrees(math.atan2(s, abs(c)))


def subspace_error(vectors, ref_basis):
    """max ||v - P_ref v||_2 over vectors, P_ref = projector on ref_basis."""
    worst = 0.0
    for v in vectors:
        proj = [0.0] * len(v)
        for u in ref_basis:
            c = dot(v, u)
            proj = [p + c * ui for p, ui in zip(proj, u)]
        worst = max(worst, norm([a - b for a, b in zip(v, proj)]))
    return worst


def clusters(values):
    """Group sorted eigenvalue indices into clusters (gaps > CLUSTER_RTOL)."""
    groups = [[0]]
    for i in range(1, len(values)):
        gap = abs(values[i] - values[i - 1])
        if gap <= CLUSTER_RTOL * max(1.0, abs(values[i]), abs(values[i - 1])):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


# --------------------------------------------------------------------------
# test-case construction
# --------------------------------------------------------------------------

def build_cases():
    cases = []

    def add(name, a, exact_vals=None, exact_vecs=None):
        cases.append(dict(name=name, a=a, exact_vals=exact_vals,
                          exact_vecs=exact_vecs))

    # 1. tiny sizes
    add("1x1", [[-3.5]], [-3.5], [[1.0]])
    add("2x2", [[2.0, 1.0], [1.0, 2.0]], [1.0, 3.0], None)

    # 2. identity: eigenvalue 1 with full multiplicity n
    add("identity I5 (lambda=1, mult 5)", identity(5),
        [1.0] * 5, [list(r) for r in identity(5)])

    # 3. zero matrix: eigenvalue 0 with full multiplicity
    add("zero 4x4 (lambda=0, mult 4)", [[0.0] * 4 for _ in range(4)],
        [0.0] * 4, None)

    # 4. repeated roots (multiplicities 3 and 2), randomly rotated
    a, v, q = from_spectrum([3.0, 2.0, 2.0, 2.0, -1.0, -1.0], seed=11)
    add("repeated roots {3,2x3,-1x2}, rotated", a, v, q)

    # 5. negative + repeated spectrum
    a, v, q = from_spectrum([-5.0, -2.0, -2.0, 1.0, 4.0], seed=12)
    add("indefinite with double root {-5,-2x2,1,4}", a, v, q)

    # 6. near-singular: eigenvalues down to 1e-15 (cond ~ 1e15)
    a, v, q = from_spectrum([1.0, 1e-8, 1e-12, 1e-15], seed=13)
    add("near-singular {1,1e-8,1e-12,1e-15}", a, v, q)

    # 7. extreme scale differences: 1e-9 .. 1e8 (17 orders of magnitude)
    a, v, q = from_spectrum([1e8, 1e3, 1.0, 1e-5, 1e-9], seed=14)
    add("huge scale gap {1e8,1e3,1,1e-5,1e-9}", a, v, q)

    # 8. clustered (not exactly equal) eigenvalues
    a, v, q = from_spectrum([1.0, 1.0 + 1e-9, 1.0 - 1e-9, 5.0], seed=15)
    add("tight cluster {1,1+1e-9,1-1e-9,5}", a, v, q)

    # 9. random symmetric 8x8 (no analytic answer -> Jacobi is reference)
    rng = random.Random(16)
    n = 8
    r = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    a = [[r[i][j] + r[j][i] for j in range(n)] for i in range(n)]
    add("random symmetric 8x8", a)

    # 10. Hilbert matrix 6x6 (ill-conditioned, cond ~ 1.5e7)
    n = 6
    a = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
    add("Hilbert 6x6", a)

    return cases


# --------------------------------------------------------------------------
# per-case driver
# --------------------------------------------------------------------------

def run_case(case):
    a = case["a"]
    n = len(a)
    norm_a = max(1.0, frobenius(a))

    vals, vecs = qr_eigen(a)
    jvals, jvecs = jacobi_eigen(a)

    res = residuals(a, vals, vecs)
    max_res = max(res)
    orth = orthogonality_error(vecs)

    # eigenvalue errors
    err_jacobi = max(abs(x - y) for x, y in zip(vals, jvals))
    if case["exact_vals"] is not None:
        err_exact = max(abs(x - y) for x, y in zip(vals, case["exact_vals"]))
    else:
        err_exact = None

    # eigenvector comparison: angle for separated eigenvalues,
    # subspace projection error for clusters.  Tolerances are
    # condition-aware: a symmetric eigenvector/subspace is only
    # determined up to ~ eps*||A||/gap, where gap is the distance of
    # the eigenvalue (cluster) to the rest of the spectrum.
    eps = 2.220446049250313e-16
    cond_factor = 200.0 * eps * norm_a
    groups = clusters(vals)
    max_angle = 0.0
    max_sub = 0.0
    worst_vec_ratio = 0.0   # max( error / condition-aware tolerance )
    for g in groups:
        if len(g) == 1:
            i = g[0]
            if n == 1:
                continue
            gap = min(abs(vals[i] - vals[j]) for j in range(n) if j != i)
            tol_rad = max(1e-12, cond_factor / gap)
            refs = [jvecs[i]]
            if case["exact_vecs"] is not None:
                refs.append(case["exact_vecs"][i])
            for ref in refs:
                ang = math.radians(angle_deg(vecs[i], ref))
                max_angle = max(max_angle, math.degrees(ang))
                worst_vec_ratio = max(worst_vec_ratio, ang / tol_rad)
        else:
            outside = [j for j in range(n) if j not in g]
            if outside:
                gap = min(abs(vals[i] - vals[j]) for i in g for j in outside)
                tol_sub = max(1e-12, cond_factor / gap)
            else:
                tol_sub = 1e-12  # whole spectrum is one cluster (I, 0, ...)
            ref = [jvecs[i] for i in g]
            got = [vecs[i] for i in g]
            err = max(subspace_error(got, ref), subspace_error(ref, got))
            if case["exact_vecs"] is not None:
                exref = [case["exact_vecs"][i] for i in g]
                err = max(err, subspace_error(got, exref),
                          subspace_error(exref, got))
            max_sub = max(max_sub, err)
            worst_vec_ratio = max(worst_vec_ratio, err / tol_sub)

    # ---- report ----
    print("-" * 74)
    print("case: %s   (n=%d, ||A||_F=%.3e)" % (case["name"], n, norm_a))
    print("  eigenvalues        : %s" % ", ".join("%.6e" % v for v in vals))
    if err_exact is not None:
        print("  max |dlam| vs exact : %.3e  (scaled %.3e)"
              % (err_exact, err_exact / norm_a))
    print("  max |dlam| vs jacobi: %.3e  (scaled %.3e)"
          % (err_jacobi, err_jacobi / norm_a))
    print("  max residual ||Av-lv||: %.3e  (scaled %.3e)"
          % (max_res, max_res / norm_a))
    print("  orthogonality |V^TV-I|_max: %.3e" % orth)
    print("  max vector angle (separated): %.3e deg" % max_angle)
    print("  max cluster subspace error  : %.3e" % max_sub)
    print("  vector error / condition-aware tolerance: %.3e (must be < 1)"
          % worst_vec_ratio)

    # ---- assertions ----
    assert err_jacobi / norm_a < ASSERT_TOL["eig_scaled"], "eig vs jacobi"
    if err_exact is not None:
        assert err_exact / norm_a < ASSERT_TOL["eig_scaled"], "eig vs exact"
    assert max_res / norm_a < ASSERT_TOL["residual_scaled"], "residual"
    assert orth < ASSERT_TOL["orthogonality"], "orthogonality"
    assert worst_vec_ratio < 1.0, "eigenvector / subspace accuracy"
    return True


def main():
    print("=" * 74)
    print("eigen.py self-test: QL orthogonal iteration vs Jacobi reference")
    print("=" * 74)
    cases = build_cases()
    for case in cases:
        run_case(case)
    print("-" * 74)
    print("ALL %d CASES PASSED  (assert tolerances: %s)"
          % (len(cases), ASSERT_TOL))


if __name__ == "__main__":
    main()
