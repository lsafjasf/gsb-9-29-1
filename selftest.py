"""CG 库自测：前提检查 + 边界用例 + 与直接解对拍。仅标准库 unittest。

运行: python3 selftest.py [-v]
"""

import math
import random
import unittest

from cglib import (
    CGResult,
    IdentityPreconditioner,
    JacobiPreconditioner,
    NonSymmetricError,
    NotPositiveDefiniteError,
    SparseMatrix,
    SSORPreconditioner,
    pcg,
)
from direct import solve_dense


def random_spd(n, seed=42, cond_exp=2):
    """构造随机 SPD 矩阵 A = Q D Q^T，特征值在 [1, 10^cond_exp] 对数均匀。"""
    rng = random.Random(seed)
    # 随机矩阵 Gram-Schmidt 得正交 Q
    cols = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    q = []
    for k in range(n):
        v = cols[k][:]
        for u in q:
            d = sum(a * b for a, b in zip(v, u))
            v = [a - d * b for a, b in zip(v, u)]
        nv = math.sqrt(sum(a * a for a in v))
        q.append([a / nv for a in v])
    eig = [10.0 ** (cond_exp * i / (n - 1)) for i in range(n)]
    dense = [[sum(q[k][i] * eig[k] * q[k][j] for k in range(n))
              for j in range(n)] for i in range(n)]
    return SparseMatrix.from_dense(dense), dense


class TestPremiseChecks(unittest.TestCase):
    """前提检查：不满足对称/正定时必须明确报错。"""

    def test_zero_matrix_raises(self):
        """零矩阵：非正定，b != 0 时第一次迭代即被 p^T A p <= 0 检出。"""
        A = SparseMatrix(4)
        with self.assertRaises(NotPositiveDefiniteError):
            pcg(A, [1.0, 2.0, 3.0, 4.0])

    def test_zero_matrix_jacobi_rejected(self):
        """零矩阵对角元为 0，Jacobi 预条件构造阶段即拒绝。"""
        with self.assertRaises(NotPositiveDefiniteError):
            JacobiPreconditioner(SparseMatrix(3))

    def test_nonsymmetric_raises(self):
        A = SparseMatrix.from_dense([[2.0, 1.0],
                                     [0.5, 2.0]])  # A[0][1] != A[1][0]
        with self.assertRaises(NonSymmetricError):
            pcg(A, [1.0, 1.0])

    def test_indefinite_raises(self):
        """不定矩阵 diag(1, -1, 2)：迭代中 p^T A p <= 0 检出。"""
        A = SparseMatrix.from_dense([[1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, 2.0]])
        with self.assertRaises(NotPositiveDefiniteError):
            pcg(A, [1.0, 1.0, 1.0])

    def test_negative_diagonal_jacobi_rejected(self):
        A = SparseMatrix.from_dense([[1.0, 0.0], [0.0, -2.0]])
        with self.assertRaises(NotPositiveDefiniteError):
            JacobiPreconditioner(A)

    def test_dimension_mismatch_raises(self):
        A = SparseMatrix.from_dense([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            pcg(A, [1.0, 2.0, 3.0])


class TestEdgeCases(unittest.TestCase):
    """边界用例：单位阵、零右端、病态矩阵、迭代上限耗尽。"""

    def test_identity_converges_in_one_iteration(self):
        n = 50
        A = SparseMatrix.from_dense(
            [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])
        b = [float(i + 1) for i in range(n)]
        res = pcg(A, b)
        self.assertTrue(res.converged)
        self.assertLessEqual(res.iterations, 1)
        for xi, bi in zip(res.x, b):
            self.assertAlmostEqual(xi, bi, places=12)

    def test_zero_rhs_gives_zero_solution(self):
        A, _ = random_spd(8, seed=1)
        res = pcg(A, [0.0] * 8)
        self.assertTrue(res.converged)
        self.assertEqual(res.iterations, 0)
        self.assertTrue(all(x == 0.0 for x in res.x))

    def test_illconditioned_preconditioner_speedup(self):
        """病态矩阵（对角尺度 1e-8..1e8）：Jacobi 应显著减少迭代次数。"""
        n = 40
        rng = random.Random(7)
        # A = S K S，K 为良态 SPD（三对角 2,-1 加小扰动），S 对角尺度悬殊
        scale = [10.0 ** rng.uniform(-3, 3) for _ in range(n)]
        dense = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dense[i][i] = 2.0
            if i > 0:
                dense[i][i - 1] = dense[i - 1][i] = -1.0
        dense = [[dense[i][j] * scale[i] * scale[j]
                  for j in range(n)] for i in range(n)]
        A = SparseMatrix.from_dense(dense)
        b = [rng.uniform(-1, 1) for _ in range(n)]

        plain = pcg(A, b, precond=IdentityPreconditioner(),
                    tol=1e-8, max_iter=10000)
        jacobi = pcg(A, b, precond=JacobiPreconditioner(A), tol=1e-8)
        ssor = pcg(A, b, precond=SSORPreconditioner(A), tol=1e-8)

        self.assertTrue(plain.converged)
        self.assertTrue(jacobi.converged)
        self.assertTrue(ssor.converged)
        self.assertLess(jacobi.iterations, plain.iterations)
        print("\n  [病态矩阵] 无预条件 %d 次, Jacobi %d 次, SSOR %d 次"
              % (plain.iterations, jacobi.iterations, ssor.iterations))

    def test_max_iter_exhaustion(self):
        """迭代上限耗尽：converged=False，仍返回当前最优迭代点与残差历史。"""
        A, dense = random_spd(30, seed=3, cond_exp=6)
        b = [1.0] * 30
        res = pcg(A, b, max_iter=3, tol=1e-12)
        self.assertFalse(res.converged)
        self.assertEqual(res.iterations, 3)
        self.assertEqual(len(res.residual_history), 4)  # 初始 + 3 步
        self.assertIn("上限", res.message)
        # 返回的 x 必须是真实迭代点：真实残差与记录的末步残差一致
        Ax = A.matvec(res.x)
        true_res = math.sqrt(sum((bi - ai) ** 2 for bi, ai in zip(b, Ax)))
        self.assertTrue(math.isclose(true_res, res.residual_history[-1],
                                     rel_tol=1e-6))


class TestAgainstDirectSolver(unittest.TestCase):
    """与直接解对拍。"""

    def test_matches_direct_solution(self):
        n = 25
        A, dense = random_spd(n, seed=11, cond_exp=4)
        rng = random.Random(5)
        b = [rng.uniform(-10, 10) for _ in range(n)]
        x_ref = solve_dense(dense, b)
        for precond in (IdentityPreconditioner(),
                        JacobiPreconditioner(A),
                        SSORPreconditioner(A)):
            res = pcg(A, b, precond=precond, tol=1e-12)
            self.assertTrue(res.converged, precond.name)
            err = max(abs(a - r) for a, r in zip(res.x, x_ref))
            rel = err / max(abs(v) for v in x_ref)
            self.assertLess(rel, 1e-6, "%s 相对误差 %g" % (precond.name, rel))


if __name__ == "__main__":
    unittest.main(verbosity=2)
