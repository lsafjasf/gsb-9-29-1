"""test_eig.py — eiglib 的自测/对拍脚本（仅标准库，Python 3）。

对拍方式
--------
主求解器 qr_eigen（Householder 三对角化 + Wilkinson 位移 QR）与
独立参照实现 jacobi_eigen（循环 Jacobi 扫描）互相校验，并在
有解析解的用例上与真值比较。

每个用例输出：
  * 特征值误差：QR vs Jacobi、QR vs 真值（若已知）的最大绝对/相对偏差
  * 残差：max_j ||A v_j - lambda_j v_j||_2 / max(1, ||A||_F)
  * 正交性：max |V^T V - I|
  * 向量夹角：按特征值间隙聚类后，QR 与 Jacobi 对应特征子空间之间的
    最大主夹角（度）。重根簇内单个向量不可比，必须比较子空间。

断言阈值见文件底部 THRESHOLDS；任一断言失败则以非零码退出。
运行：python3 test_eig.py
"""

import math
import random
import sys

from eiglib import qr_eigen, jacobi_eigen

_MACHINE_EPS = 2.220446049250313e-16
ANGLE_FLOOR_RAD = 1e-12   # 测量/舍入噪声地板（弧度）
ANGLE_SAFETY = 1000.0     # 扰动界安全系数

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def mat_from_spectral(q, diag):
    """A = Q diag Q^T，构造已知谱的测试矩阵。"""
    n = len(q)
    return [[sum(q[i][k] * diag[k] * q[j][k] for k in range(n))
             for j in range(n)] for i in range(n)]


def random_orthogonal(n, rng):
    """对随机矩阵做两次修正 Gram-Schmidt，得到数值正交阵。"""
    m = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    cols = [[m[i][j] for i in range(n)] for j in range(n)]
    basis = []
    for col in cols:
        v = col[:]
        for _ in range(2):  # 重正交化
            for u in basis:
                d = sum(a * b for a, b in zip(v, u))
                v = [a - d * b for a, b in zip(v, u)]
        nv = math.sqrt(sum(x * x for x in v))
        basis.append([x / nv for x in v])
    return [[basis[j][i] for j in range(n)] for i in range(n)]


def random_symmetric(n, rng):
    m = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    return [[0.5 * (m[i][j] + m[j][i]) for j in range(n)] for i in range(n)]


def frobenius(a):
    return math.sqrt(sum(x * x for row in a for x in row))


def max_residual(a, values, vectors):
    """max_j ||A v_j - lambda_j v_j||_2"""
    n = len(a)
    worst = 0.0
    for j in range(n):
        s = 0.0
        for i in range(n):
            av = sum(a[i][k] * vectors[k][j] for k in range(n))
            d = av - values[j] * vectors[i][j]
            s += d * d
        worst = max(worst, math.sqrt(s))
    return worst


def max_orthogonality_error(vectors):
    """max |V^T V - I|"""
    n = len(vectors[0]) if vectors else 0
    rows = len(vectors)
    worst = 0.0
    for a in range(n):
        for b in range(n):
            dot = sum(vectors[i][a] * vectors[i][b] for i in range(rows))
            target = 1.0 if a == b else 0.0
            worst = max(worst, abs(dot - target))
    return worst


def cluster_indices(values, gap_tol=1e-6):
    """按相对间隙把升序特征值聚成簇（重根/近重根同簇）。"""
    clusters = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or abs(values[i] - values[i - 1]) > \
                gap_tol * max(1.0, abs(values[i]), abs(values[i - 1])):
            clusters.append(list(range(start, i)))
            start = i
    return clusters


def principal_angle_rad(v1, idx1, v2, idx2):
    """两组列子空间之间的最大主夹角（弧度）。

    R = V1 - V2 (V2^T V1)，sin^2(theta_max) = lambda_max(R^T R)。
    直接构造"垂直连通分量" R（本身 ~sin(theta) 量级），避免
    I - B B^T 中两个 ~1 量相消带来的 sqrt(eps) 测量噪声地板，
    可分辨到 ~1e-15 rad。
    """
    rows = len(v1)
    k = len(idx1)
    b = [[sum(v2[r][idx2[i]] * v1[r][idx1[j]] for r in range(rows))
          for j in range(k)] for i in range(k)]
    r = [[v1[rr][idx1[j]] - sum(v2[rr][idx2[t]] * b[t][j] for t in range(k))
          for j in range(k)] for rr in range(rows)]
    s = [[sum(r[rr][i] * r[rr][j] for rr in range(rows)) for j in range(k)]
         for i in range(k)]
    w, _ = jacobi_eigen(s)
    sin2 = min(1.0, max(0.0, max(w)))
    return math.asin(math.sqrt(sin2))


def cluster_angle_report(values_qr, v_qr, values_ref, v_ref, norm_a):
    """逐簇计算子空间主夹角，并给出与该簇分离度相称的容差。

    特征子空间的扰动界为 sin(theta) ~ ||E|| / gap，其中 ||E|| ~ eps*||A||
    是算法的后向误差、gap 是簇外间隙。因此每个簇的容差取
        tol = max(ANGLE_FLOOR_RAD, ANGLE_SAFETY * eps * ||A|| / gap)
    返回 (最大夹角 deg, 最差 夹角/容差 比值)。
    """
    clusters = cluster_indices(values_qr)
    worst_angle = 0.0
    worst_ratio = 0.0
    n = len(values_qr)
    for cl in clusters:
        angle = principal_angle_rad(v_qr, cl, v_ref, cl)
        in_cluster = set(cl)
        outside = [abs(values_qr[i] - values_qr[j])
                   for i in cl for j in range(n) if j not in in_cluster]
        gap = min(outside) if outside else norm_a
        tol = max(ANGLE_FLOOR_RAD,
                  ANGLE_SAFETY * _MACHINE_EPS * norm_a / max(gap, 1e-300))
        worst_angle = max(worst_angle, math.degrees(angle))
        worst_ratio = max(worst_ratio, angle / tol)
    return worst_angle, worst_ratio


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def build_cases():
    rng = random.Random(20260930)
    cases = []

    def add(name, a, true_values=None):
        cases.append((name, a, true_values))

    # 边界：1x1 / 2x2 / 零矩阵 / 单位阵 / 已对角
    add("edge_1x1", [[3.25]], [3.25])
    add("edge_2x2_repeated", [[2.0, 0.0], [0.0, 2.0]], [2.0, 2.0])
    add("edge_2x2_general", [[1.0, 2.0], [2.0, 3.0]])
    add("edge_zero_4x4", [[0.0] * 4 for _ in range(4)], [0.0] * 4)
    ident = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    add("identity_6x6", ident, [1.0] * 6)
    add("diagonal_5x5",
        [[-2.0 if (i, j) == (0, 0) else
          0.5 if (i, j) == (1, 1) else
          1.0 if (i, j) == (2, 2) else
          3.0 if (i, j) == (3, 3) else
          7.0 if (i, j) == (4, 4) else 0.0
          for j in range(5)] for i in range(5)],
        [-2.0, 0.5, 1.0, 3.0, 7.0])

    # 重根：谱 {2,2,2,5,5,7}，随机正交相似变换隐藏结构
    q = random_orthogonal(6, rng)
    add("repeated_{2,2,2,5,5,7}",
        mat_from_spectral(q, [2.0, 2.0, 2.0, 5.0, 5.0, 7.0]),
        [2.0, 2.0, 2.0, 5.0, 5.0, 7.0])

    # 近似重根（近简并）：两对间距 ~1e-9 的特征值
    q = random_orthogonal(4, rng)
    add("nearly_degenerate",
        mat_from_spectral(q, [1.0, 1.0 + 1e-9, 3.0, 3.0 + 1e-9]),
        sorted([1.0, 1.0 + 1e-9, 3.0, 3.0 + 1e-9]))

    # 近似奇异：最小特征值 ~1e-13
    q = random_orthogonal(5, rng)
    add("near_singular",
        mat_from_spectral(q, [1e-13, 1.0, 2.0, 3.0, 4.0]),
        [1e-13, 1.0, 2.0, 3.0, 4.0])

    # 量级差异极大：1e-10 .. 1e10
    q = random_orthogonal(5, rng)
    add("wide_scale_1e-10..1e10",
        mat_from_spectral(q, [1e-10, 1e-5, 1.0, 1e5, 1e10]),
        [1e-10, 1e-5, 1.0, 1e5, 1e10])

    # 梯度对角阵（Hilbert 风格的极端条件数）
    q = random_orthogonal(8, rng)
    graded = [10.0 ** k for k in range(-7, 1)]
    add("graded_1e-7..1", mat_from_spectral(q, graded), sorted(graded))

    # 胶合块：两个相同块用 eps 耦合，产生近重根对
    eps = 1e-10
    b = [[2.0, 1.0], [1.0, 2.0]]
    glued = [[0.0] * 4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            glued[i][j] = b[i][j]
            glued[i + 2][j + 2] = b[i][j]
    glued[0][2] = glued[2][0] = eps
    glued[1][3] = glued[3][1] = eps
    add("glued_blocks_eps=1e-10", glued)

    # Wilkinson W+ 型三对角阵：特征值成对极度接近，QR 的经典难题
    m = 5
    w_diag = [float(abs(k - m)) for k in range(2 * m + 1)]
    wil = [[0.0] * (2 * m + 1) for _ in range(2 * m + 1)]
    for i in range(2 * m + 1):
        wil[i][i] = w_diag[i]
        if i + 1 < 2 * m + 1:
            wil[i][i + 1] = wil[i + 1][i] = 1.0
    add("wilkinson_W+_11x11", wil)

    # 随机对称阵（无解析解，纯对拍）
    for k, n in enumerate([3, 5, 8, 12]):
        add("random_sym_%dx%d#%d" % (n, n, k), random_symmetric(n, rng))

    return cases


# ---------------------------------------------------------------------------
# 断言阈值
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "residual_rel": 1e-12,   # 残差 / max(1, ||A||_F)
    "orthogonality": 1e-12,  # max |V^T V - I|
    "eig_abs_vs_ref": 1e-8,  # |lambda_QR - lambda_Jacobi|，相对 max(1,||A||)
    "eig_abs_vs_true": 1e-8, # 有真值时同上
    "angle_ratio": 1.0,     # 子空间夹角 / 分离度自适应容差
}


def run():
    cases = build_cases()
    failures = []
    header = ("%-26s %4s %11s %11s %11s %11s %11s %6s"
              % ("case", "n", "resid(rel)", "orthQR", "orthJAC",
                 "dEig(QR,J)", "dEig(QR,true)", "ang(deg)"))
    print(header)
    print("-" * len(header))
    for name, a, true_values in cases:
        n = len(a)
        norm_a = max(1.0, frobenius(a))
        w_qr, v_qr = qr_eigen(a)
        w_jc, v_jc = jacobi_eigen(a)

        res_qr = max_residual(a, w_qr, v_qr) / norm_a
        res_jc = max_residual(a, w_jc, v_jc) / norm_a
        orth_qr = max_orthogonality_error(v_qr)
        orth_jc = max_orthogonality_error(v_jc)
        d_eig_ref = max(abs(x - y) for x, y in zip(w_qr, w_jc)) / norm_a
        if true_values is not None:
            tv = sorted(true_values)
            d_eig_true = max(abs(x - y)
                             for x, y in zip(w_qr, tv)) / norm_a
        else:
            d_eig_true = float("nan")
        angle, angle_ratio = cluster_angle_report(
            w_qr, v_qr, w_jc, v_jc, norm_a)

        print("%-26s %4d %11.3e %11.3e %11.3e %11.3e %11.3e %9.2e"
              % (name, n, max(res_qr, res_jc), orth_qr, orth_jc,
                 d_eig_ref, d_eig_true, angle))

        problems = []
        if res_qr > THRESHOLDS["residual_rel"]:
            problems.append("QR residual %.2e" % res_qr)
        if res_jc > THRESHOLDS["residual_rel"]:
            problems.append("Jacobi residual %.2e" % res_jc)
        if orth_qr > THRESHOLDS["orthogonality"]:
            problems.append("QR orthogonality %.2e" % orth_qr)
        if orth_jc > THRESHOLDS["orthogonality"]:
            problems.append("Jacobi orthogonality %.2e" % orth_jc)
        if d_eig_ref > THRESHOLDS["eig_abs_vs_ref"]:
            problems.append("eig vs ref %.2e" % d_eig_ref)
        if true_values is not None and \
                d_eig_true > THRESHOLDS["eig_abs_vs_true"]:
            problems.append("eig vs true %.2e" % d_eig_true)
        if angle_ratio > THRESHOLDS["angle_ratio"]:
            problems.append("subspace angle %.2e deg (%.1fx tolerance)"
                            % (angle, angle_ratio))
        if problems:
            failures.append((name, problems))

    # 非对称输入必须被拒绝
    try:
        qr_eigen([[1.0, 2.0], [0.0, 1.0]])
        failures.append(("nonsymmetric_rejected", ["no ValueError raised"]))
    except ValueError:
        print("%-26s %4s %s" % ("nonsymmetric_rejected", "-",
                                "ValueError raised as expected"))

    print("-" * len(header))
    if failures:
        print("FAIL: %d case(s) violated assertions" % len(failures))
        for name, problems in failures:
            print("  %s: %s" % (name, "; ".join(problems)))
        return 1
    print("PASS: all %d cases, all assertions satisfied" % (len(cases) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(run())
