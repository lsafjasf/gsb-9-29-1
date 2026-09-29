"""演示：尺度悬殊的病态 SPD 系统上，预条件 CG vs 普通 CG。

产出:
  - 终端打印残差随迭代次数的 ASCII 曲线（log10 相对残差）
  - residual_curve.csv   三种方法的逐步残差数据
  - 与直接解（高斯消元）对拍的最终误差表

运行: python3 demo.py
"""

import math
import random

from cglib import (
    IdentityPreconditioner,
    JacobiPreconditioner,
    SparseMatrix,
    SSORPreconditioner,
    pcg,
)
from direct import solve_dense

N = 80          # 矩阵阶数
SCALE_EXP = 4   # 对角尺度 1e-4 .. 1e4，即系数量级相差 1e8
TOL = 1e-6
SEED = 7


def build_problem():
    """A = S K S：K 为良态 SPD（三对角 2,-1），S 为尺度悬殊的对角阵。"""
    rng = random.Random(SEED)
    scale = [10.0 ** rng.uniform(-SCALE_EXP, SCALE_EXP) for _ in range(N)]
    dense = [[0.0] * N for _ in range(N)]
    for i in range(N):
        dense[i][i] = 2.0
        if i > 0:
            dense[i][i - 1] = dense[i - 1][i] = -1.0
    dense = [[dense[i][j] * scale[i] * scale[j]
              for j in range(N)] for i in range(N)]
    b = [rng.uniform(-1.0, 1.0) for _ in range(N)]
    return SparseMatrix.from_dense(dense), dense, b


def ascii_curve(histories, bnorm, width=64, rows=18):
    """把多条残差曲线画成 ASCII 图。

    纵轴 log10(||r||/||b||)，横轴 log10(迭代次数+1)，快慢方法都能看清。
    """
    series = []
    for name, hist in histories:
        pts = [math.log10(max(h / bnorm, 1e-17)) for h in hist]
        series.append((name, pts))
    xmax = max(len(p) - 1 for _, p in series)
    lxmax = math.log10(xmax + 1)
    ymax = 0.0
    ymin = math.floor(min(v for _, p in series for v in p)) - 0.5
    grid = [[" "] * (width + 1) for _ in range(rows + 1)]
    marks = "oxs"
    for si, (name, pts) in enumerate(series):
        for k, v in enumerate(pts):
            col = round(math.log10(k + 1) / lxmax * width)
            row = round((v - ymax) / (ymin - ymax) * rows)
            row = max(0, min(rows, row))
            grid[row][col] = marks[si % len(marks)]
    print("  log10(||r||/||b||) vs 迭代次数（横轴为对数刻度）")
    for r, line in enumerate(grid):
        axis = ymin + (ymax - ymin) * (rows - r) / rows
        print("  %+6.1f |%s" % (axis, "".join(line)))
    print("        +" + "-" * (width + 1))
    tickrow = [" "] * (width + 8)
    for t in sorted({0, 1, 10, 100, 1000, 10000, xmax}):
        if t > xmax:
            continue
        col = round(math.log10(t + 1) / lxmax * width)
        for ci, ch in enumerate(str(t)):
            if col + ci < len(tickrow):
                tickrow[col + ci] = ch
    print("         " + "".join(tickrow) + " (迭代次数)")
    for si, (name, _) in enumerate(series):
        print("  图例 %s = %s" % (marks[si % len(marks)], name))


def main():
    A, dense, b = build_problem()
    bnorm = math.sqrt(sum(v * v for v in b))
    print("问题: n=%d, A = S K S, 对角尺度 1e-%d..1e%d (量级差 1e%d)"
          % (N, SCALE_EXP, SCALE_EXP, 2 * SCALE_EXP))
    print("收敛阈值: ||r|| <= %g * ||b||\n" % TOL)

    runs = [
        ("无预条件 CG", IdentityPreconditioner()),
        ("Jacobi 预条件", JacobiPreconditioner(A)),
        ("SSOR 预条件", SSORPreconditioner(A)),
    ]

    x_ref = solve_dense(dense, b)  # 直接解（对拍基准）

    histories = []
    table = []
    for name, precond in runs:
        res = pcg(A, b, precond=precond, tol=TOL, max_iter=500000)
        err_inf = max(abs(a - r) for a, r in zip(res.x, x_ref))
        xnorm = max(abs(v) for v in x_ref)
        Ax = A.matvec(res.x)
        true_res = math.sqrt(sum((bi - ai) ** 2 for bi, ai in zip(b, Ax)))
        histories.append((name, res.residual_history))
        table.append((name, res.iterations, res.converged,
                      true_res / bnorm, err_inf, err_inf / xnorm))

    ascii_curve(histories, bnorm)

    print("\n与直接解（部分主元高斯消元）对拍:")
    print("  %-12s %8s %6s %14s %14s %14s"
          % ("方法", "迭代次数", "收敛", "相对残差", "误差inf范数", "相对误差"))
    for name, iters, conv, relres, err, relerr in table:
        print("  %-12s %8d %6s %14.3e %14.3e %14.3e"
              % (name, iters, "是" if conv else "否", relres, err, relerr))

    with open("residual_curve.csv", "w", encoding="utf-8") as f:
        f.write("iteration,plain_cg,jacobi,ssor\n")
        maxlen = max(len(h) for _, h in histories)
        for k in range(maxlen):
            row = [str(k)]
            for _, h in histories:
                row.append("%.6e" % (h[k] / bnorm) if k < len(h) else "")
            f.write(",".join(row) + "\n")
    print("\n残差数据已写入 residual_curve.csv")


if __name__ == "__main__":
    main()
