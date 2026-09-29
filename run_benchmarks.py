#!/usr/bin/env python3
"""基准与对拍：生成 results/ 下的残差 CSV 与 REPORT.md。

用法：python3 run_benchmarks.py
"""

import csv
import os
import time

from steady_field import (
    optimal_omega,
    jacobi,
    gauss_seidel,
    sor,
    multigrid,
    max_error,
    max_diff,
    sine_problem,
    uniform_boundary_problem,
    point_source_problem,
)

TOL = 1e-6
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save_residuals(name, res):
    path = os.path.join(RESULTS, name + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "rel_residual"])
        for k, r in enumerate(res.residuals):
            w.writerow([k, "%.6e" % r])
    return path


def fmt_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    report = []
    out = lambda s="": (print(s), report.append(s))

    out("# 稳态场迭代求解：基准与对拍报告")
    out("")
    out("问题：单位正方形上 -Δu = f，五点差分，Dirichlet 边界，"
        "相对残差 ||r||/||b|| ≤ %.0e 判收敛。" % TOL)
    out("")

    # ------------------------------------------------ 1. 加速对比 + 误差对拍
    out("## 1. 解析解对拍与加速对比（u = sin(πx)sin(πy)）")
    out("")
    header = ["N", "方法", "迭代轮数", "最终相对残差", "最大误差", "耗时(s)"]
    out(fmt_row(header))
    out(fmt_row(["---"] * len(header)))
    plan = [(15, "all"), (31, "all"), (63, "all"),
            (127, "fast"), (255, "fast"), (511, "mg"), (1023, "mg")]
    for n, level in plan:
        f, bc, exact = sine_problem(n)
        runs = []
        if level == "all":
            runs.append(("Jacobi", jacobi(n, f, bc, tol=TOL)))
            runs.append(("Gauss-Seidel", gauss_seidel(n, f, bc, tol=TOL)))
        if level in ("all", "fast"):
            runs.append(("SOR(ω*)", sor(n, f, bc, tol=TOL)))
        runs.append(("Multigrid", multigrid(n, f, bc, tol=TOL)))
        for name, res in runs:
            save_residuals("sine_N%d_%s" % (n, name.split("(")[0].lower()), res)
            err = max_error(res.u, exact, n)
            out(fmt_row([n, name, res.iterations, "%.3e" % res.residuals[-1],
                         "%.3e" % err, "%.2f" % res.wall_time]))
    out("")
    out("误差随网格加密按 O(h²) 下降（N 加倍误差约 ÷4）；Jacobi/GS 迭代轮数")
    out("随 N² 增长，SOR(ω*) 随 N 增长，多重网格轮数与 N 基本无关。")
    out("")

    # ------------------------------------------------ 2. 边界全同
    out("## 2. 边界全同（四周 u≡1，f=0，解析解 u≡1）")
    out("")
    out(fmt_row(["方法", "迭代轮数", "最终相对残差", "最大误差"]))
    out(fmt_row(["---"] * 4))
    n = 63
    f, bc, exact = uniform_boundary_problem(n, value=1.0)
    for name, res in [("Jacobi", jacobi(n, f, bc, tol=TOL)),
                      ("Gauss-Seidel", gauss_seidel(n, f, bc, tol=TOL)),
                      ("SOR(ω*)", sor(n, f, bc, tol=TOL)),
                      ("Multigrid", multigrid(n, f, bc, tol=TOL))]:
        save_residuals("uniform_N%d_%s" % (n, name.split("(")[0].lower()), res)
        out(fmt_row([name, res.iterations, "%.3e" % res.residuals[-1],
                     "%.3e" % max_error(res.u, exact, n)]))
    out("")

    # ------------------------------------------------ 3. 单点源
    out("## 3. 单点源（中心 δ 源，零边界，无解析解 → 与 MG 高精度解对拍）")
    out("")
    n = 63
    f, bc, c = point_source_problem(n)
    ref = multigrid(n, f, bc, tol=1e-10)
    out("参考解：Multigrid tol=1e-10，中心值 u(c) = %.6f" % ref.u[c][c])
    out("")
    out(fmt_row(["方法", "迭代轮数", "最终相对残差", "与参考解最大差"]))
    out(fmt_row(["---"] * 4))
    for name, res in [("Jacobi", jacobi(n, f, bc, tol=TOL)),
                      ("Gauss-Seidel", gauss_seidel(n, f, bc, tol=TOL)),
                      ("SOR(ω*)", sor(n, f, bc, tol=TOL)),
                      ("Multigrid", multigrid(n, f, bc, tol=TOL))]:
        save_residuals("point_N%d_%s" % (n, name.split("(")[0].lower()), res)
        out(fmt_row([name, res.iterations, "%.3e" % res.residuals[-1],
                     "%.3e" % max_diff(res.u, ref.u, n)]))
    out("")

    # ------------------------------------------------ 4. 极细网格
    out("## 4. 极细网格（N=1023，约 105 万未知数，仅多重网格可行）")
    out("")
    n = 1023
    f, bc, exact = sine_problem(n)
    res = multigrid(n, f, bc, tol=TOL)
    save_residuals("sine_N1023_multigrid", res)
    out("V 循环数：%d，最终相对残差 %.3e，最大误差 %.3e，耗时 %.1fs"
        % (res.iterations, res.residuals[-1],
           max_error(res.u, exact, n), res.wall_time))
    out("逐循环残差：" + ", ".join("%.1e" % r for r in res.residuals))
    out("")

    # ------------------------------------------------ 5. 不收敛参数
    out("## 5. 松弛因子选取（N=63，ω* = %.4f）" % optimal_omega(63))
    out("")
    n = 63
    f, bc, _ = sine_problem(n)
    out(fmt_row(["ω", "迭代轮数", "状态", "说明"]))
    out(fmt_row(["---"] * 4))
    notes = {1.0: "退化为 GS", 1.5: "欠松弛", 1.8: "接近最优",
             1.99: "过松弛，极慢", 2.0: "谱半径=1，不收敛", 2.05: "发散"}
    for w in (1.0, 1.5, 1.8, None, 1.99, 2.0, 2.05):
        label = "ω*" if w is None else w
        res = sor(n, f, bc, omega=w, tol=TOL, max_iter=30000)
        save_residuals("omega_%s" % (label if label != "ω*" else "opt"), res)
        status = "收敛" if res.converged else ("发散" if res.reason == "diverged"
                                               else "超迭代上限")
        note = "最优 ω*=2/(1+sin(πh))" if w is None else notes.get(w, "")
        out(fmt_row([label, res.iterations, status, note]))
    out("")
    out("ω≥2 时迭代矩阵谱半径≥1：ω=2.0 残差停滞跑满上限，ω=2.05 被发散检测提前截获。")

    with open(os.path.join(RESULTS, "REPORT.md"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print("\n报告与残差 CSV 已写入 results/")


if __name__ == "__main__":
    main()
