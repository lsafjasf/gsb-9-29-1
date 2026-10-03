"""覆盖率蒙特卡洛模拟。

对每个 (分布, 统计量, 样本量) 单元格：
- 重复 R 次：从已知分布抽样本 -> 对同一样本计算三种 bootstrap 95% 区间
  （B 次重采样，BCa 含 n 次 jackknife）-> 判断真实参数是否落入区间；
- 用 Wilson 得分区间给出"实际覆盖率本身"的 95% 置信区间，
  使覆盖率差异可以区分是蒙特卡洛噪声还是方法本身欠覆盖。

全程使用整数种子派生（sim_seed, cell_id, rep_id, 用途位），结果可复现。

用法：
    python3 -m bootstrap_ci.simulate --R 2000 --B 1999 --n 15 50 --jobs 14
结果写入 results/coverage.csv 与 results/coverage.md。
"""

import argparse
import csv
import math
import multiprocessing
import os

from .bootstrap import bootstrap_ci
from .distributions import SAMPLERS, STATISTICS, TRUTHS
from .rngutil import make_rng, derive_seed

METHODS = ("percentile", "bc", "bca")
NOMINAL = 0.95
Z95 = 1.959963984540054

CELLS = [
    ("normal", ["mean", "median", "variance", "q90"]),
    ("exponential", ["mean", "median", "variance", "q90"]),
    ("bimodal", ["mean", "median", "variance", "q90"]),
    ("outlier", ["mean", "median", "variance", "q90"]),
    ("ratio_pairs", ["ratio"]),
]


def wilson_ci(k, n, z=Z95):
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return center - half, center + half


def run_cell(args):
    dist_name, stat_name, n, R, B, alpha, sim_seed, cell_id = args
    sampler = SAMPLERS[dist_name]
    statistic = STATISTICS[stat_name]
    truth = TRUTHS[dist_name][stat_name]

    counts = {m: 0 for m in METHODS}
    nan_cells = 0
    for rep in range(R):
        data_rng = make_rng(sim_seed, cell_id, rep, 0)
        boot_seed = derive_seed(sim_seed, cell_id, rep, 1)
        data = sampler(data_rng, n)
        res = bootstrap_ci(data, statistic, B=B, alpha=alpha, seed=boot_seed,
                           methods=METHODS)
        for m in METHODS:
            lo, hi = res["intervals"][m]
            if math.isnan(lo) or math.isnan(hi):
                continue
            if lo <= truth <= hi:
                counts[m] += 1

    row = {
        "distribution": dist_name,
        "statistic": stat_name,
        "n": n,
        "R": R,
        "B": B,
        "nominal": 1.0 - alpha,
    }
    for m in METHODS:
        k = counts[m]
        lo95, hi95 = wilson_ci(k, R)
        row["cover_" + m] = k / R
        row["cover_" + m + "_wilson_lo"] = lo95
        row["cover_" + m + "_wilson_hi"] = hi95
    return row


def all_tasks(n_list, R, B, alpha, sim_seed):
    tasks = []
    cell_id = 0
    for n in n_list:
        for dist_name, stats in CELLS:
            for stat_name in stats:
                tasks.append((dist_name, stat_name, n, R, B, alpha,
                              sim_seed, cell_id))
                cell_id += 1
    return tasks


CSV_FIELDS = ["distribution", "statistic", "n", "R", "B", "nominal"] + [
    m + suffix for m in ("percentile", "bc", "bca")
    for suffix in ("", "_wilson_lo", "_wilson_hi")
]
# 与上面实际 row 键对齐
CSV_FIELDS = ["distribution", "statistic", "n", "R", "B", "nominal"]
for m in METHODS:
    CSV_FIELDS += ["cover_" + m, "cover_" + m + "_wilson_lo",
                   "cover_" + m + "_wilson_hi"]


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "coverage.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path = os.path.join(out_dir, "coverage.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 覆盖率模拟结果\n\n")
        f.write("名义水平 95%%；方括号为覆盖率的 95%% Wilson 置信区间；"
                "B=%d 次重采样，每个单元格 R 次蒙特卡洛重复。\n\n"
                % rows[0]["B"])
        f.write("| 分布 | 统计量 | n | R | 百分位法 | BC | BCa |\n")
        f.write("|---|---|---:|---:|---|---|---|\n")
        for row in rows:
            rendered = []
            for m in METHODS:
                p = row["cover_" + m]
                lo = row["cover_" + m + "_wilson_lo"]
                hi = row["cover_" + m + "_wilson_hi"]
                flag = ""
                if hi < NOMINAL:
                    flag = " (欠覆盖)"
                elif lo > NOMINAL:
                    flag = " (过覆盖)"
                rendered.append("%.3f [%.3f, %.3f]%s" % (p, lo, hi, flag))
            f.write("| %s | %s | %d | %d | %s | %s | %s |\n" % (
                row["distribution"], row["statistic"], row["n"], row["R"],
                rendered[0], rendered[1], rendered[2]))
    return csv_path, md_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="bootstrap 覆盖率模拟")
    ap.add_argument("--R", type=int, default=2000, help="蒙特卡洛重复次数")
    ap.add_argument("--B", type=int, default=1999, help="bootstrap 重采样次数")
    ap.add_argument("--n", type=int, nargs="+", default=[15, 50],
                    help="样本量，可给多个，如 --n 15 50")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    tasks = all_tasks(args.n, args.R, args.B, args.alpha, args.seed)
    print("共 %d 个单元格，R=%d, B=%d, n=%s, jobs=%d"
          % (len(tasks), args.R, args.B, args.n, args.jobs), flush=True)

    if args.jobs == 1:
        rows = []
        for i, task in enumerate(tasks):
            rows.append(run_cell(task))
            print("  [%2d/%2d] %s/%s n=%d 完成"
                  % (i + 1, len(tasks), task[0], task[1], task[2]), flush=True)
    else:
        with multiprocessing.Pool(processes=args.jobs) as pool:
            rows = pool.map(run_cell, tasks)

    rows.sort(key=lambda r: (r["n"], r["distribution"], r["statistic"]))
    csv_path, md_path = write_outputs(rows, args.out)

    print("\n覆盖率（名义 %.2f）：" % (1.0 - args.alpha))
    print("%-13s %-9s %3s %6s | %-22s %-22s %-22s"
          % ("分布", "统计量", "n", "R", "percentile", "bc", "bca"))
    for row in rows:
        vals = []
        for m in METHODS:
            p = row["cover_" + m]
            lo = row["cover_" + m + "_wilson_lo"]
            hi = row["cover_" + m + "_wilson_hi"]
            vals.append("%.3f [%.3f,%.3f]" % (p, lo, hi))
        print("%-13s %-9s %3d %6d | %s %s %s"
              % (row["distribution"], row["statistic"], row["n"], row["R"],
                 vals[0], vals[1], vals[2]))
    print("\n已写出: %s, %s" % (csv_path, md_path))


if __name__ == "__main__":
    main()
