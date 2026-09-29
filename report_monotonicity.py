"""report_monotonicity.py — 生成单调性检测数据 monotonicity_report.json

覆盖三类典型数据：单调上升、U 形（非单调）、极端长尾。
对每类数据分别给出：无约束分箱的单调性诊断（违约点对 + PAVA 合并建议）
与强制单调约束后的分箱结果。

运行：python3 report_monotonicity.py
"""

import json
import random

from binning import Binner


def gen_monotone(n=3000, seed=11):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(0, 10)
        xs.append(x)
        ys.append(1 if rng.random() < 0.05 + 0.9 * (x / 10.0) else 0)
    return xs, ys


def gen_u_shape(n=4000, seed=13):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        x = rng.uniform(0, 10)
        xs.append(x)
        ys.append(1 if rng.random() < 0.05 + 0.8 * ((x - 5) / 5.0) ** 2 else 0)
    return xs, ys


def gen_long_tail(n=5000, seed=7):
    rng = random.Random(seed)
    xs = [rng.expovariate(1.0) for _ in range(n)] + [1e6, 2e6, 5e5]
    ys = [1 if rng.random() < min(0.9, x / 50.0) else 0 for x in xs]
    return xs, ys


def run_case(name, xs, ys, n_bins):
    free = Binner(n_bins=n_bins, method="optimal").fit(xs, ys)
    locked = Binner(n_bins=n_bins, method="optimal",
                    monotonic="auto").fit(xs, ys)
    return {
        "dataset": name,
        "n_samples": len(xs),
        "unconstrained": {
            "edges": free.edges_,
            "bin_stats": free.bin_stats_,
            "missing_bin": free.missing_stats_,
            "outlier_count": free.outlier_count_,
            "iv": free.iv_,
            "monotonicity": free.monotonicity_report(),
        },
        "monotone_constrained": {
            "edges": locked.edges_,
            "direction": locked.monotonic_direction_,
            "bin_stats": locked.bin_stats_,
            "iv": locked.iv_,
            "monotonicity": locked.monotonicity_report(),
        },
    }


def main():
    report = [
        run_case("monotone_increasing", *gen_monotone(), n_bins=6),
        run_case("u_shape_non_monotone", *gen_u_shape(), n_bins=8),
        run_case("extreme_long_tail", *gen_long_tail(), n_bins=5),
    ]
    with open("monotonicity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    for case in report:
        unc = case["unconstrained"]["monotonicity"]
        con = case["monotone_constrained"]
        print(f"[{case['dataset']}] "
              f"无约束: monotonic={unc['is_monotonic']} "
              f"violations={len(unc.get('violations', []))} "
              f"suggestions={unc.get('merge_suggestions')} | "
              f"强约束: direction={con['direction']} "
              f"monotonic={con['monotonicity']['is_monotonic']} "
              f"bins={len(con['edges']) + 1} iv={con['iv']:.4f}")


if __name__ == "__main__":
    main()
