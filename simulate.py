"""错误率与样本量模拟：在真实无差异数据上重复跑 SPRT，实证检验 I 类错误。

运行：
    python3 simulate.py                 # 默认规模（约 1-3 分钟）
    python3 simulate.py --reps 100000   # 自定义重复次数
    python3 simulate.py --seed 2026

输出：
    data/simulation_runs.csv     每次重复的原始结果
    data/simulation_summary.txt  汇总（误判率、置信区间、撞顶率、样本量）
仅依赖标准库。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics

from seqtest import SPRTConfig, FamilySPRT, bonferroni_configs

REJECT = "reject_h0"
ACCEPT = "accept_h0"


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_single(cfg: SPRTConfig, p_t: float, p_c: float, reps: int, seed: int):
    """重复 reps 次单检验，返回行列表 (rep, decision, n, capped)。"""
    rng = random.Random(seed)
    rand = rng.random
    steps = (cfg.llr_pair(0, 0), cfg.llr_pair(0, 1),
             cfg.llr_pair(1, 0), cfg.llr_pair(1, 1))
    a, b, nmax = cfg.log_upper, cfg.log_lower, cfg.nmax
    rows = []
    for r in range(reps):
        z = 0.0
        decision, n, capped = ACCEPT, nmax, True
        for n in range(1, nmax + 1):
            xt = 1 if rand() < p_t else 0
            xc = 1 if rand() < p_c else 0
            z += steps[2 * xt + xc]
            if z >= a:
                decision, capped = REJECT, False
                break
            if z <= b:
                decision, capped = ACCEPT, False
                break
        rows.append((r, decision, n, capped))
    return rows


def run_family(configs, truths, reps: int, seed: int, alpha: float,
               use_bonferroni: bool):
    """重复 reps 次 K 指标族检验。truths: [(p_t,p_c), ...]；返回族级行。"""
    cfgs = (bonferroni_configs(configs, alpha) if use_bonferroni
            else list(configs))
    step_lists = [(c.llr_pair(0, 0), c.llr_pair(0, 1),
                   c.llr_pair(1, 0), c.llr_pair(1, 1)) for c in cfgs]
    a_list = [c.log_upper for c in cfgs]
    b_list = [c.log_lower for c in cfgs]
    nmax_list = [c.nmax for c in cfgs]
    rng = random.Random(seed)
    rand = rng.random
    rows = []
    k = len(cfgs)
    for r in range(reps):
        zs = [0.0] * k
        active = set(range(k))
        family_dec = ACCEPT
        t = pairs_used = 0
        capped = False
        while active:
            t += 1
            for i in list(active):
                p_t, p_c = truths[i]
                xt = 1 if rand() < p_t else 0
                xc = 1 if rand() < p_c else 0
                pairs_used += 1
                zs[i] += step_lists[i][2 * xt + xc]
                if zs[i] >= a_list[i]:
                    family_dec = REJECT
                    active.clear()  # 族级一旦阳性即结束
                    break
                if zs[i] <= b_list[i]:
                    active.discard(i)
                elif t >= nmax_list[i]:
                    active.discard(i)
                    capped = True
        rows.append((r, family_dec, pairs_used, capped))
    return rows


def summarize(name, rows, target_bound, extra=""):
    reps = len(rows)
    rejects = sum(1 for _, d, _, _ in rows if d == REJECT)
    capped = sum(1 for _, _, _, c in rows if c)
    ns = [n for _, _, n, _ in rows]
    lo, hi = wilson_ci(rejects, reps)
    ns_sorted = sorted(ns)
    q = lambda f: ns_sorted[min(len(ns_sorted) - 1, int(f * len(ns_sorted)))]
    return {
        "name": name, "reps": reps, "reject": rejects, "rate": rejects / reps,
        "ci_lo": lo, "ci_hi": hi, "target": target_bound,
        "cap_rate": capped / reps,
        "n_mean": statistics.fmean(ns), "n_med": q(0.5), "n_p90": q(0.9),
        "n_max": max(ns), "extra": extra,
    }


def fmt_summary(s):
    verdict = "OK ≤ 目标" if s["ci_hi"] <= s["target"] + 1e-12 else \
              ("边界附近" if s["rate"] <= s["target"] else "超限!")
    return (
        f"{s['name']}\n"
        f"  重复次数={s['reps']}, 判阳性={s['reject']}, "
        f"实测比例={s['rate']:.4f} (95% Wilson CI [{s['ci_lo']:.4f},"
        f"{s['ci_hi']:.4f}]), 目标上界={s['target']:.4f}  => {verdict}\n"
        f"  样本量(配对数) mean={s['n_mean']:.1f}, median={s['n_med']}, "
        f"p90={s['n_p90']}, max={s['n_max']}; 撞顶率={s['cap_rate']*100:.2f}%\n"
        f"  {s['extra']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--family-reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()

    base = dict(p0=0.5, delta=0.1)
    cfg_05 = SPRTConfig.for_effect(alpha=0.05, beta=0.20, **base)
    cfg_01 = SPRTConfig.for_effect(alpha=0.01, beta=0.05, **base)

    scenarios = []  # (name, kind, ...)
    print(f"nmax(alpha=.05)={cfg_05.nmax}, nmax(alpha=.01)={cfg_01.nmax}")
    print("H0 理论 ASN ≈ %.0f 配对，H1 理论 ASN ≈ %.0f 配对"
          % (cfg_05.asn_h0(), cfg_05.asn_h1()))

    runs = {}
    runs["S1_null_a05"] = run_single(cfg_05, 0.5, 0.5, args.reps, args.seed)
    runs["S2_h1_a05"] = run_single(cfg_05, 0.6, 0.4, args.reps, args.seed + 1)
    runs["S3_negative"] = run_single(cfg_05, 0.4, 0.6, args.reps, args.seed + 2)
    runs["S4_null_a01"] = run_single(cfg_01, 0.5, 0.5, args.reps, args.seed + 3)

    cfgs_k2 = [SPRTConfig.for_effect(0.5, 0.1, alpha=0.05, beta=0.20, nmax=cfg_05.nmax)
               for _ in range(2)]
    cfgs_k4 = [SPRTConfig.for_effect(0.5, 0.1, alpha=0.05, beta=0.20, nmax=cfg_05.nmax)
               for _ in range(4)]
    null_truths = [(0.5, 0.5)] * 4
    runs["S5_k2_bonf_null"] = run_family(
        cfgs_k2, [(0.5, 0.5)] * 2, args.family_reps, args.seed + 4,
        alpha=0.05, use_bonferroni=True)
    runs["S6_k4_bonf_null"] = run_family(
        cfgs_k4, null_truths, args.family_reps, args.seed + 5,
        alpha=0.05, use_bonferroni=True)
    runs["S7_k4_uncorrected_null"] = run_family(
        cfgs_k4, null_truths, args.family_reps, args.seed + 6,
        alpha=0.05, use_bonferroni=False)
    one_pos = [(0.6, 0.4)] + [(0.5, 0.5)] * 3
    runs["S8_k4_bonf_onepositive"] = run_family(
        cfgs_k4, one_pos, args.family_reps, args.seed + 7,
        alpha=0.05, use_bonferroni=True)
    runs["S9_k4_uncorrected_onepositive"] = run_family(
        cfgs_k4, one_pos, args.family_reps, args.seed + 8,
        alpha=0.05, use_bonferroni=False)

    summaries = [
        summarize("S1 无差异 H0 (alpha=.05, beta=.20)", runs["S1_null_a05"],
                  0.05, "关键场景：实测假阳性率应 ≤ 5%"),
        summarize("S2 真实效应等于设计效应 (0.6 vs 0.4)", runs["S2_h1_a05"],
                  1.0, "功效场景：阳性率应接近或高于 80%（撞顶损失功率）"),
        summarize("S3 效果为负 (0.4 vs 0.6)", runs["S3_negative"],
                  0.05, "负效应不应误报为正向；通常很早碰下界"),
        summarize("S4 无差异 H0 (alpha=.01, beta=.05)", runs["S4_null_a01"],
                  0.01, "更严水平：假阳性率应 ≤ 1%"),
        summarize("S5 K=2 Bonferroni 全无效 (FWER 目标 5%)",
                  runs["S5_k2_bonf_null"], 0.05,
                  "每指标 alpha=.025；族系假阳性 ≤ 5%"),
        summarize("S6 K=4 Bonferroni 全无效 (FWER 目标 5%)",
                  runs["S6_k4_bonf_null"], 0.05,
                  "每指标 alpha=.0125；族系假阳性 ≤ 5%"),
        summarize("S7 K=4 不校正 全无效 (反面教材)",
                  runs["S7_k4_uncorrected_null"], 0.05,
                  "每指标仍 alpha=.05：FWER 膨胀到约 1-.95^4≈18.5%"),
        summarize("S8 K=4 Bonferroni 恰好 1 个指标有效",
                  runs["S8_k4_bonf_onepositive"], 1.0,
                  "校正后族级功效；n_pairs 为族内实际消耗的跨臂配对总数"),
        summarize("S9 K=4 不校正 恰好 1 个指标有效（对照 S8）",
                  runs["S9_k4_uncorrected_onepositive"], 1.0,
                  "上界 log A 从 4.38(alpha=.0125) 降到 2.996(alpha=.05)，"
                  "平均样本量更小——这就是校正的样本量代价"),
    ]

    os.makedirs("data", exist_ok=True)
    with open("data/simulation_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "rep", "metric", "decision",
                    "n_pairs", "capped"])
        for name, rows in runs.items():
            for rep, dec, n, cap in rows:
                w.writerow([name, rep, "family" if name.startswith(("S5", "S6", "S7", "S8"))
                            else "m0", dec, n, int(cap)])

    lines = [
        "# 序贯检验错误率模拟汇总",
        f"# 种子={args.seed}, 单指标重复={args.reps}, "
        f"多指标重复={args.family_reps}",
        f"# 设计: 双臂伯努利 H0=(.5,.5), H1=(.6,.4), "
        f"nmax={cfg_05.nmax}(a=.05), {cfg_01.nmax}(a=.01)",
        "",
    ]
    for s in summaries:
        line = fmt_summary(s)
        print(line)
        lines.append(line)
    with open("data/simulation_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n原始数据: data/simulation_runs.csv")
    print("汇总:     data/simulation_summary.txt")


if __name__ == "__main__":
    main()
