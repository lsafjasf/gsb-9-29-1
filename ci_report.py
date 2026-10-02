"""置信区间报告：在固定种子的样例数据集上计算指标与自助法区间。

运行：python3 ci_report.py
输出：终端打印 + ci_data.json
"""

import json
import random

import ranking_metrics as M

N_BOOT = 2000
ALPHA = 0.05


def sample_dataset(seed=20261002, num_queries=12):
    rng = random.Random(seed)
    dataset = []
    for qi in range(num_queries):
        n_docs = rng.randint(3, 10)
        docs = [f"d{qi}_{j}" for j in range(n_docs)]
        ranking = list(zip(docs, (rng.randint(0, 4) for _ in docs)))
        rng.shuffle(ranking)
        labels = {}
        for doc_id in docs:
            if rng.random() < 0.75:
                labels[doc_id] = rng.choice([0, 0, 1, 1, 2, 3])
        if rng.random() < 0.3:
            labels[f"ghost_{qi}"] = rng.randint(1, 3)
        dataset.append({"qid": f"q{qi}", "ranking": ranking, "labels": labels})
    return dataset


def build_report(k=10):
    dataset = sample_dataset()
    result = M.evaluate(dataset, k)
    per_query = result["per_query"]

    report = {
        "config": {"k": k, "n_boot": N_BOOT, "alpha": ALPHA,
                   "num_queries": result["num_queries"],
                   "num_zero_relevance": result["num_zero_relevance"]},
        "macro": {},
        "micro": {},
        "aggregation_explanation": M.explain_aggregation(result),
    }

    for metric in M.METRICS:
        values = [item[metric] for item in per_query]
        low, high, info = M.bootstrap_ci(values, n_boot=N_BOOT, alpha=ALPHA)
        report["macro"][metric] = {
            "point": result["macro"][metric],
            "ci": [low, high],
            "ci_info": info,
            "per_query_values": values,
        }

    micro_pairs = {
        "ap": [(item["ap_numerator"], float(item["relevant"])) for item in per_query],
        "ndcg": [(item["dcg"], item["idcg"]) for item in per_query],
    }
    for metric in M.METRICS:
        entry = {"point": result["micro"][metric]}
        if metric in micro_pairs:
            low, high, info = M.bootstrap_ci_micro(
                micro_pairs[metric], n_boot=N_BOOT, alpha=ALPHA)
            entry["ci"] = [low, high]
            entry["ci_info"] = info
        else:
            entry["ci"] = report["macro"][metric]["ci"]
            entry["ci_info"] = dict(report["macro"][metric]["ci_info"])
            entry["note"] = "MRR/Hit 微平均与宏平均恒等，复用宏平均区间。"
        report["micro"][metric] = entry

    return report


def main():
    report = build_report()
    with open("ci_data.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    cfg = report["config"]
    print(f"k={cfg['k']}  查询数={cfg['num_queries']}  "
          f"零相关查询={cfg['num_zero_relevance']}  "
          f"自助重采样={cfg['n_boot']}  alpha={cfg['alpha']}")
    print(f"{'指标':<6}{'宏平均':>10}{'宏 95% CI':>24}{'微平均':>10}{'微 95% CI':>24}")
    for metric in M.METRICS:
        macro = report["macro"][metric]
        micro = report["micro"][metric]
        print(f"{metric:<6}{macro['point']:>10.4f}"
              f"  [{macro['ci'][0]:.4f}, {macro['ci'][1]:.4f}]"
              f"{micro['point']:>10.4f}"
              f"  [{micro['ci'][0]:.4f}, {micro['ci'][1]:.4f}]")
    print()
    for metric in M.METRICS:
        warning = report["macro"][metric]["ci_info"].get("warning")
        if warning:
            print(f"[稳健性提示] {metric}: {warning}")
    print()
    print("宏/微口径归因：")
    print(report["aggregation_explanation"])
    print("\n区间数据已写入 ci_data.json")


if __name__ == "__main__":
    main()
