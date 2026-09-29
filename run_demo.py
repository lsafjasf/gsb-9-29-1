#!/usr/bin/env python3
"""Generate the sparse dataset, run every strategy and emit offline metrics.

Outputs:
    results/metrics.csv      all strategies x all evaluation slices
    results/metrics.json     same data plus dataset statistics
    stdout                   formatted comparison tables
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from recsys.data import generate_dataset
from recsys.metrics import evaluate
from recsys.recommenders import STRATEGIES

TOP_N = 10
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

METRIC_ORDER = [
    "precision@10",
    "recall@10",
    "ndcg@10",
    "catalog_coverage",
    "category_coverage",
    "diversity",
    "intra_list_similarity",
    "novelty",
    "gini",
    "cold_item_exposure",
    "cold_item_recall",
    "users_evaluated",
]

STRATEGY_DESCRIPTIONS = {
    "popularity": "纯热门兜底（基线）",
    "user_cf": "基于用户的协同过滤（Pearson）",
    "item_cf": "基于物品的协同过滤（Adjusted Cosine）",
    "content": "内容特征（类别+标签，加权 Jaccard）",
    "cat_pop": "类别热门（个人偏好类别内取热门）",
    "explore": "探索位（epsilon-greedy 长尾随机）",
    "hybrid": "混合（item-CF + 内容 + 热门，按支持度加权）",
}


def format_table(title: str, rows: List[Dict[str, float]]) -> str:
    headers = ["strategy", *[m.replace("@10", "@%d" % TOP_N) for m in METRIC_ORDER[:-1]]]
    lines = [title, "-" * len(title)]
    lines.append(" | ".join(f"{h:>17}" for h in headers))
    for row in rows:
        cells = [f"{row['strategy']:>17}"]
        for metric in METRIC_ORDER[:-1]:
            cells.append(f"{row[metric]:>17.4f}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main() -> None:
    dataset = generate_dataset()
    test_matrix = dataset.test_matrix()

    warm_users = sorted(user for user in test_matrix if user not in dataset.cold_users)
    cold_users = sorted(user for user in test_matrix if user in dataset.cold_users)

    data_stats = {
        "num_users": dataset.num_users,
        "num_items": dataset.num_items,
        "train_ratings": len(dataset.train),
        "test_ratings": len(dataset.test),
        "matrix_density": dataset.density,
        "matrix_sparsity": 1.0 - dataset.density,
        "cold_users": sorted(dataset.cold_users),
        "cold_items": sorted(dataset.cold_items),
        "cold_users_in_test": len(cold_users),
        "cold_item_test_interactions": sum(
            1 for _u, item, _r in dataset.test if item in dataset.cold_items
        ),
    }

    print("== 数据集 ==")
    print(
        f"用户 {dataset.num_users} / 物品 {dataset.num_items} / "
        f"训练评分 {len(dataset.train)} / 测试评分 {len(dataset.test)}"
    )
    print(f"矩阵密度 {dataset.density:.4%}（稀疏度 {1 - dataset.density:.4%}）")
    print(f"冷启动用户 {len(dataset.cold_users)} 个，冷启动物品 {len(dataset.cold_items)} 个")
    print()

    slices = {
        "all": sorted(test_matrix.keys()),
        "warm_users": warm_users,
        "cold_users": cold_users,
    }

    all_rows: List[Dict[str, float]] = []
    table_rows: Dict[str, List[Dict[str, float]]] = {}
    for name, cls in STRATEGIES.items():
        recommender = cls(dataset)
        for slice_name, users in slices.items():
            metrics = evaluate(recommender, dataset, top_n=TOP_N, users=users)
            row = {"strategy": name, "slice": slice_name,
                   "description": STRATEGY_DESCRIPTIONS[name], **metrics}
            all_rows.append(row)
            table_rows.setdefault(slice_name, []).append(row)

    for slice_name, title in (
        ("all", "全部用户（含冷启动）"),
        ("warm_users", "老用户切片"),
        ("cold_users", "冷启动用户切片"),
    ):
        print(format_table(title, table_rows[slice_name]))
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    fieldnames = ["strategy", "slice", "description", *METRIC_ORDER]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"dataset": data_stats, "rows": all_rows},
                  handle, ensure_ascii=False, indent=2)

    print(f"指标已写入 {csv_path} 和 {json_path}")


if __name__ == "__main__":
    main()
