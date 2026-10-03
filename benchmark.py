#!/usr/bin/env python3
"""Offline benchmark: CF paths vs cold-start strategies on a sparse matrix.

Writes:
    results/metrics.json         -- every metric, structured
    results/warm_metrics.csv     -- warm-user comparison (P/R/NDCG + RMSE)
    results/coldstart_metrics.csv-- semi-cold / cold-user / cold-item +
                                   coverage / diversity / novelty
and prints the same tables.

Run:  python3 benchmark.py
"""

import csv
import json
import os

from cfrec.dataset import make_split
from cfrec.metrics import (
    aggregate_ranking,
    catalog_coverage,
    category_entropy,
    intra_list_diversity,
    long_tail_share,
    novelty,
    rmse,
)
from cfrec.recommenders import (
    CategoryPopularityRecommender,
    ContentRecommender,
    HybridColdStartRecommender,
    ItemCFRecommender,
    PopularityRecommender,
    UserCFRecommender,
)
from cfrec.similarity import binary_cosine

K = 10
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_models(train, catalog):
    user_cf = UserCFRecommender().fit(train, catalog)
    item_cf = ItemCFRecommender().fit(train, catalog)
    content = ContentRecommender().fit(train, catalog)
    cat_pop = CategoryPopularityRecommender().fit(train, catalog)
    popularity = PopularityRecommender().fit(train, catalog)
    hybrid = HybridColdStartRecommender(
        user_cf, item_cf, content, cat_pop).fit(train, catalog)
    return [popularity, user_cf, item_cf, content, cat_pop, hybrid]


def recs_for(model, test_matrix, train, k):
    out = {}
    for user_id in test_matrix:
        seen = set(train.user_ratings(user_id))
        rec = model.recommend(user_id, k=k, exclude=seen)
        out[user_id] = rec
    return out


def rmse_for(model, test_matrix):
    pairs = []
    for user_id, items in test_matrix.items():
        for item_id, rating in items.items():
            pairs.append((model.predict(user_id, item_id), rating))
    return rmse(pairs) if pairs else None


def cold_item_recall(model, cold_item_test, cold_items, train, k):
    """Recall@k restricted to held-out interactions with unseen items."""
    hits = 0
    targets = 0
    for user_id, items in cold_item_test.items():
        relevant = {i for i, r in items.items() if r >= 4.0 and i in cold_items}
        if not relevant:
            continue
        seen = set(train.user_ratings(user_id))
        rec = [i for i, _ in model.recommend(user_id, k=k, exclude=seen)]
        hits += len(set(rec) & relevant)
        targets += len(relevant)
    return hits / float(targets) if targets else 0.0


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split = make_split()
    train, catalog = split.train, split.catalog
    models = build_models(train, catalog)

    item_sim = {}
    ids = catalog.ids()
    for a_idx, a in enumerate(ids):
        for b in ids[a_idx + 1:]:
            s = binary_cosine(catalog.get(a).features, catalog.get(b).features)
            if s > 0.0:
                item_sim.setdefault(a, {})[b] = s
                item_sim.setdefault(b, {})[a] = s
    popularity = train.item_popularity()

    buckets = {
        "warm": split.warm_test,
        "semi_cold": split.semi_cold_test,
        "cold_user": split.cold_user_test,
    }

    all_recs = {}
    results = {}
    for model in models:
        model_results = {}
        for bucket_name, test_matrix in buckets.items():
            recs = recs_for(model, test_matrix, train, K)
            all_recs[(model.name, bucket_name)] = recs
            metrics = aggregate_ranking(recs, test_matrix, K)
            served = {u: r for u, r in recs.items() if r}
            metrics["catalog_coverage"] = catalog_coverage(served, len(catalog))
            metrics["intra_list_diversity"] = intra_list_diversity(served, item_sim)
            metrics["category_entropy"] = category_entropy(served, catalog)
            metrics["novelty"] = novelty(served, popularity)
            metrics["long_tail_share"] = long_tail_share(served, popularity)
            if bucket_name in ("warm", "semi_cold"):
                score = rmse_for(model, test_matrix)
                metrics["rmse"] = score
            model_results[bucket_name] = metrics
        model_results["cold_item_recall@10"] = cold_item_recall(
            model, split.cold_item_test, split.cold_items, train, K)
        results[model.name] = model_results

    print_dataset_stats(split)
    print_table_1(results)
    print_table_2(results)
    write_outputs(results, split)
    return results


def print_dataset_stats(split):
    print("=" * 78)
    print("DATASET (synthetic, seeded; standard library only)")
    print("-" * 78)
    print("train ratings=%d  users=%d  items=%d  density=%.3f%%"
          % (len(split.train), split.train.n_users, split.train.n_items,
             split.train.density() * 100))
    print("warm eval users=%d  semi-cold=%d  cold users=%d  cold-item queries=%d"
          % (len(split.warm_test), len(split.semi_cold_test),
             len(split.cold_user_test),
             sum(len(v) for v in split.cold_item_test.values())))
    print("=" * 78)


def print_table_1(results):
    print("\nWARM USERS: two CF paths vs baselines (top-10)")
    header = ("model", "P@10", "R@10", "NDCG@10", "RMSE",
              "cat_cov", "ILD", "entropy", "novelty", "tail%")
    print("%-15s %6s %6s %7s %6s %7s %6s %7s %7s %6s" % header)
    for name, row in results.items():
        m = row["warm"]
        print("%-15s %.3f  %.3f  %.3f   %s %.3f  %.3f  %.3f   %.3f  %.3f"
              % (name, m["precision@10"], m["recall@10"], m["ndcg@10"],
                 ("%.3f" % m["rmse"]) if m["rmse"] is not None else "  n/a",
                 m["catalog_coverage"], m["intra_list_diversity"],
                 m["category_entropy"], m["novelty"], m["long_tail_share"]))


def print_table_2(results):
    print("\nCOLD START: semi-cold (1-3 train) / cold user (0 train) / cold item")
    header = ("model", "semi_R@10", "semi_RMSE", "colduser_R@10",
              "user_cov", "cat_cov", "colditem_R@10", "tail%")
    print("%-15s %9s %9s %13s %8s %7s %14s %6s" % header)
    for name, row in results.items():
        semi = row["semi_cold"]
        cu = row["cold_user"]
        print("%-15s %.3f     %s   %.3f        %.2f   %.3f  %.3f          %.3f"
              % (name, semi["recall@10"],
                 ("%.3f" % semi["rmse"]) if semi["rmse"] is not None else " n/a",
                 cu["recall@10"], cu["user_coverage"], cu["catalog_coverage"],
                 row["cold_item_recall@10"], cu["long_tail_share"]))
    print("\n(user_cov = share of cold users the model can serve at all; "
          "tail% = long-tail exposure among cold-user recs)")


def write_outputs(results, split):
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as fh:
        json.dump({
            "dataset": {
                "train_ratings": len(split.train),
                "train_users": split.train.n_users,
                "train_items": split.train.n_items,
                "density": split.train.density(),
                "warm_eval_users": len(split.warm_test),
                "semi_cold_users": len(split.semi_cold_test),
                "cold_users": len(split.cold_user_test),
                "cold_item_queries": sum(len(v) for v in split.cold_item_test.values()),
            },
            "k": K,
            "models": results,
        }, fh, indent=2, sort_keys=True)

    with open(os.path.join(RESULTS_DIR, "warm_metrics.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "precision@10", "recall@10", "ndcg@10",
                         "rmse", "catalog_coverage", "intra_list_diversity",
                         "category_entropy", "novelty", "long_tail_share"])
        for name, row in results.items():
            m = row["warm"]
            writer.writerow([name, m["precision@10"], m["recall@10"],
                             m["ndcg@10"], m["rmse"], m["catalog_coverage"],
                             m["intra_list_diversity"], m["category_entropy"],
                             m["novelty"], m["long_tail_share"]])

    with open(os.path.join(RESULTS_DIR, "coldstart_metrics.csv"), "w",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "semi_cold_recall@10", "semi_cold_rmse",
                         "semi_cold_catalog_coverage",
                         "cold_user_recall@10", "cold_user_user_coverage",
                         "cold_user_catalog_coverage",
                         "cold_item_recall@10", "cold_user_long_tail_share"])
        for name, row in results.items():
            semi = row["semi_cold"]
            cu = row["cold_user"]
            writer.writerow([name, semi["recall@10"], semi["rmse"],
                             semi["catalog_coverage"], cu["recall@10"],
                             cu["user_coverage"], cu["catalog_coverage"],
                             row["cold_item_recall@10"],
                             cu["long_tail_share"]])
    print("\nWrote results/metrics.json, results/warm_metrics.csv, "
          "results/coldstart_metrics.csv")


if __name__ == "__main__":
    main()
