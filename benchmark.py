"""Recall / candidate-count / timing benchmark: LSH vs brute force.

Generates a clustered dataset (near-duplicate groups plus random distractors),
computes ground truth by exhaustive cosine comparison, then measures:

  * recall        = |retrieved & truth| / |truth|, averaged over queries
  * avg candidates per query (before exact re-ranking)
  * avg query time (signature + bucket lookup + exact re-rank of candidates)
  * avg brute-force query time (full scan) for comparison

A parameter sweep over (num_hashes k, num_tables L) shows the recall /
candidate / speed trade-off. Run: python3 benchmark.py
"""

import random
import time
import tracemalloc

from lsh import LSHIndex, cosine_similarity

THRESHOLD = 0.9
SEED = 99


def unit(vec):
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def random_unit_vector(rng, dim):
    return unit([rng.gauss(0.0, 1.0) for _ in range(dim)])


def noisy_copy(rng, vec, rel_noise):
    sigma = rel_noise / len(vec) ** 0.5
    return unit([x + rng.gauss(0.0, sigma) for x in vec])


def build_dataset(rng, dim, n_bases, copies_per_base, n_singletons, rel_noise):
    items, bases = {}, []
    for b in range(n_bases):
        base = random_unit_vector(rng, dim)
        bases.append(base)
        for c in range(copies_per_base):
            items[f"b{b}-c{c}"] = noisy_copy(rng, base, rel_noise)
    for s in range(n_singletons):
        items[f"rand-{s}"] = random_unit_vector(rng, dim)
    return bases, items


def brute_force(items, query):
    return {key for key, vec in items.items()
            if cosine_similarity(query, vec) >= THRESHOLD}


def evaluate(items, queries, truths, num_hashes, num_tables):
    index = LSHIndex(num_hashes=num_hashes, num_tables=num_tables, seed=SEED)
    start = time.perf_counter()
    for key, vec in items.items():
        index.add(key, vec)
    build_s = time.perf_counter() - start

    total_candidates = 0
    recall_sum = 0.0
    n_scored = 0
    query_s = 0.0
    for query, truth in zip(queries, truths):
        start = time.perf_counter()
        candidates = index.query(query)
        retrieved = {key for key in candidates
                     if cosine_similarity(query, items[key]) >= THRESHOLD}
        query_s += time.perf_counter() - start
        total_candidates += len(candidates)
        if truth:
            recall_sum += len(retrieved & truth) / len(truth)
            n_scored += 1
    return {
        "recall": recall_sum / n_scored if n_scored else 1.0,
        "avg_candidates": total_candidates / len(queries),
        "query_ms": query_s / len(queries) * 1000,
        "build_s": build_s,
        "index": index,
    }


def main():
    rng = random.Random(2024)
    dim = 128

    # ---- main benchmark: n = 2000, 100 queries ----
    bases, items = build_dataset(rng, dim, n_bases=400, copies_per_base=4,
                                 n_singletons=400, rel_noise=0.2)
    queries = [noisy_copy(rng, bases[rng.randrange(len(bases))], 0.2)
               for _ in range(100)]

    start = time.perf_counter()
    truths = [brute_force(items, q) for q in queries]
    brute_ms = (time.perf_counter() - start) / len(queries) * 1000

    print(f"dataset: {len(items)} items, dim={dim}, "
          f"{len(queries)} queries, threshold={THRESHOLD}")
    print(f"avg ground-truth neighbours per query: "
          f"{sum(len(t) for t in truths) / len(truths):.2f}")
    print(f"brute-force scan: {brute_ms:.2f} ms/query\n")

    for k, L in ((8, 16), (12, 16)):
        result = evaluate(items, queries, truths, num_hashes=k, num_tables=L)
        stats = result["index"].memory_stats()
        print(f"main config: k={k}, L={L}")
        print(f"  recall           : {result['recall']:.4f}")
        print(f"  avg candidates   : {result['avg_candidates']:.1f} / {len(items)}")
        print(f"  query time       : {result['query_ms']:.2f} ms "
              f"(brute force: {brute_ms:.2f} ms, "
              f"speedup {brute_ms / result['query_ms']:.1f}x)")
        print(f"  build time       : {result['build_s']:.2f} s")
        print(f"  nonempty buckets : {stats['nonempty_buckets']} "
              f"(bound L*min(n,2^k) = {L * min(len(items), 1 << k)})")
        print(f"  memberships      : {stats['bucket_memberships']} "
              f"(= n*L = {len(items) * L})")
        print(f"  weight cache     : {stats['weight_cache_entries']} floats "
              f"(bound L*k*dim = {L * k * dim})")
        print()

    # Peak allocation measured in a separate build pass: tracemalloc
    # instruments every allocation and would corrupt the timing above.
    tracemalloc.start()
    mem_index = LSHIndex(num_hashes=12, num_tables=16, seed=SEED)
    for key, vec in items.items():
        mem_index.add(key, vec)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"peak alloc building k=12/L=16 index on {len(items)} vectors "
          f"(tracemalloc, incl. the vectors themselves): {peak / 2**20:.1f} MiB")

    # ---- parameter sweep: smaller dataset for speed ----
    rng = random.Random(7)
    dim_s = 96
    bases_s, items_s = build_dataset(rng, dim_s, n_bases=250, copies_per_base=4,
                                     n_singletons=200, rel_noise=0.2)
    queries_s = [noisy_copy(rng, bases_s[rng.randrange(len(bases_s))], 0.2)
                 for _ in range(60)]
    truths_s = [brute_force(items_s, q) for q in queries_s]
    start = time.perf_counter()
    for q in queries_s:
        brute_force(items_s, q)
    brute_ms_s = (time.perf_counter() - start) / len(queries_s) * 1000

    print(f"\nsweep dataset: {len(items_s)} items, dim={dim_s}, "
          f"{len(queries_s)} queries, brute force {brute_ms_s:.2f} ms/query")
    print("| k (hashes) | L (tables) | recall | avg candidates | query ms |")
    print("|---|---|---|---|---|")
    for k in (4, 8, 12):
        for L in (4, 8, 16, 32):
            r = evaluate(items_s, queries_s, truths_s, k, L)
            print(f"| {k} | {L} | {r['recall']:.4f} | "
                  f"{r['avg_candidates']:.1f} | {r['query_ms']:.2f} |")


if __name__ == "__main__":
    main()
