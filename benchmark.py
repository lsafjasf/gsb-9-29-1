"""召回率 / 候选数 / 耗时基准：LSH vs 暴力全量比对。

生成带"植入近重复"的合成数据集：
  - 150 个簇，每簇 1 个基向量 + 9 个加噪副本（簇内余弦相似度约 0.85~0.95）
  - 500 个随机孤立向量
以暴力比对（全量 O(n) 余弦扫描）的结果作为 ground truth 计算召回率。
"""

import random
import time

from lsh import LSH, cosine

DIM = 128
N_CLUSTERS = 150
DUPS_PER_CLUSTER = 9
N_SINGLES = 500
NOISE = 0.5
THRESHOLD = 0.8
N_QUERIES = 300
SEED = 7


def gen_data():
    rng = random.Random(SEED)
    vecs = []
    for _ in range(N_CLUSTERS):
        base = [rng.gauss(0, 1) for _ in range(DIM)]
        vecs.append(base)
        for _ in range(DUPS_PER_CLUSTER):
            vecs.append([b + rng.gauss(0, NOISE) for b in base])
    for _ in range(N_SINGLES):
        vecs.append([rng.gauss(0, 1) for _ in range(DIM)])
    rng.shuffle(vecs)
    return vecs


def brute_force(vecs, q, threshold):
    return {i for i, v in enumerate(vecs) if cosine(q, v) >= threshold}


def main():
    vecs = gen_data()
    n = len(vecs)
    rng = random.Random(SEED + 1)
    qidx = rng.sample(range(n), N_QUERIES)
    queries = [vecs[i] for i in qidx]
    print(f"数据集: n={n}, dim={DIM}, 阈值={THRESHOLD}, 查询数={N_QUERIES}")

    # ---- 暴力比对：ground truth + 基准耗时 ----
    t0 = time.perf_counter()
    gt = [brute_force(vecs, q, THRESHOLD) for q in queries]
    t_brute = (time.perf_counter() - t0) / N_QUERIES
    nonempty = sum(1 for g in gt if g)
    print(f"暴力比对: 平均 {t_brute*1e3:.2f} ms/查询, "
          f"平均命中 {sum(map(len, gt))/N_QUERIES:.1f} 个/查询, "
          f"{nonempty}/{N_QUERIES} 个查询有真近邻")

    # ---- 不同 (k, L) 参数下的 LSH 表现 ----
    header = (f"{'k':>3} {'L':>3} | {'召回率':>7} | {'平均候选数':>9} | "
              f"{'查询ms':>7} | {'构建s':>6} | {'加速比':>6} | {'索引MB':>7}")
    print(header)
    print("-" * len(header))
    for k, L in [(8, 16), (10, 24), (10, 48), (12, 48)]:
        idx = LSH(num_hash=k, num_tables=L, threshold=THRESHOLD, dim=DIM, seed=SEED)
        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            idx.insert(i, v)
        t_build = time.perf_counter() - t0

        recalls, cands = [], []
        t0 = time.perf_counter()
        for q, g in zip(queries, gt):
            res = {key for key, _ in idx.query(q)}
            cands.append(idx.last_num_candidates)
            if g:
                recalls.append(len(res & g) / len(g))
        t_query = (time.perf_counter() - t0) / N_QUERIES

        recall = sum(recalls) / len(recalls)
        avg_cand = sum(cands) / len(cands)
        mb = idx.stats()["estimated_index_bytes"] / 1e6
        print(f"{k:>3} {L:>3} | {recall:>7.4f} | {avg_cand:>9.1f} | "
              f"{t_query*1e3:>7.3f} | {t_build:>6.2f} | "
              f"{t_brute/t_query:>5.0f}x | {mb:>7.2f}")


if __name__ == "__main__":
    main()
