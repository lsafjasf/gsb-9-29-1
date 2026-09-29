"""LSH：随机超平面局部敏感哈希，用于高维向量的近似重复检索（余弦相似度）。

仅依赖 Python 标准库。支持动态插入 / 删除，桶数与哈希函数数可配置。

原理：
    用 h = num_hash * num_tables 个随机超平面把向量投影到 h 个符号位上。
    每张表取其中 num_hash 个比特作为桶键，相似向量（夹角小）以高概率落入同一桶。
    余弦相似度 s 对应的单比特碰撞概率 p = 1 - arccos(s) / pi，
    单表命中概率 p^k，L 张表至少命中一次的概率 P = 1 - (1 - p^k)^L。
"""

import math
from collections import defaultdict

_MASK64 = (1 << 64) - 1


def _splitmix64(x):
    """64 位整数混合哈希，确定性、无状态。"""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _hash_gauss(seed, h, i):
    """由 (seed, 超平面编号 h, 维度 i) 确定性生成 N(0,1) 样本（Box-Muller）。

    哈希函数由此隐式定义：不需要为稀疏 / 超高维向量物化投影矩阵。
    """
    key = _splitmix64((seed ^ (h * 0x9E3779B1) ^ (i * 0x85EBCA77)) & _MASK64)
    u1 = (_splitmix64(key) + 0.5) / (1 << 64)
    u2 = (_splitmix64(key ^ 0xD1B54A32D192ED03) + 0.5) / (1 << 64)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _norm(vec):
    if isinstance(vec, dict):
        return math.sqrt(sum(x * x for x in vec.values()))
    return math.sqrt(sum(x * x for x in vec))


def cosine(a, b, na=None, nb=None):
    """余弦相似度，支持 list（稠密）与 dict（稀疏 {dim: value}）任意组合。"""
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) > len(b):
            a, b = b, a
        dot = sum(x * b.get(i, 0.0) for i, x in a.items())
    elif isinstance(a, dict):
        dot = sum(x * b[i] for i, x in a.items() if i < len(b))
    elif isinstance(b, dict):
        dot = sum(x * a[i] for i, x in b.items() if i < len(a))
    else:
        dot = sum(x * y for x, y in zip(a, b))
    if na is None:
        na = _norm(a)
    if nb is None:
        nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LSH:
    """随机超平面 LSH 索引。

    参数:
        num_hash (k):  每张表的签名位数（哈希函数数/表）。桶数上界为 2^k。
                       k 越大，桶越细，候选越少、越快，但召回率越低。
        num_tables (L): 独立哈希表数量。L 越大召回率越高，
                        候选数与内存随 L 线性增长。
        threshold:     query 返回结果的余弦相似度阈值（精确重判定）。
        dim:           稠密向量维度；为 None 时由首个稠密向量推断。
        seed:          随机种子，保证结果可复现。

    内存上界:
        每个 id 在每张表恰好出现一次，因此桶成员条目总数恒为 n * L，
        与 2^k 无关；每张表的桶数 <= min(2^k, n)。
        总内存 O(n * L + n * d)，详见 stats() 与 README。
    """

    def __init__(self, num_hash=10, num_tables=16, threshold=0.8, dim=None, seed=42):
        if num_hash < 1 or num_tables < 1:
            raise ValueError("num_hash 和 num_tables 必须为正整数")
        self.k = num_hash
        self.L = num_tables
        self.h = num_hash * num_tables
        self.threshold = threshold
        self.dim = dim
        self._seed = seed
        self._tables = [defaultdict(set) for _ in range(self.L)]
        self._vectors = {}   # id -> 向量引用（不复制）
        self._norms = {}     # id -> 范数
        self._sigs = {}      # id -> 每张表的桶键列表
        self._rows = None    # 稠密投影行缓存（惰性物化），上界 h * dim 个 float
        self.last_num_candidates = 0  # 最近一次 query 的候选数（供基准测试）

    # ---- 签名计算 ----

    def _ensure_rows(self):
        if self._rows is None:
            self._rows = [
                [_hash_gauss(self._seed, h, i) for i in range(self.dim)]
                for h in range(self.h)
            ]

    def _signature(self, vec):
        if isinstance(vec, dict):
            items = list(vec.items())
            dots = [0.0] * self.h
            for h in range(self.h):
                s = 0.0
                for i, x in items:
                    s += x * _hash_gauss(self._seed, h, i)
                dots[h] = s
        else:
            if self.dim is None:
                self.dim = len(vec)
            self._ensure_rows()
            rows = self._rows
            dots = [0.0] * self.h
            for h in range(self.h):
                row = rows[h]
                s = 0.0
                for j in range(self.dim):
                    s += row[j] * vec[j]
                dots[h] = s
        keys = []
        for t in range(self.L):
            key = 0
            base = t * self.k
            for j in range(self.k):
                if dots[base + j] >= 0.0:
                    key |= 1 << j
            keys.append(key)
        return keys

    # ---- 动态增删 ----

    def insert(self, key, vec):
        """插入向量；key 已存在时先删除旧值（即支持更新）。"""
        if key in self._vectors:
            self.delete(key)
        sig = self._signature(vec)
        self._vectors[key] = vec
        self._norms[key] = _norm(vec)
        self._sigs[key] = sig
        for t in range(self.L):
            self._tables[t][sig[t]].add(key)

    def delete(self, key):
        """删除向量；key 不存在返回 False，否则返回 True。"""
        if key not in self._vectors:
            return False
        sig = self._sigs.pop(key)
        del self._vectors[key]
        del self._norms[key]
        for t in range(self.L):
            bucket = self._tables[t][sig[t]]
            bucket.discard(key)
            if not bucket:
                del self._tables[t][sig[t]]  # 空桶即时回收
        return True

    # ---- 查询 ----

    def candidates(self, vec):
        """返回候选 id 集合（不做精确重判定）。"""
        sig = self._signature(vec)
        cand = set()
        for t in range(self.L):
            bucket = self._tables[t].get(sig[t])
            if bucket:
                cand.update(bucket)
        return cand

    def query(self, vec, threshold=None, max_results=None):
        """返回 [(id, 余弦相似度), ...]，按相似度降序，仅保留 >= threshold 的项。"""
        if threshold is None:
            threshold = self.threshold
        cand = self.candidates(vec)
        self.last_num_candidates = len(cand)
        if not cand:
            return []
        nq = _norm(vec)
        scored = []
        for key in cand:
            s = cosine(vec, self._vectors[key], na=nq, nb=self._norms[key])
            if s >= threshold:
                scored.append((key, s))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        if max_results is not None:
            scored = scored[:max_results]
        return scored

    # ---- 统计 ----

    def __len__(self):
        return len(self._vectors)

    def stats(self):
        """返回桶与内存估算信息。

        内存估算（字节，CPython 经验值）：
          - 桶成员条目：n * L 条，每条 ~80B（set 槽位 + 引用摊销）
          - 桶 dict 对象：每桶 ~232B，桶数 <= min(2^k, n) / 表
          - 稠密投影行缓存：h * dim 个 float（仅使用稠密向量时物化）
          - 向量本体只存引用，不计入索引开销
        """
        buckets_per_table = [len(t) for t in self._tables]
        entries = sum(len(s) for t in self._tables for s in t.values())
        est = entries * 80 + sum(buckets_per_table) * 232
        row_bytes = 0
        if self._rows is not None:
            row_bytes = self.h * (self.dim or 0) * 8
        return {
            "n": len(self._vectors),
            "num_hash (k)": self.k,
            "num_tables (L)": self.L,
            "max_buckets_per_table": min(1 << self.k, max(len(self._vectors), 1)),
            "buckets_per_table": buckets_per_table,
            "bucket_entries (== n*L)": entries,
            "projection_row_bytes": row_bytes,
            "estimated_index_bytes": est + row_bytes,
        }
