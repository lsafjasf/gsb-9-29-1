"""边界用例测试：空集合、单点、全相同向量、高维稀疏向量、动态增删。"""

import random

from lsh import LSH, cosine


def test_empty():
    idx = LSH(num_hash=8, num_tables=4)
    assert len(idx) == 0
    assert idx.query([1.0, 2.0, 3.0]) == []          # 空索引查询不报错
    assert idx.delete("missing") is False            # 删除不存在的 key
    s = idx.stats()
    assert s["n"] == 0 and s["bucket_entries (== n*L)"] == 0
    print("ok 空集合：查询返回空、删除安全、统计正常")


def test_single_point():
    idx = LSH(num_hash=8, num_tables=4, threshold=0.8)
    idx.insert("only", [1.0, 0.0, 0.0, 0.0])
    res = idx.query([1.0, 0.0, 0.0, 0.0])
    assert [k for k, _ in res] == ["only"] and abs(res[0][1] - 1.0) < 1e-12
    assert idx.query([0.0, 1.0, 0.0, 0.0]) == []     # 正交向量不命中
    print("ok 单点：自查询命中（sim=1.0），正交查询为空")


def test_all_identical():
    n = 200
    v = [0.5] * 64
    idx = LSH(num_hash=10, num_tables=8, threshold=0.99)
    for i in range(n):
        idx.insert(i, v)                             # 同一向量对象，只存引用
    res = idx.query(v)
    assert len(res) == n and all(abs(s - 1.0) < 1e-12 for _, s in res)
    assert idx.last_num_candidates == n              # 全部落同桶，候选=全集
    # 与暴力比对一致（ground truth 也是全集）
    gt = {i for i in range(n) if cosine(v, v) >= 0.99}
    assert {k for k, _ in res} == gt
    print(f"ok 全相同向量：{n} 个全部召回，recall=1.0，候选数={n}")


def test_high_dim_sparse():
    dim, nnz, n_base, n_dup = 1_000_000, 30, 40, 3
    rng = random.Random(1)
    idx = LSH(num_hash=8, num_tables=12, threshold=0.9)
    bases = []
    for b in range(n_base):
        base = {rng.randrange(dim): rng.gauss(0, 1) for _ in range(nnz)}
        bases.append(base)
        idx.insert(f"base{b}", base)
        for d in range(n_dup):                       # 缩放+微噪 -> 余弦 ~0.99
            dup = {i: x * 0.98 + rng.gauss(0, 0.01) for i, x in base.items()}
            idx.insert(f"base{b}_dup{d}", dup)
    # 每个基向量应召回自身 + 3 个副本
    recalls = []
    for b, base in enumerate(bases):
        got = {k for k, _ in idx.query(base)}
        want = {f"base{b}"} | {f"base{b}_dup{d}" for d in range(n_dup)}
        recalls.append(len(got & want) / len(want))
    recall = sum(recalls) / len(recalls)
    assert recall >= 0.99, f"稀疏召回率过低: {recall}"
    print(f"ok 高维稀疏：dim={dim}, nnz={nnz}, 近重复召回率={recall:.4f}")


def test_dynamic_insert_delete():
    idx = LSH(num_hash=8, num_tables=8, threshold=0.9)
    idx.insert("a", [1.0, 0.0, 0.0])
    idx.insert("b", [0.99, 0.01, 0.0])
    idx.insert("c", [0.0, 1.0, 0.0])
    assert {k for k, _ in idx.query([1.0, 0.0, 0.0])} == {"a", "b"}
    assert idx.delete("b") is True                   # 删除后不再出现
    assert {k for k, _ in idx.query([1.0, 0.0, 0.0])} == {"a"}
    idx.insert("a", [0.0, 0.0, 1.0])                 # 同 key 更新向量
    assert {k for k, _ in idx.query([1.0, 0.0, 0.0])} == set()
    assert {k for k, _ in idx.query([0.0, 0.0, 1.0])} == {"a"}
    assert len(idx) == 2
    # 删除后桶条目数必须收缩（内存随删除释放）
    for i in range(100):
        idx.insert(f"tmp{i}", [float(i % 3 == 0), float(i % 3 == 1), 0.5])
    before = idx.stats()["bucket_entries (== n*L)"]
    for i in range(100):
        idx.delete(f"tmp{i}")
    after = idx.stats()["bucket_entries (== n*L)"]
    assert before - after == 100 * idx.L and after == 2 * idx.L
    print("ok 动态增删：删除即时生效、同 key 可更新、桶条目随删除收缩")


def test_zero_vector():
    idx = LSH(num_hash=6, num_tables=4, threshold=0.5)
    idx.insert("zero", [0.0, 0.0, 0.0])
    idx.insert("nz", [1.0, 1.0, 1.0])
    res = dict(idx.query([1.0, 1.0, 1.0]))
    assert res.get("zero", 0.0) == 0.0               # 零向量相似度定义为 0，不崩溃
    print("ok 零向量：相似度按 0 处理，无除零异常")


if __name__ == "__main__":
    test_empty()
    test_single_point()
    test_all_identical()
    test_high_dim_sparse()
    test_dynamic_insert_delete()
    test_zero_vector()
    print("\n全部边界用例通过")
