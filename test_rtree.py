"""R-tree 自测：正确性对拍、边界用例、删除一致性、退化告警、百万对象与内存实测。

运行：
    python3 test_rtree.py -v
"""

from __future__ import annotations

import gc
import random
import resource
import sys
import time
import tracemalloc
import unittest
import warnings

from rtree import RTree, rects_intersect, rect_contains

SPAN = 10000.0


def make_dataset(n, rng, max_side=50.0, same=False):
    data = []
    for i in range(n):
        if same:
            x = y = 0.0
            w = h = 100.0
        else:
            x = rng.uniform(0, SPAN)
            y = rng.uniform(0, SPAN)
            w = rng.uniform(0.0, max_side)
            h = rng.uniform(0.0, max_side)
        data.append(((x, y, x + w, y + h), i))
    return data


def random_window(rng, max_side=200.0):
    x = rng.uniform(-100, SPAN)
    y = rng.uniform(-100, SPAN)
    w = rng.uniform(0.0, max_side)
    h = rng.uniform(0.0, max_side)
    return (x, y, x + w, y + h)


def brute_intersect(data, q):
    return {it for r, it in data if rects_intersect(r, q)}


def brute_within(data, q):
    return {it for r, it in data if rect_contains(q, r)}


# ============================================================ 边界与正确性

class TestEdgeCases(unittest.TestCase):

    def test_empty_index(self):
        t = RTree()
        self.assertEqual(len(t), 0)
        self.assertEqual(t.query_intersect((0, 0, 1, 1)), [])
        self.assertEqual(t.query_within((0, 0, 1, 1)), [])
        self.assertEqual(t.items(), [])
        self.assertEqual(t.stats()["nodes"], 0)
        self.assertFalse(t.check_degradation())
        with self.assertRaises(KeyError):
            t.delete((0, 0, 1, 1), 0)
        # 对空集合 bulk_load 也应正常
        t.load(iter(()))
        self.assertEqual(len(t), 0)

    def test_single_huge_object(self):
        t = RTree()
        huge = (-1e9, -1e9, 1e9, 1e9)
        t.insert(huge, "big")
        self.assertEqual(len(t), 1)
        self.assertEqual(t.query_intersect((0.0, 0.0, 1.0, 1.0)), ["big"])
        self.assertEqual(t.query_intersect((2e9, 2e9, 3e9, 3e9)), [])
        # 边界相接也算相交
        self.assertEqual(t.query_intersect((1e9, 1e9, 1e9 + 1, 1e9 + 1)), ["big"])
        self.assertEqual(t.query_within((-2e9, -2e9, 2e9, 2e9)), ["big"])
        self.assertEqual(t.query_within((0.0, 0.0, 1.0, 1.0)), [])
        t.delete(huge, "big")
        self.assertEqual(len(t), 0)
        self.assertEqual(t.query_intersect(huge), [])
        # 删错对象要报错
        t.insert(huge, "big")
        with self.assertRaises(KeyError):
            t.delete(huge, "other")

    def test_point_and_reversed_rects(self):
        t = RTree()
        t.insert((5.0, 5.0, 5.0, 5.0), 0)   # 零面积点
        t.insert((9.0, 9.0, 1.0, 1.0), 1)   # 逆序矩形，插入时归一化
        # 点 (5,5) 同时落在归一化后的矩形 (1,1)-(9,9) 内
        self.assertEqual(set(t.query_intersect((5, 5, 5, 5))), {0, 1})
        self.assertEqual(set(t.query_intersect((10, 10, 10, 10))), set())
        self.assertEqual(set(t.query_intersect((1, 1, 9, 9))), {0, 1})


class TestCorrectness(unittest.TestCase):

    def _assert_same_brute(self, tree, data, windows):
        for q in windows:
            self.assertEqual(set(tree.query_intersect(q)), brute_intersect(data, q),
                             "相交查询与暴力扫描不一致")
            self.assertEqual(set(tree.query_within(q)), brute_within(data, q),
                             "范围查询与暴力扫描不一致")

    def test_dynamic_insert_small(self):
        rng = random.Random(20240101)
        data = make_dataset(3000, rng, max_side=80.0)
        tree = RTree()
        for r, it in data:
            tree.insert(r, it)
        windows = [random_window(rng) for _ in range(200)]
        self._assert_same_brute(tree, data, windows)
        self.assertEqual(len(tree), len(data))
        s = tree.stats()
        self.assertEqual(s["size"], len(data))
        self.assertFalse(tree.check_degradation())  # 分散数据不应告警

    def test_bulk_load_medium(self):
        rng = random.Random(424242)
        data = make_dataset(50000, rng, max_side=40.0)
        tree = RTree.bulk_load(data)
        windows = [random_window(rng) for _ in range(200)]
        self._assert_same_brute(tree, data, windows)
        s = tree.stats()
        # STR 打包后内部节点应接近满载
        self.assertGreaterEqual(s["avg_fill"], 0.9 * tree.max_children)
        self.assertFalse(tree.check_degradation())

    def test_dynamic_delete(self):
        rng = random.Random(777)
        data = make_dataset(5000, rng, max_side=80.0)
        tree = RTree.bulk_load(data)
        alive = {it: r for r, it in data}
        victims = rng.sample(list(alive), 1500)
        for it in victims:
            tree.delete(alive.pop(it), it)
        self.assertEqual(len(tree), len(alive))
        windows = [random_window(rng) for _ in range(300)]
        cur = [(r, it) for it, r in alive.items()]
        self._assert_same_brute(tree, cur, windows)
        # 索引内剩余项集合也要一致
        self.assertEqual({it for _r, it in tree.items()}, set(alive))

    def test_insert_delete_mixed(self):
        rng = random.Random(31337)
        data = make_dataset(2000, rng, max_side=120.0)
        tree = RTree()
        model = {}  # item -> rect
        next_id = 0
        for _ in range(8000):
            if not model or rng.random() < 0.6:
                r = (rng.uniform(0, SPAN), rng.uniform(0, SPAN))
                w = rng.uniform(0, 120)
                h = rng.uniform(0, 120)
                rect = (r[0], r[1], r[0] + w, r[1] + h)
                tree.insert(rect, next_id)
                model[next_id] = rect
                next_id += 1
            else:
                it = rng.choice(list(model))
                tree.delete(model.pop(it), it)
        self.assertEqual(len(tree), len(model))
        cur = [(r, it) for it, r in model.items()]
        windows = [random_window(rng, max_side=400.0) for _ in range(300)]
        self._assert_same_brute(tree, cur, windows)

    def test_all_overlapping_warns(self):
        n = 4000
        data = make_dataset(n, random.Random(0), same=True)
        tree = RTree.bulk_load(data)
        print(f"\n[全部重叠] {n} 个完全相同矩形, 平均成对 IoU = "
              f"{tree.stats()['avg_overlap']:.3f}")
        # 所有矩形相同：结果仍必须与暴力扫描一致
        q = (50, 50, 60, 60)
        self.assertEqual(set(tree.query_intersect(q)), brute_intersect(data, q))
        self.assertEqual(set(tree.query_within((-1, -1, 101, 101))),
                         brute_within(data, (-1, -1, 101, 101)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            degraded = tree.check_degradation()
        self.assertTrue(degraded)
        self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught))
        # 查询放大：几乎每个节点都要访问
        self.assertGreater(tree.last_query_visited, 0.9 * tree.stats()["nodes"])

    def test_duplicates_and_rebuild(self):
        t = RTree()
        r = (0.0, 0.0, 10.0, 10.0)
        for _ in range(5):
            t.insert(r, "x")
        self.assertEqual(len(t), 5)
        t.delete(r, "x")  # 只删一条
        self.assertEqual(len(t), 4)
        # load 替换现有内容
        data = [((float(i), 0.0, float(i) + 1.0, 1.0), i) for i in range(100)]
        t.load(data)
        self.assertEqual(len(t), 100)
        self.assertEqual(set(t.query_intersect((10, 0, 20, 1))),
                         {i for i in range(9, 21)})


# ============================================================ 百万对象与内存

def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class TestMillionAndMemory(unittest.TestCase):

    def test_million_objects(self):
        n = 1_000_000
        rng = random.Random(20251001)
        print("\n[百万对象] 生成数据 ...", flush=True)
        data = make_dataset(n, rng, max_side=20.0)

        rss0 = _rss_mb()
        t0 = time.perf_counter()
        tree = RTree.bulk_load(data)
        build_s = time.perf_counter() - t0
        rss1 = _rss_mb()
        struct_bytes = tree.memory_bytes()
        s = tree.stats()

        print("=" * 64)
        print(f"对象数 N            : {n:,}")
        print(f"STR 批量加载耗时   : {build_s:.2f} s")
        print(f"树高 / 节点数      : {s['height']} / {s['nodes']:,} "
              f"(叶 {s['leaf_nodes']:,}, 内部 {s['internal_nodes']:,})")
        print(f"内部节点平均填充   : {s['avg_fill']:.2f} / {tree.max_children}")
        print(f"平均兄弟重叠度     : {s['avg_overlap']:.4f}")
        print(f"结构占用(memory_bytes, 不含源data): {struct_bytes/1e6:.1f} MB, "
              f"{struct_bytes/n:.1f} 字节/对象")
        print(f"进程峰值 RSS 增量  : {rss1 - rss0:.1f} MB")
        print("=" * 64)

        # 对拍：10 个随机窗口逐项比较
        windows = [random_window(rng, max_side=100.0) for _ in range(10)]
        t0 = time.perf_counter()
        for q in windows:
            got = set(tree.query_intersect(q))
            expect = brute_intersect(data, q)
            self.assertEqual(got, expect, "百万数据相交查询与暴力扫描不一致")
            self.assertEqual(set(tree.query_within(q)), brute_within(data, q))
        qs = (time.perf_counter() - t0) / len(windows) * 1000
        print(f"索引+暴力对拍平均每窗口: {qs:.1f} ms（含百万级暴力扫描）")

        # 索引查询单独计时 + 查询放大
        t0 = time.perf_counter()
        for q in windows:
            tree.query_intersect(q)
        idx_ms = (time.perf_counter() - t0) / len(windows) * 1000
        t0 = time.perf_counter()
        for q in windows:
            brute_intersect(data, q)
        brute_ms = (time.perf_counter() - t0) / len(windows) * 1000
        print(f"纯索引平均 {idx_ms:.3f} ms/窗口 vs 暴力扫描 {brute_ms:.1f} ms/窗口"
              f"（加速 {brute_ms/idx_ms:.0f}x），访问节点 {tree.last_query_visited:,}")
        self.assertFalse(tree.check_degradation())
        self.assertEqual(len(tree), n)

        # 批量加载后仍支持动态插入
        before = set(tree.query_intersect((0.0, 0.0, 1.0, 1.0)))
        tree.insert((0.0, 0.0, 1.0, 1.0), n)
        after = set(tree.query_intersect((0.0, 0.0, 1.0, 1.0)))
        self.assertEqual(after - before, {n})
        tree.delete((0.0, 0.0, 1.0, 1.0), n)
        self.assertEqual(len(tree), n)

    def test_memory_growth_table(self):
        """节点数随对象数增长：线性内存、节点数 ≈ N/(Mf) 的几何级数和。"""
        print("\n[内存与节点增长]（tracemalloc 精确计量）")
        print(f"{'N':>10} | {'叶节点':>8} | {'内部':>7} | {'高':>2} | "
              f"{'结构MB':>8} | {'B/对象':>7}")
        rows = []
        for n in (10_000, 50_000, 200_000):
            data = make_dataset(n, random.Random(n), max_side=20.0)
            gc.collect()
            tracemalloc.start()
            cur0, peak0 = tracemalloc.get_traced_memory()
            tree = RTree.bulk_load(data)
            cur1, peak1 = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            s = tree.stats()
            mb = tree.memory_bytes() / 1e6
            rows.append((n, s, mb, (peak1 - peak0) / 1e6))
            print(f"{n:>10,} | {s['leaf_nodes']:>8,} | {s['internal_nodes']:>7,} | "
                  f"{s['height']:>2} | {mb:>8.2f} | {tree.memory_bytes()/n:>7.1f}")
        # 动态插入（非打包）的内存与填充率对比
        dn = 20_000
        ddata = make_dataset(dn, random.Random(dn), max_side=20.0)
        t0 = time.perf_counter()
        dt = RTree()
        for r, it in ddata:
            dt.insert(r, it)
        dyn_s = time.perf_counter() - t0
        ds = dt.stats()
        print(f"{'动态插入':>10} | {ds['leaf_nodes']:>8,} | {ds['internal_nodes']:>7,} | "
              f"{ds['height']:>2} | {dt.memory_bytes()/1e6:>8.2f} | "
              f"{dt.memory_bytes()/dn:>7.1f}  (20,000 条逐条插入 {dyn_s:.2f}s, "
              f"平均填充 {ds['avg_fill']:.1f}/{dt.max_children})")
        # 动态树正确性抽验
        drng = random.Random(99)
        for _ in range(50):
            q = random_window(drng)
            assert set(dt.query_intersect(q)) == brute_intersect(ddata, q)
            assert set(dt.query_within(q)) == brute_within(ddata, q)

        # 线性检验：20 万数据的每对象字节数与 1 万数据相差不超过 15%
        bps_lo = rows[0][2] * 1e6 / rows[0][0]
        bps_hi = rows[-1][2] * 1e6 / rows[-1][0]
        self.assertLess(abs(bps_hi - bps_lo) / bps_lo, 0.15)
        print(f"每对象字节数 1万 vs 20万: {bps_lo:.1f} vs {bps_hi:.1f} "
              f"（线性 O(N)，偏差 <15%）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
