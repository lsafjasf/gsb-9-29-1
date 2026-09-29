"""自测：边界用例 + 与精确几何（Fraction 裁剪）全集合对拍。

运行：python3 test_voxelgrid.py [-v]
"""

import random
import unittest

from voxelgrid import SparseVoxelGrid, tri_box_overlap
from exact import voxelize_exact


def make_grid(triangles, h, origin=(0.0, 0.0, 0.0)):
    g = SparseVoxelGrid(h, origin)
    g.voxelize(triangles)
    return g


class TestEdgeCases(unittest.TestCase):
    def test_empty_scene(self):
        g = make_grid([], h=1.0)
        self.assertEqual(len(g), 0)
        self.assertIsNone(g.bounds())
        self.assertEqual(g.count_range((-10, -10, -10), (10, 10, 10)), 0)
        self.assertEqual(list(g.iter_range((0, 0, 0), (5, 5, 5))), [])

    def test_single_voxel(self):
        # 小三角形完全落在体素 (0,0,0) 内
        tri = [((0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1))]
        g = make_grid(tri, h=1.0)
        self.assertEqual(set(g.cells()), {(0, 0, 0)})
        self.assertEqual(g.count_range((0, 0, 0), (0, 0, 0)), 1)
        self.assertEqual(g.count_range((1, 0, 0), (3, 3, 3)), 0)

    def test_thin_face_traversal_no_vertex_inside(self):
        # 大三角形平面 z=0.5，顶点远在 ±10：体素 (3,4,0) 内没有任何顶点，
        # 但三角形穿过它，必须被标记；相邻层 (k=±1) 不得标记。
        tri = [((-10.0, -10.0, 0.5), (10.0, -10.0, 0.5), (0.0, 10.0, 0.5))]
        g = make_grid(tri, h=1.0)
        for v in tri[0]:
            self.assertNotEqual(g.index_of(v), (3, 4, 0))  # 顶点确实不在该体素
        self.assertIn((3, 4, 0), g)
        self.assertNotIn((3, 4, 1), g)
        self.assertNotIn((3, 4, -1), g)
        # 整层 k=0 被占据，其它层为空
        self.assertTrue(all(idx[2] == 0 for idx in g.cells()))

    def test_degenerate_needle_triangle(self):
        # 退化（零面积）三角形 = 线段，斜穿体素 (1,1,1)，端点都在其外
        tri = [((0.0, 0.0, 0.0), (3.0, 3.0, 3.0), (1.5, 1.5, 1.5))]
        g = make_grid(tri, h=1.0)
        self.assertIn((1, 1, 1), g)
        # 线段穿过整数角点时会接触相邻 8 个体素（接触算占据），
        # 完整占据集合与精确基线对拍
        self.assertEqual(set(g.cells()), voxelize_exact(tri, 1.0))

    def test_voxel_much_smaller_than_object(self):
        # h=0.05，物体尺寸 ~1：占据数 ~ 面积/h^2，且与精确结果完全一致
        tri = [((0.0, 0.0, 0.3), (1.0, 0.0, 0.3), (0.0, 1.0, 0.7))]
        h = 0.05
        g = make_grid(tri, h=h)
        exact = voxelize_exact(tri, h)
        self.assertEqual(set(g.cells()), exact)
        self.assertGreater(len(g), 100)  # 远多于单体素

    def test_huge_scene(self):
        # 坐标 ~1e6，h=1：稠密数组需 ~(2e6)^3 个体素，稀疏只存占据部分
        tris = [
            ((1e6, 1e6, 1e6), (1e6 + 3, 1e6, 1e6), (1e6, 1e6 + 3, 1e6 + 1)),
            ((-1e6, -1e6, -1e6), (-1e6 + 2, -1e6, -1e6), (-1e6, -1e6 + 2, -1e6)),
            ((5e5, -5e5, 2.5), (5e5 + 4, -5e5, 2.5), (5e5, -5e5 + 4, 2.5)),
        ]
        g = make_grid(tris, h=1.0)
        exact = voxelize_exact(tris, 1.0)
        self.assertEqual(set(g.cells()), exact)
        self.assertLess(g.estimated_bytes(), 1 << 20)  # 稀疏内存 < 1MB

    def test_negative_and_offset_origin(self):
        origin = (10.0, -5.0, 2.0)
        tri = [((10.2, -4.8, 2.2), (10.4, -4.8, 2.2), (10.2, -4.6, 2.4))]
        g = make_grid(tri, h=0.5, origin=origin)
        exact = voxelize_exact(tri, 0.5, origin)
        self.assertEqual(set(g.cells()), exact)
        self.assertIn((0, 0, 0), g)


class TestCrossCheck(unittest.TestCase):
    """小场景随机对拍：SAT 体素化结果必须与精确裁剪结果完全一致。"""

    def test_random_small_scene(self):
        rng = random.Random(20260930)
        for trial in range(30):
            h = rng.choice([0.3, 0.5, 0.7, 1.0])
            tris = [
                tuple(
                    (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))
                    for _ in range(3)
                )
                for _ in range(20)
            ]
            g = make_grid(tris, h)
            exact = voxelize_exact(tris, h)
            self.assertEqual(
                set(g.cells()), exact, f"trial={trial} h={h} 对拍不一致"
            )

    def test_random_thin_slivers(self):
        # 细长/薄三角形（穿越高发情形）重点对拍
        rng = random.Random(7)
        for trial in range(20):
            h = rng.choice([0.4, 0.6])
            tris = []
            for _ in range(10):
                p0 = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2))
                p1 = tuple(p0[c] + rng.uniform(-4, 4) for c in range(3))
                p2 = tuple(p0[c] + rng.uniform(-0.01, 0.01) for c in range(3))
                tris.append((p0, p1, p2))
            g = make_grid(tris, h)
            exact = voxelize_exact(tris, h)
            self.assertEqual(set(g.cells()), exact, f"trial={trial} 薄面穿越对拍不一致")


class TestRangeQuery(unittest.TestCase):
    def test_range_query_matches_bruteforce(self):
        rng = random.Random(1)
        tris = [
            tuple(
                (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))
                for _ in range(3)
            )
            for _ in range(15)
        ]
        g = make_grid(tris, 0.5)
        exact = voxelize_exact(tris, 0.5)
        lo, hi = (-3, -2, -1), (2, 3, 1)
        expect = sum(1 for idx in exact if all(lo[c] <= idx[c] <= hi[c] for c in range(3)))
        self.assertEqual(g.count_range(lo, hi), expect)
        got = {idx for idx, _ in g.iter_range(lo, hi)}
        self.assertEqual(got, {idx for idx in exact if all(lo[c] <= idx[c] <= hi[c] for c in range(3))})


if __name__ == "__main__":
    unittest.main(verbosity=2)
