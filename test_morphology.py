"""
test_morphology.py — 形态学库自测（纯标准库 unittest）。

运行：python3 test_morphology.py   或   python3 -m unittest test_morphology -v

覆盖：
  * van Herk 1D 滤波 vs 朴素滑窗
  * 快速实现 vs 暴力实现 对拍（随机二值/灰度图、随机形状 SE、矩形 SE）
  * 开/闭运算幂等性断言（含非对称 SE）
  * 边界用例：单像素图像、全前景、全背景、SE 大于图像、1x1 SE 恒等
"""

import random
import unittest
from math import inf

from morphology import (
    _van_herk_1d, erode, dilate, opening, closing,
    erode_brute, dilate_brute, opening_brute, closing_brute,
    decompose,
)


def rand_image(h, w, binary, rng):
    if binary:
        return [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
    return [[rng.randint(0, 255) for _ in range(w)] for _ in range(h)]


def rand_se(rng, max_size=6, ensure_origin=True):
    h = rng.randint(1, max_size)
    w = rng.randint(1, max_size)
    se = [[rng.random() < 0.5 for _ in range(w)] for _ in range(h)]
    if ensure_origin:
        se[h // 2][w // 2] = True
    if not any(any(row) for row in se):
        se[rng.randrange(h)][rng.randrange(w)] = True
    return se


def rect_se(h, w):
    return [[True] * w for _ in range(h)]


def disk_se(r):
    return [[(dy * dy + dx * dx) <= r * r for dx in range(-r, r + 1)]
            for dy in range(-r, r + 1)]


class TestVanHerk(unittest.TestCase):
    def test_against_naive(self):
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 40)
            data = [rng.randint(0, 99) for _ in range(n)]
            k = rng.randint(1, 12)          # 含 k > n 的情形
            origin = rng.randrange(k)
            for op, neutral in ((min, inf), (max, -inf)):
                got = _van_herk_1d(data, k, origin, op, neutral)
                want = []
                for i in range(n):
                    lo, hi = i - origin, i - origin + k - 1
                    vals = [data[j] for j in range(max(lo, 0), min(hi, n - 1) + 1)]
                    want.append(op(vals) if vals else neutral)
                self.assertEqual(got, want)


class TestEquivalence(unittest.TestCase):
    """快速实现与暴力实现逐像素对拍。"""

    def check_all(self, img, se):
        self.assertEqual(erode(img, se), erode_brute(img, se))
        self.assertEqual(dilate(img, se), dilate_brute(img, se))
        self.assertEqual(opening(img, se), opening_brute(img, se))
        self.assertEqual(closing(img, se), closing_brute(img, se))

    def test_random_shapes(self):
        rng = random.Random(42)
        for _ in range(60):
            img = rand_image(rng.randint(1, 12), rng.randint(1, 12),
                             rng.random() < 0.5, rng)
            self.check_all(img, rand_se(rng))

    def test_rectangles_separable_path(self):
        rng = random.Random(1)
        for _ in range(30):
            img = rand_image(rng.randint(1, 15), rng.randint(1, 15),
                             rng.random() < 0.5, rng)
            se = rect_se(rng.randint(1, 9), rng.randint(1, 9))
            self.assertEqual(decompose(se)["kind"], "rectangle")
            self.check_all(img, se)

    def test_disk_run_decomposition(self):
        rng = random.Random(2)
        for r in (1, 2, 3, 5):
            se = disk_se(r)
            self.assertEqual(decompose(se)["kind"], "runs")
            img = rand_image(14, 14, False, rng)
            self.check_all(img, se)
            img = rand_image(14, 14, True, rng)
            self.check_all(img, se)

    def test_se_larger_than_image(self):
        rng = random.Random(3)
        for _ in range(20):
            img = rand_image(rng.randint(1, 4), rng.randint(1, 4),
                             rng.random() < 0.5, rng)
            se = rand_se(rng, max_size=9)   # 很可能大于图像
            self.check_all(img, se)
        # 明确构造：3x3 图，7x7 实心 SE
        img = rand_image(3, 3, False, rng)
        se = rect_se(7, 7)
        self.check_all(img, se)
        flat = [v for row in img for v in row]
        self.assertEqual(erode(img, se), [[min(flat)] * 3 for _ in range(3)])
        self.assertEqual(dilate(img, se), [[max(flat)] * 3 for _ in range(3)])


class TestIdempotency(unittest.TestCase):
    """开/闭运算幂等：重复执行结果不变（含非对称 SE）。"""

    def test_opening_idempotent(self):
        rng = random.Random(11)
        for _ in range(40):
            img = rand_image(rng.randint(2, 12), rng.randint(2, 12),
                             rng.random() < 0.5, rng)
            se = rand_se(rng)               # 一般非对称
            once = opening(img, se)
            self.assertEqual(opening(once, se), once)

    def test_closing_idempotent(self):
        rng = random.Random(12)
        for _ in range(40):
            img = rand_image(rng.randint(2, 12), rng.randint(2, 12),
                             rng.random() < 0.5, rng)
            se = rand_se(rng)
            once = closing(img, se)
            self.assertEqual(closing(once, se), once)

    def test_idempotent_edge_images(self):
        se = rect_se(3, 3)
        for img in ([[0]], [[1]], [[0] * 5 for _ in range(5)],
                    [[1] * 5 for _ in range(5)]):
            self.assertEqual(opening(opening(img, se), se), opening(img, se))
            self.assertEqual(closing(closing(img, se), se), closing(img, se))


class TestEdgeCases(unittest.TestCase):
    def test_single_pixel(self):
        se = rect_se(3, 3)
        self.assertEqual(erode([[7]], se), [[7]])
        self.assertEqual(dilate([[7]], se), [[7]])
        self.assertEqual(opening([[7]], se), [[7]])
        self.assertEqual(closing([[7]], se), [[7]])
        # SE 不含锚点的极端情形：窗口完全出界 -> 中性元
        se_far = [[False, False, False],
                  [False, False, False],
                  [False, False, True]]     # 仅右下角，锚点(1,1)不在内
        img = [[1]]
        self.assertEqual(erode(img, se_far), [[inf]])
        self.assertEqual(dilate(img, se_far), [[-inf]])

    def test_all_foreground_all_background(self):
        se = disk_se(2)
        ones = [[1] * 6 for _ in range(6)]
        zeros = [[0] * 6 for _ in range(6)]
        for img, v in ((ones, 1), (zeros, 0)):
            expect = [[v] * 6 for _ in range(6)]
            self.assertEqual(erode(img, se), expect)
            self.assertEqual(dilate(img, se), expect)
            self.assertEqual(opening(img, se), expect)
            self.assertEqual(closing(img, se), expect)

    def test_identity_se(self):
        rng = random.Random(5)
        img = rand_image(8, 8, False, rng)
        se = [[True]]
        self.assertEqual(erode(img, se), img)
        self.assertEqual(dilate(img, se), img)
        self.assertEqual(opening(img, se), img)
        self.assertEqual(closing(img, se), img)

    def test_binary_known_result(self):
        img = [[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]]
        se = rect_se(3, 3)
        self.assertEqual(erode(img, se),
                         [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]])
        self.assertEqual(dilate(img, se), [[1] * 5 for _ in range(5)])
        self.assertEqual(opening(img, se), img)   # 矩形块对矩形 SE 开运算是稳定的

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            erode([], [[True]])
        with self.assertRaises(ValueError):
            erode([[1]], [[False]])
        with self.assertRaises(ValueError):
            erode([[1, 2], [3]], [[True]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
