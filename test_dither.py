"""dither 库自测（仅标准库 unittest）。

运行：python3 test_dither.py  或  python3 -m unittest test_dither -v

覆盖：
  - 纯渐变 / 纯色 / 随机噪声 / 单像素宽（1xN、Nx1、1x1）图像
  - 误差守恒：|总亮度差| <= step/2（误差扩散，含边界重归一化）
  - 边界：误差不得扩散到画布之外（守恒断言 + 角落像素用例）
  - 输出合法性：所有像素落在量化级别集合内
"""

import random
import unittest

from dither import (
    bayer_matrix,
    error_diffusion,
    error_stats,
    ordered_dither,
    quantize_plain,
    quantize_value,
    read_pgm,
    write_pgm,
)

CONSERVATION_SLACK = 1e-6


def make_gradient(width, height):
    """水平线性渐变 0 -> 255。"""
    return [[255.0 * x / (width - 1) for x in range(width)] for _ in range(height)]


def make_solid(width, height, value):
    return [[float(value)] * width for _ in range(height)]


def make_noise(width, height, seed=42):
    rng = random.Random(seed)
    return [[rng.uniform(0, 255) for _ in range(width)] for _ in range(height)]


def step_of(levels):
    return 255.0 / (levels - 1)


def assert_conserved(testcase, before, after, levels):
    """误差扩散的守恒不变量：总亮度差不超过 step/2（最后像素的残差）。"""
    stats = error_stats(before, after)
    testcase.assertLessEqual(
        abs(stats["total_diff"]),
        step_of(levels) / 2 + CONSERVATION_SLACK,
        "total brightness not conserved: diff=%r" % stats["total_diff"],
    )
    return stats


def assert_valid_levels(testcase, image, levels):
    valid = {round(k * step_of(levels), 9) for k in range(levels)}
    for row in image:
        for v in row:
            testcase.assertIn(round(v, 9), valid)
            testcase.assertGreaterEqual(v, 0.0)
            testcase.assertLessEqual(v, 255.0)


class TestQuantize(unittest.TestCase):
    def test_levels_endpoints(self):
        self.assertEqual(quantize_value(0, 4), 0.0)
        self.assertEqual(quantize_value(255, 4), 255.0)
        self.assertEqual(quantize_value(128, 2), 255.0)
        self.assertEqual(quantize_value(100, 2), 0.0)

    def test_out_of_range_input_clamped(self):
        self.assertEqual(quantize_value(-50, 2), 0.0)
        self.assertEqual(quantize_value(300, 2), 255.0)

    def test_invalid_levels(self):
        with self.assertRaises(ValueError):
            quantize_value(10, 1)
        with self.assertRaises(ValueError):
            quantize_plain(make_solid(2, 2, 0), 0)


class TestBayerMatrix(unittest.TestCase):
    def test_contents_are_permutation(self):
        for size in (2, 4, 8):
            m = bayer_matrix(size)
            flat = sorted(v for row in m for v in row)
            self.assertEqual(flat, list(range(size * size)))

    def test_mean_is_centered(self):
        size = 4
        m = bayer_matrix(size)
        mean = sum(v for row in m for v in row) / (size * size)
        self.assertAlmostEqual(mean, (size * size - 1) / 2)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            bayer_matrix(3)


class TestErrorDiffusionConservation(unittest.TestCase):
    """核心不变量：误差 100% 留在画布内，总亮度守恒。"""

    def test_gradient(self):
        img = make_gradient(64, 64)
        for levels in (2, 3, 4, 8):
            out = error_diffusion(img, levels)
            assert_conserved(self, img, out, levels)
            assert_valid_levels(self, out, levels)

    def test_solid(self):
        for value in (0, 100, 128, 200, 255):
            img = make_solid(32, 32, value)
            out = error_diffusion(img, 2)
            assert_conserved(self, img, out, 2)
            assert_valid_levels(self, out, 2)

    def test_noise(self):
        img = make_noise(48, 48, seed=7)
        for levels in (2, 4):
            out = error_diffusion(img, levels)
            assert_conserved(self, img, out, levels)
            assert_valid_levels(self, out, levels)

    def test_non_serpentine_also_conserves(self):
        img = make_gradient(33, 17)
        out = error_diffusion(img, 4, serpentine=False)
        assert_conserved(self, img, out, 4)


class TestBoundary(unittest.TestCase):
    """边界用例：画布四周的误差不得泄漏到画布外。

    若实现把误差加到越界邻居（或丢失权重不重归一化），总亮度
    将不再守恒，下列断言会失败。
    """

    def test_corner_heavy_image(self):
        # 四角为中间灰，量化误差最大，且角落邻居最少
        img = make_solid(8, 8, 0)
        for y, x in ((0, 0), (0, 7), (7, 0), (7, 7)):
            img[y][x] = 128.0
        out = error_diffusion(img, 2)
        assert_conserved(self, img, out, 2)
        assert_valid_levels(self, out, 2)

    def test_single_pixel_wide_column(self):
        img = make_solid(1, 64, 100)  # 宽 1，高 64：误差只能向下传
        out = error_diffusion(img, 2)
        assert_conserved(self, img, out, 2)
        assert_valid_levels(self, out, 2)

    def test_single_pixel_wide_row(self):
        img = make_solid(64, 1, 100)  # 宽 64，高 1：误差只能横向传
        out = error_diffusion(img, 2)
        assert_conserved(self, img, out, 2)
        assert_valid_levels(self, out, 2)

    def test_single_pixel_image(self):
        img = make_solid(1, 1, 100)
        out = error_diffusion(img, 2)
        # 1x1 误差无处可去，全部成为残差，但仍 <= step/2
        self.assertEqual(out, [[0.0]])
        assert_conserved(self, img, out, 2)

    def test_bottom_right_residual_bounded(self):
        # 构造最后一行最后一列累积误差的场景，验证残差上限
        img = make_solid(5, 5, 64)
        out = error_diffusion(img, 2)
        stats = error_stats(img, out)
        self.assertLessEqual(abs(stats["total_diff"]), step_of(2) / 2 + CONSERVATION_SLACK)

    def test_empty_image_rejected(self):
        with self.assertRaises(ValueError):
            error_diffusion([], 2)
        with self.assertRaises(ValueError):
            ordered_dither([[]], 2)


class TestOrderedDither(unittest.TestCase):
    def test_valid_levels(self):
        img = make_gradient(32, 32)
        out = ordered_dither(img, 2, matrix_size=4)
        assert_valid_levels(self, out, 2)

    def test_unbiased_on_average(self):
        # 阈值矩阵均值为 0.5 => 大图上平均亮度偏差应远小于半个 step
        img = make_noise(64, 64, seed=3)
        out = ordered_dither(img, 2, matrix_size=4)
        stats = error_stats(img, out)
        mean_diff = abs(stats["total_diff"]) / stats["pixels"]
        self.assertLess(mean_diff, step_of(2) / 8)

    def test_solid_midtone_alternates(self):
        # 128 在 2 级量化下应产生黑白交织而非全黑或全白
        img = make_solid(16, 16, 128)
        out = ordered_dither(img, 2, matrix_size=2)
        values = {v for row in out for v in row}
        self.assertEqual(values, {0.0, 255.0})

    def test_matrix_size_must_be_power_of_two(self):
        with self.assertRaises(ValueError):
            ordered_dither(make_solid(4, 4, 1), 2, matrix_size=3)


class TestStatsAndIO(unittest.TestCase):
    def test_error_stats_identical_images(self):
        img = make_noise(8, 8, seed=1)
        stats = error_stats(img, [row[:] for row in img])
        self.assertEqual(stats["total_diff"], 0.0)
        self.assertEqual(stats["mean_abs_error"], 0.0)
        self.assertEqual(stats["rmse"], 0.0)

    def test_error_stats_shape_mismatch(self):
        with self.assertRaises(ValueError):
            error_stats(make_solid(2, 2, 0), make_solid(3, 2, 0))

    def test_pgm_roundtrip(self):
        import os
        import tempfile

        img = make_gradient(17, 9)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.pgm")
            write_pgm(img, path)
            back = read_pgm(path)
        stats = error_stats(img, back)
        self.assertLessEqual(stats["max_abs_error"], 0.5)  # 仅四舍五入误差


if __name__ == "__main__":
    unittest.main(verbosity=2)
