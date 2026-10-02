#!/usr/bin/env python3
"""edge_detect 自测：纯色图、纯噪声、单像素宽线条、超大图、边界规则等。"""

import os
import random
import tempfile
import time
import unittest

import edge_detect as ed


def solid_image(w, h, value):
    return [[value] * w for _ in range(h)]


def noise_image(w, h, seed=42):
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(w)] for _ in range(h)]


class TestBorderRule(unittest.TestCase):
    """边界规则：边缘复制填充，不得在边框产生虚假边缘。"""

    def test_constant_nonzero_image_has_no_border_edges(self):
        # 非零常数图：零填充会在边框产生假边缘，边缘复制不会
        img = solid_image(64, 64, 200)
        edges, low, high = ed.detect_edges(img)
        self.assertEqual(ed.count_edges(edges), 0)

    def test_gradient_zero_on_constant_image(self):
        img = solid_image(32, 32, 128)
        gx, gy = ed.sobel([[float(v) for v in row] for row in img])
        self.assertTrue(all(v == 0 for row in gx for v in row))
        self.assertTrue(all(v == 0 for row in gy for v in row))

    def test_border_pixel_can_be_edge(self):
        # 亮块贴到图像左边界：边界列应能检出边缘（NMS 对越界邻居取复制值）
        img = solid_image(64, 64, 0)
        for r in range(64):
            for c in range(0, 10):
                img[r][c] = 255
        edges, _, _ = ed.detect_edges(img)
        self.assertTrue(any(edges[r][9] or edges[r][10] for r in range(10, 54)))


class TestSolidColor(unittest.TestCase):
    def test_solid_black(self):
        edges, low, high = ed.detect_edges(solid_image(50, 50, 0))
        self.assertEqual(ed.count_edges(edges), 0)
        self.assertEqual((low, high), (0.0, 0.0))

    def test_solid_white(self):
        edges, _, _ = ed.detect_edges(solid_image(50, 50, 255))
        self.assertEqual(ed.count_edges(edges), 0)


class TestNoise(unittest.TestCase):
    def test_pure_noise_bounded_density(self):
        # 纯噪声：自动阈值应把边缘密度控制在有限范围（不是满屏噪点）
        img = noise_image(128, 128)
        edges, low, high = ed.detect_edges(img)
        frac = ed.count_edges(edges) / (128 * 128)
        self.assertGreater(high, 0.0)
        self.assertLess(frac, 0.35, '噪声图边缘密度过高: %.2f' % frac)

    def test_step_signal_survives_noise(self):
        # 同一噪声水平：含阶跃信号时，阶跃位置边缘应连续，且平坦区被抑制
        rng = random.Random(7)
        sig = [[(255 if c >= 64 else 0) + rng.randrange(30) for c in range(128)]
               for _ in range(128)]
        edges, _, high = ed.detect_edges(sig)
        boundary_hits = sum(any(edges[r][c] for c in range(62, 67))
                            for r in range(128))
        flat_pixels = sum(edges[r][c] for r in range(128)
                          for c in list(range(8, 56)) + list(range(72, 120)))
        self.assertGreater(boundary_hits, 0.8 * 128,
                           '阶跃边缘不连续: %d/128' % boundary_hits)
        self.assertLess(flat_pixels / (128 * 96), 0.05,
                        '平坦区噪声未被有效抑制')


class TestThinLine(unittest.TestCase):
    def test_single_pixel_vertical_line(self):
        w = h = 128
        img = solid_image(w, h, 0)
        cx = w // 2
        for r in range(h):
            img[r][cx] = 255
        edges, _, _ = ed.detect_edges(img)
        # 线条两侧各应检出一列边缘，且位置在线条 ±2 像素内
        cols = set()
        for r in range(8, h - 8):
            for c in range(w):
                if edges[r][c]:
                    cols.add(c)
        self.assertTrue(cols, '未检出任何边缘')
        self.assertTrue(all(abs(c - cx) <= 2 for c in cols),
                        '边缘偏离线条: %s' % sorted(cols))
        # 两侧都有边缘
        self.assertTrue(any(c < cx for c in cols))
        self.assertTrue(any(c > cx for c in cols))
        # 边缘像素数约为 2*h（两侧各一列），允许 50% 容差
        n = ed.count_edges(edges)
        self.assertGreater(n, h)
        self.assertLess(n, 3 * h)

    def test_single_pixel_horizontal_line(self):
        w = h = 128
        img = solid_image(w, h, 0)
        cy = h // 2
        for c in range(w):
            img[cy][c] = 255
        edges, _, _ = ed.detect_edges(img)
        rows = set()
        for r in range(h):
            for c in range(8, w - 8):
                if edges[r][c]:
                    rows.add(r)
        self.assertTrue(all(abs(r - cy) <= 2 for r in rows))
        self.assertTrue(any(r < cy for r in rows))
        self.assertTrue(any(r > cy for r in rows))


class TestAutoThreshold(unittest.TestCase):
    def test_threshold_monotonicity(self):
        # 阈值升高，边缘数单调不增
        img = noise_image(96, 96, seed=3)
        nms = ed.compute_nms(img)
        low, high = ed.auto_thresholds(nms)
        counts = [ed.count_edges(ed.hysteresis(nms, low * s, high * s))
                  for s in (0.5, 1.0, 2.0, 4.0)]
        self.assertTrue(all(a >= b for a, b in zip(counts, counts[1:])),
                        '非单调: %s' % counts)

    def test_sensitivity_miss_ratio(self):
        img = noise_image(96, 96, seed=5)
        nms = ed.compute_nms(img)
        low, high = ed.auto_thresholds(nms)
        rows = ed.sensitivity(nms, low, high)
        self.assertAlmostEqual(rows[0]['miss_ratio'], 0.0)
        self.assertTrue(all(0.0 <= r['miss_ratio'] <= 1.0 for r in rows))
        self.assertTrue(all(a['edge_pixels'] >= b['edge_pixels']
                            for a, b in zip(rows, rows[1:])))

    def test_contrast_adaptation(self):
        # 同一形状、不同对比度：自动阈值都应检出边缘（自适应于图像统计）
        for contrast in (20, 80, 200):
            img = solid_image(64, 64, 0)
            for r in range(64):
                for c in range(32, 64):
                    img[r][c] = contrast
            edges, low, high = ed.detect_edges(img)
            self.assertGreater(ed.count_edges(edges), 0,
                               '对比度 %d 未检出边缘' % contrast)


class TestHugeImage(unittest.TestCase):
    def test_huge_image(self):
        # 2048x2048 对角线 + 噪声：验证正确性与可接受的运行时间/内存
        w = h = 2048
        rng = random.Random(1)
        img = []
        for r in range(h):
            row = [rng.randrange(40) for _ in range(w)]
            row[min(r, w - 1)] = 255
            img.append(row)
        t0 = time.time()
        edges, low, high = ed.detect_edges(img)
        elapsed = time.time() - t0
        n = ed.count_edges(edges)
        print('\n超大图 2048x2048: %.1fs, 边缘像素 %d, low=%.2f high=%.2f'
              % (elapsed, n, low, high))
        self.assertGreater(n, h // 2, '超大图上对角线边缘丢失')
        # 对角线附近应有边缘
        found = sum(1 for r in range(0, h, 16)
                    for dc in range(-2, 3)
                    if 0 <= r + dc < w and edges[r][r + dc])
        self.assertGreater(found, h // 32)


class TestPgmIo(unittest.TestCase):
    def test_roundtrip(self):
        img = noise_image(40, 30, seed=9)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 't.pgm')
            ed.write_pgm(p, img)
            back = ed.read_pgm(p)
        self.assertEqual(back, img)


if __name__ == '__main__':
    unittest.main(verbosity=2)
