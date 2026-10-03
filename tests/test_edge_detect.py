#!/usr/bin/env python3
"""Unit + boundary tests for the edge_detect package (stdlib unittest)."""

import os
import random
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edge_detect import (auto_thresholds, count_edges, detect_edges,
                         hysteresis, non_max_suppression, otsu_threshold,
                         read_pgm, sensitivity, sobel_gradient, write_pgm)

HUGE_SIZE = int(os.environ.get("EDGE_HUGE_SIZE", "2048"))


def const_image(w, h, v):
    return [[v] * w for _ in range(h)]


def noise_image(w, h, seed=7, sigma=25.0):
    rng = random.Random(seed)
    return [[max(0, min(255, int(round(rng.gauss(128.0, sigma)))))
             for _ in range(w)] for _ in range(h)]


class TestGradientBorderRule(unittest.TestCase):
    """Replicate padding: no spurious border edges, real ones preserved."""

    def test_constant_image_has_zero_gradient_everywhere(self):
        mag, _ = sobel_gradient(const_image(30, 20, 128))
        self.assertEqual(max(max(row) for row in mag), 0.0)

    def test_border_pixels_have_zero_gradient_on_constant_image(self):
        mag, _ = sobel_gradient(const_image(10, 10, 200))
        border = (mag[0] + mag[-1]
                  + [row[0] for row in mag] + [row[-1] for row in mag])
        self.assertEqual(max(border), 0.0)

    def test_step_at_image_boundary_is_detected_not_invented(self):
        rows = [[0] + [255] * 19 for _ in range(10)]
        mag, _ = sobel_gradient(rows)
        self.assertGreater(mag[5][0], 0.0)
        self.assertGreater(mag[5][1], 0.0)
        self.assertEqual(mag[5][10], 0.0)
        self.assertEqual(mag[5][19], 0.0)

    def test_vertical_step_gradient_location(self):
        rows = [[0] * 10 + [255] * 10 for _ in range(10)]
        mag, _ = sobel_gradient(rows)
        for y in range(10):
            self.assertGreater(mag[y][9], 0.0)
            self.assertGreater(mag[y][10], 0.0)
            self.assertEqual(mag[y][5], 0.0)

    def test_ragged_and_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            sobel_gradient([])
        with self.assertRaises(ValueError):
            sobel_gradient([[1, 2], [3]])


class TestNonMaxSuppression(unittest.TestCase):
    def test_single_pixel_line_yields_two_thin_parallel_edges(self):
        # A 1px bright line on a dark background has zero gradient on the
        # line itself and one gradient ridge on each side (y +/- 1); both
        # ridges must be thinned to one pixel (no three-row blobs).
        rows = const_image(40, 40, 0)
        for x in range(5, 35):
            rows[20][x] = 255
        mag, sector = sobel_gradient(rows)
        keep = non_max_suppression(mag, sector)
        for y in range(40):
            for x in range(40):
                if keep[y][x]:
                    if y == 20:
                        # only the two line endpoints have a real
                        # horizontal gradient on the line itself; the
                        # asymmetric tie rule keeps one pixel per end
                        self.assertIn(x, (5, 35))
                    else:
                        self.assertIn(y, (19, 21))
                    self.assertIn(x, range(5, 36))
        for x in range(10, 30):
            self.assertTrue(keep[19][x] and keep[21][x])

    def test_flat_region_survives_nothing(self):
        mag, sector = sobel_gradient(const_image(20, 20, 77))
        keep = non_max_suppression(mag, sector)
        self.assertFalse(any(any(row) for row in keep))


class TestHysteresis(unittest.TestCase):
    def test_weak_connected_to_strong_is_kept_isolated_is_dropped(self):
        mag = [[0.0] * 10 for _ in range(10)]
        keep = [[False] * 10 for _ in range(10)]
        mag[5][2] = 100.0
        keep[5][2] = True
        for x in (3, 4, 5):
            mag[5][x] = 20.0
            keep[5][x] = True
        mag[1][8] = 20.0
        keep[1][8] = True
        edges = hysteresis(keep, mag, low=10.0, high=50.0)
        self.assertTrue(all(edges[5][x] for x in (2, 3, 4, 5)))
        self.assertFalse(edges[1][8])

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            hysteresis([[True]], [[1.0]], low=5.0, high=1.0)


class TestAutoThresholds(unittest.TestCase):
    def test_otsu_separates_bimodal_population(self):
        vals = [10.0] * 900 + [100.0] * 100
        t = otsu_threshold(vals)
        self.assertGreater(t, 10.0)
        self.assertLess(t, 100.0)

    def test_empty_and_zero_inputs(self):
        self.assertEqual(auto_thresholds([]), (0.0, 0.0))
        self.assertEqual(auto_thresholds([0.0, 0.0]), (0.0, 0.0))

    def test_noise_floor_guards_unimodal_distribution(self):
        rng = random.Random(1)
        vals = [abs(rng.gauss(50.0, 10.0)) for _ in range(20000)]
        low, high = auto_thresholds(vals)
        self.assertGreater(high, 50.0 + 10.0)
        self.assertGreater(low, 0.0)
        self.assertLessEqual(low, high)


class TestPipelineCases(unittest.TestCase):
    def test_pure_color_image_yields_no_edges(self):
        edges, (low, high) = detect_edges(const_image(100, 80, 128))
        self.assertEqual(count_edges(edges), 0)
        self.assertEqual((low, high), (0.0, 0.0))

    def test_pure_noise_edge_density_is_bounded(self):
        rows = noise_image(200, 200, sigma=25.0)
        edges, (low, high) = detect_edges(rows)
        density = count_edges(edges) / (200 * 200)
        self.assertGreater(high, 0.0)
        self.assertLess(density, 0.05,
                        "auto threshold failed to suppress noise: %.3f"
                        % density)

    def test_single_pixel_line_fully_detected(self):
        w, h = 120, 60
        rows = const_image(w, h, 0)
        for x in range(10, w - 10):
            rows[30][x] = 255
        edges, _ = detect_edges(rows)
        detected = {x for x in range(w) if edges[29][x] or edges[31][x]}
        expected = set(range(12, w - 12))
        missed = expected - detected
        self.assertEqual(len(missed), 0, "missed line pixels: %s" % missed)

    def test_low_contrast_edge_found_where_fixed_threshold_fails(self):
        rows = const_image(60, 60, 100)
        for y in range(60):
            for x in range(30, 60):
                rows[y][x] = 112
        edges_auto, (low, high) = detect_edges(rows)
        edges_fixed, _ = detect_edges(rows, low=25.0, high=50.0)
        self.assertGreater(count_edges(edges_auto), 0)
        self.assertEqual(count_edges(edges_fixed), 0)
        self.assertLess(high, 50.0)


class TestHugeImage(unittest.TestCase):
    def test_huge_image(self):
        n = HUGE_SIZE
        rows = const_image(n, n, 0)
        q = n // 4
        for y in range(q, n - q):
            row = rows[y]
            for x in range(q, n - q):
                row[x] = 255
        t0 = time.time()
        edges, (low, high) = detect_edges(rows)
        elapsed = time.time() - t0
        count = count_edges(edges)
        perimeter = 4 * (n - 2 * q)
        self.assertGreater(count, int(0.9 * perimeter))
        self.assertLess(count, int(2.5 * perimeter))
        print("\nhuge image %dx%d: %.1fs, %d edge px, auto=(%.2f, %.2f)"
              % (n, n, elapsed, count, low, high))


class TestPGM(unittest.TestCase):
    def test_roundtrip_p5(self):
        rows = [[(x * 7 + y * 13) % 256 for x in range(37)]
                for y in range(23)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.pgm")
            write_pgm(path, rows)
            self.assertEqual(read_pgm(path), rows)

    def test_read_p2_ascii(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.pgm")
            with open(path, "w") as f:
                f.write("P2\n# comment\n3 2\n255\n0 1 2 3 4 5\n")
            self.assertEqual(read_pgm(path), [[0, 1, 2], [3, 4, 5]])


class TestSensitivity(unittest.TestCase):
    def test_sweep_is_monotonic_and_reports_miss_ratio(self):
        rows = const_image(80, 80, 0)
        for y in range(80):
            for x in range(20, 60):
                rows[y][x] = 200
        report = sensitivity(rows)
        counts = [r["edge_pixels"] for r in report["rows"]]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(report["rows"][0]["miss_ratio"], 0.0)
        for r in report["rows"]:
            self.assertGreaterEqual(r["miss_ratio"], 0.0)
            self.assertLessEqual(r["miss_ratio"], 1.0)

    def test_ground_truth_miss_ratio(self):
        rows = const_image(50, 50, 0)
        gt = [[False] * 50 for _ in range(50)]
        for x in range(5, 45):
            rows[25][x] = 255
            gt[25][x] = True
        report = sensitivity(rows, gt=gt, scales=(1.0,))
        self.assertEqual(report["reference_kind"], "ground_truth")
        self.assertEqual(report["rows"][0]["miss_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
