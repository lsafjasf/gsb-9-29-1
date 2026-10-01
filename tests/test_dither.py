"""Self-tests for ditherlib. Run: python3 -m unittest discover -s tests -v"""

import os
import random
import sys
import tracemalloc
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ditherlib import (
    StreamingDitherer,
    diffuse,
    error_stats,
    fixed_palette,
    gray_palette,
    median_cut_palette,
    ordered_dither_image,
    pack_indices,
    pack_rows,
    packed_row_size,
    quantize_image,
    unpack_indices,
    unpack_rows,
)
from ditherlib import gen


class TestPacking(unittest.TestCase):
    def test_roundtrip_all_depths(self):
        rng = random.Random(42)
        for bits in (1, 2, 4, 8):
            for count in (0, 1, 3, 7, 8, 9, 63, 64, 65, 257):
                values = [rng.randrange(1 << bits) for _ in range(count)]
                packed = pack_indices(values, bits)
                self.assertEqual(len(packed), packed_row_size(count, bits))
                self.assertEqual(unpack_indices(packed, bits, count), values)

    def test_bit_exact_positions(self):
        rng = random.Random(7)
        for bits in (1, 2, 4, 8):
            values = [rng.randrange(1 << bits) for _ in range(53)]
            packed = pack_indices(values, bits)
            for i, expected in enumerate(values):
                bitpos = i * bits
                actual = 0
                for b in range(bits):
                    p = bitpos + b
                    actual = (actual << 1) | ((packed[p // 8] >> (7 - p % 8)) & 1)
                self.assertEqual(actual, expected, f"bits={bits} pixel {i}")

    def test_rows_roundtrip(self):
        rng = random.Random(1)
        for bits in (1, 2, 4, 8):
            rows = [[rng.randrange(1 << bits) for _ in range(37)]
                    for _ in range(11)]
            packed = pack_rows(rows, bits)
            self.assertEqual(unpack_rows(packed, bits, 37, 11), rows)

    def test_rejects_invalid(self):
        self.assertRaises(ValueError, pack_indices, [0], 3)
        self.assertRaises(ValueError, pack_indices, [2], 1)
        self.assertRaises(ValueError, pack_indices, [-1], 4)
        self.assertRaises(ValueError, unpack_indices, b"\x00", 8, 2)


class TestPalettes(unittest.TestCase):
    def test_fixed_palette_sizes(self):
        for bits in (1, 2, 4, 8):
            pal = fixed_palette(bits)
            self.assertEqual(len(pal), 1 << bits)
            self.assertEqual(pal.bits, bits)

    def test_median_cut_recovers_flat_colors(self):
        img = gen.solid(16, 16, (200, 10, 10))
        img += gen.solid(16, 16, (10, 200, 10))
        img += gen.solid(16, 16, (10, 10, 200))
        img += gen.solid(16, 16, (240, 240, 240))
        pal = median_cut_palette((p for row in img for p in row), 2)
        self.assertEqual(len(pal), 4)
        for color in ((200, 10, 10), (10, 200, 10),
                      (10, 10, 200), (240, 240, 240)):
            idx = pal.nearest_index(*color)
            for ch in range(3):
                self.assertAlmostEqual(pal.colors[idx][ch], color[ch], delta=2)

    def test_adaptive_beats_fixed_on_color_image(self):
        img = gen.color_zones(96, 48)
        flat = [p for row in img for p in row]
        adaptive = median_cut_palette(flat, 4)
        fixed = fixed_palette(4)
        out_a = quantize_image(img, adaptive)
        out_f = quantize_image(img, fixed)
        mae_a = error_stats(img, out_a, adaptive)["mae"]
        mae_f = error_stats(img, out_f, fixed)["mae"]
        self.assertLess(mae_a, mae_f)


class TestDiffusion(unittest.TestCase):
    def test_streaming_matches_batch_byte_for_byte(self):
        for bits in (1, 2, 4, 8):
            pal = fixed_palette(bits)
            img = gen.noise(41, 29, seed=bits)
            batch, _ = diffuse(img, 41, pal)
            streamer = StreamingDitherer(41, pal)
            streamed = [streamer.process_row(row) for row in img]
            self.assertEqual(batch, streamed)
            self.assertEqual(pack_rows(batch, bits), pack_rows(streamed, bits))
            again = StreamingDitherer(41, pal)
            self.assertEqual([again.process_row(r) for r in img], streamed)

    def test_error_conservation_identity(self):
        for bits in (1, 2, 4, 8):
            for serpentine in (True, False):
                pal = fixed_palette(bits)
                img = gen.noise(37, 53, seed=bits * 2 + serpentine)
                d = StreamingDitherer(37, pal, serpentine=serpentine)
                for row in img:
                    d.process_row(row)
                gap = d.conservation_residual()
                scale = max(1.0, sum(d.sum_in))
                for ch in range(3):
                    self.assertAlmostEqual(gap[ch], 0.0, delta=1e-6 * scale)

    def test_per_row_error_bounded_vs_height(self):
        width = 64
        pal = fixed_palette(1)
        step = pal.max_gray_step
        maxima = []
        for height in (256, 1024):
            img = gen.vertical_gradient(width, height)
            out, d = diffuse(img, width, pal)
            stats = error_stats(img, out, pal)
            maxima.append(stats["max_abs_row_error"])
            self.assertLessEqual(stats["max_abs_row_error"], width * 255.0)
            for ch in range(3):
                self.assertLessEqual(abs(d.residual()[ch]), width * 255.0)
            last_in = sum(p[0] for p in img[-1]) / width
            last_out = sum(pal.colors[i][0] for i in out[-1]) / width
            self.assertLessEqual(abs(last_out - last_in), step)
        self.assertLessEqual(maxima[1], maxima[0] * 1.5 + width * step * 0.05)

    def test_no_horizontal_wrap_between_rows(self):
        pal = gray_palette(2)
        width = 6
        row0 = [(85, 85, 85)] * (width - 1) + [(200, 200, 200)]
        row1 = [(40, 40, 40)] * width
        out, _ = diffuse([row0, row1], width, pal, serpentine=True)
        self.assertEqual(out[1][0], 0)
        self.assertNotEqual(out[1][width - 1], 0)

    def test_no_wrap_on_reverse_pass(self):
        pal = gray_palette(2)
        width = 6
        base = gen.solid(width, 3, (85, 85, 85))
        changed = [list(r) for r in base]
        changed[1][0] = (200, 200, 200)
        out_a, _ = diffuse(base, width, pal, serpentine=True)
        out_b, _ = diffuse(changed, width, pal, serpentine=True)
        self.assertEqual(out_a[2][width - 1], out_b[2][width - 1])
        self.assertNotEqual(out_a[1][0], out_b[1][0])

    def test_causality_later_rows_never_change_earlier(self):
        img = gen.noise(23, 40, seed=5)
        pal = fixed_palette(2)
        d = StreamingDitherer(23, pal)
        prefix = [d.process_row(r) for r in img[:20]]
        d2 = StreamingDitherer(23, pal)
        full = [d2.process_row(r) for r in img]
        self.assertEqual(prefix, full[:20])


class TestBoundaries(unittest.TestCase):
    def test_degenerate_sizes(self):
        pal = fixed_palette(2)
        for w, h in ((1, 1), (1, 64), (64, 1), (2, 4096), (4096, 2)):
            img = gen.noise(w, h, seed=w + h)
            out, d = diffuse(img, w, pal)
            self.assertEqual(len(out), h)
            self.assertTrue(all(len(r) == w for r in out))
            for row in out:
                for idx in row:
                    self.assertTrue(0 <= idx < len(pal))
            packed = pack_rows(out, 2)
            self.assertEqual(unpack_rows(packed, 2, w, h), out)
            gap = d.conservation_residual()
            scale = max(1.0, sum(d.sum_in))
            for ch in range(3):
                self.assertAlmostEqual(gap[ch], 0.0, delta=1e-6 * scale)

    def test_solid_palette_color_is_exact(self):
        pal = fixed_palette(2)
        img = gen.solid(32, 32, (255, 255, 255))
        out, d = diffuse(img, 32, pal)
        stats = error_stats(img, out, pal)
        self.assertEqual(stats["total_lum_diff"], 0.0)
        self.assertEqual(stats["mae"], 0.0)

    def test_output_only_uses_palette(self):
        for bits in (1, 2, 4, 8):
            pal = fixed_palette(bits)
            img = gen.gradient(50, 30)
            out, _ = diffuse(img, 50, pal)
            ordered = ordered_dither_image(img, pal)
            for rows in (out, ordered):
                for row in rows:
                    for idx in row:
                        self.assertTrue(0 <= idx < len(pal))


class TestStreamingMemory(unittest.TestCase):
    def _peak(self, width, height, bits):
        pal = fixed_palette(bits)
        ditherer = StreamingDitherer(width, pal)
        row = [(120, 60, 200)] * width
        tracemalloc.start()
        sink = 0
        for _ in range(height):
            sink += sum(ditherer.process_row(row))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertGreater(sink, -1)
        return peak

    def test_peak_memory_independent_of_height(self):
        small = self._peak(64, 2000, 2)
        large = self._peak(64, 16000, 2)
        self.assertLessEqual(large, small + 64 * 1024)
        self.assertLessEqual(large, small * 1.5 + 4096)


if __name__ == "__main__":
    unittest.main()
