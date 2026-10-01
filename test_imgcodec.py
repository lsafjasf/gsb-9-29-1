#!/usr/bin/env python3
"""Self-test / roundtrip verification for the RIF codec (stdlib unittest).

Coverage:
  * single-pixel images (1x1, all supported channel counts)
  * solid-color images
  * random noise (uniform, incompressible)
  * smooth synthetic gradient
  * non-multiple-of-8 widths (1, 3, 7, 13, 127, 1023, ...)
  * extreme aspect ratios (1xN, Nx1)
  * a very large image
  * malformed / truncated input and bad parameters
"""

import os
import random
import time
import unittest

import imgcodec
from imgcodec import FILTER_NAMES, FormatError, encode, encode_with_stats, decode


def roundtrip_checked(test, pixels, width, height, channels, level=9):
    """Encode -> decode and assert byte-exact equality; return (enc, stats)."""
    encoded, stats = encode_with_stats(pixels, width, height, channels, level)
    decoded, dw, dh, dc = decode(encoded)
    test.assertEqual((dw, dh, dc), (width, height, channels))
    test.assertEqual(decoded, pixels, "roundtrip mismatch: bytes differ")
    test.assertEqual(stats.rows, height)
    test.assertEqual(sum(stats.counts), height)
    return encoded, stats


def gradient(width, height, channels):
    total = width * height * channels
    return bytes((i * 73 + (i // 17) * 11) & 0xFF for i in range(total))


class RoundtripTests(unittest.TestCase):
    def test_single_pixel(self):
        for ch in imgcodec.VALID_CHANNELS:
            px = bytes(range(1, ch + 1))  # 0x01, 0x0102, ... distinct values
            enc, _ = roundtrip_checked(self, px, 1, 1, ch)
            self.assertTrue(enc.startswith(b"RIF1"))

    def test_solid_color(self):
        for ch in imgcodec.VALID_CHANNELS:
            color = bytes((0xAA * (i + 1)) & 0xFF for i in range(ch))
            for w, h in [(1, 1), (64, 64), (100, 3), (3, 100), (255, 17)]:
                px = color * (w * h)
                enc, stats = roundtrip_checked(self, px, w, h, ch)
                # A constant image must compress dramatically (for sizes
                # where the 14-byte header is not dominant).
                if w * h * ch >= 4096:
                    self.assertLess(len(enc) * 20, w * h * ch)

    def test_random_noise(self):
        rng = random.Random(42)
        for w, h, ch in [(257, 131, 3), (129, 129, 1), (64, 64, 4), (53, 71, 2)]:
            px = bytes(rng.randrange(256) for _ in range(w * h * ch))
            enc, _ = roundtrip_checked(self, px, w, h, ch)
            # Noise is incompressible: no format can be "much smaller" than the
            # entropy; encoded size must stay in the same order of magnitude.
            self.assertLess(len(enc), len(px) * 1.05)

    def test_gradient_synthetic(self):
        for w, h, ch in [(320, 240, 3), (321, 117, 1), (200, 200, 4)]:
            px = gradient(w, h, ch)
            enc, stats = roundtrip_checked(self, px, w, h, ch)
            self.assertLess(len(enc), len(px) * 0.25)  # gradients compress hard

    def test_non_multiple_of_8_widths(self):
        # Widths that are not multiples of 8 must be packed exactly, no padding.
        rng = random.Random(7)
        widths = [1, 2, 3, 5, 6, 7, 9, 11, 13, 17, 31, 63, 127, 255, 1023, 4095]
        heights = [1, 2, 3, 7, 13]
        for w in widths:
            for h in heights:
                for ch in (1, 3):
                    px = bytes(rng.randrange(256) for _ in range(w * h * ch))
                    roundtrip_checked(self, px, w, h, ch)

    def test_extreme_aspect_ratios(self):
        rng = random.Random(99)
        cases = [(1, 20000, 1), (20000, 1, 1), (1, 10000, 3),
                 (2, 9999, 3), (9999, 2, 4)]
        for w, h, ch in cases:
            px = bytes(rng.randrange(256) for _ in range(w * h * ch))
            roundtrip_checked(self, px, w, h, ch)

    def test_each_filter_reconstructs_exactly(self):
        # For every filter f: recon_f(filter_f(cur, prev), prev) == cur.
        rng = random.Random(31337)
        for bpp in imgcodec.VALID_CHANNELS:
            for _ in range(20):
                n = bpp * rng.randrange(1, 40)
                cur = bytes(rng.randrange(256) for _ in range(n))
                prev = bytes(rng.randrange(256) for _ in range(n))
                for ftype in range(5):
                    filt = imgcodec._FILTERS[ftype](cur, prev, bpp)
                    back = imgcodec._RECONS[ftype](filt, prev, bpp)
                    self.assertEqual(back, cur,
                                     f"filter {FILTER_NAMES[ftype]} bpp={bpp}")

    def test_encoder_picks_min_l1_score(self):
        # The documented criterion: pick the filter whose filtered row has the
        # smallest sum of |signed byte|; ties go to the lower filter number.
        rng = random.Random(777)
        w, h, ch = 37, 23, 3
        px = bytes(rng.randrange(256) for _ in range(w * h * ch))
        encoded, stats = encode_with_stats(px, w, h, ch)
        import zlib
        packed = zlib.decompress(encoded[14:])
        row_len = w * ch
        prev = bytes(row_len)
        pos = 0
        for y in range(h):
            cur = px[y * row_len:(y + 1) * row_len]
            scores = [sum(imgcodec._FILTERS[f](cur, prev, ch)
                          .translate(imgcodec._ABS_SIGNED))
                      for f in range(5)]
            best = min(range(5), key=lambda f: scores[f])
            chosen = packed[pos]
            self.assertEqual(chosen, best, f"row {y}: scores={scores}")
            pos += 1 + row_len
            prev = cur

    def test_multiple_filters_used_on_synthetic(self):
        # A gradient + flat-rect image should exercise more than one filter.
        w, h = 200, 120
        buf = bytearray(w * h * 3)
        i = 0
        for y in range(h):
            for x in range(w):
                buf[i] = (x * 255) // w
                buf[i + 1] = (y * 255) // h
                buf[i + 2] = ((x + y) * 255) // (w + h)
                i += 3
        _, stats = encode_with_stats(bytes(buf), w, h, 3)
        used = sum(1 for c in stats.counts if c > 0)
        self.assertGreaterEqual(used, 2)

    def test_large_image(self):
        # 2560x1440 RGB ~= 11 MB raw: exercises every filter at scale.
        w, h, ch = 2560, 1440, 3
        buf = bytearray(w * h * ch)
        rng = random.Random(2024)
        i = 0
        for y in range(h):
            for x in range(w):
                buf[i] = (x + rng.randint(-3, 3)) & 0xFF
                buf[i + 1] = (y + rng.randint(-3, 3)) & 0xFF
                buf[i + 2] = (x + y + rng.randint(-3, 3)) & 0xFF
                i += 3
        start = time.time()
        enc, stats = roundtrip_checked(self, bytes(buf), w, h, ch)
        elapsed = time.time() - start
        print(f"\nlarge image: {w}x{h}x{ch} raw={len(buf)} B "
              f"enc={len(enc)} B ratio={len(enc)/len(buf):.4f} "
              f"roundtrip={elapsed:.1f}s  {stats}")

    def test_gray_alpha_and_rgba(self):
        rng = random.Random(555)
        for w, h, ch in [(17, 19, 1), (17, 19, 2), (17, 19, 4)]:
            px = bytes(rng.randrange(256) for _ in range(w * h * ch))
            roundtrip_checked(self, px, w, h, ch)


class MalformedInputTests(unittest.TestCase):
    def setUp(self):
        self.good = encode(bytes([1, 2, 3, 4, 5, 6]), 2, 1, 3)

    def test_bad_magic(self):
        bad = b"XXXX" + self.good[4:]
        with self.assertRaises(FormatError):
            decode(bad)

    def test_truncated_header(self):
        with self.assertRaises(FormatError):
            decode(b"RIF1\x00\x00")

    def test_truncated_payload(self):
        with self.assertRaises(FormatError):
            decode(self.good[:-3])

    def test_bit_flipped_payload(self):
        corrupted = self.good[:-2] + bytes([self.good[-2] ^ 0xFF]) + self.good[-1:]
        # A flip inside the zlib stream either fails zlib/checksum or, very
        # rarely, lands in the last bytes; the decoder must raise or differ.
        try:
            out, _, _, _ = decode(corrupted)
        except FormatError:
            return
        self.assertNotEqual(out, bytes([1, 2, 3, 4, 5, 6]))

    def test_invalid_channels(self):
        with self.assertRaises(FormatError):
            encode(b"\x00\x00", 1, 1, 5)

    def test_zero_dimensions(self):
        with self.assertRaises(FormatError):
            encode(b"", 0, 0, 3)

    def test_wrong_buffer_length(self):
        with self.assertRaises(FormatError):
            encode(b"\x00\x00", 2, 2, 3)  # need 12 bytes

    def test_non_multiple_width_has_no_padding_in_stream(self):
        # For a 3x1 RGB image the packed stream must be exactly:
        # 1 filter byte + 9 residual bytes.
        data = encode_with_stats(bytes(9), 3, 1, 3)[0]
        import zlib
        packed = zlib.decompress(data[14:])
        self.assertEqual(len(packed), 1 + 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
