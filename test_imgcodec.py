"""ICF1 编解码往返对拍测试：python3 -m unittest test_imgcodec -v"""

import random
import unittest

import imgcodec
from imgcodec import CodecError


def roundtrip(w, h, ch, bd, pixels):
    blob = imgcodec.encode(w, h, ch, bd, pixels)
    dw, dh, dch, dbd, dpix = imgcodec.decode(blob)
    assert (dw, dh, dch, dbd) == (w, h, ch, bd), "头部参数不一致"
    assert list(dpix) == list(pixels), "像素不一致"
    return blob


class TestRoundtrip(unittest.TestCase):
    def test_matrix_bitdepth_width(self):
        """位深 x 宽度（含非 8 倍数）x 通道 的随机对拍矩阵。"""
        rng = random.Random(20260930)
        for bd in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16):
            for w in (1, 2, 3, 7, 8, 9, 13, 16, 17, 31, 33):
                for h, ch in ((1, 1), (2, 3), (5, 4)):
                    limit = 1 << bd
                    px = [rng.randrange(limit) for _ in range(w * h * ch)]
                    roundtrip(w, h, ch, bd, px)

    def test_single_pixel(self):
        for bd in (1, 8, 16):
            px = [(1 << bd) - 1]
            roundtrip(1, 1, 1, bd, px)

    def test_solid_color(self):
        for bd in (1, 4, 8, 16):
            px = [12345 % (1 << bd)] * (64 * 48 * 3)
            blob = roundtrip(64, 48, 3, bd, px)
            raw = (64 * 48 * 3 * bd + 7) // 8
            self.assertLess(len(blob), raw // 4,
                            "全同色图应大幅压缩")

    def test_random_noise_roundtrip(self):
        rng = random.Random(7)
        for bd in (1, 8, 12):
            w, h, ch = 97, 53, 3
            px = [rng.randrange(1 << bd) for _ in range(w * h * ch)]
            roundtrip(w, h, ch, bd, px)

    def test_extreme_aspect(self):
        rng = random.Random(3)
        for w, h in ((1, 1000), (1000, 1), (1, 1), (2, 999), (999, 2)):
            px = [rng.randrange(256) for _ in range(w * h)]
            roundtrip(w, h, 1, 8, px)

    def test_non_multiple_of_8_width_1bit(self):
        """1 位位深 + 宽度非 8 倍数：验证位打包边界。"""
        rng = random.Random(11)
        for w in (1, 7, 9, 13, 15, 17, 63, 65):
            px = [rng.randrange(2) for _ in range(w * 10)]
            roundtrip(w, 10, 1, 1, px)

    def test_all_extremes_values(self):
        """每行交替全 0 / 全最大值，考验滤波器进位边界。"""
        bd = 12
        w, h, ch = 17, 6, 2
        hi = (1 << bd) - 1
        px = []
        for y in range(h):
            px.extend([hi if y % 2 else 0] * (w * ch))
        roundtrip(w, h, ch, bd, px)

    def test_large_image(self):
        """超大图：1920x1080 RGB 8bit（约 6.2MB 原始像素）。"""
        w, h = 1920, 1080
        row = bytes(((x * 7 + 13) & 0xFF) for x in range(w * 3))
        px = b"".join(
            bytes(((v + y) & 0xFF) for v in row) for y in range(h)
        )
        blob = roundtrip(w, h, 3, 8, px)
        self.assertLess(len(blob), len(px))

    def test_large_1bit_image(self):
        """超大 1 位图：4096x4096，宽度是 8 的倍数且含随机内容。"""
        rng = random.Random(5)
        w = h = 4096
        px = [rng.randrange(2) for _ in range(w * h)]
        roundtrip(w, h, 1, 1, px)

    def test_bytes_and_list_input_equivalent(self):
        px_list = [(i * 37) & 0xFF for i in range(40 * 30 * 3)]
        a = imgcodec.encode(40, 30, 3, 8, px_list)
        b = imgcodec.encode(40, 30, 3, 8, bytes(px_list))
        self.assertEqual(a, b)

    def test_file_roundtrip(self):
        import os, tempfile
        px = bytes((i * 11) & 0xFF for i in range(50 * 40))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.icf")
            imgcodec.write_file(p, 50, 40, 1, 8, px)
            w, h, ch, bd, out = imgcodec.read_file(p)
        self.assertEqual((w, h, ch, bd), (50, 40, 1, 8))
        self.assertEqual(list(out), list(px))


class TestRobustness(unittest.TestCase):
    def test_bad_magic(self):
        blob = imgcodec.encode(4, 4, 1, 8, [0] * 16)
        with self.assertRaises(CodecError):
            imgcodec.decode(b"XXXX" + blob[4:])

    def test_truncated(self):
        blob = imgcodec.encode(4, 4, 1, 8, [0] * 16)
        with self.assertRaises(CodecError):
            imgcodec.decode(blob[: len(blob) // 2])

    def test_corrupted_payload(self):
        blob = bytearray(imgcodec.encode(8, 8, 1, 8, list(range(64))))
        blob[-3] ^= 0xFF
        with self.assertRaises(CodecError):
            imgcodec.decode(bytes(blob))

    def test_invalid_params(self):
        with self.assertRaises(CodecError):
            imgcodec.encode(0, 4, 1, 8, [])
        with self.assertRaises(CodecError):
            imgcodec.encode(4, 4, 1, 17, [0] * 16)
        with self.assertRaises(CodecError):
            imgcodec.encode(4, 4, 1, 8, [0] * 15)      # 数量不符
        with self.assertRaises(CodecError):
            imgcodec.encode(4, 4, 1, 4, [16] * 16)     # 超出位深

    def test_stats_returned(self):
        px = bytes(100 * 50 * 3)
        blob, stats = imgcodec.encode(100, 50, 3, 8, px, want_stats=True)
        self.assertEqual(sum(stats["filter_counts"].values()), 50)
        self.assertEqual(stats["file_bytes"], len(blob))
        self.assertGreater(stats["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
