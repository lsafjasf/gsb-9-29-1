"""
test_resampler.py -- resampler.py 的自测（仅标准库 unittest）。

运行：
    python3 -m unittest -v test_resampler
或：
    python3 test_resampler.py
"""

import math
import random
import unittest

from resampler import (
    StreamResampler, resample, resample_channels, output_length,
)


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0


def sine(n, freq, phase=0.0):
    """freq 为归一化频率（周期/样本，奈奎斯特=0.5）。"""
    return [math.sin(2.0 * math.pi * freq * i + phase) for i in range(n)]


def stream_by_chunks(samples, ratio, chunk_sizes):
    """按给定块大小序列喂入，拼接所有输出（含 flush）。"""
    rs = StreamResampler(ratio)
    out = []
    pos = 0
    ci = 0
    while pos < len(samples):
        size = chunk_sizes[ci % len(chunk_sizes)]
        ci += 1
        out.extend(rs.process(samples[pos:pos + size]))
        pos += size
    out.extend(rs.flush())
    return out


def stream_multi(channels, ratio, chunk_sizes):
    rs = StreamResampler(ratio, channels=len(channels))
    out = [[] for _ in channels]
    n = len(channels[0])
    pos, ci = 0, 0
    while pos < n:
        size = chunk_sizes[ci % len(chunk_sizes)]
        ci += 1
        block = [ch[pos:pos + size] for ch in channels]
        part = rs.process(block)
        for c in range(len(channels)):
            out[c].extend(part[c])
        pos += size
    tail = rs.flush()
    for c in range(len(channels)):
        out[c].extend(tail[c])
    return out


class StreamingConsistency(unittest.TestCase):
    """分块喂入必须与整体处理逐位一致（块边界无爆音）。"""

    def setUp(self):
        random.seed(20261001)
        self.signal = [
            0.8 * math.sin(0.07 * i) + 0.3 * random.uniform(-1, 1)
            for i in range(1000)
        ]

    def _check(self, ratio, chunk_sizes):
        whole = resample(self.signal, ratio)
        chunked = stream_by_chunks(self.signal, ratio, chunk_sizes)
        self.assertEqual(len(whole), len(chunked))
        # 要求逐位相同：浮点值用 == 比较，而不是近似
        diff = [i for i, (a, b) in enumerate(zip(whole, chunked)) if a != b]
        self.assertEqual(diff, [],
                         "ratio=%r chunks=%r 存在 %d 个不一致样本"
                         % (ratio, chunk_sizes, len(diff)))

    def test_downsample(self):
        for chunks in ([1], [1, 1, 1], [7], [3, 5, 8], [64, 13, 1], [1000]):
            self._check(44100 / 48000, chunks)

    def test_upsample(self):
        for chunks in ([1], [2, 9], [33], [5, 50, 3], [1000]):
            self._check(48000 / 44100, chunks)

    def test_half_ratio(self):
        for chunks in ([1], [3], [2, 1, 4], [17, 29]):
            self._check(0.5, chunks)

    def test_double_ratio(self):
        for chunks in ([1], [3], [2, 1, 4], [17, 29]):
            self._check(2.0, chunks)

    def test_irrational_ratio(self):
        self._check(math.sqrt(2.0), [1, 7, 13, 100])

    def test_sample_jump_at_boundary(self):
        """边界两侧故意放跳变信号，确认边界处不产生额外爆音。"""
        sig = [1.0] * 400 + [-1.0] * 400
        whole = resample(sig, 0.5)
        # 随机切分点多次验证（含紧邻跳变沿的切分）
        cuts = [random.randint(1, len(sig) - 1) for _ in range(30)]
        cuts += [399, 400, 401, 1, len(sig) - 1]
        for b in cuts:
            rs = StreamResampler(0.5)
            got = rs.process(sig[:b]) + rs.process(sig[b:]) + rs.flush()
            self.assertEqual(got, whole)


class MultiChannel(unittest.TestCase):
    def test_stereo_independent_and_consistent(self):
        random.seed(7)
        left = sine(600, 0.05, 0.0)
        right = sine(600, 0.11, 1.3)
        whole = resample_channels([left, right], 0.7)
        self.assertEqual(whole[0], resample(left, 0.7))
        self.assertEqual(whole[1], resample(right, 0.7))
        chunked = stream_multi([left, right], 0.7, [1, 7, 64])
        self.assertEqual(chunked[0], whole[0])
        self.assertEqual(chunked[1], whole[1])

    def test_mono_matches_multichannel_api(self):
        sig = sine(300, 0.09)
        self.assertEqual(resample(sig, 1.6),
                         resample_channels([sig], 1.6)[0])


class RatioOne(unittest.TestCase):
    def test_identity_whole_and_stream(self):
        sig = [0.3 * math.sin(0.1 * i) + 0.01 * (i % 7)
               for i in range(500)]
        self.assertEqual(resample(sig, 1.0), sig)
        rs = StreamResampler(1.0)
        got = (rs.process(sig[:37]) + rs.process(sig[37:300])
               + rs.process(sig[300:]) + rs.flush())
        self.assertEqual(got, sig)

    def test_identity_multichannel(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        rs = StreamResampler(1.0, channels=2)
        head = rs.process(data)
        tail = rs.flush()
        self.assertEqual([h + t for h, t in zip(head, tail)], data)


class ShortAudio(unittest.TestCase):
    """极短输入（含空输入、短于滤波器长度）不得崩溃且结果有限。"""

    def test_empty(self):
        self.assertEqual(resample([], 0.5), [])
        self.assertEqual(resample([], 2.0), [])
        rs = StreamResampler(0.5)
        self.assertEqual(rs.process([]), [])
        self.assertEqual(rs.flush(), [])

    def test_one_sample(self):
        for ratio in (0.25, 0.5, 0.99, 1.0, 1.01, 2.0, 8.0):
            out = resample([0.75], ratio)
            self.assertEqual(len(out), output_length(1, ratio))
            self.assertTrue(all(math.isfinite(v) for v in out))
            # 理想带限模型：单样本展开为衰减 sinc，绝不放大
            self.assertTrue(all(abs(v) <= 0.75 + 1e-9 for v in out))

    def test_two_samples(self):
        for ratio in (0.3, 0.7, 1.0, 1.5, 3.3):
            out = resample([1.0, -1.0], ratio)
            self.assertEqual(len(out), output_length(2, ratio))
            self.assertTrue(all(math.isfinite(v) for v in out))
            # 允许理想低通的 Gibbs 过冲（约 9%），但不允许异常放大
            self.assertTrue(all(abs(v) <= 1.2 for v in out))

    def test_shorter_than_filter(self):
        sig = [0.5, -0.5, 0.2]
        for ratio in (0.5, 2.0):
            out = resample(sig, ratio)
            chunked = stream_by_chunks(sig, ratio, [1])
            self.assertEqual(out, chunked)


class ExtremeRatios(unittest.TestCase):
    """极端倍率：长度正确、有限、有界、可流式逐位复现。"""

    def _check(self, n, ratio):
        random.seed(int(1000 * ratio) + n)
        sig = [random.uniform(-1, 1) for _ in range(n)]
        out = resample(sig, ratio)
        self.assertEqual(len(out), output_length(n, ratio))
        self.assertTrue(all(math.isfinite(v) for v in out))
        self.assertTrue(all(abs(v) <= 1.6 for v in out))
        self.assertEqual(stream_by_chunks(sig, ratio, [1, 3, 16]), out)

    def test_heavy_downsample(self):
        self._check(4096, 1 / 64)   # 48000 -> 750 量级
        self._check(1000, 1 / 100)

    def test_heavy_upsample(self):
        self._check(32, 64)
        self._check(10, 100)

    def test_near_one(self):
        self._check(500, 0.999)
        self._check(500, 1.001)


class FrequencyBehavior(unittest.TestCase):
    def test_passband_preserved(self):
        """通带内正弦波重采样后幅度与频率应保持。"""
        f0 = 0.12
        sig = sine(2000, f0)
        ratio = 0.5  # 输出奈奎斯特 = 0.25，f0 安全位于通带
        out = resample(sig, ratio)
        # 去掉两端各约一个滤波器半长的瞬态区
        L = StreamResampler(ratio).half_length
        edge = int(L * ratio) + 2
        # 与"重采样时刻上的理论正弦"逐点比较（注意 out 的全局索引）
        err = max(abs(out[edge + m]
                      - math.sin(2 * math.pi * f0 * (edge + m) / ratio))
                  for m in range(len(out) - 2 * edge))
        self.assertLess(err, 2e-3)

    def test_alias_attenuated(self):
        """高于输出奈奎斯特的成分必须被抗混叠滤掉（而非折叠回基带）。"""
        f0 = 0.45                 # 输入侧归一化频率
        ratio = 0.5               # 输出奈奎斯特 = 0.25
        sig = sine(4000, f0)
        out = resample(sig, ratio)
        L = StreamResampler(ratio).half_length
        edge = int(L * ratio) + 5
        core = out[edge:-edge]
        # 若不抗混叠，f0 会折叠到 0.5-0.45=0.05，幅度约 1；
        # Blackman 窗 32 过零点设计下残余应远小于 -50 dB
        self.assertLess(rms(core), 3e-3)

    def test_image_attenuated_on_upsample(self):
        """上采样镜像必须被低通滤除：输出在整数输入位置间应平滑。"""
        sig = sine(1000, 0.4)
        out = resample(sig, 4.0)
        L = StreamResampler(4.0).half_length
        edge = L * 4 + 5
        # 与理论正弦比较（注意 out 的全局索引；0.4 接近 fc=0.5，余量放宽）
        err = max(abs(out[edge + m]
                      - math.sin(2 * math.pi * 0.4 * (edge + m) / 4.0))
                  for m in range(len(out) - 2 * edge))
        self.assertLess(err, 5e-3)


class ApiGuard(unittest.TestCase):
    def test_bad_ratio_and_channels(self):
        with self.assertRaises(ValueError):
            StreamResampler(0.0)
        with self.assertRaises(ValueError):
            StreamResampler(-1.0)
        with self.assertRaises(ValueError):
            StreamResampler(0.5, channels=0)

    def test_double_flush_and_late_process(self):
        rs = StreamResampler(0.5)
        rs.process([1.0, 2.0])
        rs.flush()
        with self.assertRaises(RuntimeError):
            rs.flush()
        with self.assertRaises(RuntimeError):
            rs.process([1.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
