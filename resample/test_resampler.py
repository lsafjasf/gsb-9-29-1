"""重采样库测试：流式一致性、边界用例、抗混叠、相位、参照对拍。

运行：python3 test_resampler.py  （或 python3 -m unittest test_resampler -v）
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resampler import Resampler, resample, resample_mono
from reference import fft_resample


def tone(freq, rate, n, amp=1.0, phase=0.0):
    return [amp * math.sin(2.0 * math.pi * freq * i / rate + phase)
            for i in range(n)]


def fit_amp(samples, freq, rate):
    """最小二乘拟合已知频率正弦的幅度（解 2x2 正规方程，非整周期无偏）。"""
    cc = ss = cs = yc = ys = 0.0
    for i, v in enumerate(samples):
        w = 2.0 * math.pi * freq * i / rate
        c = math.cos(w)
        s = math.sin(w)
        cc += c * c
        ss += s * s
        cs += c * s
        yc += v * c
        ys += v * s
    det = cc * ss - cs * cs
    a = (yc * ss - ys * cs) / det
    b = (ys * cc - yc * cs) / det
    return math.hypot(a, b)


def chunked_process(frames, in_rate, out_rate, channels, chunk_sizes, **kw):
    """按 chunk_sizes 循环切片喂入，返回拼接后的全部输出。"""
    r = Resampler(in_rate, out_rate, channels=channels, **kw)
    out = []
    pos = 0
    idx = 0
    while pos < len(frames):
        size = chunk_sizes[idx % len(chunk_sizes)]
        out.extend(r.process(frames[pos:pos + size]))
        pos += size
        idx += 1
    out.extend(r.flush())
    return out


class TestRatioOne(unittest.TestCase):
    def test_passthrough_exact(self):
        x = [math.sin(i * 0.37) + 0.1 * i for i in range(1000)]
        y = resample_mono(x, 44100, 44100)
        self.assertEqual(x, y)  # 倍率 1 必须逐位直通

    def test_passthrough_streaming(self):
        frames = [(float(i), -float(i)) for i in range(500)]
        out = chunked_process(frames, 48000, 48000, 2, [1, 7, 64])
        self.assertEqual(frames, out)


class TestStreaming(unittest.TestCase):
    """分块喂入必须与整体处理逐位一致（块边界无爆音的根本保证）。"""

    def setUp(self):
        self.in_rate, self.out_rate = 48000, 44100
        n = 3000
        self.mono = [0.6 * math.sin(2 * math.pi * 997 * i / self.in_rate)
                     + 0.3 * math.sin(2 * math.pi * 5003 * i / self.in_rate + 1.0)
                     for i in range(n)]

    def test_chunked_equals_whole_bitexact(self):
        whole = resample_mono(self.mono, self.in_rate, self.out_rate)
        for chunks in ([1], [7], [100], [1, 2, 3, 4], [997, 3, 512]):
            frames = [(v,) for v in self.mono]
            got = [f[0] for f in chunked_process(
                frames, self.in_rate, self.out_rate, 1, chunks)]
            self.assertEqual(len(whole), len(got))
            self.assertEqual(whole, got)  # 逐位相等，非近似

    def test_chunked_equals_whole_multichannel(self):
        frames = [(v, -v * 0.5, 0.1) for v in self.mono]
        whole = resample(frames, self.in_rate, self.out_rate, channels=3)
        got = chunked_process(frames, self.in_rate, self.out_rate, 3,
                              [1, 13, 256])
        self.assertEqual(whole, got)

    def test_chunked_extreme_ratio(self):
        frames = [(v,) for v in self.mono]
        whole = resample(frames, self.in_rate, 1000, channels=1)
        got = chunked_process(frames, self.in_rate, 1000, 1, [5, 1000])
        self.assertEqual(whole, got)

    def test_process_after_flush_raises(self):
        r = Resampler(48000, 44100)
        r.process([(0.0,)] * 10)
        r.flush()
        with self.assertRaises(RuntimeError):
            r.process([(0.0,)])


class TestShortInputs(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(resample_mono([], 48000, 44100), [])
        self.assertEqual(resample_mono([], 48000, 100), [])
        self.assertEqual(resample_mono([], 100, 48000), [])

    def test_tiny_lengths(self):
        for n in (1, 2, 3, 5):
            x = [0.5 * (i + 1) for i in range(n)]
            for out_rate in (100, 44100, 48000 * 16):
                y = resample_mono(x, 48000, out_rate)
                self.assertTrue(all(math.isfinite(v) for v in y))
                # 输出长度应接近 n*ratio（端部零填充区不计）
                self.assertGreaterEqual(len(y), 1)

    def test_dc_preserved(self):
        # 常数信号重采样后仍为同一常数（直流增益严格为 1）
        for in_rate, out_rate in ((48000, 44100), (8000, 48000), (48000, 1000)):
            x = [0.5] * 3000
            r = Resampler(in_rate, out_rate)
            y = [f[0] for f in r.process([(v,) for v in x]) + r.flush()]
            lo = int(math.ceil(r.half_width * r.ratio)) + 2
            hi = int(math.floor((len(x) - 1 - r.half_width) * r.ratio)) - 2
            mid = y[lo:hi]
            self.assertTrue(mid)
            # 截断核的直流增益与 1 的偏差在 Blackman 纹波量级
            self.assertLess(max(abs(v - 0.5) for v in mid), 1e-4)


class TestMultichannel(unittest.TestCase):
    def test_channels_independent(self):
        n = 800
        ch0 = tone(1000, 48000, n)
        ch1 = tone(2000, 48000, n, amp=0.5)
        ch2 = [0.25] * n
        frames = list(zip(ch0, ch1, ch2))
        out = resample(frames, 48000, 32000, channels=3)
        cols = list(zip(*out))
        for ch, src in enumerate((ch0, ch1, ch2)):
            mono = resample_mono(list(src), 48000, 32000)
            self.assertEqual(list(cols[ch]), mono)

    def test_channel_count_mismatch_passthrough(self):
        frames = [(1.0, 2.0)] * 10
        out = resample(frames, 44100, 44100, channels=2)
        self.assertEqual(out, frames)


class TestExtremeRatios(unittest.TestCase):
    def test_down_48x_antialias_and_gain(self):
        in_rate, out_rate = 48000, 1000
        n = 9600
        x = [0.5 * math.sin(2 * math.pi * 100 * i / in_rate)      # 带内
             + 0.8 * math.sin(2 * math.pi * 5300 * i / in_rate)  # 远超新奈奎斯特
             for i in range(n)]
        r = Resampler(in_rate, out_rate, zero_crossings=32)
        y = [f[0] for f in r.process([(v,) for v in x]) + r.flush()]
        # 稳态区：距内容两端各 half_width（输入采样）+ 余量
        lo = int(math.ceil(r.half_width * r.ratio)) + 2
        hi = int(math.floor((n - 1 - r.half_width) * r.ratio)) - 2
        steady = y[lo:hi]
        inband = fit_amp(steady, 100, out_rate)
        # 5300 Hz 在 1 kHz 采样率下混叠到 300 Hz；拟合该处幅度
        alias = fit_amp(steady, 300.0, out_rate)
        self.assertAlmostEqual(inband, 0.5, delta=0.01)   # 带内增益保持
        self.assertLess(alias, 0.5 * 10 ** (-55 / 20))    # 混叠抑制 > 55 dB

    def test_up_48x_gain_and_waveform(self):
        in_rate, out_rate = 1000, 48000
        n = 200
        x = tone(50, in_rate, n, amp=0.7)
        r = Resampler(in_rate, out_rate, zero_crossings=32)
        y = [f[0] for f in r.process([(v,) for v in x]) + r.flush()]
        lo = int(math.ceil(r.half_width * r.ratio)) + 2
        hi = int(math.floor((n - 1 - r.half_width) * r.ratio)) - 2
        amp = fit_amp(y[lo:hi], 50, out_rate)
        self.assertAlmostEqual(amp, 0.7, delta=0.005)


class TestPhase(unittest.TestCase):
    def test_zero_group_delay(self):
        """零相位：输出峰值位置应与输入峰值位置按倍率对齐（无群延迟）。"""
        in_rate, out_rate = 8000, 48000
        n = 400
        # 低频包络峰值位于正中部，便于定位
        x = [math.sin(2 * math.pi * 20 * i / in_rate) for i in range(n)]
        y = resample_mono(x, in_rate, out_rate)
        ratio = out_rate / in_rate
        in_peak = max(range(n // 4, 3 * n // 4), key=lambda i: x[i])
        out_peak = max(range(len(y)), key=lambda i: y[i])
        self.assertAlmostEqual(out_peak / ratio, in_peak, delta=1.0)


class TestAgainstReference(unittest.TestCase):
    """与 FFT 理想带限参照对拍（稳态区，信号峰值约 1.0）。"""

    def _compare(self, in_rate, out_rate, n, components, tol_max, tol_rms):
        x = [sum(a * math.sin(2 * math.pi * f * i / in_rate + p)
                 for f, a, p in components) for i in range(n)]
        r = Resampler(in_rate, out_rate, zero_crossings=32)
        got = [f[0] for f in r.process([(v,) for v in x]) + r.flush()]
        ref = fft_resample(x, in_rate, out_rate)
        m = min(len(got), len(ref))
        skip = int(math.ceil(r.half_width * r.ratio)) + 8
        errs = [abs(got[i] - ref[i]) for i in range(skip, m - skip)]
        max_err = max(errs)
        rms_err = math.sqrt(sum(e * e for e in errs) / len(errs))
        self.assertLess(max_err, tol_max)
        self.assertLess(rms_err, tol_rms)
        return max_err, rms_err

    def test_down(self):
        self._compare(48000, 16000, 4096,
                      [(500, 0.5, 0.0), (2000, 0.3, 1.0), (6000, 0.2, 2.0),
                       (12000, 0.6, 0.3)],
                      tol_max=5e-3, tol_rms=2e-3)

    def test_up(self):
        self._compare(8000, 48000, 2048,
                      [(300, 0.5, 0.0), (1000, 0.3, 1.0), (3000, 0.2, 2.0)],
                      tol_max=5e-3, tol_rms=1e-3)

    def test_near_unity(self):
        self._compare(48000, 44100, 4096,
                      [(1000, 0.5, 0.0), (5000, 0.3, 1.0), (15000, 0.2, 2.0)],
                      tol_max=5e-3, tol_rms=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
