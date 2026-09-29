"""对拍：被测库 vs 参照实现（FFT 理想带限重采样）。

对每组倍率输出：
  * 逐样本误差表（稳态区前若干样本，同时写 CSV 到 error_data/）
  * 稳态区 max / RMS / mean 绝对误差
运行：python3 compare.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resampler import Resampler, resample_mono
from reference import fft_resample

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "error_data")


def make_signal(n, components):
    """components: [(freq_hz, amp, phase), ...]，采样率见调用处。"""
    return lambda t: sum(a * math.sin(2.0 * math.pi * f * t + p)
                         for f, a, p in components)


def run_case(name, in_rate, out_rate, n_in, components, zero_crossings=32,
             table_rows=12):
    gen = make_signal(n_in, components)
    x = [gen(i / in_rate) for i in range(n_in)]

    r = Resampler(in_rate, out_rate, zero_crossings=zero_crossings)
    got = [f[0] for f in (r.process([(v,) for v in x]) + r.flush())]
    ref = fft_resample(x, in_rate, out_rate)

    n = min(len(got), len(ref))
    # 跳过两端过渡区：被测库边缘长度 = half_width（输入采样），
    # 参照实现的周期延拓边缘效应也集中在两端，各留 8 样本余量。
    skip = int(math.ceil(r.half_width * r.ratio)) + 8
    lo, hi = skip, n - skip
    errs = [abs(got[i] - ref[i]) for i in range(lo, hi)]
    max_err = max(errs)
    rms_err = math.sqrt(sum(e * e for e in errs) / len(errs))
    mean_err = sum(errs) / len(errs)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "%s.csv" % name)
    with open(csv_path, "w") as f:
        f.write("out_index,library,reference,abs_error\n")
        for i in range(lo, hi):
            f.write("%d,%.17g,%.17g,%.17g\n" % (i, got[i], ref[i], errs[i - lo]))

    print("=" * 72)
    print("用例 %-22s %g Hz -> %g Hz (ratio=%.6f)  输入 %d 样本 -> 输出 %d 样本"
          % (name, in_rate, out_rate, out_rate / in_rate, n_in, n))
    print("稳态区 [%d, %d) 共 %d 样本" % (lo, hi, len(errs)))
    print("  max 绝对误差 : %.3e" % max_err)
    print("  RMS 绝对误差 : %.3e" % rms_err)
    print("  平均绝对误差 : %.3e" % mean_err)
    print("  逐样本误差（稳态区前 %d 个，完整数据见 error_data/%s.csv）："
          % (table_rows, name))
    print("    %-10s %-18s %-18s %-12s" % ("out_idx", "library", "reference", "abs_err"))
    for i in range(lo, lo + table_rows):
        print("    %-10d %-18.10f %-18.10f %.3e" % (i, got[i], ref[i], errs[i - lo]))
    return max_err, rms_err


def main():
    print("对拍：windowed-sinc 重采样库 vs FFT 理想带限参照")
    print("信号峰值幅度约 1.0；误差为稳态区绝对误差。")
    run_case("down_48k_to_44.1k", 48000, 44100, 4096,
             [(1000, 0.5, 0.0), (3000, 0.3, 1.0), (10000, 0.2, 2.0),
              (15000, 0.4, 0.5)])   # 15kHz：带内（cutoff≈21.6k，过渡带下缘≈20k）
    run_case("up_44.1k_to_48k", 44100, 48000, 4096,
             [(1000, 0.5, 0.0), (5000, 0.3, 1.0), (15000, 0.2, 2.0)])
    run_case("down_48k_to_16k", 48000, 16000, 4096,
             [(500, 0.5, 0.0), (2000, 0.3, 1.0), (6000, 0.2, 2.0),
              (12000, 0.6, 0.3)])   # 12kHz 超过 8k 奈奎斯特，应被双方滤除
    run_case("up_8k_to_48k", 8000, 48000, 2048,
             [(300, 0.5, 0.0), (1000, 0.3, 1.0), (3000, 0.2, 2.0)])
    run_case("down_48k_to_1k_extreme", 48000, 1000, 8192,
             [(50, 0.5, 0.0), (200, 0.3, 1.0), (400, 0.2, 2.0),
              (5000, 0.6, 0.3)])    # 5kHz 远超新奈奎斯特 500Hz
    run_case("up_1k_to_48k_extreme", 1000, 48000, 512,
             [(50, 0.5, 0.0), (200, 0.3, 1.0), (400, 0.2, 2.0)])


if __name__ == "__main__":
    main()
