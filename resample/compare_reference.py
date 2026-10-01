"""
compare_reference.py -- 与高精度参照实现对拍，并给出逐样本误差数据。

参照实现（高精度理想带限重采样）
================================
信号两端补零到 2 的幂长度后，用 FFT 在频域精确完成重采样：
  * 降采样：频谱保留 |f| < 输出奈奎斯特 的分量，其余置零（理想低通）；
  * 上采样：在频谱高半区插入零（理想零值插补）。
FFT 为标准库 cmath 实现的基-2 迭代算法，双精度复数运算。
被测实现（resampler.py，Blackman 窗 sinc FIR，32 过零点）处理的是
同一份"两端补零"的信号，因此两者的数学模型一致，差异只来自 FIR 的
有限长度/窗函数截断，可直接逐样本比较。

输出：
  error_data_down.csv  ratio=0.7 （降采样）逐样本误差
  error_data_up.csv    ratio=1.5 （上采样）逐样本误差
  CSV 列：out_index, input_time, reference, resampler, abs_error

运行：python3 compare_reference.py
"""

import cmath
import csv
import math
import os

from resampler import StreamResampler, resample, output_length


# ---------- 标准库基-2 FFT 与任意长度 Bluestein FFT ----------
def _fft_radix2(a):
    """就地基-2 DIT FFT，len(a) 必须是 2 的幂。返回新列表。"""
    n = len(a)
    a = list(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2j * math.pi / length
        wlen = cmath.exp(ang)
        half = length >> 1
        for i in range(0, n, length):
            w = 1 + 0j
            for k in range(half):
                u = a[i + k]
                v = a[i + k + half] * w
                a[i + k] = u + v
                a[i + k + half] = u - v
                w *= wlen
        length <<= 1
    return a


def bluestein(x, inverse=False):
    """任意长度 DFT（Bluestein/chirp-z 算法，卷积用基-2 FFT）。

    利用 kn = (k^2 + n^2 - (k-n)^2) / 2，把任意长度 DFT 化为
    两个 chirp 序列的线性卷积，因此参照实现不需要长度为 2 的幂，
    可处理任意分数倍率对应的任意输出长度。
    """
    n = len(x)
    if n == 1:
        return [x[0]]
    alpha = 1.0 if inverse else -1.0
    m = 1
    while m < 2 * n - 1:
        m <<= 1
    a = [0j] * m
    b = [0j] * m
    for i in range(n):
        a[i] = x[i] * cmath.exp(alpha * 1j * math.pi * i * i / n)
    for q in range(-(n - 1), n):
        b[q % m] = cmath.exp(-alpha * 1j * math.pi * q * q / n)
    A = _fft_radix2(a)
    B = _fft_radix2(b)
    C = [A[i] * B[i] for i in range(m)]
    c = _fft_radix2([z.conjugate() for z in C])
    out = []
    for k in range(n):
        v = c[k].conjugate() / m
        v *= cmath.exp(alpha * 1j * math.pi * k * k / n)
        out.append(v / n if inverse else v)
    return out


def reference_resample(x, ratio):
    """FFT 域理想带限重采样。x 长度需为 2 的幂。"""
    n = len(x)
    m = output_length(n, ratio)
    spec = bluestein(x)
    ys = [0j] * m
    if m < n:  # 降采样：保留低频 m 个谱线（理想抗混叠低通）
        hi = m // 2
        for k in range(hi + 1):
            ys[k] = spec[k]
        for k in range(1, m - hi):
            ys[m - k] = spec[n - k]
    else:      # 上采样：谱中插零（理想零值插补）
        hi = n // 2
        for k in range(hi + 1):
            ys[k] = spec[k]
        for k in range(1, hi):
            ys[m - k] = spec[n - k]
    ys = bluestein(ys, inverse=True)
    scale = m / n
    return [v.real * scale for v in ys]


# ---------- 对拍 ----------
def run_case(name, ratio, freqs, nsig):
    L = StreamResampler(ratio).half_length
    pad = L + 4  # 两端补零，宽度超过滤波器半长

    # 测试信号：多个正弦之和，全部能量远低于输出奈奎斯特
    sig = [
        sum(amp * math.sin(2.0 * math.pi * f * i + ph)
            for (f, amp, ph) in freqs)
        for i in range(nsig)
    ]
    padded = [0.0] * pad + sig + [0.0] * pad

    # 输出长度取整使"有效倍率"与名义倍率略有差异
    # (|n_out/n_in - ratio| <= 0.5/n_in)。参照与被测必须使用同一
    # 有效倍率，否则时间映射不同，相位误差随样本索引线性累积。
    n_in = len(padded)
    n_out = output_length(n_in, ratio)
    ratio_eff = n_out / n_in

    ref = reference_resample(padded, ratio_eff)
    got = resample(padded, ratio_eff)

    # 只比较信号内部区域（两端各再留一个滤波器半长，避开边界瞬态）
    margin_in = L + 4
    m_lo = int(math.ceil((pad + margin_in) * ratio))
    m_hi = int(math.floor((pad + nsig - margin_in) * ratio))

    rows = []
    for m in range(m_lo, m_hi + 1):
        center = m / ratio
        err = abs(got[m] - ref[m])
        rows.append((m, center, ref[m], got[m], err))

    errs = [r[4] for r in rows]
    n = len(errs)
    rms = math.sqrt(sum(e * e for e in errs) / n)
    ordered = sorted(errs)
    stats = {
        "samples": n,
        "max_abs": max(errs),
        "rms": rms,
        "mean": sum(errs) / n,
        "p999": ordered[int(0.999 * (n - 1))],
        "max_db": 20 * math.log10(max(errs)) if max(errs) > 0 else float("-inf"),
    }

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "error_data_%s.csv" % name)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["out_index", "input_time", "reference",
                    "resampler", "abs_error"])
        w.writerows(rows)

    print("=" * 72)
    print("对拍用例 %s  名义ratio=%.4f 有效ratio=%.8f  FIR 半长 L=%d 输入样本"
          % (name, ratio, ratio_eff, L))
    print("测试信号频率(归一化): %s"
          % ", ".join("%.4f" % f for f, _, _ in freqs))
    print("比较区间: 输出样本 %d..%d（共 %d 个，已剔除两端边界瞬态）"
          % (m_lo, m_hi, n))
    print("逐样本误差: max=%.3e  RMS=%.3e  mean=%.3e  p99.9=%.3e"
          % (stats["max_abs"], stats["rms"], stats["mean"], stats["p999"]))
    print("最大误差折合 %.2f dB（相对满量程 1.0）；逐样本数据 -> %s"
          % (stats["max_db"], os.path.basename(csv_path)))
    print("前 8 个样本预览:")
    print("  idx   reference      resampler      abs_error")
    for r in rows[:8]:
        print("  %5d %+ .8f  %+ .8f  %.3e" % (r[0], r[2], r[3], r[4]))
    return stats


def main():
    # 降采样 0.7：输出奈奎斯特=0.35，测试能量全部 <= 0.23
    run_case("down", 0.7,
             [(0.045, 0.60, 0.0),
              (0.121, 0.40, 1.1),
              (0.230, 0.25, 2.3)],
             nsig=2048)
    # 上采样 1.5：奈奎斯特=0.5（输入侧），测试能量 <= 0.33
    run_case("up", 1.5,
             [(0.050, 0.60, 0.0),
              (0.173, 0.40, 0.7),
              (0.330, 0.25, 1.9)],
             nsig=2048)


if __name__ == "__main__":
    main()
