"""参照实现：基于 DFT 的理想带限重采样（仅用于对拍，非交付库）。

方法：零填充 -> DFT -> 频域按目标长度补零（升采样）或截断
（降采样，即理想矩形低通抗混叠）-> IDFT -> 缩放。
频域矩形窗对应时域无限长 sinc，是"高精度插值"的标准参照。

DFT 对任意长度可用：2 的幂走基-2 FFT，其余走 Bluestein chirp-z。

被测库与参照都假定信号在 [0, N) 之外为零，因此两者只在
滤波器截断/过渡带与浮点舍入上存在差异。
"""

import math
from fractions import Fraction

__all__ = ["fft_resample"]


def _fft_pow2(x, invert):
    """递归基-2 Cooley-Tukey DFT，len(x) 必须为 2 的幂。逆变换不除以 n。"""
    n = len(x)
    if n == 1:
        return list(x)
    even = _fft_pow2(x[0::2], invert)
    odd = _fft_pow2(x[1::2], invert)
    sign = 1.0 if invert else -1.0
    out = [0j] * n
    for k in range(n // 2):
        ang = sign * 2.0 * math.pi * k / n
        tw = complex(math.cos(ang), math.sin(ang)) * odd[k]
        out[k] = even[k] + tw
        out[k + n // 2] = even[k] - tw
    return out


def _dft(x, invert):
    """任意长度 DFT。invert=True 时为逆变换（不除以 n）。"""
    n = len(x)
    if n == 0:
        return []
    if n & (n - 1) == 0:
        return _fft_pow2(x, invert)
    # Bluestein：X[k] = w^{k^2/2} * (a * c)[k]，w = e^{sign*2*pi*i/n}
    sign = 1.0 if invert else -1.0

    def w_half(j):
        ang = sign * math.pi * ((j * j) % (2 * n)) / n
        return complex(math.cos(ang), math.sin(ang))

    a = [x[j] * w_half(j) for j in range(n)]
    size = 1
    while size < 2 * n - 1:
        size *= 2
    c = [0j] * size
    for m in range(n):
        v = w_half(m).conjugate()      # c[m] = w^{-m^2/2}
        c[m] = v
        if m:
            c[size - m] = v            # c 偶对称：c[-m] = c[m]
    a = a + [0j] * (size - n)
    fa = _fft_pow2(a, False)
    fc = _fft_pow2(c, False)
    conv = _fft_pow2([fa[i] * fc[i] for i in range(size)], True)
    return [w_half(k) * conv[k] / size for k in range(n)]


def fft_resample(samples, in_rate, out_rate):
    """理想带限重采样，返回 float 列表，长度 round(N * out_rate/in_rate)。"""
    n = len(samples)
    if n == 0:
        return []
    ratio = float(out_rate) / float(in_rate)
    # 精确有理倍率：令 nfft 为分母 q 的倍数，使 m = nfft*p/q 为整数，
    # 保证输出样本 i 恰好对应输入位置 i/ratio（无累积相位漂移）。
    frac = Fraction(out_rate, in_rate)          # ratio = out/in 的最简分数
    p, q = frac.numerator, frac.denominator     # ratio = p/q
    nfft = q * ((2 * n + q - 1) // q)           # >= 2n 的 q 的倍数：零填充降端部串扰
    spec = _dft([complex(v, 0.0) for v in samples] + [0j] * (nfft - n), False)
    m = nfft * p // q
    if m < 2:
        m = 2
    out_spec = [0j] * m
    half_in = nfft // 2
    out_spec[0] = spec[0]
    if ratio >= 1.0:
        # 升采样：频谱中部补零；奈奎斯特 bin 对半分到两侧（实信号惯例）
        for k in range(1, half_in):
            out_spec[k] = spec[k]
            out_spec[m - k] = spec[nfft - k]
        out_spec[half_in] = spec[half_in] * 0.5
        out_spec[m - half_in] = spec[half_in] * 0.5
    else:
        # 降采样：丢弃新奈奎斯特以上的频谱 = 理想抗混叠低通。
        # 正频率 bin 为 1..floor((m-1)/2)；m 为偶数时另有奈奎斯特 bin m/2。
        pos = (m - 1) // 2
        for k in range(1, pos + 1):
            out_spec[k] = spec[k]
            out_spec[m - k] = spec[nfft - k]
        if m % 2 == 0:
            out_spec[m // 2] = spec[m // 2]
    time = _dft(out_spec, True)
    # _dft(invert=True) 未归一化；y = ifft(Y) * (m/nfft)，ifft 自带 1/m，
    # 故净缩放为 1/nfft（DC 检验：全 1 信号重采样后仍为全 1）。
    count = int(round(n * ratio))
    return [time[i].real / nfft for i in range(count)]
