"""任意倍率音频重采样器（仅标准库）。

设计要点
--------
* 核函数：加 Blackman 窗的 sinc 低通（windowed-sinc FIR），
  h(t) = 2*fc * sinc(2*fc*t) * blackman(t / W),  |t| <= W
  - fc   : 截止频率（单位：周期/输入采样），fc = 0.5 * min(1, ratio) * cutoff_scale。
           降采样时 fc < 0.5，即自动完成抗混叠；升采样时 fc 接近输入奈奎斯特，
           用于抑制镜像。
  - W    : 核半径（输入采样数），W = zero_crossings / (2*fc)，
           即两侧各保留 zero_crossings 个 sinc 过零点。降采样越多，核越宽，
           过渡带越窄，抗混叠越充分。
  - Blackman 窗阻带衰减约 -74 dB，截断 sinc 引起的吉布斯纹波被压到该量级。
* 相位：核关于 t=0 偶对称（零相位 / 线性相位且群延迟已被对齐补偿）。
  第 n 个输出样本对应输入时间 t_n = n / ratio（输入采样为单位），
  滤波器以 t_n 为中心对称取值，因此输出与输入时间轴严格对齐，无相位偏移。
* 流式：内部维护输入帧缓冲与绝对序号，process() 可任意分块喂入；
  只有右窗缘已落在已收到数据内的输出样本才会产出，flush() 用零填充
  收尾。每个输出样本的计算只依赖其绝对序号，与分块方式无关，
  因此分块结果与一次性处理逐位（bit-exact）一致，块边界无爆音。
* 端点：信号在 [0, N) 之外按零处理（零填充），两端各有一个长度约 W
  的淡入/淡出过渡区，这是任何因果有限长滤波器的固有边界效应。
* ratio == 1 时直通，样本原样返回。
"""

import math

__all__ = ["Resampler", "resample", "resample_mono"]


def _sinc(x):
    """归一化 sinc：sin(pi*x)/(pi*x)。"""
    if x == 0.0:
        return 1.0
    xp = math.pi * x
    return math.sin(xp) / xp


def _blackman(x):
    """中心形式的 Blackman 窗，x in [-1, 1]。"""
    return 0.42 + 0.5 * math.cos(math.pi * x) + 0.08 * math.cos(2.0 * math.pi * x)


class Resampler:
    """有状态的多声道重采样器，支持分块（流式）喂入。

    frames 为帧序列，每帧是长度为 channels 的 float 序列，如
    [(l0, r0), (l1, r1), ...]。单声道可用 resample_mono() 便捷入口。
    """

    def __init__(self, in_rate, out_rate, channels=1,
                 zero_crossings=16, cutoff_scale=0.98):
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError("采样率必须为正数")
        if channels < 1:
            raise ValueError("声道数必须 >= 1")
        if zero_crossings < 2:
            raise ValueError("zero_crossings 太小，滤波器长度过短")
        if not (0.0 < cutoff_scale <= 1.0):
            raise ValueError("cutoff_scale 必须在 (0, 1] 内")
        self.in_rate = float(in_rate)
        self.out_rate = float(out_rate)
        self.channels = int(channels)
        self.ratio = self.out_rate / self.in_rate
        self._passthrough = (self.ratio == 1.0)
        # 截止频率（周期/输入采样）：降采样时收缩到新奈奎斯特以下 -> 抗混叠
        self.cutoff = 0.5 * min(1.0, self.ratio) * cutoff_scale
        # 核半径（输入采样数）：两侧各 zero_crossings 个过零点
        self.half_width = zero_crossings / (2.0 * self.cutoff)
        self._buf = []        # 尚未消费的输入帧，_buf[i] 的绝对序号为 _base + i
        self._base = 0
        self._total_in = 0    # 已累计喂入的帧数
        self._out_index = 0   # 下一个待产出的输出帧绝对序号
        self._flushed = False

    # ---- 内部 ----

    def _kernel(self, dt):
        """核在距中心 dt（输入采样）处的权重。"""
        w = self.half_width
        if dt <= -w or dt >= w:
            return 0.0
        return (2.0 * self.cutoff
                * _sinc(2.0 * self.cutoff * dt)
                * _blackman(dt / w))

    def _compute(self, t, lo, hi):
        """计算输入位置 t 处的一个输出帧，窗覆盖输入序号 [lo, hi]。"""
        acc = [0.0] * self.channels
        buf = self._buf
        base = self._base
        nbuf = len(buf)
        for k in range(lo, hi + 1):
            g = self._kernel(t - k)
            if g == 0.0:
                continue
            idx = k - base
            if 0 <= idx < nbuf:
                frame = buf[idx]
                for ch in range(self.channels):
                    acc[ch] += g * frame[ch]
            # 缓冲之外（仅 flush 后的尾部零填充区）按 0 处理
        return tuple(acc)

    def _emit(self, flushed):
        """产出所有当前可确定的输出帧。"""
        out = []
        if self._total_in == 0:
            return out
        width = self.half_width
        last_in = self._total_in - 1
        while True:
            t = self._out_index / self.ratio
            lo = math.ceil(t - width)
            hi = math.floor(t + width)
            if flushed:
                if lo > last_in:      # 左窗缘已越过末尾，后续输出全为 0
                    break
            else:
                if hi > last_in:      # 右窗缘超出已有数据，等更多输入
                    break
            out.append(self._compute(t, lo, hi))
            self._out_index += 1
        # 丢弃不再需要的头部历史
        if self._out_index > 0:
            keep_from = math.ceil(self._out_index / self.ratio - width)
            drop = min(keep_from - self._base, len(self._buf))
            if drop > 0:
                del self._buf[:drop]
                self._base += drop
        return out

    # ---- 公开接口 ----

    def process(self, frames):
        """喂入一块输入帧，返回本次可产出的输出帧列表。"""
        if self._flushed:
            raise RuntimeError("flush() 之后不能再 process()")
        if self._passthrough:
            out = [tuple(f) for f in frames]
            self._total_in += len(out)
            return out
        for f in frames:
            self._buf.append(tuple(f))
        self._total_in = self._base + len(self._buf)
        return self._emit(flushed=False)

    def flush(self):
        """结束输入，用零填充产出全部剩余输出帧。"""
        if self._flushed:
            return []
        self._flushed = True
        if self._passthrough:
            return []
        return self._emit(flushed=True)


def resample(frames, in_rate, out_rate, channels=1, **kwargs):
    """一次性重采样：frames -> 输出帧列表。"""
    r = Resampler(in_rate, out_rate, channels=channels, **kwargs)
    return r.process(frames) + r.flush()


def resample_mono(samples, in_rate, out_rate, **kwargs):
    """单声道便捷接口：float 列表 -> float 列表。"""
    frames = [(x,) for x in samples]
    return [f[0] for f in resample(frames, in_rate, out_rate, channels=1, **kwargs)]
