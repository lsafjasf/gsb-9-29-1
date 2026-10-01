"""
resampler.py -- 仅依赖 Python 3 标准库的带限重采样（库）。

核心设计
========
倍率 ratio = 输出采样率 / 输入采样率（可为任意正实数，不必为整数）。

抗混叠滤波器
------------
统一用一个"窗函数 sinc 低通 FIR"同时承担：
  * 降采样 (ratio < 1)：截止频率 fc = 0.5 * ratio（输出端奈奎斯特频率，
    归一化到输入采样率），滤除高于输出奈奎斯特的分量，抑制混叠；
  * 上采样 (ratio > 1)：截止频率 fc = 0.5（输入端奈奎斯特），
    消除零值保持/插值产生的镜像；
  * ratio == 1：直接直通（见下）。
窗函数用 Blackman 窗（a0=0.42, a1=0.5, a2=0.08），第一旁瓣约 -58 dB、
阻带衰减约 74 dB，过渡带较窄，纯标准库即可实现（无需 Kaiser 所需的
贝塞尔函数）。sinc 每侧保留 32 个过零点，半长（输入样本数）
L = 32 / (2*fc)，并用 max_half_length=8192 封顶以限制极端倍率下的开销。

滤波器对每个输出样本"按需"在非整数位置重新采样：
    h(d) = 2*fc * sinc(2*fc*d) * blackman(d / L),  -L <= d <= L
其中 d = 输入样本索引 - 输出样本对应的连续时刻 center = m / ratio。
每个输出样本再用该相位下抽头权重之和做归一化，保证任意分数延迟相位下
直流增益恒为 1（不会随相位出现幅度调制）。

相位处理
--------
滤波器关于 d=0 对称（零相位 / 线性相位，群延迟为 0）：输出样本 m
严格对齐输入连续时刻 m/ratio，不引入任何样本延迟或相位失真。实现上
不采用因果延迟补偿，而是在流式缓冲中等到所需输入窗口全部到齐后再
计算该输出样本；flush 时信号两端缺失的输入严格按 0 处理（理想带限
模型，不做增益补偿，永不放大），因此块边界不会有任何拼接差异。

ratio == 1
----------
此时 fc=0.5 恰好压在奈奎斯特频率上，任何有限长滤波器都会轻微衰减
频带顶端，没有意义，故直接原样返回（逐位一致）。

流式
----
StreamResampler.process() 只输出"整个滤波窗口所需输入都已到齐"的样本，
历史样本按窗口下界滑动丢弃；flush() 输出剩余样本（端点之外视为 0）。
每个输出样本的抽头遍历顺序与分块方式无关，因此分块处理与整体处理
得到的浮点数逐位相同（不只是"近似相等"）。

仅使用标准库 math。
"""

import math


def _sinc(x):
    """归一化 sinc：sin(pi*x)/(pi*x)，x=0 时为 1。"""
    if x == 0.0:
        return 1.0
    px = math.pi * x
    return math.sin(px) / px


def _blackman(t):
    """Blackman 窗，t 归一化到 [-1, 1]，端点处恰好为 0。"""
    return 0.42 + 0.5 * math.cos(math.pi * t) + 0.08 * math.cos(2.0 * math.pi * t)


def output_length(n_in, ratio):
    """给定输入样本数与倍率，返回输出样本数（四舍五入）。"""
    return int(math.floor(n_in * float(ratio) + 0.5))


class StreamResampler:
    """流式重采样器。

    用法（单声道，process 可吃普通 list）::

        rs = StreamResampler(44100 / 48000)
        out  = rs.process(block)      # 可反复调用
        tail = rs.flush()

    多声道（process 吃"每声道一个 list"的列表，声道相互独立）::

        rs = StreamResampler(2.0, channels=2)
        out  = rs.process([left_block, right_block])
        tail = rs.flush()             # [left_tail, right_tail]
    """

    def __init__(self, ratio, channels=1, zero_crossings=32,
                 max_half_length=8192):
        if ratio <= 0.0:
            raise ValueError("ratio 必须为正数")
        if channels < 1:
            raise ValueError("channels 必须 >= 1")
        self.ratio = float(ratio)
        self.channels = int(channels)
        self.identity = (self.ratio == 1.0)
        # 截止频率：降采样取输出奈奎斯特，上采样取输入奈奎斯特（输入侧归一化）
        self.fc = 0.5 * min(1.0, self.ratio)
        half = zero_crossings / (2.0 * self.fc)
        self.half_length = int(min(max_half_length, math.ceil(half)))
        self._bufs = [[] for _ in range(self.channels)]
        self._buf_start = 0       # buf[0] 对应的绝对输入索引
        self._out_idx = 0         # 下一个待产出的输出索引
        self._in_count = 0        # 已接收输入总数
        self._flushed = False

    # -- 单个输出样本的卷积 -----------------------------------------------
    def _weight(self, d):
        if abs(d) >= self.half_length:
            return 0.0
        return 2.0 * self.fc * _sinc(2.0 * self.fc * d) * _blackman(
            d / self.half_length)

    def _convolve(self, ch, center, lo, hi, av_lo, av_hi, normalize):
        """计算一个输出样本。[lo,hi] 为理论窗口，[av_lo,av_hi] 为实际
        可用的绝对输入索引范围（端点之外按 0）。

        normalize=True（内部样本，窗口完整）：按权重和归一化，保证
        任意分数延迟相位下直流增益恒为 1。
        normalize=False（flush 尾部、窗口被截断）：严格按零延伸处理，
        不做增益补偿（理想带限插值模型，绝不会放大；代价是信号结尾
        存在约一个滤波器半长的自然衰减/Gibbs 振铃）。
        """
        buf = self._bufs[ch]
        base = self._buf_start
        acc = 0.0
        wsum = 0.0
        start = max(lo, av_lo)
        end = min(hi, av_hi)
        for idx in range(start, end + 1):
            w = self._weight(idx - center)
            acc += w * buf[idx - base]
            wsum += w
        if not normalize:
            return acc
        return acc / wsum if wsum != 0.0 else 0.0

    # -- 流式 API ----------------------------------------------------------
    def process(self, chunks):
        """喂入一个数据块，返回当前可以确定的全部输出样本。

        单声道可直接传 list[float]；多声道传 list[list[float]]。
        本方法只产出整个滤波窗口都已到达的样本，因此永远不"预支"
        未来输入，块边界不需要交叉淡化也不会产生爆音。
        """
        if self._flushed:
            raise RuntimeError("flush 之后不能再 process")
        if not isinstance(chunks, (list, tuple)):
            raise TypeError("chunks 必须是 list")
        # 单声道便捷形式：普通数值 list
        if self.channels == 1 and not (
                chunks and isinstance(chunks[0], (list, tuple))):
            chunks = [chunks]
        if len(chunks) != self.channels:
            raise ValueError("声道数不匹配：期望 %d，收到 %d"
                             % (self.channels, len(chunks)))
        n = len(chunks[0])
        if any(len(c) != n for c in chunks):
            raise ValueError("各声道块长度必须一致")

        if self.identity:
            self._in_count += n
            self._out_idx += n
            res = [list(c) for c in chunks]
            return res[0] if self.channels == 1 else res

        for c in range(self.channels):
            self._bufs[c].extend(chunks[c])
        self._in_count += n

        out = [[] for _ in range(self.channels)]
        L = self.half_length
        buf_end = self._buf_start + len(self._bufs[0])
        while True:
            center = self._out_idx / self.ratio
            hi = int(math.floor(center + L))
            if hi > buf_end - 1:
                break  # 窗口右端还缺输入，等下一块
            lo = int(math.ceil(center - L))
            # 丢弃当前及未来所有输出都不再需要的历史样本
            drop = lo - self._buf_start
            if drop > 0:
                for c in range(self.channels):
                    del self._bufs[c][:drop]
                self._buf_start += drop
            for c in range(self.channels):
                out[c].append(self._convolve(
                    c, center, lo, hi, self._buf_start, buf_end - 1, True))
            self._out_idx += 1
        return out[0] if self.channels == 1 else out

    def flush(self):
        """信号结束：按端点之外为 0，输出剩余样本。只能调用一次。"""
        if self._flushed:
            raise RuntimeError("flush 已调用过")
        self._flushed = True

        if self.identity:
            return [[] for _ in range(self.channels)][0] \
                if self.channels == 1 else [[] for _ in range(self.channels)]

        total_out = output_length(self._in_count, self.ratio)
        out = [[] for _ in range(self.channels)]
        L = self.half_length
        av_lo = self._buf_start
        av_hi = self._buf_start + len(self._bufs[0]) - 1
        while self._out_idx < total_out:
            center = self._out_idx / self.ratio
            lo = int(math.ceil(center - L))
            hi = int(math.floor(center + L))
            for c in range(self.channels):
                out[c].append(self._convolve(
                    c, center, lo, hi, av_lo, av_hi, False))
            self._out_idx += 1
        for c in range(self.channels):
            self._bufs[c] = []
        return out[0] if self.channels == 1 else out


def resample_channels(channels, ratio, **kwargs):
    """一次性整体重采样。channels: list[list[float]]，返回同结构。"""
    rs = StreamResampler(ratio, channels=len(channels), **kwargs)
    head = rs.process(channels)
    tail = rs.flush()
    if len(channels) == 1:  # 单声道时 process/flush 返回扁平 list
        head, tail = [head], [tail]
    return [h + t for h, t in zip(head, tail)]


def resample(samples, ratio, **kwargs):
    """一次性整体重采样（单声道）。samples: list[float]。"""
    return resample_channels([samples], ratio, **kwargs)[0]
