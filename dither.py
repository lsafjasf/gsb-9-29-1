"""抖动与量化库（仅 Python 标准库）。

图像表示：灰度图，list[list[float]]，取值 0..255。
彩色图可对每个通道分别调用同一组函数。

提供两种抖动：
  1. ordered_dither   —— Bayer 有序抖动（阈值矩阵）
  2. error_diffusion  —— Floyd-Steinberg 误差扩散（蛇形扫描 + 边界权重重归一化）

边界处理约定（误差扩散）：
  扫描到画布边缘时，越界邻居被丢弃，剩余邻居的权重按比例
  重归一化（权重和恒为 1），因此量化误差绝不会扩散到画布之外，
  整图总亮度严格守恒（唯一残差是最后一个被处理像素的未分配误差，
  其绝对值 <= step/2，step 为相邻量化级间距）。
"""

from __future__ import annotations

import math
import struct
import zlib

__all__ = [
    "quantize_value",
    "quantize_plain",
    "bayer_matrix",
    "ordered_dither",
    "error_diffusion",
    "error_stats",
    "read_pgm",
    "write_pgm",
    "write_png",
]


# ---------------------------------------------------------------- 量化

def quantize_value(value, levels):
    """把 0..255 的灰度量化为 levels 个等距级别之一，返回 float。

    输出被钳制在 [0, 255]；输入允许越界（误差扩散中累积误差
    可能把像素推出范围），钳制发生在输出端，误差本身不丢失。
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = 255.0 / (levels - 1)
    q = int(math.floor(value / step + 0.5))
    q = max(0, min(levels - 1, q))
    return q * step


def quantize_plain(image, levels):
    """无抖动直接量化（对照组，用于展示色带）。"""
    return [[quantize_value(v, levels) for v in row] for row in image]


# ---------------------------------------------------------------- 有序抖动

def bayer_matrix(size):
    """生成 size x size 的 Bayer 阈值矩阵，size 必须为 2 的幂。"""
    if size < 2 or size & (size - 1):
        raise ValueError("bayer size must be a power of two >= 2")
    m = [[0, 2], [3, 1]]
    n = 2
    while n < size:
        big = [[0] * (2 * n) for _ in range(2 * n)]
        for y in range(n):
            for x in range(n):
                base = 4 * m[y][x]
                big[y][x] = base
                big[y][x + n] = base + 2
                big[y + n][x] = base + 3
                big[y + n][x + n] = base + 1
        m = big
        n *= 2
    return m


def ordered_dither(image, levels, matrix_size=4):
    """Bayer 有序抖动。

    每个像素按 (x, y) 查阈值矩阵，把 (threshold - 0.5) * step 的偏移
    加到像素值上再量化。阈值矩阵均值为 0.5（偏移均值为 0），
    因此整体无系统性偏亮/偏暗，但不保证逐图严格守恒。
    """
    if not image or not image[0]:
        raise ValueError("image must be non-empty")
    bayer = bayer_matrix(matrix_size)
    n = matrix_size * matrix_size
    step = 255.0 / (levels - 1)
    height = len(image)
    width = len(image[0])
    out = []
    for y in range(height):
        row = []
        brow = bayer[y % matrix_size]
        for x in range(width):
            offset = ((brow[x % matrix_size] + 0.5) / n - 0.5) * step
            row.append(quantize_value(image[y][x] + offset, levels))
        out.append(row)
    return out


# ---------------------------------------------------------------- 误差扩散

# Floyd-Steinberg 权重（向右扫描方向）；蛇形扫描时水平镜像。
_FS_WEIGHTS = ((0, 1, 7.0 / 16), (1, -1, 3.0 / 16), (1, 0, 5.0 / 16), (1, 1, 1.0 / 16))


def error_diffusion(image, levels, serpentine=True):
    """Floyd-Steinberg 误差扩散抖动。

    serpentine=True 时按蛇形（之字形）扫描，消除单向扫描的方向性纹理。

    边界处理：越界邻居被剔除，剩余邻居权重重归一化（权重和 = 1），
    误差 100% 留在画布内。整图满足
        |sum(输出) - sum(输入)| <= 最后一个像素的未分配残差
    无钳制发生时该残差 <= step/2。
    """
    if not image or not image[0]:
        raise ValueError("image must be non-empty")
    height = len(image)
    width = len(image[0])
    buf = [[float(v) for v in row] for row in image]
    out = [[0.0] * width for _ in range(height)]
    for y in range(height):
        left_to_right = (not serpentine) or (y % 2 == 0)
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        direction = 1 if left_to_right else -1
        for x in xs:
            old = buf[y][x]
            new = quantize_value(old, levels)
            out[y][x] = new
            err = old - new
            if err == 0.0:
                continue
            neighbors = []
            total_w = 0.0
            for dy, dx, w in _FS_WEIGHTS:
                nx, ny = x + dx * direction, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbors.append((ny, nx, w))
                    total_w += w
            if total_w == 0.0:
                continue  # 1x1 图像：误差无处可去，成为全局残差
            for ny, nx, w in neighbors:
                buf[ny][nx] += err * w / total_w
    return out


# ---------------------------------------------------------------- 误差统计

def error_stats(before, after):
    """量化误差统计。

    返回 dict：
      pixels             像素数
      total_before/after 总亮度（守恒性指标）
      total_diff         总亮度差（after - before）
      mean_abs_error     平均绝对误差
      rmse               均方根误差
      max_abs_error      单像素最大绝对误差
    """
    if len(before) != len(after) or len(before[0]) != len(after[0]):
        raise ValueError("shape mismatch")
    total_before = total_after = 0.0
    abs_err = sq_err = max_err = 0.0
    for row_b, row_a in zip(before, after):
        for vb, va in zip(row_b, row_a):
            total_before += vb
            total_after += va
            e = abs(va - vb)
            abs_err += e
            sq_err += e * e
            if e > max_err:
                max_err = e
    n = len(before) * len(before[0])
    return {
        "pixels": n,
        "total_before": total_before,
        "total_after": total_after,
        "total_diff": total_after - total_before,
        "mean_abs_error": abs_err / n,
        "rmse": math.sqrt(sq_err / n),
        "max_abs_error": max_err,
    }


# ---------------------------------------------------------------- PGM 读写（样例输出用）

def write_pgm(image, path):
    """写二进制 PGM (P5)。像素四舍五入并钳制到 0..255。"""
    height = len(image)
    width = len(image[0])
    data = bytearray()
    for row in image:
        for v in row:
            data.append(max(0, min(255, int(v + 0.5))))
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (width, height))
        f.write(bytes(data))


def read_pgm(path):
    """读二进制 PGM (P5)，返回 list[list[float]]。"""
    with open(path, "rb") as f:
        raw = f.read()
    tokens = []
    i = 0
    while len(tokens) < 4:
        while i < len(raw) and raw[i:i + 1].isspace():
            i += 1
        if raw[i:i + 1] == b"#":
            while i < len(raw) and raw[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j:j + 1].isspace():
            j += 1
        tokens.append(raw[i:j])
        i = j
    if tokens[0] != b"P5":
        raise ValueError("only binary PGM (P5) is supported")
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if maxval != 255:
        raise ValueError("only maxval=255 is supported")
    i += 1
    pixels = raw[i:i + width * height]
    if len(pixels) != width * height:
        raise ValueError("truncated PGM")
    return [[float(pixels[y * width + x]) for x in range(width)] for y in range(height)]


def write_png(image, path):
    """写 8-bit 灰度 PNG（仅用 zlib/struct，方便任意看图工具打开）。"""
    height = len(image)
    width = len(image[0])
    raw = bytearray()
    for row in image:
        raw.append(0)  # 每行滤波器类型 0
        for v in row:
            raw.append(max(0, min(255, int(v + 0.5))))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit 灰度
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
