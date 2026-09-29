"""ICF1 - 内部传输用无损图像格式（纯标准库实现）。

格式布局（大端）：
    头部 14 字节: magic(4)="ICF1" width(u32) height(u32) channels(u8) bit_depth(u8)
    负载: zlib 压缩的逐行数据，每行 = 1 字节滤波器类型 + 滤波后的打包行字节

特性：
    - 任意宽高（>=1），任意通道数（1..255），任意位深（1..16 bit/通道）
    - 像素按位紧密打包（MSB first），宽度不需要是 8 的倍数
    - 5 种行滤波器（None/Sub/Up/Average/Paeth，与 PNG 相同），逐行自适应选择
    - 选择判据：滤波后字节的"有符号绝对值之和"最小（PNG 推荐启发式），
      平局时取编号较小的滤波器
    - 往返严格无损：解码结果与原始像素逐字节一致
"""

import struct
import zlib

MAGIC = b"ICF1"
_HEADER = struct.Struct(">4sIIBB")

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVERAGE = 3
FILTER_PAETH = 4
FILTER_NAMES = ["None", "Sub", "Up", "Average", "Paeth"]

# 判据查找表：把字节当作有符号残差时的绝对值（0..127 -> 0..127, 128..255 -> 128..1）
_SCORE = [min(b, 256 - b) for b in range(256)]


class CodecError(ValueError):
    """编码参数非法或数据损坏时抛出。"""


def _paeth(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


# ---------------------------------------------------------------- 位打包

def _pack_row(vals, bit_depth):
    """把一行的像素值（每个 < 2**bit_depth）紧密打包成字节（MSB first）。"""
    if bit_depth == 8:
        return bytes(vals)
    if bit_depth == 16:
        return struct.pack(">%dH" % len(vals), *vals)
    out = bytearray()
    acc = 0
    nbits = 0
    for v in vals:
        acc = (acc << bit_depth) | v
        nbits += bit_depth
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
            acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


def _unpack_row(data, count, bit_depth):
    """_pack_row 的逆操作，从 data 中解出 count 个值。"""
    if bit_depth == 8:
        return list(data[:count])
    if bit_depth == 16:
        return list(struct.unpack(">%dH" % count, data[: count * 2]))
    out = []
    acc = 0
    nbits = 0
    mask = (1 << bit_depth) - 1
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= bit_depth and len(out) < count:
            nbits -= bit_depth
            out.append((acc >> nbits) & mask)
            acc &= (1 << nbits) - 1
        if len(out) == count:
            break
    return out


# ---------------------------------------------------------------- 滤波器

def _filter_row(ftype, row, prev, bpp):
    n = len(row)
    if ftype == FILTER_NONE:
        return bytes(row)
    out = bytearray(n)
    if ftype == FILTER_SUB:
        out[:bpp] = row[:bpp]
        for i in range(bpp, n):
            out[i] = (row[i] - row[i - bpp]) & 0xFF
    elif ftype == FILTER_UP:
        for i in range(n):
            out[i] = (row[i] - prev[i]) & 0xFF
    elif ftype == FILTER_AVERAGE:
        for i in range(n):
            a = row[i - bpp] if i >= bpp else 0
            out[i] = (row[i] - ((a + prev[i]) >> 1)) & 0xFF
    elif ftype == FILTER_PAETH:
        for i in range(n):
            a = row[i - bpp] if i >= bpp else 0
            c = prev[i - bpp] if i >= bpp else 0
            out[i] = (row[i] - _paeth(a, prev[i], c)) & 0xFF
    else:
        raise CodecError("未知滤波器类型: %d" % ftype)
    return bytes(out)


def _unfilter_row(ftype, frow, prev, bpp):
    n = len(frow)
    if ftype == FILTER_NONE:
        return bytes(frow)
    out = bytearray(n)
    if ftype == FILTER_SUB:
        out[:bpp] = frow[:bpp]
        for i in range(bpp, n):
            out[i] = (frow[i] + out[i - bpp]) & 0xFF
    elif ftype == FILTER_UP:
        for i in range(n):
            out[i] = (frow[i] + prev[i]) & 0xFF
    elif ftype == FILTER_AVERAGE:
        for i in range(n):
            a = out[i - bpp] if i >= bpp else 0
            out[i] = (frow[i] + ((a + prev[i]) >> 1)) & 0xFF
    elif ftype == FILTER_PAETH:
        for i in range(n):
            a = out[i - bpp] if i >= bpp else 0
            c = prev[i - bpp] if i >= bpp else 0
            out[i] = (frow[i] + _paeth(a, prev[i], c)) & 0xFF
    else:
        raise CodecError("未知滤波器类型: %d" % ftype)
    return bytes(out)


def _score(filtered):
    """滤波判据：有符号残差绝对值之和，越小越容易被 zlib 压缩。"""
    return sum(map(_SCORE.__getitem__, filtered))


# ---------------------------------------------------------------- 编解码

def _validate(width, height, channels, bit_depth, pixels):
    if not (1 <= width <= 0xFFFFFFFF and 1 <= height <= 0xFFFFFFFF):
        raise CodecError("宽高必须在 [1, 2**32-1] 内")
    if not (1 <= channels <= 255):
        raise CodecError("通道数必须在 [1, 255] 内")
    if not (1 <= bit_depth <= 16):
        raise CodecError("位深必须在 [1, 16] 内")
    if len(pixels) != width * height * channels:
        raise CodecError(
            "像素数量不符: 期望 %d, 实际 %d"
            % (width * height * channels, len(pixels))
        )
    limit = 1 << bit_depth
    if len(pixels) and (min(pixels) < 0 or max(pixels) >= limit):
        raise CodecError("像素值超出位深范围 [0, %d)" % limit)


def encode(width, height, channels, bit_depth, pixels, level=9, want_stats=False):
    """编码像素序列为 ICF1 字节串。

    pixels: 长度为 width*height*channels 的整数序列（bytes/list 均可），
            按行优先、通道交错的顺序排列。
    返回: bytes；want_stats=True 时返回 (bytes, stats dict)。
    """
    _validate(width, height, channels, bit_depth, pixels)

    bits_per_pixel = channels * bit_depth
    bpp = max(1, (bits_per_pixel + 7) // 8)  # 滤波用的"字节步长"，与 PNG 一致
    row_bytes = (width * bits_per_pixel + 7) // 8
    row_vals = width * channels

    payload = bytearray()
    filter_counts = [0] * 5
    prev = bytes(row_bytes)
    for y in range(height):
        row = _pack_row(pixels[y * row_vals:(y + 1) * row_vals], bit_depth)
        best_type = FILTER_NONE
        best_data = None
        best_score = None
        for ftype in range(5):
            filtered = _filter_row(ftype, row, prev, bpp)
            score = _score(filtered)
            if best_score is None or score < best_score:
                best_type, best_data, best_score = ftype, filtered, score
        filter_counts[best_type] += 1
        payload.append(best_type)
        payload += best_data
        prev = row

    compressed = zlib.compress(bytes(payload), level)
    blob = _HEADER.pack(MAGIC, width, height, channels, bit_depth) + compressed

    if want_stats:
        raw = row_bytes * height
        stats = {
            "width": width,
            "height": height,
            "channels": channels,
            "bit_depth": bit_depth,
            "raw_bytes": raw,                       # 打包后未压缩的像素字节数
            "filtered_bytes": len(payload),         # 加滤波器标记后的负载大小
            "compressed_bytes": len(compressed),
            "file_bytes": len(blob),
            "ratio": raw / len(blob),               # 压缩率：原始 / 文件
            "filter_counts": dict(zip(FILTER_NAMES, filter_counts)),
        }
        return blob, stats
    return blob


def decode(blob, want_stats=False):
    """解码 ICF1 字节串。

    返回 (width, height, channels, bit_depth, pixels)。
    bit_depth == 8 时 pixels 为 bytes，否则为 list[int]。
    数据损坏时抛出 CodecError。
    """
    if len(blob) < _HEADER.size:
        raise CodecError("数据太短，不是合法的 ICF1 文件")
    magic, width, height, channels, bit_depth = _HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise CodecError("magic 不匹配，不是 ICF1 文件")
    if not (1 <= channels <= 255 and 1 <= bit_depth <= 16):
        raise CodecError("头部参数非法")

    bits_per_pixel = channels * bit_depth
    bpp = max(1, (bits_per_pixel + 7) // 8)
    row_bytes = (width * bits_per_pixel + 7) // 8
    row_vals = width * channels
    stride = 1 + row_bytes

    try:
        payload = zlib.decompress(blob[_HEADER.size:])
    except zlib.error as exc:
        raise CodecError("zlib 解压失败: %s" % exc)
    if len(payload) != height * stride:
        raise CodecError(
            "负载长度不符: 期望 %d, 实际 %d" % (height * stride, len(payload))
        )

    if bit_depth == 8:
        pixels = bytearray(width * height * channels)
    else:
        pixels = [0] * (width * height * channels)

    filter_counts = [0] * 5
    prev = bytes(row_bytes)
    pos = 0
    for y in range(height):
        ftype = payload[pos]
        if ftype > FILTER_PAETH:
            raise CodecError("第 %d 行滤波器类型非法: %d" % (y, ftype))
        filter_counts[ftype] += 1
        frow = payload[pos + 1: pos + stride]
        pos += stride
        row = _unfilter_row(ftype, frow, prev, bpp)
        vals = _unpack_row(row, row_vals, bit_depth)
        base = y * row_vals
        if bit_depth == 8:
            pixels[base: base + row_vals] = bytes(vals)
        else:
            pixels[base: base + row_vals] = vals
        prev = row

    if bit_depth == 8:
        pixels = bytes(pixels)
    if want_stats:
        stats = {"filter_counts": dict(zip(FILTER_NAMES, filter_counts))}
        return (width, height, channels, bit_depth, pixels), stats
    return width, height, channels, bit_depth, pixels


# ---------------------------------------------------------------- 文件便捷接口

def write_file(path, width, height, channels, bit_depth, pixels, level=9):
    blob, stats = encode(width, height, channels, bit_depth, pixels,
                         level=level, want_stats=True)
    with open(path, "wb") as f:
        f.write(blob)
    return stats


def read_file(path):
    with open(path, "rb") as f:
        return decode(f.read())
