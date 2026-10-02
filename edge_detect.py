#!/usr/bin/env python3
"""edge_detect.py -- Canny 风格边缘检测库（仅 Python 3 标准库）。

流程：高斯平滑 -> Sobel 梯度 -> 方向非极大抑制 -> Otsu 自动双阈值 -> 滞后连接。

边界规则（明确约定）：
  所有卷积/差分均使用“边缘复制”(replicate/clamp) 填充。
  常数图像在边界处梯度为 0，不会像零填充那样在图像边框产生虚假边缘。
  非极大抑制同样对越界邻居取边缘复制值，因此边界像素可以正常成为边缘。

自动阈值依据（默认 robust 法）：
  NMS 非零幅值中绝大多数来自噪声/纹理，真实边缘只占很小比例，因此用
  稳健统计量刻画“噪声底”：high = median + 3 * 1.4826 * MAD
  （MAD 为中位数绝对偏差，1.4826 使其对高斯分布等价于标准差，
  3 倍等效标准差对应约 0.1% 的噪声误检率），low = low_ratio * high。
  该阈值随图像自身噪声水平缩放，且不被少量强边缘拉抬——同一张图里
  对比度差异很大的边缘（强边缘与弱边缘并存）都能保留，而纯噪声被抑制。
  另提供 Otsu 法（method='otsu'）：对 NMS 非零幅值做最大类间方差分割，
  适合边缘占比高、对比度较一致的图像。
"""

import argparse
import csv
import math
import statistics
from collections import deque

# ---------------------------------------------------------------- PGM I/O

def read_pgm(path):
    """读取 P2/P5 PGM，返回 list[list[int]]。"""
    with open(path, 'rb') as f:
        data = f.read()
    tokens = []
    i = 0
    # 解析头部：magic, width, height, maxval（跳过注释）
    while len(tokens) < 4:
        while i < len(data) and data[i:i+1].isspace():
            i += 1
        if data[i:i+1] == b'#':
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j+1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    magic = tokens[0]
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1  # 头部后单个空白字符
    if magic == b'P5':
        if maxval > 255:
            raise ValueError('仅支持 maxval<=255')
        pix = data[i:i + w * h]
        if len(pix) < w * h:
            raise ValueError('PGM 数据长度不足')
        return [list(pix[r * w:(r + 1) * w]) for r in range(h)]
    elif magic == b'P2':
        body = data[i:].split()
        vals = [int(t) for t in body[:w * h]]
        return [vals[r * w:(r + 1) * w] for r in range(h)]
    raise ValueError('不支持的 PGM 格式: %r' % magic)


def write_pgm(path, img):
    """写 P5 二值/灰度 PGM。img 为 list[list[number]]，自动裁剪到 [0,255]。"""
    h = len(img)
    w = len(img[0])
    buf = bytearray()
    for row in img:
        for v in row:
            iv = int(round(v))
            buf.append(0 if iv < 0 else (255 if iv > 255 else iv))
    with open(path, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (w, h))
        f.write(bytes(buf))


def edges_to_image(edges):
    """0/1 边缘图 -> 0/255 图像。"""
    return [[255 if v else 0 for v in row] for row in edges]

# ---------------------------------------------------------------- 卷积

def gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(3.0 * sigma + 0.5))
    k = [math.exp(-(x * x) / (2.0 * sigma * sigma)) for x in range(-radius, radius + 1)]
    s = sum(k)
    return [v / s for v in k]


def _conv_horizontal(img, k):
    r = len(k) // 2
    out = []
    for row in img:
        n = len(row)
        ext = [row[0]] * r + row + [row[-1]] * r
        acc = [k[0] * v for v in ext[0:n]]
        for j in range(1, len(k)):
            kj = k[j]
            acc = [a + kj * b for a, b in zip(acc, ext[j:j + n])]
        out.append(acc)
    return out


def _conv_vertical(img, k):
    r = len(k) // 2
    n = len(img)
    ext = [img[0]] * r + img + [img[-1]] * r
    out = []
    for i in range(n):
        acc = [k[0] * v for v in ext[i]]
        for j in range(1, len(k)):
            kj = k[j]
            acc = [a + kj * b for a, b in zip(acc, ext[i + j])]
        out.append(acc)
    return out


def gaussian_blur(img, sigma=1.4):
    """可分离高斯平滑，边界使用边缘复制填充。"""
    k = gaussian_kernel1d(sigma)
    return _conv_vertical(_conv_horizontal(img, k), k)

# ---------------------------------------------------------------- 梯度

def sobel(img):
    """Sobel 梯度，边界使用边缘复制填充（常数区域边界梯度为 0，无虚假边缘）。

    返回 (gx, gy)：gx 沿 x 方向（列），gy 沿 y 方向（行）。
    """
    h = len(img)
    w = len(img[0])
    ext = [img[0]] + img + [img[-1]]
    gx = []
    gy = []
    for i in range(h):
        up = [ext[i][0]] + ext[i] + [ext[i][-1]]
        mid = [ext[i + 1][0]] + ext[i + 1] + [ext[i + 1][-1]]
        dn = [ext[i + 2][0]] + ext[i + 2] + [ext[i + 2][-1]]
        gx.append([(up[j + 2] + 2.0 * mid[j + 2] + dn[j + 2])
                   - (up[j] + 2.0 * mid[j] + dn[j]) for j in range(w)])
        gy.append([(dn[j] + 2.0 * dn[j + 1] + dn[j + 2])
                   - (up[j] + 2.0 * up[j + 1] + up[j + 2]) for j in range(w)])
    return gx, gy


def gradient_magnitude(gx, gy):
    hypot = math.hypot
    return [[hypot(a, b) for a, b in zip(rx, ry)] for rx, ry in zip(gx, gy)]

# ---------------------------------------------------------------- 非极大抑制

_TAN_22_5 = 0.41421356237309503  # tan(pi/8)


def nonmax_suppression(gx, gy, mag):
    """沿梯度方向做非极大抑制，方向量化为 0/45/90/135 四个扇区。

    越界邻居取边缘复制值，保证边界像素也可被保留为边缘。
    """
    h = len(mag)
    w = len(mag[0])
    out = [[0.0] * w for _ in range(h)]
    for i in range(h):
        iu = i - 1 if i > 0 else 0
        idn = i + 1 if i < h - 1 else h - 1
        row_m = mag[i]
        row_up = mag[iu]
        row_dn = mag[idn]
        row_gx = gx[i]
        row_gy = gy[i]
        row_out = out[i]
        for j in range(w):
            m = row_m[j]
            if m == 0.0:
                continue
            jl = j - 1 if j > 0 else 0
            jr = j + 1 if j < w - 1 else w - 1
            gxi = row_gx[j]
            gyi = row_gy[j]
            ax = gxi if gxi >= 0.0 else -gxi
            ay = gyi if gyi >= 0.0 else -gyi
            if ay <= _TAN_22_5 * ax:            # 0 度：左右
                n1 = row_m[jl]
                n2 = row_m[jr]
            elif ax <= _TAN_22_5 * ay:          # 90 度：上下
                n1 = row_up[j]
                n2 = row_dn[j]
            elif (gxi >= 0.0) == (gyi >= 0.0):  # +45 度
                n1 = row_up[jr]
                n2 = row_dn[jl]
            else:                               # -45 度
                n1 = row_up[jl]
                n2 = row_dn[jr]
            if m >= n1 and m >= n2:
                row_out[j] = m
    return out

# ---------------------------------------------------------------- 自动阈值

def otsu_threshold(values, nbins=256):
    """一维 Otsu：最大化类间方差。values 为正浮点序列，返回阈值（浮点）。"""
    vals = list(values)
    if not vals:
        return 0.0
    vmax = max(vals)
    if vmax <= 0.0:
        return 0.0
    hist = [0] * nbins
    scale = (nbins - 1) / vmax
    for v in vals:
        hist[int(v * scale)] += 1
    total = len(vals)
    sum_all = 0.0
    for b in range(nbins):
        sum_all += b * hist[b]
    sum_b = 0.0
    w_b = 0
    best_var = -1.0
    best_bin = 0
    for b in range(nbins):
        w_b += hist[b]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += b * hist[b]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f)
        if var_between > best_var:
            best_var = var_between
            best_bin = b
    return (best_bin + 0.5) / scale


def auto_thresholds(nms, low_ratio=0.5, method='robust', k=3.0):
    """由 NMS 幅值统计自动确定 (low, high)。

    method='robust'（默认）: high = median + k * 1.4826 * MAD，
        以噪声底为基准，对同图内对比度差异大的边缘最稳健。
    method='otsu': high = NMS 非零幅值的 Otsu 最大类间方差阈值。
    low = low_ratio * high。
    图像无非零梯度（如纯色图）时返回 (0.0, 0.0)，此时不产生任何边缘。
    """
    vals = [v for row in nms for v in row if v > 0.0]
    if not vals:
        return 0.0, 0.0
    if method == 'otsu':
        high = otsu_threshold(vals)
    elif method == 'robust':
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals)
        high = med + k * 1.4826 * mad
    else:
        raise ValueError('未知阈值方法: %r' % method)
    return high * low_ratio, high

# ---------------------------------------------------------------- 滞后连接

def hysteresis(nms, low, high):
    """双阈值滞后连接：>=high 为强边缘，>=low 且与强边缘 8-连通者保留。"""
    h = len(nms)
    w = len(nms[0])
    edges = [[0] * w for _ in range(h)]
    queue = deque()
    for i in range(h):
        row = nms[i]
        erow = edges[i]
        for j in range(w):
            if row[j] >= high and high > 0.0:
                erow[j] = 1
                queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        for di in (-1, 0, 1):
            ni = i + di
            if ni < 0 or ni >= h:
                continue
            nrow = nms[ni]
            erow = edges[ni]
            for dj in (-1, 0, 1):
                nj = j + dj
                if nj < 0 or nj >= w or (di == 0 and dj == 0):
                    continue
                if not erow[nj] and nrow[nj] >= low:
                    erow[nj] = 1
                    queue.append((ni, nj))
    return edges

# ---------------------------------------------------------------- 顶层管线

def compute_nms(img, sigma=1.4):
    """平滑 + 梯度 + NMS，返回 nms 幅值图（供阈值扫描复用）。"""
    blurred = gaussian_blur(img, sigma)
    gx, gy = sobel(blurred)
    mag = gradient_magnitude(gx, gy)
    return nonmax_suppression(gx, gy, mag)


def detect_edges(img, sigma=1.4, low=None, high=None, low_ratio=0.5,
                 method='robust'):
    """完整管线。返回 (edges, low, high)；low/high 为 None 时自动确定。"""
    nms = compute_nms(img, sigma)
    if low is None or high is None:
        auto_low, auto_high = auto_thresholds(nms, low_ratio, method)
        if low is None:
            low = auto_low
        if high is None:
            high = auto_high
    return hysteresis(nms, low, high), low, high


def count_edges(edges):
    return sum(sum(row) for row in edges)


def sensitivity(nms, auto_low, auto_high, scales=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """阈值敏感性：不同缩放系数下的边缘像素数与漏检比例。

    漏检比例 = 1 - count(T) / count(T_min)，即以最低测试阈值（最宽松）
    得到的边缘集合为基准，阈值升高所丢失的边缘占比。
    """
    rows = []
    baseline = None
    degenerate = False
    for s in scales:
        low = auto_low * s
        high = auto_high * s
        edges = hysteresis(nms, low, high)
        count = count_edges(edges)
        if baseline is None:
            baseline = count
            if count == 0:  # 纯色等退化情形：无基准边缘，漏检比例定义为 0
                degenerate = True
        rows.append({
            'scale': s,
            'low': low,
            'high': high,
            'edge_pixels': count,
            'miss_ratio': 0.0 if degenerate else 1.0 - count / baseline,
        })
    return rows

# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description='Canny 风格边缘检测（纯标准库）')
    ap.add_argument('input', help='输入 PGM (P2/P5)')
    ap.add_argument('-o', '--output', help='输出边缘图 PGM')
    ap.add_argument('--sigma', type=float, default=1.4, help='高斯平滑 sigma（默认 1.4）')
    ap.add_argument('--low', type=float, default=None, help='手动低阈值')
    ap.add_argument('--high', type=float, default=None, help='手动高阈值')
    ap.add_argument('--low-ratio', type=float, default=0.5, help='low/high 比例（默认 0.5）')
    ap.add_argument('--method', choices=('robust', 'otsu'), default='robust',
                    help='自动阈值方法（默认 robust）')
    ap.add_argument('--sensitivity', metavar='CSV', help='输出阈值敏感性 CSV')
    args = ap.parse_args(argv)

    img = read_pgm(args.input)
    nms = compute_nms(img, args.sigma)
    auto_low, auto_high = auto_thresholds(nms, args.low_ratio, args.method)
    low = args.low if args.low is not None else auto_low
    high = args.high if args.high is not None else auto_high
    edges = hysteresis(nms, low, high)
    n = count_edges(edges)
    total = len(img) * len(img[0])
    print('图像: %dx%d  自动阈值: low=%.3f high=%.3f  使用: low=%.3f high=%.3f'
          % (len(img[0]), len(img), auto_low, auto_high, low, high))
    print('边缘像素: %d (%.2f%%)' % (n, 100.0 * n / total))
    if args.output:
        write_pgm(args.output, edges_to_image(edges))
        print('已写出: %s' % args.output)
    if args.sensitivity:
        rows = sensitivity(nms, auto_low, auto_high)
        with open(args.sensitivity, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['scale', 'low', 'high',
                                                   'edge_pixels', 'miss_ratio'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print('已写出: %s' % args.sensitivity)


if __name__ == '__main__':
    main()
