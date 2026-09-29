"""压缩率与滤波器分布统计：python3 demo_stats.py

对四类典型图像（合成图 / 随机噪声 / 全同色 / 照片式渐变+噪声）分别编码，
输出压缩率和各滤波器的选择分布，并验证往返无损。
"""

import random

import imgcodec


def make_synthetic(w, h):
    """合成图：渐变背景 + 纯色矩形 + 圆形，模拟 UI 截图/图表。"""
    px = bytearray(w * h * 3)
    for y in range(h):
        base = y * w * 3
        for x in range(w):
            i = base + x * 3
            px[i] = (x * 255) // (w - 1)          # R 水平渐变
            px[i + 1] = (y * 255) // (h - 1)      # G 垂直渐变
            px[i + 2] = 128
    def rect(x0, y0, x1, y1, rgb):
        for y in range(y0, y1):
            for x in range(x0, x1):
                i = (y * w + x) * 3
                px[i:i + 3] = bytes(rgb)
    rect(50, 50, 300, 200, (200, 30, 30))
    rect(400, 100, 700, 350, (30, 30, 200))
    rect(100, 400, 500, 550, (240, 240, 240))
    cx, cy, r = 600, 450, 100                    # 圆
    for y in range(max(0, cy - r), min(h, cy + r)):
        for x in range(max(0, cx - r), min(w, cx + r)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                i = (y * w + x) * 3
                px[i:i + 3] = b"\x00\xa0\x00"
    return bytes(px)


def make_noise(w, h, seed=42):
    """随机噪声图。"""
    return random.Random(seed).randbytes(w * h * 3)


def make_solid(w, h):
    """全同色图。"""
    return bytes((70, 130, 180)) * (w * h)


def make_photo_like(w, h, seed=1):
    """照片式：平滑渐变叠加小幅噪声，模拟自然图像。"""
    rng = random.Random(seed)
    px = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            n = rng.randrange(-12, 13)
            px[i] = min(255, max(0, (x * 255) // w + n))
            px[i + 1] = min(255, max(0, (y * 255) // h + n))
            px[i + 2] = min(255, max(0, ((x + y) * 255) // (w + h) + n))
    return bytes(px)


def report(name, w, h, ch, bd, px):
    blob, st = imgcodec.encode(w, h, ch, bd, px, want_stats=True)
    # 往返校验
    dw, dh, dch, dbd, dpix = imgcodec.decode(blob)
    ok = (dw, dh, dch, dbd) == (w, h, ch, bd) and list(dpix) == list(px)
    print(f"== {name}  ({w}x{h}, {ch}ch, {bd}bit) ==")
    print(f"  往返无损校验 : {'PASS' if ok else 'FAIL'}")
    print(f"  原始像素     : {st['raw_bytes']:>10,} 字节")
    print(f"  ICF1 文件    : {st['file_bytes']:>10,} 字节"
          f"  (含 14 字节头)")
    print(f"  压缩率       : {st['ratio']:>10.2f}x"
          f"  (文件为原始的 {100.0 / st['ratio']:.2f}%)")
    total = sum(st["filter_counts"].values())
    dist = "  ".join(
        f"{k}:{v}({100.0 * v / total:.1f}%)"
        for k, v in st["filter_counts"].items()
    )
    print(f"  滤波器分布   : {dist}")
    print()
    assert ok


def main():
    w, h = 800, 600
    report("合成图 (渐变+色块+圆)", w, h, 3, 8, make_synthetic(w, h))
    report("随机噪声图", w, h, 3, 8, make_noise(w, h))
    report("全同色图", w, h, 3, 8, make_solid(w, h))
    report("照片式图 (渐变+小噪声)", w, h, 3, 8, make_photo_like(w, h))

    # 非 8 位倍数宽度 + 低位深示例
    rng = random.Random(9)
    w2, h2 = 803, 601
    px2 = [rng.randrange(2) for _ in range(w2 * h2)]
    report("1bit 抖动图 (宽803, 非8倍数)", w2, h2, 1, 1, px2)


if __name__ == "__main__":
    main()
