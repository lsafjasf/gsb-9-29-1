"""
benchmark.py — 暴力实现 vs 可分离/分段分解实现 的耗时对比与等价性对拍。

运行：python3 benchmark.py
"""

import random
import time

from morphology import (
    erode, dilate, erode_brute, dilate_brute, decompose,
)


def rand_image(h, w, rng):
    return [[rng.randint(0, 255) for _ in range(w)] for _ in range(h)]


def rect_se(h, w):
    return [[True] * w for _ in range(h)]


def disk_se(r):
    return [[(dy * dy + dx * dx) <= r * r for dx in range(-r, r + 1)]
            for dy in range(-r, r + 1)]


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def main():
    rng = random.Random(2026)
    size = 200
    img = rand_image(size, size, rng)
    print(f"图像: {size}x{size} 灰度 (0..255)，Python 纯标准库\n")
    header = f"{'结构元素':<22}{'|SE|':>6} {'暴力(s)':>10} {'分解(s)':>10} {'加速比':>8} {'结果一致':>8}"
    print(header)
    print("-" * len(header))

    cases = [("矩形 5x5", rect_se(5, 5)),
             ("矩形 15x15", rect_se(15, 15)),
             ("矩形 31x31", rect_se(31, 31)),
             ("矩形 63x63", rect_se(63, 63)),
             ("圆盘 r=10", disk_se(10)),
             ("圆盘 r=20", disk_se(20))]
    for name, se in cases:
        n_pix = sum(sum(row) for row in se)
        e_fast, t_fast = timed(erode, img, se)
        e_slow, t_slow = timed(erode_brute, img, se)
        ok = e_fast == e_slow
        print(f"{name:<22}{n_pix:>6} {t_slow:>10.3f} {t_fast:>10.3f} "
              f"{t_slow / t_fast:>7.1f}x {str(ok):>8}")
        assert ok, name

    print("\n超大结构元素（仅分解实现，暴力已不现实）：")
    for name, se in [("矩形 127x127", rect_se(127, 127)),
                     ("矩形 255x255", rect_se(255, 255)),
                     ("圆盘 r=60", disk_se(60))]:
        _, t_fast = timed(erode, img, se)
        print(f"  {name:<20} 分解实现耗时 {t_fast:.3f}s")

    # 膨胀 + 灰度/二值 各抽一组对拍
    print("\n膨胀对拍抽查：", end=" ")
    for se in (rect_se(17, 9), disk_se(8)):
        d_fast, _ = timed(dilate, img, se)
        d_slow, _ = timed(dilate_brute, img, se)
        assert d_fast == d_slow
    bin_img = [[v & 1 for v in row] for row in img]
    assert erode(bin_img, disk_se(6)) == erode_brute(bin_img, disk_se(6))
    assert dilate(bin_img, rect_se(11, 11)) == dilate_brute(bin_img, rect_se(11, 11))
    print("全部一致 ✓")

    print("\n分解信息示例：")
    print("  矩形 63x63 ->", decompose(rect_se(63, 63)))
    info = decompose(disk_se(10))
    print(f"  圆盘 r=10  -> kind={info['kind']}, 水平段数={len(info['runs'])}")


if __name__ == "__main__":
    main()
