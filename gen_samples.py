#!/usr/bin/env python3
"""生成样例图像、运行边缘检测、输出结果与阈值敏感性数据。"""

import math
import os
import random

import edge_detect as ed

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, 'samples')
RESULTS = os.path.join(HERE, 'results')


def add_gaussian_noise(img, sigma, seed):
    rng = random.Random(seed)
    return [[max(0, min(255, int(round(v + rng.gauss(0, sigma))))) for v in row]
            for row in img]


def make_solid(w=256, h=256, value=128):
    return [[value] * w for _ in range(h)]


def make_noise(w=256, h=256, seed=42):
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(w)] for _ in range(h)]


def make_line(w=256, h=256):
    img = [[0] * w for _ in range(h)]
    for r in range(h):
        img[r][w // 2] = 255
    return add_gaussian_noise(img, 10, seed=11)


def make_shapes(w=256, h=256):
    """不同对比度的矩形、圆形轮廓 + 高斯噪声（对比度差异大的场景）。"""
    img = [[30.0] * w for _ in range(h)]
    for r in range(30, 110):            # 高对比矩形
        for c in range(30, 110):
            img[r][c] = 230.0
    for r in range(h):                  # 低对比矩形
        for c in range(150, 230):
            if 30 <= r < 110:
                img[r][c] = 62.0
    cx, cy, rad = 128, 180, 45          # 圆形轮廓（中等对比）
    for r in range(h):
        for c in range(w):
            d = abs(math.hypot(r - cy, c - cx) - rad)
            if d < 1.5:
                img[r][c] = 170.0
    return add_gaussian_noise(img, 12, seed=5)


def make_big(w=2048, h=2048):
    rng = random.Random(1)
    img = []
    for r in range(h):
        row = [rng.randrange(40) for _ in range(w)]
        row[min(r, w - 1)] = 255
        img.append(row)
    return img


def run_case(name, img, save_input=True):
    print('--- %s (%dx%d) ---' % (name, len(img[0]), len(img)))
    if save_input:
        ed.write_pgm(os.path.join(SAMPLES, name + '.pgm'), img)
    nms = ed.compute_nms(img)
    low, high = ed.auto_thresholds(nms)
    edges = ed.hysteresis(nms, low, high)
    n = ed.count_edges(edges)
    total = len(img) * len(img[0])
    print('auto low=%.3f high=%.3f  edges=%d (%.2f%%)'
          % (low, high, n, 100.0 * n / total))
    ed.write_pgm(os.path.join(RESULTS, name + '_edges.pgm'),
                 ed.edges_to_image(edges))
    rows = ed.sensitivity(nms, low, high)
    csv_path = os.path.join(RESULTS, name + '_sensitivity.csv')
    import csv as _csv
    with open(csv_path, 'w', newline='') as f:
        wtr = _csv.DictWriter(f, fieldnames=['scale', 'low', 'high',
                                             'edge_pixels', 'miss_ratio'])
        wtr.writeheader()
        wtr.writerows(rows)
    for r in rows:
        print('  x%-4.2f low=%8.2f high=%8.2f edges=%7d miss=%.3f'
              % (r['scale'], r['low'], r['high'], r['edge_pixels'],
                 r['miss_ratio']))


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    run_case('solid', make_solid())
    run_case('noise', make_noise())
    run_case('line', make_line())
    run_case('shapes', make_shapes())
    run_case('big', make_big(), save_input=False)  # 2048x2048 超大图，不落盘输入


if __name__ == '__main__':
    main()
