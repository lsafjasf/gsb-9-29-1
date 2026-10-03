#!/usr/bin/env python3
"""Generate sample images, run edge detection, and write results.

Writes to samples/:
  *_input.pgm        synthetic inputs (pure color, noise, line, scene)
  *_edges.pgm        detected edge maps
  *_gt.pgm           ground truth where known (line, scene)
  *_sensitivity.csv  threshold-sensitivity sweep per sample
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edge_detect import (detect_edges, edges_to_image, sensitivity, to_csv,
                         write_pgm)

OUT = os.path.join(os.path.dirname(__file__), "..", "samples")


def pure_color(w=200, h=200, value=128):
    return [[value] * w for _ in range(h)], None


def pure_noise(w=256, h=256, seed=7, sigma=25.0):
    rng = random.Random(seed)
    rows = [[max(0, min(255, int(round(rng.gauss(128.0, sigma)))))
             for _ in range(w)] for _ in range(h)]
    return rows, None


def single_pixel_line(w=200, h=200):
    rows = [[0] * w for _ in range(h)]
    gt = [[False] * w for _ in range(h)]
    y = h // 2
    for x in range(10, w - 10):
        rows[y][x] = 255
        gt[y][x] = True
    x = 150
    for yy in range(20, h - 20):
        rows[yy][x] = 255
        gt[yy][x] = True
    return rows, gt


def mixed_contrast_scene(w=512, h=512, seed=3, noise_sigma=3.0):
    """Rectangles of very different contrast on a gradient background."""
    rng = random.Random(seed)
    rows = [[int(40 + 60 * x / w) for x in range(w)] for _ in range(h)]
    gt = [[False] * w for _ in range(h)]
    boxes = [  # (x0, y0, x1, y1, delta) -- contrast varies 8..200
        (40, 40, 160, 160, 200),
        (200, 40, 320, 160, 60),
        (360, 40, 480, 160, 20),
        (40, 220, 160, 340, 8),
        (200, 220, 320, 340, 120),
    ]
    for (x0, y0, x1, y1, delta) in boxes:
        for y in range(y0, y1):
            for x in range(x0, x1):
                rows[y][x] = max(0, min(255, rows[y][x] + delta))
        for x in range(x0, x1):
            gt[y0][x] = gt[y1 - 1][x] = True
        for y in range(y0, y1):
            gt[y][x0] = gt[y][x1 - 1] = True
    cx, cy, r = 420, 300, 60
    for y in range(h):
        for x in range(w):
            d = math.hypot(x - cx, y - cy)
            if d < r:
                rows[y][x] = max(0, min(255, rows[y][x] + 90))
            if abs(d - r) < 0.5:
                gt[y][x] = True
    for y in range(h):
        for x in range(w):
            rows[y][x] = max(0, min(255,
                             int(round(rows[y][x]
                                       + rng.gauss(0.0, noise_sigma)))))
    return rows, gt


def run_case(name, rows, gt):
    write_pgm(os.path.join(OUT, name + "_input.pgm"), rows)
    edges, (low, high) = detect_edges(rows)
    write_pgm(os.path.join(OUT, name + "_edges.pgm"), edges_to_image(edges))
    n = sum(row.count(True) for row in edges)
    total = len(rows) * len(rows[0])
    print("%-12s %4dx%-4d auto=(%.3f, %.3f) edges=%d (%.3f%%)"
          % (name, len(rows[0]), len(rows), low, high, n, 100.0 * n / total))
    if gt is not None:
        write_pgm(os.path.join(OUT, name + "_gt.pgm"), edges_to_image(gt))
    report = sensitivity(rows, gt=gt)
    with open(os.path.join(OUT, name + "_sensitivity.csv"), "w") as f:
        f.write(to_csv(report))


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (rows, gt) in [
        ("pure_color", pure_color()),
        ("pure_noise", pure_noise()),
        ("line", single_pixel_line()),
        ("scene", mixed_contrast_scene()),
    ]:
        run_case(name, rows, gt)
    print("samples written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
