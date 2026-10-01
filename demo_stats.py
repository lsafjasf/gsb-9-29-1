#!/usr/bin/env python3
"""Compression-ratio and filter-distribution demo for the RIF codec.

Runs a lossless roundtrip on three kinds of images and reports how often each
row filter wins plus the achieved compression ratio:

  1. synthetic  -- smooth gradients + solid rectangles (easy to compress)
  2. photo-like -- smooth bands plus mild additive noise (mixed)
  3. noise      -- uniform random bytes (incompressible)
"""

import math
import os
import random

import imgcodec

WIDTH = 800
HEIGHT = 600
CHANNELS = 3


def make_synthetic(width, height):
    """Diagonal gradient background with a few flat rectangles."""
    buf = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            buf[i] = (x * 255) // width
            buf[i + 1] = (y * 255) // height
            buf[i + 2] = ((x + y) * 255) // (width + height)
            i += 3
    rects = [(120, 90, 260, 200, (210, 30, 30)),
             (420, 140, 300, 260, (30, 180, 60)),
             (80, 380, 480, 120, (40, 60, 220))]
    for x0, y0, rw, rh, color in rects:
        for y in range(y0, min(y0 + rh, height)):
            start = (y * width + x0) * 3
            row_part = bytes(color) * min(rw, width - x0)
            buf[start:start + len(row_part)] = row_part
    return bytes(buf)


def make_photo_like(width, height):
    """Smooth sinusoidal-ish bands plus mild +-12 additive noise."""
    rng = random.Random(1234)
    buf = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            r = 120 + 100 * math.sin(x / 57.0)
            g = 120 + 100 * math.sin(y / 83.0)
            b = 120 + 100 * math.sin((x + y) / 101.0)
            buf[i] = max(0, min(255, int(r) + rng.randint(-12, 12)))
            buf[i + 1] = max(0, min(255, int(g) + rng.randint(-12, 12)))
            buf[i + 2] = max(0, min(255, int(b) + rng.randint(-12, 12)))
            i += 3
    return bytes(buf)


def make_noise(width, height):
    """Uniform random RGB bytes -- maximum zero-order entropy."""
    return os.urandom(width * height * 3)


def report(name, pixels, width, height, channels):
    raw_size = len(pixels)
    encoded, stats = imgcodec.encode_with_stats(pixels, width, height, channels)
    decoded, dw, dh, dc = imgcodec.decode(encoded)
    assert decoded == pixels, f"roundtrip mismatch for {name}"
    assert (dw, dh, dc) == (width, height, channels)
    ratio = len(encoded) / raw_size
    print(f"== {name} ==")
    print(f"  image      : {width}x{height}x{channels}  raw={raw_size} B  "
          f"encoded={len(encoded)} B")
    print(f"  ratio      : {ratio:.4f}  (encoded/raw, smaller is better; "
          f"{(1 - ratio) * 100:.2f}% saved)")
    print("  filters    :")
    for fname, (count, frac) in stats.distribution().items():
        bar = "#" * int(frac * 40)
        print(f"    {fname:<8s}: {count:5d} rows ({frac * 100:5.1f}%) {bar}")
    print()


def main():
    print(f"RIF codec demo  ({WIDTH}x{HEIGHT}x{CHANNELS}, zlib level 9)\n")
    report("synthetic (gradient + flat rects)",
           make_synthetic(WIDTH, HEIGHT), WIDTH, HEIGHT, CHANNELS)
    report("photo-like (smooth + mild noise)",
           make_photo_like(WIDTH, HEIGHT), WIDTH, HEIGHT, CHANNELS)
    report("random noise (os.urandom)",
           make_noise(WIDTH, HEIGHT), WIDTH, HEIGHT, CHANNELS)


if __name__ == "__main__":
    main()
