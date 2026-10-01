"""Deterministic test/demo image generators (rows of RGB tuples)."""

import random


def gradient(width, height):
    """Horizontal gray ramp 0..255, repeated for every row."""
    row = [(round(x * 255 / max(1, width - 1)),) * 3
           for x in range(width)]
    return [list(row) for _ in range(height)]


def vertical_gradient(width, height):
    return [
        [(round(y * 255 / max(1, height - 1)),) * 3 for _ in range(width)]
        for y in range(height)
    ]


def solid(width, height, color):
    return [[tuple(color) for _ in range(width)] for _ in range(height)]


def noise(width, height, seed=0):
    """Uniformly distributed independent RGB noise."""
    rng = random.Random(seed)
    return [[(rng.randrange(256), rng.randrange(256), rng.randrange(256))
             for _ in range(width)] for _ in range(height)]


def color_zones(width, height):
    """Four flat saturated zones with smooth ramps between them."""
    zones = ((220, 30, 40), (40, 200, 90), (30, 60, 210), (230, 200, 40))
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            t = x * 4 // max(1, width)
            c0 = zones[min(t, 3)]
            c1 = zones[min(t + 1, 3)]
            f = (x * 4 / max(1, width)) - t
            row.append(tuple(round(a + (b - a) * f) for a, b in zip(c0, c1)))
        rows.append(row)
    return rows
