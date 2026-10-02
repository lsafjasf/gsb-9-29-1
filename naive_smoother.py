"""Buggy reference implementation (pre-fix), kept to reproduce the defect.

Two defects, matching the reported symptoms:

1. `naive_smooth`: Laplacian iterations are applied without any collision
   check. Small per-iteration violations accumulate downstream, so after
   many iterations / several smoothing passes the path is inside a wall.

2. `naive_smooth_with_sampled_check`: the collision test only looks at
   discrete sample points along each segment. When obstacle records are
   sparse (or sampling is slightly too coarse), a thin obstacle sitting
   between two sample points is never seen, even though the segment
   passes straight through it.
"""

import math

from smoother import laplacian_smooth


def naive_smooth(path, iterations=60, alpha=0.6, closed=False):
    """Defect 1: smoothing with no collision checking at all."""
    return laplacian_smooth(path, iterations, alpha, closed)


def sampled_segment_clearance(a, b, obstacles, sample_step):
    """Defect 2: clearance evaluated only at discrete sample points."""
    length = math.dist(a, b)
    best = math.inf
    samples = 0 if length == 0.0 else int(length / sample_step)
    for k in range(samples + 1):
        t = min(1.0, (k * sample_step) / length) if length else 0.0
        sx = a[0] + t * (b[0] - a[0])
        sy = a[1] + t * (b[1] - a[1])
        for ox, oy, r in obstacles:
            d = math.hypot(ox - sx, oy - sy) - r
            if d < best:
                best = d
    return best


def naive_smooth_with_sampled_check(path, obstacles, *, iterations=60, alpha=0.6,
                                    min_clearance=0.1, sample_step=1.0, closed=False):
    """Smooth, then 'validate' with the buggy sampled check.

    Returns (smoothed_path, reported_safe, sample_based_checks).
    `reported_safe` is what the buggy pipeline believes; it can be True
    while a continuous check proves the path actually collides.
    """
    candidate = laplacian_smooth(path, iterations, alpha, closed)
    worst = math.inf
    checks = 0
    n = len(candidate)
    last = n if closed else n - 1
    for i in range(last):
        a = candidate[i]
        b = candidate[(i + 1) % n]
        length = math.dist(a, b)
        samples = 0 if length == 0.0 else int(length / sample_step)
        checks += (samples + 1) * len(list(obstacles))
        c = sampled_segment_clearance(a, b, obstacles, sample_step)
        if c < worst:
            worst = c
    return candidate, worst >= min_clearance, checks
