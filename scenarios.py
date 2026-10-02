"""Test scenarios.

Reproduction shapes (bug-triggering):
  narrow_corridor    - narrow slalom corridor, corner cutting
  sharp_turns        - consecutive tight turns with obstacles at inner corners
  hugging_obstacle   - path runs tight alongside an obstacle
  sparse_obstacles   - obstacle wall recorded as a few sparse small discs
  multi_iteration    - slight per-iteration drift accumulates over many passes

Boundary cases:
  single_point, straight_line, closed_loop, enclosed
"""

import math


def narrow_corridor():
    # Slalom through a corridor made of obstacle discs. Rounding the
    # slalom straightens the path into the pinch discs.
    path = [(0.0, 0.0), (2.0, 0.6), (4.0, -0.6),
            (6.0, 0.6), (8.0, -0.6), (10.0, 0.0)]
    obstacles = [(2.0, -0.5, 0.55), (4.0, 0.5, 0.55),
                 (6.0, -0.5, 0.55), (8.0, 0.5, 0.55)]
    return dict(name="narrow_corridor", path=path, obstacles=obstacles,
                closed=False, min_clearance=0.1)


def sharp_turns():
    # Tight switchback: consecutive 90-degree turns with short legs;
    # discs pinch every inner corner, so rounding any corner cuts
    # through. A noisy straight stretch at the end can still be
    # smoothed locally (segment-wise fallback).
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
            (4.0, 1.0), (4.0, 2.0), (6.0, 2.0),
            (7.0, 2.15), (8.0, 1.9), (9.0, 2.1), (10.0, 2.0)]
    obstacles = [(1.5, 0.5, 0.39), (2.5, 0.5, 0.39), (3.5, 1.5, 0.39)]
    return dict(name="sharp_turns", path=path, obstacles=obstacles,
                closed=False, min_clearance=0.1)


def hugging_obstacle():
    # Path arches slightly around a disc that sits very close to it.
    # Smoothing flattens the arch straight into the disc.
    path = [(0.0, 0.0), (2.5, 0.35), (5.0, 0.4),
            (7.5, 0.35), (10.0, 0.0)]
    obstacles = [(5.0, -0.05, 0.2)]
    return dict(name="hugging_obstacle", path=path, obstacles=obstacles,
                closed=False, min_clearance=0.1)


def sparse_obstacles():
    # A wall recorded sparsely: only two small discs, far apart.
    # Coarse segment sampling steps over them; the continuous check sees them.
    path = [(0.0, 0.0), (5.0, 0.6), (10.0, 0.0)]
    obstacles = [(4.4, 0.08, 0.12), (5.6, 0.08, 0.12)]
    return dict(name="sparse_obstacles", path=path, obstacles=obstacles,
                closed=False, min_clearance=0.1, sample_step=1.0)


def multi_iteration():
    # Same shape as hugging_obstacle: a few iterations barely move the
    # path, but many iterations accumulate drift deep into the disc.
    s = hugging_obstacle()
    s = dict(s)
    s["name"] = "multi_iteration"
    s["few_iterations"] = 1
    s["many_iterations"] = 120
    return s


def closed_loop():
    # Hexagon loop with a disc in the middle; smoothing shrinks the
    # loop toward the center disc.
    path = [(3.0 * math.cos(math.radians(a)), 3.0 * math.sin(math.radians(a)))
            for a in range(0, 360, 60)]
    obstacles = [(0.0, 0.0, 1.2)]
    return dict(name="closed_loop", path=path, obstacles=obstacles,
                closed=True, min_clearance=0.1)


def enclosed():
    # Obstacles cage the whole path: peak discs sit 0.12 above and below
    # every vertex (blocking any vertical move), and zero-line discs sit
    # on the chord between vertices (blocking the fully flattened
    # candidate). Every smoothing level violates the margin, so the
    # smoother must return the original path (L3).
    path = [(float(i), 0.3 if i % 2 == 0 else -0.3) for i in range(9)]
    obstacles = []
    for x, y in path:
        obstacles.append((x, y + 0.12, 0.0))
        obstacles.append((x, y - 0.12, 0.0))
    for i in range(1, 8):
        obstacles.append((float(i), 0.0, 0.0))
    return dict(name="enclosed", path=path, obstacles=obstacles,
                closed=False, min_clearance=0.1)


def single_point():
    return dict(name="single_point", path=[(1.0, 2.0)], obstacles=[],
                closed=False, min_clearance=0.1)


def straight_line():
    return dict(name="straight_line",
                path=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
                obstacles=[(1.5, 0.4, 0.2)], closed=False, min_clearance=0.1)


REPRO_SHAPES = [narrow_corridor, sharp_turns, hugging_obstacle,
                sparse_obstacles, multi_iteration]

EDGE_CASES = [single_point, straight_line, closed_loop, enclosed]

ALL_SCENARIOS = REPRO_SHAPES + EDGE_CASES
