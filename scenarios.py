"""Shared scenario definitions for tests and the comparison report.

Each scenario: name, path (list of (x, y)), obstacles (list of (cx, cy, r)).
"""

import math


def _wall_circles(x0, y0, x1, y1, r, spacing):
    """A row of circles from (x0, y0) to (x1, y1) forming a wall."""
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(1, int(length / spacing))
    return [
        (x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, r)
        for k in range(n + 1)
    ]




def _offset_polyline(points, d):
    """Miter-joined offset of a polyline at signed distance d."""
    n = len(points)
    out = []
    for i, p in enumerate(points):
        if i == 0:
            dx = points[1][0] - p[0]
            dy = points[1][1] - p[1]
            L = math.hypot(dx, dy)
            nx, ny = -dy / L, dx / L
            out.append((p[0] + nx * d, p[1] + ny * d))
        elif i == n - 1:
            dx = p[0] - points[i - 1][0]
            dy = p[1] - points[i - 1][1]
            L = math.hypot(dx, dy)
            nx, ny = -dy / L, dx / L
            out.append((p[0] + nx * d, p[1] + ny * d))
        else:
            dx1 = p[0] - points[i - 1][0]
            dy1 = p[1] - points[i - 1][1]
            L1 = math.hypot(dx1, dy1)
            n1 = (-dy1 / L1, dx1 / L1)
            dx2 = points[i + 1][0] - p[0]
            dy2 = points[i + 1][1] - p[1]
            L2 = math.hypot(dx2, dy2)
            n2 = (-dy2 / L2, dx2 / L2)
            mx, my = n1[0] + n2[0], n1[1] + n2[1]
            Lm = math.hypot(mx, my)
            if Lm < 1e-9:
                mx, my = n1
                Lm = 1.0
            # miter length = d / cos(half-angle)
            cos_half = Lm / 2.0
            miter = d / max(cos_half, 1e-6)
            out.append((p[0] + mx / Lm * miter, p[1] + my / Lm * miter))
    return out


def _wall_along_polyline(points, d, r, spacing):
    offset = _offset_polyline(points, d)
    circles = []
    for a, b in zip(offset, offset[1:]):
        circles += _wall_circles(a[0], a[1], b[0], b[1], r, spacing)
    return circles

def narrow_corridor():
    """L-shaped corridor, ~0.8 wide, with a 90-degree turn. Smoothing loves
    to cut the inner corner straight into the wall."""
    obstacles = []
    obstacles += _wall_circles(0.0, -0.8, 6.8, -0.8, 0.4, 0.4)   # bottom wall
    obstacles += _wall_circles(0.0, 0.8, 5.2, 0.8, 0.4, 0.4)      # top wall (stops at turn)
    obstacles += _wall_circles(6.8, -0.8, 6.8, 6.0, 0.4, 0.4)     # right wall
    obstacles += _wall_circles(5.2, 0.8, 5.2, 6.0, 0.4, 0.4)      # left wall of vertical leg
    path = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0), (6.0, 2.0),
            (6.0, 4.0), (6.0, 6.0)]
    return "narrow_corridor", path, obstacles


def sharp_turns():
    """Four consecutive 90-degree turns; an obstacle sits at every inner
    corner, exactly where a corner-cutting smoother will go."""
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (8.0, 2.0),
            (8.0, 4.0), (12.0, 4.0)]
    obstacles = [
        (3.3, 0.7, 0.5),   # inner corner of turn at (4, 0)
        (4.7, 1.3, 0.5),   # inner corner of turn at (4, 2)
        (7.3, 2.7, 0.5),   # inner corner of turn at (8, 2)
        (8.7, 3.3, 0.5),   # inner corner of turn at (8, 4)
    ]
    return "sharp_turns", path, obstacles


def obstacle_hugging():
    """Path squeezes past a circle with only 0.1 clearance. Any smoothing
    of the apex dives straight through the obstacle."""
    path = [(0.0, 0.0), (5.0, 1.1), (10.0, 0.0)]
    obstacles = [(5.0, 0.0, 1.0)]
    return "obstacle_hugging", path, obstacles


def single_point():
    return "single_point", [(3.0, 3.0)], [(5.0, 5.0, 1.0)]


def straight_line():
    path = [(0.0, 0.0), (10.0, 0.0)]
    obstacles = [(5.0, 5.0, 1.0)]
    return "straight_line", path, obstacles


def unsmoothable():
    """S-shaped corridor so tight (clearance 0.02 < margin 0.05) that NO
    smoothing move is legal: shortcuts cut across the bends into the walls,
    and every gradient step drops clearance below the safety margin.
    The smoother MUST return the original path unchanged."""
    center = [(0.0, 0.0), (2.0, 1.0), (4.0, 0.0), (6.0, 1.0), (8.0, 0.0)]
    obstacles = []
    obstacles += _wall_along_polyline(center, +0.37, 0.35, 0.3)
    obstacles += _wall_along_polyline(center, -0.37, 0.35, 0.3)
    return "unsmoothable", list(center), obstacles


ALL = [narrow_corridor, sharp_turns, obstacle_hugging,
       single_point, straight_line, unsmoothable]
