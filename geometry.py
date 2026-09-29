"""Geometry helpers: segment/circle distance, path metrics, per-segment validation.

Obstacles are circles: (cx, cy, radius). Paths are lists of (x, y) points.
Standard library only.
"""

import math


def dist_point_segment(p, a, b):
    """Exact distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def segment_clearance(a, b, obstacle):
    """Clearance between segment ab and one circular obstacle (negative = collision)."""
    ox, oy, r = obstacle
    return dist_point_segment((ox, oy), a, b) - r


def segment_collides(a, b, obstacles, margin=0.0):
    """True if segment ab comes within (radius + margin) of any obstacle."""
    return any(segment_clearance(a, b, ob) < margin for ob in obstacles)


def path_length(path):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
    )


def path_max_curvature(path):
    """Max Menger curvature (1/radius of circumcircle) over interior points.

    Collinear triples give 0. A path with fewer than 3 points has curvature 0.
    """
    max_k = 0.0
    for a, b, c in zip(path, path[1:], path[2:]):
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        if ab == 0.0 or bc == 0.0 or ca == 0.0:
            continue
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area2 = abs(cross)  # twice the triangle area
        max_k = max(max_k, 2.0 * area2 / (ab * bc * ca))
    return max_k


def path_min_clearance(path, obstacles):
    """Exact minimum clearance over all segments vs all obstacles."""
    best = math.inf
    for a, b in zip(path, path[1:]):
        for ob in obstacles:
            best = min(best, segment_clearance(a, b, ob))
    return best


def validate_path(path, obstacles, margin=0.0):
    """Per-segment validation. Returns a list of violations:
    (segment_index, obstacle, clearance). Empty list means the path is safe.
    """
    violations = []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        for ob in obstacles:
            clearance = segment_clearance(a, b, ob)
            if clearance < margin:
                violations.append((i, ob, clearance))
    return violations
