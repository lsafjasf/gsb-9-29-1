"""Safe path smoothing with continuous collision checking and graceful degradation.

Fix for: smoothed paths occasionally cutting through obstacles because
collision checks were only done at sampled points (or not at all), and
multiple smoothing passes accumulated drift downstream.

Design:
- Obstacles are discrete records: (x, y, radius) discs.
- Every smoothed segment is validated with a *conservative continuous*
  check: the exact distance from each obstacle center to the whole
  segment (not to sample points) must keep `min_clearance`.
- If constraints cannot be satisfied simultaneously, the smoother
  degrades step by step and records which level was triggered:
    L0  full-strength smoothing, whole path accepted after validation
    L1  reduced smoothing strength (alpha and iterations scaled down)
    L2  segment-wise smoothing (only locally safe moves are applied)
    L3  constraints unsatisfiable -> the original path is returned
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple    # (x, y)
Obstacle = tuple  # (x, y, radius)

# Floating-point comparison tolerance; greedy acceptance additionally keeps
# a positive buffer so results never park exactly on the safety boundary.
_EPS = 1e-9
_GREEDY_BUFFER = 1e-8

LEVEL_NAMES = {
    0: "L0_full_strength",
    1: "L1_reduced_strength",
    2: "L2_segmentwise",
    3: "L3_original",
}

# (level, label, scale applied to alpha and iterations)
DEGRADATION_LADDER = (
    (0, "full_strength", 1.0),
    (1, "reduced_strength", 0.5),
    (1, "reduced_strength", 0.25),
    (1, "reduced_strength", 0.1),
)


# --------------------------------------------------------------------- geometry

def point_segment_distance(p, a, b):
    """Exact distance from point p to the *whole* closed segment a-b.

    This is the continuous check: it never samples, so an obstacle can
    never hide between two sample points.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def iter_segments(path, closed=False):
    """Yield (index, a, b) for every segment; includes the closing segment if closed."""
    n = len(path)
    for i in range(n - 1):
        yield i, path[i], path[i + 1]
    if closed and n > 2:
        yield n - 1, path[n - 1], path[0]


def _segment_clearance_uncounted(a, b, obstacles):
    best = math.inf
    for ox, oy, r in obstacles:
        d = point_segment_distance((ox, oy), a, b) - r
        if d < best:
            best = d
    return best


class CollisionChecker:
    """Conservative continuous collision checker with a query counter."""

    def __init__(self, obstacles, min_clearance=0.0):
        self.obstacles = [tuple(map(float, o)) for o in obstacles]
        self.min_clearance = float(min_clearance)
        self.checks = 0

    def segment_clearance(self, a, b):
        """Exact minimum clearance of the whole segment against all obstacles."""
        best = math.inf
        for ox, oy, r in self.obstacles:
            self.checks += 1
            d = point_segment_distance((ox, oy), a, b) - r
            if d < best:
                best = d
        return best

    def segment_safe(self, a, b):
        return self.segment_clearance(a, b) + _EPS >= self.min_clearance


# --------------------------------------------------------------------- safety data / assertions

def segment_clearances(path, obstacles, closed=False):
    """Per-segment minimum clearance data: list of (segment_index, clearance)."""
    return [
        (i, _segment_clearance_uncounted(a, b, obstacles))
        for i, a, b in iter_segments(path, closed)
    ]


def min_path_clearance(path, obstacles, closed=False):
    best = math.inf
    for _, a, b in iter_segments(path, closed):
        c = _segment_clearance_uncounted(a, b, obstacles)
        if c < best:
            best = c
    return best


def assert_path_safe(path, obstacles, min_clearance, closed=False):
    """Per-segment safety assertion (continuous, conservative)."""
    for i, a, b in iter_segments(path, closed):
        c = _segment_clearance_uncounted(a, b, obstacles)
        assert c + _EPS >= min_clearance, (
            f"segment {i} violates safety margin: clearance {c:.4f} "
            f"< required {min_clearance:.4f} (segment {a} -> {b})"
        )
    return True


# --------------------------------------------------------------------- metrics

def path_length(path, closed=False):
    return sum(math.dist(a, b) for _, a, b in iter_segments(path, closed))


def max_curvature(path, closed=False):
    """Discrete curvature: turning angle at each vertex over mean edge length."""
    n = len(path)
    if n < 3:
        return 0.0
    indices = range(n) if closed else range(1, n - 1)
    worst = 0.0
    for i in indices:
        p0 = path[(i - 1) % n]
        p1 = path[i]
        p2 = path[(i + 1) % n]
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        lu = math.hypot(ux, uy)
        lv = math.hypot(vx, vy)
        if lu < 1e-9 or lv < 1e-9:
            continue
        cosang = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
        turn = math.acos(cosang)
        curvature = turn / (0.5 * (lu + lv))
        if curvature > worst:
            worst = curvature
    return worst


def compute_metrics(path, obstacles, closed=False):
    return {
        "length": path_length(path, closed),
        "max_curvature": max_curvature(path, closed),
        "min_clearance": min_path_clearance(path, obstacles, closed),
    }


# --------------------------------------------------------------------- smoothing core

def laplacian_smooth(path, iterations, alpha, closed=False):
    """Plain Laplacian smoothing. Endpoints of open paths stay fixed.

    NOTE: this performs NO collision checking; callers must validate the
    result (that is exactly what smooth_path_safely does).
    """
    pts = [tuple(map(float, p)) for p in path]
    n = len(pts)
    if n < 3:
        return pts
    movable = range(n) if closed else range(1, n - 1)
    for _ in range(max(0, int(iterations))):
        src = pts
        nxt = list(src)
        for i in movable:
            pm = src[(i - 1) % n]
            pp = src[(i + 1) % n]
            mx = 0.5 * (pm[0] + pp[0])
            my = 0.5 * (pm[1] + pp[1])
            nxt[i] = (
                src[i][0] + alpha * (mx - src[i][0]),
                src[i][1] + alpha * (my - src[i][1]),
            )
        pts = nxt
    return pts


def _validate(path, checker, closed):
    """Continuously validate every segment; return (ok, worst_clearance)."""
    worst = math.inf
    ok = True
    for _, a, b in iter_segments(path, closed):
        c = checker.segment_clearance(a, b)
        if c < worst:
            worst = c
        if c < checker.min_clearance:
            ok = False
    return ok, worst


def _greedy_segmentwise(path, checker, rounds, alpha, closed):
    """Apply per-point smoothing moves only when both adjacent segments stay safe."""
    pts = list(path)
    n = len(pts)
    movable = list(range(n)) if closed else list(range(1, n - 1))
    moved = 0
    for _ in range(max(0, int(rounds))):
        for i in movable:
            pm = pts[(i - 1) % n]
            pp = pts[(i + 1) % n]
            mx = 0.5 * (pm[0] + pp[0])
            my = 0.5 * (pm[1] + pp[1])
            cand = (
                pts[i][0] + alpha * (mx - pts[i][0]),
                pts[i][1] + alpha * (my - pts[i][1]),
            )
            if cand == pts[i]:
                continue
            # conservative continuous check of the two affected segments,
            # with a positive buffer against floating-point boundary noise
            margin = checker.min_clearance + _GREEDY_BUFFER
            if (checker.segment_clearance(pm, cand) >= margin
                    and checker.segment_clearance(cand, pp) >= margin):
                pts[i] = cand
                moved += 1
    return pts, moved


@dataclass
class SmoothResult:
    path: list
    closed: bool
    degradation_level: int
    degradation_reason: str
    attempts: list = field(default_factory=list)
    metrics_before: dict = field(default_factory=dict)
    metrics_after: dict = field(default_factory=dict)
    collision_checks: int = 0

    @property
    def degradation_level_name(self):
        return LEVEL_NAMES[self.degradation_level]


def smooth_path_safely(path, obstacles, *, iterations=60, alpha=0.6,
                       min_clearance=0.1, closed=False, checker=None):
    """Smooth `path` while guaranteeing `min_clearance` to every obstacle.

    Degradation contract (each triggered level is recorded in
    `result.attempts` and `result.degradation_level`):
      L0 accept full-strength smoothed path if every segment validates
      L1 retry with reduced smoothing strength
      L2 fall back to segment-wise (per-point, locally validated) smoothing
      L3 return the original path unchanged
    """
    pts = [tuple(map(float, p)) for p in path]
    obstacles = [tuple(map(float, o)) for o in obstacles]
    if checker is None:
        checker = CollisionChecker(obstacles, min_clearance)
    checks_before = checker.checks
    metrics_before = compute_metrics(pts, obstacles, closed)
    attempts = []

    def finish(result_pts, level, reason, verify=True):
        if verify:
            # per-segment safety assertion on the accepted result
            assert_path_safe(result_pts, obstacles, min_clearance, closed)
        return SmoothResult(
            path=[tuple(p) for p in result_pts],
            closed=closed,
            degradation_level=level,
            degradation_reason=reason,
            attempts=list(attempts),
            metrics_before=metrics_before,
            metrics_after=compute_metrics(result_pts, obstacles, closed),
            collision_checks=checker.checks - checks_before,
        )

    n = len(pts)
    if n < 3:
        return finish(pts, 0, "trivial_path_no_smoothing_needed", verify=False)

    # L0 / L1: whole-path candidates with decreasing smoothing strength.
    for level, label, scale in DEGRADATION_LADDER:
        iters = max(1, int(round(iterations * scale)))
        a = alpha * scale
        candidate = laplacian_smooth(pts, iters, a, closed)
        ok, worst = _validate(candidate, checker, closed)
        attempts.append({
            "level": level,
            "strategy": label,
            "alpha": a,
            "iterations": iters,
            "accepted": ok,
            "min_clearance": worst,
        })
        if ok:
            return finish(candidate, level, f"accepted_{label}_candidate")

    # L2: segment-wise smoothing, each move locally validated.
    rounds = max(1, iterations // 4)
    greedy, moved = _greedy_segmentwise(pts, checker, rounds, alpha * 0.25, closed)
    ok, worst = _validate(greedy, checker, closed)
    attempts.append({
        "level": 2,
        "strategy": "segmentwise",
        "alpha": alpha * 0.25,
        "iterations": rounds,
        "accepted": ok and moved > 0,
        "min_clearance": worst,
        "moves_applied": moved,
    })
    if ok and moved > 0:
        return finish(greedy, 2, "accepted_segmentwise_partial_smoothing")

    # L3: constraints unsatisfiable -> return the original path.
    attempts.append({
        "level": 3,
        "strategy": "original",
        "accepted": True,
        "min_clearance": metrics_before["min_clearance"],
    })
    return finish(pts, 3, "constraints_unsatisfiable_returned_original_path",
                  verify=False)
