"""Path smoothing.

- legacy_smooth_path: the original buggy implementation (classic gradient
  smoothing with NO collision checking). Kept only to reproduce the bug where
  the smoothed path looks nicer but cuts through obstacles.
- smooth_path: fixed implementation. Every accepted modification is guarded
  by an exact segment-vs-obstacle collision check (with a safety margin), and
  the final result is validated segment by segment; if anything is invalid
  the ORIGINAL path is returned, so the smoother can never turn a safe path
  into a colliding one.

Standard library only.
"""

from geometry import segment_collides, validate_path


def legacy_smooth_path(path, weight_data=0.5, weight_smooth=0.4, iterations=200):
    """Buggy reference: smooths blindly, never checks obstacle intersection."""
    if len(path) < 3:
        return [tuple(p) for p in path]
    original = [list(map(float, p)) for p in path]
    smoothed = [list(map(float, p)) for p in path]
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            for d in range(2):
                smoothed[i][d] += (
                    weight_data * (original[i][d] - smoothed[i][d])
                    + weight_smooth
                    * (smoothed[i - 1][d] + smoothed[i + 1][d] - 2.0 * smoothed[i][d])
                )
    return [tuple(p) for p in smoothed]


def _shortcut_pass(points, obstacles, margin):
    """One deterministic pass of shortcut smoothing. Only collision-free
    shortcuts (with margin) are accepted."""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(points) - 2:
            j = len(points) - 1
            while j > i + 1:
                if not segment_collides(points[i], points[j], obstacles, margin):
                    points[i + 1 : j] = []
                    changed = True
                    break
                j -= 1
            i += 1
    return points


def _guarded_gradient(points, obstacles, margin, weight_data, weight_smooth,
                      iterations):
    """Gradient smoothing where a candidate move is applied only if the two
    affected segments stay collision-free (with margin). Rejected moves are
    retried with halved step sizes."""
    anchor = [list(p) for p in points]
    pts = [list(p) for p in points]
    for _ in range(iterations):
        moved = False
        for i in range(1, len(pts) - 1):
            step = [
                weight_data * (anchor[i][d] - pts[i][d])
                + weight_smooth
                * (pts[i - 1][d] + pts[i + 1][d] - 2.0 * pts[i][d])
                for d in range(2)
            ]
            if step[0] == 0.0 and step[1] == 0.0:
                continue
            scale = 1.0
            while scale > 1e-3:
                cand = (pts[i][0] + step[0] * scale, pts[i][1] + step[1] * scale)
                if not segment_collides(pts[i - 1], cand, obstacles, margin) \
                        and not segment_collides(cand, pts[i + 1], obstacles, margin):
                    pts[i][0], pts[i][1] = cand
                    moved = True
                    break
                scale *= 0.5
        if not moved:
            break
    return [tuple(p) for p in pts]


def smooth_path(path, obstacles, margin=0.05, weight_data=0.1,
                weight_smooth=0.4, iterations=300):
    """Collision-safe smoothing.

    Guarantees:
    - len(path) < 3  -> returned unchanged (nothing to smooth).
    - Every segment of the returned path keeps clearance >= 0 from all
      obstacles; otherwise the original path is returned verbatim.
    - Smoothing moves keep a safety `margin` where possible; if the input
      path itself is tighter than `margin`, no risky move is accepted and
      the original path comes back unchanged.
    """
    original = [tuple(map(float, p)) for p in path]
    if len(original) < 3:
        return original

    pts = _shortcut_pass(list(original), obstacles, margin)
    pts = _guarded_gradient(pts, obstacles, margin, weight_data, weight_smooth,
                            iterations)

    # Final safety net: validate every segment; never return a colliding path.
    if validate_path(pts, obstacles, margin=0.0):
        return original
    return pts
