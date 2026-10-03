"""Deviation measurement: sampled symmetric Hausdorff distance."""
import math
from .vec import sub, dot


def _point_tri_dist2(p, a, b, c):
    """Squared distance from point p to triangle (a, b, c).

    Ericson, Real-Time Collision Detection, section 5.1.5.
    """
    ab = sub(b, a)
    ac = sub(c, a)
    ap = sub(p, a)
    d1 = dot(ab, ap)
    d2 = dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return dot(ap, ap)
    bp = sub(p, b)
    d3 = dot(ab, bp)
    d4 = dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        t = d1 / (d1 - d3)
        q = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])
        return dot(sub(p, q), sub(p, q))
    cp = sub(p, c)
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        t = d2 / (d2 - d6)
        q = (a[0] + t * ac[0], a[1] + t * ac[1], a[2] + t * ac[2])
        return dot(sub(p, q), sub(p, q))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bc = sub(c, b)
        q = (b[0] + t * bc[0], b[1] + t * bc[1], b[2] + t * bc[2])
        return dot(sub(p, q), sub(p, q))
    denom = va + vb + vc
    v = vb / denom
    w = vc / denom
    q = (
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    )
    return dot(sub(p, q), sub(p, q))


def _sample_triangle(a, b, c, k):
    """Grid of (k+1)(k+2)/2 barycentric samples on the triangle."""
    pts = []
    for i in range(k + 1):
        for j in range(k + 1 - i):
            l0 = 1.0 - (i + j) / k
            l1 = i / k
            l2 = j / k
            pts.append(
                (
                    l0 * a[0] + l1 * b[0] + l2 * c[0],
                    l0 * a[1] + l1 * b[1] + l2 * c[1],
                    l0 * a[2] + l1 * b[2] + l2 * c[2],
                )
            )
    return pts


def _one_sided(src_tris, dst_tris, k):
    max_d2 = 0.0
    total = 0.0
    count = 0
    for a, b, c in src_tris:
        for p in _sample_triangle(a, b, c, k):
            best = min(_point_tri_dist2(p, d, e, f) for d, e, f in dst_tris)
            if best > max_d2:
                max_d2 = best
            total += best
            count += 1
    return max_d2, total, count


def hausdorff_distance(tris_a, tris_b, k=4):
    """Symmetric sampled Hausdorff distance between two triangle soups.

    Each triangle is sampled on a barycentric grid of parameter k
    ((k+1)(k+2)/2 points per triangle). Returns a dict with the max
    deviation and the RMS deviation.
    """
    if not tris_a or not tris_b:
        raise ValueError("need non-empty triangle sets on both sides")
    max_ab, sum_ab, n_ab = _one_sided(tris_a, tris_b, k)
    max_ba, sum_ba, n_ba = _one_sided(tris_b, tris_a, k)
    n = n_ab + n_ba
    return {
        "max": math.sqrt(max(max_ab, max_ba)),
        "rms": math.sqrt((sum_ab + sum_ba) / n),
        "samples": n,
    }
