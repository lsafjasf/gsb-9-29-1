"""Independent analytic reference implementation (stdlib only).

Used to cross-validate voxelizer.tri_box_overlap.  Instead of the
Separating Axis Theorem, this uses exact polygon clipping: the triangle
is clipped (Sutherland-Hodgman) against the 6 planes of the closed box;
the triangle intersects the box iff the clipped polygon is non-empty.

Both implementations are exact analytic geometry, so on small scenes the
occupied voxel sets must agree *exactly*.
"""


def _clip_axis(poly, axis, c, keep_le):
    """Clip polygon against plane coord[axis] <= c (keep_le=True) or
    coord[axis] >= c (keep_le=False).  Closed half-space: points exactly
    on the plane are kept."""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        da = a[axis] - c
        db = b[axis] - c
        ina = da <= 0.0 if keep_le else da >= 0.0
        inb = db <= 0.0 if keep_le else db >= 0.0
        if ina:
            out.append(a)
        if ina != inb:
            t = da / (da - db)
            out.append(tuple(a[j] + t * (b[j] - a[j]) for j in range(3)))
    return out


def ref_tri_intersects_box(box_min, box_max, tri):
    """True iff `tri` intersects the closed axis-aligned box."""
    poly = [tuple(tri[0]), tuple(tri[1]), tuple(tri[2])]
    for axis in range(3):
        poly = _clip_axis(poly, axis, box_min[axis], keep_le=False)
        if not poly:
            return False
        poly = _clip_axis(poly, axis, box_max[axis], keep_le=True)
        if not poly:
            return False
    return True


def ref_occupied_set(triangles, voxel_size, origin, lo_idx, hi_idx):
    """Brute-force reference: for every voxel index in the inclusive range
    [lo_idx, hi_idx], test every triangle with the clipping method and
    return the set of occupied indices."""
    occupied = set()
    for i in range(lo_idx[0], hi_idx[0] + 1):
        for j in range(lo_idx[1], hi_idx[1] + 1):
            for k in range(lo_idx[2], hi_idx[2] + 1):
                bmin = tuple(origin[a] + c * voxel_size
                             for a, c in enumerate((i, j, k)))
                bmax = tuple(c + voxel_size for c in bmin)
                for tri in triangles:
                    if ref_tri_intersects_box(bmin, bmax, tri):
                        occupied.add((i, j, k))
                        break
    return occupied
