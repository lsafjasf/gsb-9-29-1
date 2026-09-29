"""Sparse voxelization library (Python 3, standard library only).

Stores occupied voxels of a (possibly huge) 3D scene in a sparse dict
keyed by integer voxel indices.  Memory is O(#occupied voxels) and is
independent of the scene extent.

Occupancy rule (closed-box convention): a voxel is occupied iff the
triangle intersects the *closed* voxel box.  A triangle lying exactly on
a voxel boundary therefore marks the voxels on both sides of the plane.
"""

import math
import sys


# ---------------------------------------------------------------------------
# Triangle / axis-aligned-box overlap (Akenine-Moeller SAT test, 13 axes)
# ---------------------------------------------------------------------------

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _separates(axis, v0, v1, v2, w):
    """True if `axis` separates the triangle (vertices relative to the box
    min corner) from the box [0, w]^3."""
    lo = min(_dot(axis, v0), _dot(axis, v1), _dot(axis, v2))
    hi = max(_dot(axis, v0), _dot(axis, v1), _dot(axis, v2))
    # Projection interval of the box onto `axis`.
    blo = 0.0
    bhi = 0.0
    for i in range(3):
        p = axis[i] * w[i]
        if p < 0.0:
            blo += p
        else:
            bhi += p
    # Inclusive comparison: touching counts as intersecting.
    return hi < blo or lo > bhi


def tri_box_overlap(box_min, box_max, tri):
    """Exact test: does `tri` intersect the closed axis-aligned box
    [box_min, box_max]?

    Separating Axis Theorem with 13 axes:
      3 box face normals, 1 triangle normal, 9 edge cross products.
    Works for degenerate (zero-area) triangles as well.

    The box-axis tests are evaluated as `v - box_min` / `v - box_max` in
    world coordinates so that boundary touches agree bit-exactly with
    plane-comparison-based reference implementations.
    """
    w = (box_max[0] - box_min[0], box_max[1] - box_min[1],
         box_max[2] - box_min[2])
    v0 = (tri[0][0] - box_min[0], tri[0][1] - box_min[1],
          tri[0][2] - box_min[2])
    v1 = (tri[1][0] - box_min[0], tri[1][1] - box_min[1],
          tri[1][2] - box_min[2])
    v2 = (tri[2][0] - box_min[0], tri[2][1] - box_min[1],
          tri[2][2] - box_min[2])

    # 3 axes: box face normals (world-coordinate comparisons).
    for a in range(3):
        if max(v0[a], v1[a], v2[a]) < 0.0:
            return False
        if min(tri[0][a] - box_max[a], tri[1][a] - box_max[a],
               tri[2][a] - box_max[a]) > 0.0:
            return False

    e0 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    e2 = (v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2])

    # 9 axes: cross products of box axes with triangle edges.
    for e in (e0, e1, e2):
        if _separates((0.0, -e[2], e[1]), v0, v1, v2, w):
            return False
        if _separates((e[2], 0.0, -e[0]), v0, v1, v2, w):
            return False
        if _separates((-e[1], e[0], 0.0), v0, v1, v2, w):
            return False

    # 1 axis: triangle normal (plane-vs-box test).
    n = _cross(e0, e1)
    nv0 = _dot(n, v0)
    clo = 0.0
    chi = 0.0
    for i in range(3):
        p = n[i] * w[i]
        if p < 0.0:
            clo += p
        else:
            chi += p
    if clo > nv0 or chi < nv0:
        return False
    return True


# ---------------------------------------------------------------------------
# Sparse voxel grid
# ---------------------------------------------------------------------------

class SparseVoxelGrid:
    """Sparse set of occupied voxels on a uniform grid.

    `voxel_size` is the edge length s of a cubic voxel, `origin` is the
    world coordinate of voxel index (0, 0, 0)'s min corner.
    """

    def __init__(self, voxel_size, origin=(0.0, 0.0, 0.0)):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.voxel_size = float(voxel_size)
        self.origin = (float(origin[0]), float(origin[1]), float(origin[2]))
        self._voxels = {}  # (i, j, k) -> True

    # -- coordinate mapping -------------------------------------------------

    def index_of(self, point):
        """Voxel index containing a world point (floor convention)."""
        s = self.voxel_size
        return (math.floor((point[0] - self.origin[0]) / s),
                math.floor((point[1] - self.origin[1]) / s),
                math.floor((point[2] - self.origin[2]) / s))

    def box_of(self, idx):
        """World-space (min_corner, max_corner) of voxel `idx`."""
        s = self.voxel_size
        lo = tuple(self.origin[a] + idx[a] * s for a in range(3))
        hi = tuple(c + s for c in lo)
        return lo, hi

    # -- voxelization -------------------------------------------------------

    def add_triangle(self, v0, v1, v2):
        """Mark every voxel the triangle intersects (including voxels the
        triangle merely passes through without containing any vertex)."""
        tri = (v0, v1, v2)
        s = self.voxel_size
        lo = [0, 0, 0]
        hi = [0, 0, 0]
        for a in range(3):
            mn = min(v[a] for v in tri) - self.origin[a]
            mx = max(v[a] for v in tri) - self.origin[a]
            l = math.floor(mn / s)
            h = math.floor(mx / s)
            # Correct float division rounding with exact comparisons so
            # that l*s <= mn and h*s <= mx < (h+1)*s hold.
            while l * s > mn:
                l -= 1
            while (l + 1) * s <= mn:
                l += 1
            while h * s > mx:
                h -= 1
            while (h + 1) * s <= mx:
                h += 1
            # Closed-box convention: a triangle whose min coordinate lies
            # exactly on a voxel plane also touches the voxel on the low
            # side.
            if l * s == mn:
                l -= 1
            lo[a] = l
            hi[a] = h
        added = 0
        for i in range(lo[0], hi[0] + 1):
            bx0 = self.origin[0] + i * s
            for j in range(lo[1], hi[1] + 1):
                by0 = self.origin[1] + j * s
                for k in range(lo[2], hi[2] + 1):
                    idx = (i, j, k)
                    if idx in self._voxels:
                        continue
                    bz0 = self.origin[2] + k * s
                    bmin = (bx0, by0, bz0)
                    bmax = (bx0 + s, by0 + s, bz0 + s)
                    if tri_box_overlap(bmin, bmax, tri):
                        self._voxels[idx] = True
                        added += 1
        return added

    def voxelize(self, triangles):
        """Voxelize an iterable of (v0, v1, v2) triangles."""
        for tri in triangles:
            self.add_triangle(tri[0], tri[1], tri[2])
        return self

    # -- queries ------------------------------------------------------------

    def __contains__(self, idx):
        return idx in self._voxels

    def __len__(self):
        return len(self._voxels)

    def count(self):
        return len(self._voxels)

    def iter_voxels(self):
        return iter(self._voxels.keys())

    def query_range(self, lo_idx, hi_idx):
        """All occupied voxel indices with lo_idx <= idx <= hi_idx
        (component-wise, inclusive).  O(#occupied)."""
        out = []
        for idx in self._voxels:
            if (lo_idx[0] <= idx[0] <= hi_idx[0]
                    and lo_idx[1] <= idx[1] <= hi_idx[1]
                    and lo_idx[2] <= idx[2] <= hi_idx[2]):
                out.append(idx)
        return out

    def count_range(self, lo_idx, hi_idx):
        """Number of occupied voxels inside the inclusive index range."""
        n = 0
        for idx in self._voxels:
            if (lo_idx[0] <= idx[0] <= hi_idx[0]
                    and lo_idx[1] <= idx[1] <= hi_idx[1]
                    and lo_idx[2] <= idx[2] <= hi_idx[2]):
                n += 1
        return n

    def query_range_world(self, lo, hi):
        """World-space range query: occupied voxels whose index lies between
        the indices of the two world corners (inclusive)."""
        return self.query_range(self.index_of(lo), self.index_of(hi))

    def bounds(self):
        """((min_i, min_j, min_k), (max_i, max_j, max_k)) or None if empty."""
        if not self._voxels:
            return None
        keys = self._voxels.keys()
        return (tuple(min(k[a] for k in keys) for a in range(3)),
                tuple(max(k[a] for k in keys) for a in range(3)))

    # -- memory -------------------------------------------------------------

    def memory_bytes(self):
        """Estimated bytes held by the sparse structure (shallow object
        sizes; shared small ints may make real usage slightly lower)."""
        total = sys.getsizeof(self._voxels)
        for key, val in self._voxels.items():
            total += sys.getsizeof(key)
            total += sum(sys.getsizeof(c) for c in key)
            total += sys.getsizeof(val)
        return total

    def stats(self):
        return {
            "voxel_size": self.voxel_size,
            "occupied": len(self._voxels),
            "bounds": self.bounds(),
            "memory_bytes": self.memory_bytes(),
        }
