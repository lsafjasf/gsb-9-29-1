"""Edge-collapse mesh simplification.

Algorithm (Garland & Heckbert 1997 style, implemented from scratch):

* Each incident face contributes a plane quadric  K_p = p p^T  where
  p = (nx, ny, nz, -d) is the unit plane equation of the face.
* Vertex quadric  Q_v = sum K_p  over incident faces.
* Contracting edge (u, v) has error  E(x) = x^T (Q_u + Q_v) x, i.e. the
  sum of squared distances from the new position x to all supporting
  planes of the triangles meeting at u or v.
* The optimal position minimizes E: solve the 3x3 system A x = -d. If
  that is singular (near-flat regions), the cheapest of u, v and the
  midpoint is used.
* Edges are processed by a lazy min-heap of costs; entries are
  re-validated on pop because quadrics change after each contraction.

Feature-edge protection (``protect_features=True``):
* boundary edges (1 incident face) and sharp edges (dihedral angle over
  ``sharp_deg``) are classified as feature edges and never contracted;
* vertices on a feature edge are feature vertices, and an edge with one
  feature endpoint may only collapse *onto that endpoint*, so the
  corner/boundary point itself never moves;
* edges joining two feature vertices are not contracted.

Every candidate contraction is additionally rejected unless:
* the edge is manifold (1 or 2 incident faces);
* the link condition holds (no duplicated/merged topology);
* no resulting triangle is degenerate or flipped (same orientation,
  nonzero area, for every affected face).
"""
import heapq
import math

from .mesh import edge_key
from .vec import sub, cross, dot, normalize
from .validate import MIN_AREA2


# ---- quadrics ----------------------------------------------------------
# Symmetric 4x4 stored as (a..j):
#  a b c d
#  b e f g
#  c f h i
#  d g i j
_ZERO_Q = (0.0,) * 10


def _face_quadric(plane_n, pos):
    nx, ny, nz = plane_n
    d = -(nx * pos[0] + ny * pos[1] + nz * pos[2])
    p = (nx, ny, nz, d)
    return tuple(p[i] * p[j] for i in range(4) for j in range(i, 4))


def _qadd(q1, q2):
    return tuple(a + b for a, b in zip(q1, q2))


def _qeval(q, x):
    xx, yy, zz = x
    a, b, c, d, e, f, g, h, i, j = q
    return (
        a * xx * xx + 2 * b * xx * yy + 2 * c * xx * zz + 2 * d * xx
        + e * yy * yy + 2 * f * yy * zz + 2 * g * yy
        + h * zz * zz + 2 * i * zz + j
    )


def _optimal_point(q):
    a, b, c, d, e, f, g, h, i, _ = q
    A = [[a, b, c], [b, e, f], [c, f, h]]
    rhs = [-d, -g, -i]
    det = (
        a * (e * h - f * f)
        - b * (b * h - f * c)
        + c * (b * f - e * c)
    )
    scale = max(1.0, max(abs(v) for row in A for v in row))
    if abs(det) > 1e-15 * scale ** 3:
        try:
            x = _solve3(A, rhs)
        except ZeroDivisionError:
            return None
        if all(math.isfinite(v) for v in x):
            return x
    return None


def _solve3(A, b):
    m = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-30:
            raise ZeroDivisionError
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(col + 1, 3):
            factor = m[r][col] / piv
            for k in range(col, 4):
                m[r][k] -= factor * m[col][k]
    x = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x[r] = (m[r][3] - sum(m[r][k] * x[k] for k in range(r + 1, 3))) / m[r][r]
    return x


# ---- feature classification ------------------------------------------
def classify_features(mesh, sharp_deg=45.0):
    """Return (feature_edges, feature_verts) sets."""
    cos_thr = math.cos(math.radians(sharp_deg))
    feature_edges = set()
    feature_verts = set()
    for key, fids in mesh.edge_faces.items():
        is_feature = False
        if len(fids) == 1:
            is_feature = True
        elif len(fids) == 2:
            f1, f2 = tuple(fids)
            n1, n2 = mesh.face_normal(f1), mesh.face_normal(f2)
            # unsigned cosine: both convex and concave creases count
            if abs(dot(n1, n2)) < cos_thr:
                is_feature = True
        if is_feature:
            feature_edges.add(key)
            feature_verts.update(key)
    return feature_edges, feature_verts


# ---- candidate validation --------------------------------------------
def _link_ok(mesh, u, v):
    common = mesh.v2v[u] & mesh.v2v[v]
    shared = len(mesh.edge_faces.get(edge_key(u, v), ()))
    return len(common) == shared


def _geometry_ok(mesh, u, v, new_pos):
    """Simulate the collapse and check every affected triangle."""
    affected = mesh.v2f[u] | mesh.v2f[v]
    for fid in affected:
        a, b, c = mesh.faces[fid]
        if u in (a, b, c) and v in (a, b, c):
            continue  # this face is removed by the collapse
        old_n = mesh.face_normal(fid)
        tri = tuple(new_pos if x in (u, v) else mesh.vertices[x] for x in (a, b, c))
        pa, pb, pc = tri
        cr = cross(sub(pb, pa), sub(pc, pa))
        if dot(cr, cr) < MIN_AREA2 ** 2:
            return False  # degenerate
        if dot(normalize(cr), old_n) <= 1e-6:
            return False  # flipped
    return True


def simplify(
    mesh,
    target_faces=None,
    target_fraction=None,
    protect_features=True,
    sharp_deg=45.0,
):
    """Simplify ``mesh`` in place by edge collapse.

    Returns a stats dict: target, target_reached, faces_before/after,
    collapses, rejected {reason: count}, first_cost, last_cost.
    """
    before = mesh.num_faces()
    if target_faces is None:
        if target_fraction is None:
            raise ValueError("give target_faces or target_fraction")
        target_faces = max(1, int(round(before * target_fraction)))
    target_faces = max(1, int(target_faces))

    feature_edges, feature_verts = classify_features(mesh, sharp_deg)

    # per-vertex quadrics
    quadrics = [_ZERO_Q] * len(mesh.vertices)
    for fid, (a, b, c) in mesh.faces.items():
        pa, pb, pc = (mesh.vertices[i] for i in (a, b, c))
        n = normalize(cross(sub(pb, pa), sub(pc, pa)))
        q = _face_quadric(n, pa)
        for vi in (a, b, c):
            quadrics[vi] = _qadd(quadrics[vi], q)

    heap = []
    seq = 0

    def push_edge(u, v):
        nonlocal seq
        if mesh.vertices[u] is None or mesh.vertices[v] is None:
            return
        fids = mesh.edge_faces.get((u, v))
        if not fids or len(fids) > 2:
            return
        if protect_features:
            if (u, v) in feature_edges:
                return
            uf = u in feature_verts
            vf = v in feature_verts
            if uf and vf:
                return
        q = _qadd(quadrics[u], quadrics[v])
        pos = None
        if not protect_features:
            pos = _optimal_point(q)
        elif u in feature_verts:
            pos = mesh.vertices[u]
        elif v in feature_verts:
            pos = mesh.vertices[v]
        if pos is None:
            candidates = (mesh.vertices[u], mesh.vertices[v])
            mid = tuple((a + b) / 2 for a, b in zip(mesh.vertices[u], mesh.vertices[v]))
            pos = min(candidates + (mid,), key=lambda p: _qeval(q, p))
        cost = max(0.0, _qeval(q, pos))
        heapq.heappush(heap, (cost, seq, u, v, pos))
        seq += 1

    for u, v in list(mesh.edge_faces.keys()):
        push_edge(u, v)

    rejected = {}
    collapses = 0
    first_cost = None
    last_cost = None

    while heap and mesh.num_faces() > target_faces:
        cost, _, u, v, pos = heapq.heappop(heap)
        key = edge_key(u, v)
        if key not in mesh.edge_faces or mesh.vertices[u] is None or mesh.vertices[v] is None:
            rejected["stale"] = rejected.get("stale", 0) + 1
            continue
        if len(mesh.edge_faces[key]) > 2:
            rejected["nonmanifold"] = rejected.get("nonmanifold", 0) + 1
            continue
        if protect_features:
            if key in feature_edges:
                rejected["feature_edge"] = rejected.get("feature_edge", 0) + 1
                continue
            uf = u in feature_verts
            vf = v in feature_verts
            if uf and vf:
                rejected["feature_both_ends"] = rejected.get("feature_both_ends", 0) + 1
                continue
            if (uf or vf) and pos != (mesh.vertices[u] if uf else mesh.vertices[v]):
                # position snapshot went stale; rebuild and retry once
                pos = mesh.vertices[u] if uf else mesh.vertices[v]
        if not _link_ok(mesh, u, v):
            rejected["link"] = rejected.get("link", 0) + 1
            continue
        if not _geometry_ok(mesh, u, v, pos):
            rejected["flip_or_degenerate"] = rejected.get("flip_or_degenerate", 0) + 1
            continue

        # commit: merge v into u
        affected_neighbors = list(mesh.v2v[u] | mesh.v2v[v])
        mesh.collapse(u, v, pos)
        quadrics[u] = _qadd(quadrics[u], quadrics[v])
        quadrics[v] = _ZERO_Q
        # feature status of u: feature if it was one; v disappears
        if first_cost is None:
            first_cost = cost
        last_cost = cost
        collapses += 1
        for w in affected_neighbors:
            if w == v or w == u or mesh.vertices[w] is None:
                continue
            if edge_key(u, w) in mesh.edge_faces:
                push_edge(u, w)

    after = mesh.num_faces()
    return {
        "target_faces": target_faces,
        "target_reached": after <= target_faces,
        "faces_before": before,
        "faces_after": after,
        "collapses": collapses,
        "rejected": rejected,
        "first_cost": first_cost,
        "last_cost": last_cost,
        "protect_features": protect_features,
        "sharp_deg": sharp_deg,
    }
