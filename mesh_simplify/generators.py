"""Small test meshes used by the self-tests / demo."""
import math
from .mesh import Mesh


def single_triangle():
    return Mesh(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
    )


def plane_grid(n=8, size=1.0):
    """Open planar triangulated grid (each cell split both diagonals ->
    4 triangles per cell). All boundary edges are feature edges."""
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            verts.append((size * i / n, size * j / n, 0.0))
    idx = lambda i, j: j * (n + 1) + i
    faces = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            m = len(verts)
            verts.append((size * (i + 0.5) / n, size * (j + 0.5) / n, 0.0))
            faces += [(a, b, m), (b, c, m), (c, d, m), (d, a, m)]
    return Mesh(verts, faces)


def cube(size=1.0):
    """Closed triangulated cube, 8 verts / 12 triangles, outward winding."""
    s = size / 2.0
    v = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),   # -z
        (4, 5, 6), (4, 6, 7),   # +z
        (0, 4, 7), (0, 7, 3),   # -x
        (1, 2, 6), (1, 6, 5),   # +x
        (0, 1, 5), (0, 5, 4),   # -y
        (3, 7, 6), (3, 6, 2),   # +y
    ]
    return Mesh(v, faces)


def thin_shell(length=2.0, width=1.0, thickness=0.02):
    """Closed but extremely thin box shell. Small thickness means any
    collapse across the shell would fold/flip triangles; the geometry
    checks must reject those and avoid the two sides merging."""
    l, w, t = length / 2.0, width / 2.0, thickness / 2.0
    v = [
        (-l, -w, -t), (l, -w, -t), (l, w, -t), (-l, w, -t),
        (-l, -w, t), (l, -w, t), (l, w, t), (-l, w, t),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1),
        (4, 5, 6), (4, 6, 7),
        (0, 4, 7), (0, 7, 3),
        (1, 2, 6), (1, 6, 5),
        (0, 1, 5), (0, 5, 4),
        (3, 7, 6), (3, 6, 2),
    ]
    return Mesh(v, faces)


def icosphere(subdiv=2, radius=1.0):
    """Closed sphere from a subdivided icosahedron, outward winding."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [tuple(scale / math.sqrt(1 + t * t) for scale in p) for p in verts]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    mids = {}

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in mids:
            pa, pb = verts[a], verts[b]
            m = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2)
            n = math.sqrt(sum(c * c for c in m))
            mids[key] = len(verts)
            verts.append((m[0] / n, m[1] / n, m[2] / n))
        return mids[key]

    for _ in range(subdiv):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    verts = [tuple(radius * c for c in p) for p in verts]
    return Mesh(verts, faces)
