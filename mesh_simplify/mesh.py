"""Triangle mesh with adjacency, supporting edge collapse."""
from .vec import sub, cross, norm, normalize


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


class Mesh:
    """Mutable triangle mesh.

    vertices : list of (x, y, z); removed vertices keep their slot (None).
    faces    : dict face_id -> (a, b, c), counter-clockwise winding.
    """

    def __init__(self, vertices, faces):
        self.vertices = [tuple(map(float, v)) for v in vertices]
        self.faces = {}
        self._next_fid = 0
        self.v2f = [set() for _ in self.vertices]
        self.v2v = [set() for _ in self.vertices]
        self.edge_faces = {}
        self._normal_cache = {}
        for f in faces:
            self.add_face(*f)

    # ---- construction / mutation -------------------------------------
    def add_face(self, a, b, c):
        if len({a, b, c}) != 3:
            raise ValueError("degenerate face (repeated vertex)")
        fid = self._next_fid
        self._next_fid += 1
        self.faces[fid] = (a, b, c)
        self.v2f[a].add(fid)
        self.v2f[b].add(fid)
        self.v2f[c].add(fid)
        self.v2v[a].update((b, c))
        self.v2v[b].update((a, c))
        self.v2v[c].update((a, b))
        for u, v in ((a, b), (b, c), (c, a)):
            self.edge_faces.setdefault(edge_key(u, v), set()).add(fid)
        return fid

    def _remove_face(self, fid):
        a, b, c = self.faces.pop(fid)
        self._normal_cache.pop(fid, None)
        self.v2f[a].discard(fid)
        self.v2f[b].discard(fid)
        self.v2f[c].discard(fid)
        for u, v in ((a, b), (b, c), (c, a)):
            s = self.edge_faces[edge_key(u, v)]
            s.discard(fid)
            if not s:
                del self.edge_faces[edge_key(u, v)]
        for u, v in ((a, b), (a, c)):
            if not any(u in self.faces[g] and v in self.faces[g] for g in self.v2f[u]):
                self.v2v[u].discard(v)
                self.v2v[v].discard(u)

    def collapse(self, u, v, new_pos):
        """Collapse edge (u, v): remove v, move u to new_pos.

        Faces incident on both u and v are removed; other faces incident
        on v are rewired to u. Assumes validity was checked beforehand.
        """
        dying = [f for f in self.v2f[v] if u in self.faces[f]]
        for f in dying:
            self._remove_face(f)
        for f in list(self.v2f[v]):
            a, b, c = self.faces[f]
            for x, y in ((a, b), (b, c), (c, a)):
                s = self.edge_faces[edge_key(x, y)]
                s.discard(f)
                if not s:
                    del self.edge_faces[edge_key(x, y)]
            tri = tuple(u if x == v else x for x in (a, b, c))
            self.faces[f] = tri
            self._normal_cache.pop(f, None)
            self.v2f[u].add(f)
            for x, y in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                self.edge_faces.setdefault(edge_key(x, y), set()).add(f)
        self.v2f[v] = set()
        for w in list(self.v2v[v]):
            if w != u:
                self.v2v[w].discard(v)
                self.v2v[w].add(u)
                self.v2v[u].add(w)
        self.v2v[v] = set()
        self.v2v[u].discard(u)
        self.vertices[u] = tuple(new_pos)
        self.vertices[v] = None

    # ---- queries ------------------------------------------------------
    def face_normal(self, fid):
        n = self._normal_cache.get(fid)
        if n is None:
            a, b, c = (self.vertices[i] for i in self.faces[fid])
            n = normalize(cross(sub(b, a), sub(c, a)))
            self._normal_cache[fid] = n
        return n

    def face_area2(self, fid):
        a, b, c = (self.vertices[i] for i in self.faces[fid])
        return norm(cross(sub(b, a), sub(c, a)))

    def num_faces(self):
        return len(self.faces)

    def num_vertices(self):
        return sum(1 for v in self.vertices if v is not None)

    def edges(self):
        return list(self.edge_faces.keys())

    def triangles(self):
        """List of position triples for all live faces."""
        return [
            (self.vertices[a], self.vertices[b], self.vertices[c])
            for a, b, c in self.faces.values()
        ]
