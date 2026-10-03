"""Topological and geometric validity checks for a Mesh."""
from .vec import sub, cross, norm

MIN_AREA2 = 1e-12  # min |cross| (2x area) for a face to be non-degenerate


def validate_topology(mesh):
    """Return a list of human-readable issues; empty list means valid.

    Checks:
      * every face references 3 distinct, live vertices
      * no degenerate (zero-area) faces
      * every edge is incident to 1 (boundary) or 2 faces (manifoldness)
      * edges shared by 2 faces are traversed in opposite directions
        (consistent winding, i.e. no flipped triangles)
      * every vertex's incident faces form a single fan (manifold vertex)
    """
    issues = []
    directed = {}
    for fid, (a, b, c) in mesh.faces.items():
        if len({a, b, c}) != 3:
            issues.append(f"face {fid} repeats a vertex: {(a, b, c)}")
            continue
        for x in (a, b, c):
            if mesh.vertices[x] is None:
                issues.append(f"face {fid} references removed vertex {x}")
        pa, pb, pc = (mesh.vertices[x] for x in (a, b, c))
        if None not in (pa, pb, pc):
            if norm(cross(sub(pb, pa), sub(pc, pa))) < MIN_AREA2:
                issues.append(f"face {fid} is degenerate (zero area)")
        for u, v in ((a, b), (b, c), (c, a)):
            directed.setdefault((u, v), []).append(fid)

    for (u, v), fids in directed.items():
        # A correctly oriented interior edge is used once each way; the
        # reverse traversal (v,u) therefore gets its own entry. Two faces
        # using the SAME direction means a flipped/inconsistent triangle.
        if len(fids) > 1:
            issues.append(
                f"edge ({u},{v}) traversed same direction by faces "
                f"{sorted(fids)}: inconsistent winding / flipped triangle"
            )

    for key, fids in mesh.edge_faces.items():
        if len(fids) > 2:
            issues.append(f"edge {key} is non-manifold: {len(fids)} faces")

    for vid, pos in enumerate(mesh.vertices):
        if pos is None:
            continue
        if not mesh.v2f[vid]:
            issues.append(f"vertex {vid} is isolated")
            continue
        # single-fan check: walk incident faces around the vertex
        start = next(iter(mesh.v2f[vid]))
        visited = {start}
        frontier = [start]
        while frontier:
            f = frontier.pop()
            tri = mesh.faces[f]
            others = [x for x in tri if x != vid]
            for g in mesh.v2f[vid]:
                if g in visited:
                    continue
                gtri = mesh.faces[g]
                if others[0] in gtri or others[1] in gtri:
                    visited.add(g)
                    frontier.append(g)
        if visited != mesh.v2f[vid]:
            issues.append(f"vertex {vid} is non-manifold (multiple fans)")

    return issues


def assert_valid(mesh, label="mesh"):
    issues = validate_topology(mesh)
    if issues:
        raise AssertionError(
            f"{label}: {len(issues)} validity issue(s):\n  " + "\n  ".join(issues)
        )
