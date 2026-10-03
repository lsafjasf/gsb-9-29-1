"""Extra mesh builders shared by tests and the demo."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mesh_simplify.mesh import Mesh


def bent_grid(n=12, size=1.0, height=0.35):
    """Open sheet curved along x (extruded along y): has a smooth interior
    plus a feature boundary, ideal for showing protection on/off."""
    verts = []
    for j in range(n + 1):
        for i in range(n + 1):
            x, y = size * i / n, size * j / n
            verts.append((x, y, height * math.sin(math.pi * x / size)))
    idx = lambda i, j: j * (n + 1) + i
    faces = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces += [(a, b, d), (b, c, d)]
    return Mesh(verts, faces)
