"""Memory / scaling measurements for the sparse voxelizer.

Run:  python3 benchmark.py
"""

import time
import tracemalloc

from voxelizer import SparseVoxelGrid
from test_voxelizer import cube_mesh


def measure(name, triangles, voxel_size, origin=(0.0, 0.0, 0.0)):
    g = SparseVoxelGrid(voxel_size, origin)
    tracemalloc.start()
    t0 = time.perf_counter()
    g.voxelize(triangles)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("%-46s s=%-6g voxels=%-9d est=%-9d peak=%-9d t=%.3fs"
          % (name, voxel_size, g.count(), g.memory_bytes(), peak, dt))
    return g


def main():
    print("%-46s %s" % ("scenario", "s      occupied  est_bytes peak_bytes time"))

    # Solid cube 4m at increasing resolution: count ~ surface / s^2.
    cube = cube_mesh(0.0, 0.0, 0.0, 4.0)
    for s in (1.0, 0.5, 0.25, 0.1, 0.05):
        measure("solid cube 4m (12 tris)", cube, s)

    # Thin plate 100x100m (2 triangles): count ~ area / s^2.
    plate = [((0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 100.0, 0.0)),
             ((0.0, 0.0, 0.0), (100.0, 100.0, 0.0), (0.0, 100.0, 0.0))]
    for s in (1.0, 0.5, 0.25):
        measure("thin plate 100x100m (2 tris)", plate, s)

    # Same cube at 1e9 offset: identical count/memory -> scene extent
    # does not affect memory.
    far = cube_mesh(1e9, -1e9, 5e8, 4.0)
    measure("cube 4m at 1e9 offset", far, 0.1)

    # Empty scene.
    measure("empty scene", [], 0.1)


if __name__ == "__main__":
    main()
