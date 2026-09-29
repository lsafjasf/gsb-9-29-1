"""Self-tests for the sparse voxelizer (stdlib unittest).

Run:  python3 test_voxelizer.py -v
"""

import random
import unittest

from voxelizer import SparseVoxelGrid, tri_box_overlap
from reference import ref_tri_intersects_box, ref_occupied_set


def cube_mesh(x0, y0, z0, size):
    """12 triangles forming a closed axis-aligned cube."""
    x1, y1, z1 = x0 + size, y0 + size, z0 + size
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
             (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
             (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


class TestEdgeCases(unittest.TestCase):
    def test_empty_scene(self):
        g = SparseVoxelGrid(0.5)
        self.assertEqual(g.count(), 0)
        self.assertIsNone(g.bounds())
        self.assertEqual(g.query_range((-10, -10, -10), (10, 10, 10)), [])
        self.assertEqual(g.count_range((-10, -10, -10), (10, 10, 10)), 0)

    def test_single_voxel(self):
        g = SparseVoxelGrid(1.0)
        tiny = ((0.4, 0.4, 0.4), (0.5, 0.4, 0.4), (0.4, 0.5, 0.4))
        self.assertEqual(g.add_triangle(*tiny), 1)
        self.assertEqual(set(g.iter_voxels()), {(0, 0, 0)})

    def test_invalid_voxel_size(self):
        with self.assertRaises(ValueError):
            SparseVoxelGrid(0.0)
        with self.assertRaises(ValueError):
            SparseVoxelGrid(-1.0)

    def test_thin_face_passing_through_voxel(self):
        # Triangle plane x = 0.5 slices voxel (0,0,0) = [0,1]^3, but no
        # vertex lies inside the voxel: it must still be marked.
        g = SparseVoxelGrid(1.0)
        tri = ((0.5, -10.0, -10.0), (0.5, 10.0, -10.0), (0.5, 0.0, 10.0))
        g.add_triangle(*tri)
        self.assertIn((0, 0, 0), g)
        # A parallel plane just outside the voxel must not mark it.
        g2 = SparseVoxelGrid(1.0)
        tri2 = ((1.0001, -10.0, -10.0), (1.0001, 10.0, -10.0),
                (1.0001, 0.0, 10.0))
        g2.add_triangle(*tri2)
        self.assertNotIn((0, 0, 0), g2)

    def test_face_exactly_on_boundary_marks_both_sides(self):
        # Closed-box convention: a triangle on the plane x = 1 touches
        # both voxel 0 and voxel 1.
        g = SparseVoxelGrid(1.0)
        tri = ((1.0, -5.0, -5.0), (1.0, 5.0, -5.0), (1.0, 0.0, 5.0))
        g.add_triangle(*tri)
        self.assertIn((0, 0, 0), g)
        self.assertIn((1, 0, 0), g)

    def test_edge_and_corner_touch(self):
        bmin = (0.0, 0.0, 0.0)
        bmax = (1.0, 1.0, 1.0)
        # Triangle edge piercing the box corner region.
        tri = ((0.0, 0.0, -5.0), (0.0, 0.0, 5.0), (0.0, 1e-9, 0.0))
        self.assertTrue(tri_box_overlap(bmin, bmax, tri))
        # Degenerate point triangle inside the box.
        self.assertTrue(tri_box_overlap(bmin, bmax,
                                        ((0.5, 0.5, 0.5),) * 3))
        # Degenerate point triangle outside the box.
        self.assertFalse(tri_box_overlap(bmin, bmax,
                                         ((2.0, 0.5, 0.5),) * 3))

    def test_degenerate_line_triangle(self):
        # Zero-area (collinear) triangle behaves like a segment.
        g = SparseVoxelGrid(1.0)
        g.add_triangle((0.5, 0.5, 0.5), (3.5, 0.5, 0.5), (2.0, 0.5, 0.5))
        for i in range(4):
            self.assertIn((i, 0, 0), g)
        self.assertNotIn((4, 0, 0), g)

    def test_voxel_size_much_smaller_than_object(self):
        # Solid cube 4m, voxel 0.05 -> dense shell, must match reference.
        size = 4.0
        s = 0.05
        g = SparseVoxelGrid(s)
        g.voxelize(cube_mesh(0.0, 0.0, 0.0, size))
        n = int(round(size / s))
        # Shell of an n^3 block: n^3 - (n-2)^3 voxels (surface is exactly
        # on voxel planes, and closed-box rule adds a 1-voxel rim on the
        # + side, so check against the analytic reference instead).
        lo, hi = (-1, -1, -1), (n, n, n)
        ref = ref_occupied_set(cube_mesh(0.0, 0.0, 0.0, size), s,
                               (0.0, 0.0, 0.0), lo, hi)
        self.assertEqual(set(g.iter_voxels()), ref)
        self.assertGreater(g.count(), 6 * (n - 1) ** 2)  # sanity: ~surface

    def test_huge_scene_coordinates(self):
        # Same cube placed at 1e9 offset: identical voxel count, tiny memory.
        g_near = SparseVoxelGrid(0.5)
        g_near.voxelize(cube_mesh(0.0, 0.0, 0.0, 4.0))
        g_far = SparseVoxelGrid(0.5)
        g_far.voxelize(cube_mesh(1e9, -1e9, 5e8, 4.0))
        self.assertEqual(g_near.count(), g_far.count())
        lo, hi = g_far.bounds()
        self.assertGreater(lo[0], 1.9e9)   # indices are huge ...
        self.assertLess(g_far.memory_bytes(), 1_000_000)  # ... memory isn't
        # Negative-direction floor indexing.
        g_neg = SparseVoxelGrid(1.0)
        g_neg.add_triangle((-0.5, -0.5, -0.5), (-0.1, -0.5, -0.5),
                           (-0.5, -0.1, -0.5))
        self.assertEqual(set(g_neg.iter_voxels()), {(-1, -1, -1)})

    def test_range_query_and_stats(self):
        g = SparseVoxelGrid(1.0)
        g.voxelize(cube_mesh(0.0, 0.0, 0.0, 4.0))
        all_vox = set(g.iter_voxels())
        got = set(g.query_range((1, 1, 1), (2, 2, 2)))
        expect = {v for v in all_vox
                  if all(1 <= c <= 2 for c in v)}
        self.assertEqual(got, expect)
        self.assertEqual(got, set(g.query_range_world((1.0, 1.0, 1.0),
                                                      (2.0, 2.0, 2.0))))
        self.assertEqual(g.count_range((1, 1, 1), (2, 2, 2)), len(expect))
        st = g.stats()
        self.assertEqual(st["occupied"], g.count())
        self.assertGreater(st["memory_bytes"], 0)


class TestCrossValidation(unittest.TestCase):
    """Occupied sets from the SAT voxelizer must exactly equal the
    independent clipping-based analytic reference."""

    def _check_scene(self, triangles, s, origin, lo, hi):
        g = SparseVoxelGrid(s, origin)
        g.voxelize(triangles)
        ref = ref_occupied_set(triangles, s, origin, lo, hi)
        # The reference only scans [lo, hi]; restrict the voxelizer's
        # output to the same region (closed-box rim voxels just outside
        # the region are legitimate and covered by dedicated tests).
        got = {v for v in g.iter_voxels()
               if all(lo[a] <= v[a] <= hi[a] for a in range(3))}
        self.assertEqual(got, ref,
                         msg="mismatch: only-voxelizer=%s only-ref=%s"
                         % (sorted(got - ref)[:5], sorted(ref - got)[:5]))

    def test_grid_aligned_scene(self):
        # Coordinates on a 0.25 grid: exact in binary floating point.
        rng = random.Random(20260930)
        s = 0.5
        origin = (0.0, 0.0, 0.0)
        tris = []
        for _ in range(60):
            tri = tuple(tuple(rng.randint(-12, 12) * 0.25 for _ in range(3))
                        for _ in range(3))
            tris.append(tri)
        self._check_scene(tris, s, origin, (-8, -8, -8), (8, 8, 8))

    def test_random_float_scene(self):
        rng = random.Random(42)
        s = 0.7
        origin = (0.13, -0.21, 0.05)
        tris = []
        for _ in range(80):
            tri = tuple(tuple(rng.uniform(-5.0, 5.0) for _ in range(3))
                        for _ in range(3))
            tris.append(tri)
        self._check_scene(tris, s, origin, (-9, -9, -9), (9, 9, 9))

    def test_adversarial_scene(self):
        # Thin faces, boundary-hugging planes, near-degenerate triangles.
        tris = [
            ((0.5, -10.0, -10.0), (0.5, 10.0, -10.0), (0.5, 0.0, 10.0)),
            ((1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 0.0, 2.0)),
            ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (1.5, 0.0, 0.0)),  # segment
            ((0.25, 0.25, 0.25),) * 3,                            # point
            ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), (2.0, -2.0, 0.5)),
            ((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        ]
        self._check_scene(tris, 1.0, (0.0, 0.0, 0.0), (-4, -4, -4), (4, 4, 4))
        self._check_scene(tris, 0.3, (0.01, 0.02, 0.03), (-8, -8, -8), (8, 8, 8))

    def test_cube_mesh_scene(self):
        tris = cube_mesh(-1.5, -1.5, -1.5, 3.0) + cube_mesh(2.0, 0.0, -1.0, 1.0)
        self._check_scene(tris, 0.4, (0.0, 0.0, 0.0), (-6, -6, -6), (9, 9, 9))


if __name__ == "__main__":
    unittest.main()
