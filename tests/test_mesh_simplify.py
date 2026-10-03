"""Self-tests for the mesh simplification library. Run: python3 -m unittest"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mesh_simplify import (
    Mesh,
    simplify,
    classify_features,
    validate_topology,
    assert_valid,
    hausdorff_distance,
    generators as G,
)
from tests_meshes import bent_grid


class TestSingleTriangle(unittest.TestCase):
    def test_single_triangle_cannot_simplify(self):
        m = G.single_triangle()
        st = simplify(m, target_faces=1)
        self.assertEqual(m.num_faces(), 1)
        self.assertTrue(st["target_reached"])
        self.assertEqual(st["collapses"], 0)
        assert_valid(m, "single triangle")

    def test_target_clamped_to_one(self):
        m = G.single_triangle()
        st = simplify(m, target_faces=0)
        self.assertEqual(st["target_faces"], 1)
        self.assertEqual(m.num_faces(), 1)
        assert_valid(m)


class TestClosedMesh(unittest.TestCase):
    def test_icosphere_half(self):
        orig = G.icosphere(2)
        m = G.icosphere(2)
        st = simplify(m, target_fraction=0.5)
        self.assertTrue(st["target_reached"])
        self.assertEqual(m.num_faces(), 160)
        assert_valid(m, "simplified icosphere")
        # closed stays closed: every edge has exactly 2 incident faces
        for key, fids in m.edge_faces.items():
            self.assertEqual(len(fids), 2, f"edge {key} opened the closed mesh")
        d = hausdorff_distance(orig.triangles(), m.triangles())
        self.assertLess(d["max"], 0.2)  # radius is 1.0
        self.assertGreater(st["last_cost"], 0.0)
        self.assertLessEqual(st["first_cost"], st["last_cost"] + 1e-9)


class TestThinShell(unittest.TestCase):
    def test_shell_protected_is_untouched(self):
        orig = G.thin_shell()
        m = G.thin_shell()
        st = simplify(m, target_faces=4, protect_features=True)
        # every edge of the box is a sharp feature edge: nothing may move
        self.assertEqual(m.num_faces(), 12)
        self.assertFalse(st["target_reached"])
        d = hausdorff_distance(orig.triangles(), m.triangles())
        self.assertLess(d["max"], 1e-9)
        assert_valid(m, "protected shell")

    def test_shell_unprotected_stays_valid(self):
        m = G.thin_shell()
        st = simplify(m, target_faces=4, protect_features=False)
        self.assertTrue(st["target_reached"])
        # no degenerate faces, no flipped triangles, manifold
        assert_valid(m, "unprotected shell")


class TestUnreachableTarget(unittest.TestCase):
    def test_cube_fully_locked_by_features(self):
        m = G.cube()
        st = simplify(m, target_faces=4, protect_features=True)
        self.assertFalse(st["target_reached"])
        self.assertEqual(m.num_faces(), 12)
        self.assertEqual(st["collapses"], 0)
        assert_valid(m, "locked cube")

    def test_cube_unprotected_reaches_target(self):
        m = G.cube()
        st = simplify(m, target_faces=4, protect_features=False)
        self.assertTrue(st["target_reached"])
        assert_valid(m, "simplified cube")


class TestFeatureProtection(unittest.TestCase):
    def test_classification(self):
        m = G.cube()
        fe, fv = classify_features(m, sharp_deg=45.0)
        # 12 sharp cube edges are features; the 6 face diagonals are
        # coplanar (dihedral 0) and therefore not features
        self.assertEqual(len(fe), 12)
        self.assertEqual(len(fv), 8)

    def test_protection_preserves_boundary_and_reduces_deviation(self):
        results = {}
        for pf in (True, False):
            orig = bent_grid()
            m = bent_grid()
            st = simplify(m, target_fraction=0.5, protect_features=pf)
            assert_valid(m, f"bent grid protect={pf}")
            d = hausdorff_distance(orig.triangles(), m.triangles())
            results[pf] = (m, st, d, orig)
        on, off = results[True], results[False]
        self.assertLess(on[2]["max"], off[2]["max"])
        # every original boundary vertex survives unmoved when protected
        n = 12
        orig_positions = set()
        for j in range(n + 1):
            for i in range(n + 1):
                if i in (0, n) or j in (0, n):
                    x, y = i / n, j / n
                    orig_positions.add((x, y, 0.35 * math.sin(math.pi * x)))
        remaining = {v for v in on[0].vertices if v is not None}
        for p in orig_positions:
            self.assertIn(p, remaining)


class TestValidityChecks(unittest.TestCase):
    def test_detects_flipped_triangle(self):
        # second face wound backwards: shared edge (0,2) is traversed in
        # the same direction by both faces
        m = Mesh(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
            [(0, 1, 2), (0, 3, 2)],
        )
        issues = validate_topology(m)
        self.assertTrue(any("winding" in i or "flipped" in i for i in issues))

    def test_detects_degenerate_face(self):
        m = Mesh(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            [(0, 1, 2)],
        )
        issues = validate_topology(m)
        self.assertTrue(any("degenerate" in i for i in issues))

    def test_detects_nonmanifold_edge(self):
        m = Mesh(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)],
            [(0, 1, 2), (0, 1, 3), (1, 0, 4)],
        )
        issues = validate_topology(m)
        self.assertTrue(any("non-manifold" in i for i in issues))

    def test_clean_mesh_passes(self):
        self.assertEqual(validate_topology(G.icosphere(1)), [])


class TestDeviation(unittest.TestCase):
    def test_identical_meshes_zero_deviation(self):
        a = G.icosphere(1).triangles()
        d = hausdorff_distance(a, a)
        self.assertAlmostEqual(d["max"], 0.0)
        self.assertAlmostEqual(d["rms"], 0.0)

    def test_known_offset(self):
        # unit triangle shifted by 0.5 in z -> deviation 0.5
        t1 = [((0, 0, 0), (1, 0, 0), (0, 1, 0))]
        t2 = [((0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5))]
        d = hausdorff_distance(t1, t2)
        self.assertAlmostEqual(d["max"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
