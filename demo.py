#!/usr/bin/env python3
"""Demo: run simplification scenarios, print + write REPORT.md."""
import copy
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_simplify import simplify, assert_valid, hausdorff_distance, generators as G
from tests.tests_meshes import bent_grid


def clone(mesh):
    return copy.deepcopy(mesh)


def run_case(name, mesh, note, **kw):
    orig = clone(mesh)
    st = simplify(mesh, **kw)
    issues_ok = True
    try:
        assert_valid(mesh, name)
    except AssertionError as exc:
        issues_ok = False
        print(exc, file=sys.stderr)
    d = hausdorff_distance(orig.triangles(), mesh.triangles())
    ratio = st["faces_after"] / st["faces_before"]
    row = {
        "name": name,
        "note": note,
        "before": st["faces_before"],
        "after": st["faces_after"],
        "ratio": f"{ratio:.1%}",
        "target": st["target_faces"],
        "reached": "yes" if st["target_reached"] else "NO (target unreachable)",
        "max_dev": f"{d['max']:.6f}",
        "rms_dev": f"{d['rms']:.6f}",
        "cost_first": f"{st['first_cost']:.3e}" if st["first_cost"] is not None else "-",
        "cost_last": f"{st['last_cost']:.3e}" if st["last_cost"] is not None else "-",
        "valid": "PASS" if issues_ok else "FAIL",
        "rejected": st["rejected"],
    }
    return row


def main():
    rows = []

    rows.append(run_case(
        "single triangle (protect=on)", G.single_triangle(),
        "degenerate input: 1 face can never be reduced",
        target_faces=1, protect_features=True))

    rows.append(run_case(
        "icosphere(2) closed (protect=on)", G.icosphere(2),
        "smooth closed mesh, 50% reduction",
        target_fraction=0.5, protect_features=True))

    rows.append(run_case(
        "icosphere(2) closed (protect=off)", G.icosphere(2),
        "same mesh without protection (no features exist: should match)",
        target_fraction=0.5, protect_features=False))

    rows.append(run_case(
        "thin shell t=0.02 (protect=on)", G.thin_shell(),
        "closed thin shell; all edges sharp -> fully locked",
        target_faces=4, protect_features=True))

    rows.append(run_case(
        "thin shell t=0.02 (protect=off)", G.thin_shell(),
        "unprotected: collapses but top/bottom get merged (see deviation)",
        target_faces=4, protect_features=False))

    rows.append(run_case(
        "cube (protect=on)", G.cube(),
        "every edge is a 90-degree feature: target 4 unreachable",
        target_faces=4, protect_features=True))

    rows.append(run_case(
        "bent sheet (protect=on)", bent_grid(),
        "open curved sheet, 50%: boundary locked",
        target_fraction=0.5, protect_features=True))

    rows.append(run_case(
        "bent sheet (protect=off)", bent_grid(),
        "boundary free to erode -> larger deviation",
        target_fraction=0.5, protect_features=False))

    rows.append(run_case(
        "plane grid 8x8 (protect=on)", G.plane_grid(8),
        "flat open sheet: QEM collapses are exact, deviation 0",
        target_fraction=0.5, protect_features=True))

    headers = ["name", "before", "after", "ratio", "target", "reached",
               "max_dev", "rms_dev", "cost_first", "cost_last", "valid"]
    lines = []
    lines.append("# Mesh simplification report\n")
    lines.append("All meshes validated after simplification: no degenerate faces, "
                 "no flipped triangles, manifold edges, consistent winding, "
                 "single-fan vertices (`validate_topology`).\n")
    lines.append("| case | F before | F after | ratio | target | reached | max dev | rms dev | first cost | last cost | topology |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["name"], r["before"], r["after"], r["ratio"], r["target"],
            r["reached"], r["max_dev"], r["rms_dev"], r["cost_first"],
            r["cost_last"], r["valid"]))
    lines.append("")
    lines.append("## Rejection counters (why candidate collapses were refused)\n")
    for r in rows:
        lines.append(f"- {r['name']}: {r['rejected']}")
    lines.append("")
    lines.append("## Notes\n")
    for r in rows:
        lines.append(f"- **{r['name']}** — {r['note']}")
    lines.append("")
    report = "\n".join(lines)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPORT.md"), "w") as fh:
        fh.write(report)
    print(report)
    if any(r["valid"] != "PASS" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
