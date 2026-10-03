# Mesh simplification report

All meshes validated after simplification: no degenerate faces, no flipped triangles, manifold edges, consistent winding, single-fan vertices (`validate_topology`).

| case | F before | F after | ratio | target | reached | max dev | rms dev | first cost | last cost | topology |
|---|---|---|---|---|---|---|---|---|---|---|
| single triangle (protect=on) | 1 | 1 | 100.0% | 1 | yes | 0.000000 | 0.000000 | - | - | PASS |
| icosphere(2) closed (protect=on) | 320 | 160 | 50.0% | 160 | yes | 0.085992 | 0.031502 | 6.275e-03 | 1.161e-02 | PASS |
| icosphere(2) closed (protect=off) | 320 | 160 | 50.0% | 160 | yes | 0.099205 | 0.023837 | 2.900e-03 | 4.857e-03 | PASS |
| thin shell t=0.02 (protect=on) | 12 | 12 | 100.0% | 4 | NO (target unreachable) | 0.000000 | 0.000000 | - | - | PASS |
| thin shell t=0.02 (protect=off) | 12 | 4 | 33.3% | 4 | yes | 0.485459 | 0.150812 | 2.000e-04 | 5.000e-01 | PASS |
| cube (protect=on) | 12 | 12 | 100.0% | 4 | NO (target unreachable) | 0.000000 | 0.000000 | - | - | PASS |
| bent sheet (protect=on) | 288 | 144 | 50.0% | 144 | yes | 0.000000 | 0.000000 | 0.000e+00 | 0.000e+00 | PASS |
| bent sheet (protect=off) | 288 | 144 | 50.0% | 144 | yes | 0.123087 | 0.015479 | 0.000e+00 | 0.000e+00 | PASS |
| plane grid 8x8 (protect=on) | 256 | 128 | 50.0% | 128 | yes | 0.000000 | 0.000000 | 0.000e+00 | 0.000e+00 | PASS |

## Rejection counters (why candidate collapses were refused)

- single triangle (protect=on): {}
- icosphere(2) closed (protect=on): {'stale': 240}
- icosphere(2) closed (protect=off): {'stale': 245}
- thin shell t=0.02 (protect=on): {}
- thin shell t=0.02 (protect=off): {'link': 1}
- cube (protect=on): {}
- bent sheet (protect=on): {'stale': 47}
- bent sheet (protect=off): {'stale': 85, 'flip_or_degenerate': 2}
- plane grid 8x8 (protect=on): {'stale': 189, 'flip_or_degenerate': 112}

## Notes

- **single triangle (protect=on)** — degenerate input: 1 face can never be reduced
- **icosphere(2) closed (protect=on)** — smooth closed mesh, 50% reduction
- **icosphere(2) closed (protect=off)** — same mesh without protection (no features exist: should match)
- **thin shell t=0.02 (protect=on)** — closed thin shell; all edges sharp -> fully locked
- **thin shell t=0.02 (protect=off)** — unprotected: collapses but top/bottom get merged (see deviation)
- **cube (protect=on)** — every edge is a 90-degree feature: target 4 unreachable
- **bent sheet (protect=on)** — open curved sheet, 50%: boundary locked
- **bent sheet (protect=off)** — boundary free to erode -> larger deviation
- **plane grid 8x8 (protect=on)** — flat open sheet: QEM collapses are exact, deviation 0
