"""Threshold-sensitivity analysis: edge counts and miss ratios.

The expensive part (Sobel + NMS) is computed once; the hysteresis
thresholds are then scaled by factors in ``scales`` (1.0 = the
automatically chosen thresholds).

Miss-ratio definitions
----------------------
* Without ground truth: the reference edge set is the result at the
  most permissive setting (smallest scale).  ``miss_ratio(s)`` is the
  fraction of reference edge pixels absent at scale ``s`` -- it
  quantifies how many candidate edges raising the threshold drops.
* With ground truth (``gt`` boolean map): a GT pixel counts as found
  if a detected edge sits within its 3x3 neighbourhood (1px
  localisation tolerance, standard for edge evaluation).  ``miss`` is
  the fraction of GT edge pixels not found.
"""

from .autothresh import auto_thresholds
from .core import hysteresis
from .pipeline import gradient_and_candidates

DEFAULT_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _edge_set(edges):
    return {(y, x) for y, row in enumerate(edges)
            for x, v in enumerate(row) if v}


def _covered_gt(edge_set, gt):
    """Set of GT pixels with an edge within their 3x3 neighbourhood."""
    h = len(gt)
    w = len(gt[0])
    covered = set()
    for (y, x) in edge_set:
        for dy in (-1, 0, 1):
            ny = y + dy
            if not (0 <= ny < h):
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                if 0 <= nx < w and gt[ny][nx]:
                    covered.add((ny, nx))
    return covered


def sensitivity(rows, gt=None, scales=DEFAULT_SCALES):
    """Sweep threshold scales.  Returns list of result dicts per scale."""
    mag, keep, survivors = gradient_and_candidates(rows)
    base_low, base_high = auto_thresholds(survivors)
    ordered = sorted(scales)

    gt_set = None
    if gt is not None:
        gt_set = {(y, x) for y, row in enumerate(gt)
                  for x, v in enumerate(row) if v}
        if not gt_set:
            raise ValueError("ground truth contains no edge pixels")

    reference_set = None
    results = {}
    for scale in ordered:
        low = base_low * scale
        high = base_high * scale
        if high <= 0.0:
            edges = [[False] * len(mag[0]) for _ in range(len(mag))]
        else:
            edges = hysteresis(keep, mag, low, high)
        found = _edge_set(edges)
        if reference_set is None and gt_set is None:
            reference_set = found  # smallest scale = most permissive
        results[scale] = {"scale": scale, "low": low, "high": high,
                          "edge_pixels": len(found), "_found": found}

    for scale in ordered:
        found = results[scale]["_found"]
        if gt_set is not None:
            covered = _covered_gt(found, gt)
            miss = 1.0 - len(covered) / len(gt_set)
        else:
            if reference_set:
                miss = 1.0 - len(found & reference_set) / len(reference_set)
            else:
                miss = 0.0
        results[scale]["miss_ratio"] = miss
        del results[scale]["_found"]

    return {
        "auto_low": base_low,
        "auto_high": base_high,
        "reference_pixels": len(gt_set) if gt_set is not None
                            else (len(reference_set) if reference_set else 0),
        "reference_kind": "ground_truth" if gt_set is not None
                          else "most_permissive_scale",
        "rows": [results[s] for s in ordered],
    }


def to_csv(report):
    """Render a sensitivity report as CSV text."""
    lines = [
        "# auto_low=%.4f auto_high=%.4f reference=%s reference_pixels=%d"
        % (report["auto_low"], report["auto_high"],
           report["reference_kind"], report["reference_pixels"]),
        "scale,low,high,edge_pixels,miss_ratio",
    ]
    for r in report["rows"]:
        lines.append("%.3g,%.4f,%.4f,%d,%.6f"
                     % (r["scale"], r["low"], r["high"],
                        r["edge_pixels"], r["miss_ratio"]))
    return "\n".join(lines) + "\n"
