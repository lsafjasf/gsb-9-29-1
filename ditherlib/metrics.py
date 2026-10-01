"""Error statistics: luminance drift, per-row and cumulative error."""

from .palettes import luminance


def error_stats(input_rows, output_rows, palette):
    """Compare input RGB rows with quantized index rows.

    Returns total luminance difference, per-row error, cumulative error
    over rows, and mean absolute error per pixel.
    """
    colors = palette.colors
    per_row = []
    cumulative = []
    cum = 0.0
    abs_sum = 0.0
    count = 0
    for rin, rout in zip(input_rows, output_rows):
        row_err = 0.0
        for pixel, idx in zip(rin, rout):
            diff = luminance(colors[idx]) - luminance(pixel)
            row_err += diff
            abs_sum += abs(diff)
            count += 1
        per_row.append(row_err)
        cum += row_err
        cumulative.append(cum)
    return {
        "total_lum_diff": cum,
        "per_row_error": per_row,
        "cumulative": cumulative,
        "max_abs_cumulative": max((abs(c) for c in cumulative), default=0.0),
        "max_abs_row_error": max((abs(r) for r in per_row), default=0.0),
        "mae": abs_sum / max(1, count),
    }
