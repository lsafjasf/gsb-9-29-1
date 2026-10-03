"""Automatic threshold selection from gradient-magnitude statistics.

Rationale
---------
The histogram of non-maximum-suppressed gradient magnitudes is
modelled as a mixture of a "non-edge" population (noise, flat areas)
and an "edge" population:

* ``high`` = max(Otsu threshold, median + 3 * sigma_robust)
* ``low``  = max(0.4 * Otsu threshold, median + 2 * sigma_robust)

Otsu's method picks the threshold that maximises between-class
variance, which is optimal for a bimodal edge/non-edge histogram --
the common case for images that actually contain edges.

When the image contains *only* noise the histogram is unimodal and
Otsu degenerates to splitting the noise body in half, which would
label half the image as edges.  The robust floor guards this case:
``sigma_robust = 1.4826 * MAD`` is a consistent estimator of the
standard deviation for Gaussian-like noise, so ``median + 3 sigma``
lies beyond the 99.5th percentile of the noise body -- only
statistically extreme responses qualify as strong seeds.  The weak
floor at ``median + 2 sigma`` keeps weak pixels sparse enough that
isolated noise seeds cannot grow into large hysteresis blobs (the
weak-pixel fraction in pure noise stays near 4%, well below the
8-neighbour percolation threshold).

The 0.4 low/high ratio follows the classic Canny recommendation of a
1:2 to 1:3 ratio between the low and high thresholds.
"""


def median(sorted_vals):
    """Median of an already-sorted sequence."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def otsu_threshold(values, bins=256):
    """Otsu's between-class-variance threshold for non-negative values.

    Returns a threshold in the same units as ``values``; returns 0.0
    for empty or all-zero input.
    """
    n = len(values)
    if n == 0:
        return 0.0
    hi = max(values)
    if hi <= 0.0:
        return 0.0
    hist = [0] * bins
    scale = (bins - 1) / hi
    for v in values:
        hist[int(v * scale)] += 1
    sum_all = 0.0
    for i in range(bins):
        sum_all += i * hist[i]
    sum_back = 0.0
    weight_back = 0
    best_var = -1.0
    best_bin = 0
    for i in range(bins):
        weight_back += hist[i]
        if weight_back == 0:
            continue
        weight_fore = n - weight_back
        if weight_fore == 0:
            break
        sum_back += i * hist[i]
        mean_back = sum_back / weight_back
        mean_fore = (sum_all - sum_back) / weight_fore
        diff = mean_back - mean_fore
        var_between = weight_back * weight_fore * diff * diff
        if var_between > best_var:
            best_var = var_between
            best_bin = i
    # Upper edge of the chosen bin: pixels with value >= the result are
    # classified as "edge", which cleanly separates the two classes.
    return (best_bin + 1) / scale


def auto_thresholds(survivor_magnitudes, low_ratio=0.4):
    """Pick (low, high) hysteresis thresholds automatically.

    ``survivor_magnitudes`` is an iterable of gradient magnitudes of the
    pixels that survived non-maximum suppression.  Returns
    ``(low, high)``; both are 0.0 when there is nothing to detect
    (no survivors, or a perfectly flat image), which the caller must
    treat as "no edges".
    """
    vals = sorted(survivor_magnitudes)
    if not vals:
        return 0.0, 0.0
    med = median(vals)
    mad = median(sorted(abs(v - med) for v in vals))
    sigma = 1.4826 * mad
    otsu = otsu_threshold(vals)
    high = max(otsu, med + 3.0 * sigma)
    low = max(low_ratio * otsu, med + 2.0 * sigma)
    if low > high:
        low = high
    if high <= 0.0:
        return 0.0, 0.0
    return low, high
