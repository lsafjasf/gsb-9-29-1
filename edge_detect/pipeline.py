"""High-level edge detection pipeline."""

from .autothresh import auto_thresholds
from .core import count_edges, hysteresis, non_max_suppression, sobel_gradient


def gradient_and_candidates(rows):
    """Run Sobel + NMS once, reusing the result across a threshold sweep."""
    mag, sector = sobel_gradient(rows)
    keep = non_max_suppression(mag, sector)
    survivors = [mag[y][x] for y in range(len(mag))
                 for x in range(len(mag[0])) if keep[y][x]]
    return mag, keep, survivors


def detect_from_candidates(mag, keep, low=None, high=None, survivors=None):
    """Apply hysteresis to precomputed gradients/NMS candidates."""
    if low is None or high is None:
        if survivors is None:
            survivors = [mag[y][x] for y in range(len(mag))
                         for x in range(len(mag[0])) if keep[y][x]]
        low, high = auto_thresholds(survivors)
    if high <= 0.0:
        return [[False] * len(mag[0]) for _ in range(len(mag))], (low, high)
    return hysteresis(keep, mag, low, high), (low, high)


def detect_edges(rows, low=None, high=None):
    """Full Canny-style pipeline.

    ``rows`` is a grayscale image (list of numeric rows).  When
    ``low``/``high`` are omitted they are chosen automatically from
    image statistics.  Returns ``(edges, (low, high))``.
    """
    mag, keep, survivors = gradient_and_candidates(rows)
    return detect_from_candidates(mag, keep, low, high, survivors)


__all__ = ["detect_edges", "detect_from_candidates",
           "gradient_and_candidates", "count_edges",
           "sobel_gradient", "non_max_suppression", "hysteresis"]
