"""Canny-style edge detection core (Python standard library only).

Border rule (replicate padding)
-------------------------------
Every gradient computation clamps out-of-range indices to the nearest
valid pixel ("replicate padding").  Consequences, verified by tests:

* A constant image has exactly zero gradient everywhere, borders
  included -- the image border itself never becomes a spurious edge.
* A genuine intensity step that reaches the image boundary still
  produces a one-sided gradient at the boundary pixel, so real
  boundary edges are preserved instead of being dropped or mirrored
  (mirroring would invent edges that do not exist).

Non-maximum suppression is evaluated for *all* pixels, border pixels
included, using the same replicate padding on the magnitude map, so no
edge is lost simply because it touches the image boundary.
"""

import math
from collections import deque

# tan(22.5 deg) and tan(67.5 deg): sector boundaries for direction
# quantization.  Using the exact algebraic values avoids libm calls in
# the hot loop.
_TAN_22_5 = math.sqrt(2.0) - 1.0
_TAN_67_5 = math.sqrt(2.0) + 1.0

# Neighbour offsets (dx, dy) along the gradient for each quantized
# direction.  y grows downward, as usual for images.
_NEIGHBOURS = {
    0: ((1, 0), (-1, 0)),
    45: ((1, 1), (-1, -1)),
    90: ((0, 1), (0, -1)),
    135: ((1, -1), (-1, 1)),
}


def sobel_gradient(rows):
    """Compute Sobel gradients of a grayscale image.

    ``rows`` is a sequence of ``h`` rows, each a sequence of ``w``
    numbers (typically 0..255).  Returns ``(mag, sector)`` where ``mag``
    is the gradient magnitude ``hypot(gx, gy)`` and ``sector`` is the
    gradient direction quantized to 0/45/90/135 degrees.  Both are
    lists of rows with the same shape as the input.

    Border rule: replicate padding (see module docstring).
    """
    h = len(rows)
    if h == 0:
        raise ValueError("image has no rows")
    w = len(rows[0])
    if w == 0:
        raise ValueError("image has no columns")
    for row in rows:
        if len(row) != w:
            raise ValueError("ragged image: rows have different widths")

    # Replicate-padded copy: each padded row has width w + 2.
    ext = [[row[0]] + list(row) + [row[w - 1]] for row in rows]

    mag = [[0.0] * w for _ in range(h)]
    sector = [[0] * w for _ in range(h)]
    for y in range(h):
        r0 = ext[y - 1] if y > 0 else ext[0]
        r1 = ext[y]
        r2 = ext[y + 1] if y < h - 1 else ext[h - 1]
        mrow = mag[y]
        srow = sector[y]
        for x in range(w):
            i = x + 1  # index into the padded rows
            gx = (r0[i + 1] + 2 * r1[i + 1] + r2[i + 1]) \
               - (r0[i - 1] + 2 * r1[i - 1] + r2[i - 1])
            gy = (r2[i - 1] + 2 * r2[i] + r2[i + 1]) \
               - (r0[i - 1] + 2 * r0[i] + r0[i + 1])
            mrow[x] = math.sqrt(gx * gx + gy * gy)
            ax = gx if gx >= 0 else -gx
            ay = gy if gy >= 0 else -gy
            if ay <= _TAN_22_5 * ax:
                srow[x] = 0
            elif ay >= _TAN_67_5 * ax:
                srow[x] = 90
            elif (gx > 0) == (gy > 0):
                srow[x] = 45
            else:
                srow[x] = 135
    return mag, sector


def non_max_suppression(mag, sector):
    """Thin gradient ridges to single-pixel-wide candidate edges.

    A pixel survives iff its magnitude is strictly positive and not
    smaller than either of its two neighbours along the quantized
    gradient direction.  Equal-magnitude plateaus are thinned to one
    pixel by an asymmetric tie rule (``m > n1 and m >= n2`` where ``n1``
    is the neighbour in the positive direction), so ridges are always
    one pixel wide.  Border pixels are evaluated against
    replicate-padded neighbours, so genuine boundary edges survive.

    Returns a boolean map (list of rows) of surviving pixels.
    """
    h = len(mag)
    w = len(mag[0])
    # Replicate-padded magnitude map, width w + 2, height h + 2.
    pmag = [[row[0]] + list(row) + [row[w - 1]] for row in mag]
    pmag = [pmag[0]] + pmag + [pmag[h - 1]]

    keep = [[False] * w for _ in range(h)]
    for y in range(h):
        mrow = pmag[y + 1]
        srow = sector[y]
        krow = keep[y]
        for x in range(w):
            m = mrow[x + 1]
            if m <= 0.0:
                continue
            (dx1, dy1), (dx2, dy2) = _NEIGHBOURS[srow[x]]
            n1 = pmag[y + 1 + dy1][x + 1 + dx1]
            n2 = pmag[y + 1 + dy2][x + 1 + dx2]
            if m > n1 and m >= n2:
                krow[x] = True
    return keep


def hysteresis(keep, mag, low, high):
    """Double-threshold edge linking (8-connectivity).

    Pixels with magnitude >= ``high`` are strong seeds.  Pixels with
    magnitude >= ``low`` that are 8-connected to a seed (through other
    weak pixels) are kept; isolated weak pixels are discarded.  Returns
    a boolean edge map.
    """
    if not 0.0 <= low <= high:
        raise ValueError("require 0 <= low <= high")
    h = len(keep)
    w = len(keep[0])
    edges = [[False] * w for _ in range(h)]
    queue = deque()
    for y in range(h):
        krow = keep[y]
        mrow = mag[y]
        erow = edges[y]
        for x in range(w):
            if krow[x] and mrow[x] >= high:
                erow[x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            ny = y + dy
            if ny < 0 or ny >= h:
                continue
            erow = edges[ny]
            krow = keep[ny]
            mrow = mag[ny]
            for dx in (-1, 0, 1):
                nx = x + dx
                if nx < 0 or nx >= w or erow[nx]:
                    continue
                if krow[nx] and mrow[nx] >= low:
                    erow[nx] = True
                    queue.append((ny, nx))
    return edges


def count_edges(edges):
    """Number of edge pixels in a boolean edge map."""
    return sum(row.count(True) for row in edges)
