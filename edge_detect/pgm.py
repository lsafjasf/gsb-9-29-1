"""Minimal PGM (P2/P5) reader and writer -- image I/O with no deps.

PGM is used because it needs no third-party libraries: samples are
plain P5 binary PGM files that any image tool (GIMP, ImageMagick,
``feh``, VS Code image viewers) can open.
"""


def _next_token(data, pos):
    """Return (int_token, new_pos), skipping whitespace and # comments."""
    n = len(data)
    while True:
        while pos < n and data[pos] in b" \t\n\r":
            pos += 1
        if pos < n and data[pos] == ord("#"):
            while pos < n and data[pos] not in b"\n\r":
                pos += 1
            continue
        break
    start = pos
    while pos < n and data[pos] not in b" \t\n\r":
        pos += 1
    return int(data[start:pos]), pos


def read_pgm(path):
    """Read a P2 (ASCII) or P5 (binary) PGM.  Returns list of rows.

    Values are ints in 0..maxval (usually 0..255).  Only one-component
    grayscale PGM is supported.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"P5" and data[:2] != b"P2":
        raise ValueError("not a P2/P5 PGM file: %r" % path)
    pos = 2
    width, pos = _next_token(data, pos)
    height, pos = _next_token(data, pos)
    maxval, pos = _next_token(data, pos)
    if maxval > 65535:
        raise ValueError("unsupported PGM maxval %d" % maxval)
    if data[:2] == b"P2":
        vals = []
        for _ in range(width * height):
            v, pos = _next_token(data, pos)
            vals.append(v)
    else:
        pos += 1  # single whitespace separating header from raster
        if maxval < 256:
            vals = list(data[pos:pos + width * height])
        else:
            vals = [data[pos + 2 * i] * 256 + data[pos + 2 * i + 1]
                    for i in range(width * height)]
    if len(vals) != width * height:
        raise ValueError("truncated PGM raster: %r" % path)
    return [vals[y * width:(y + 1) * width] for y in range(height)]


def write_pgm(path, rows, maxval=255):
    """Write an image (sequence of numeric rows) as binary P5 PGM."""
    height = len(rows)
    width = len(rows[0]) if height else 0
    if maxval > 255:
        raw = bytearray(width * height * 2)
        for y, row in enumerate(rows):
            for x, v in enumerate(row):
                v = max(0, min(maxval, int(v)))
                raw[2 * (y * width + x)] = v // 256
                raw[2 * (y * width + x) + 1] = v % 256
    else:
        raw = bytearray(width * height)
        for y, row in enumerate(rows):
            off = y * width
            for x, v in enumerate(row):
                raw[off + x] = max(0, min(255, int(v)))
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n%d\n" % (width, height, maxval))
        f.write(bytes(raw))


def edges_to_image(edges):
    """Convert a boolean edge map to an 8-bit image (255 = edge)."""
    return [[255 if v else 0 for v in row] for row in edges]
