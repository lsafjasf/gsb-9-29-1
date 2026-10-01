"""Minimal PGM/PPM/PNG writers (stdlib only) for artifact samples."""

import struct
import zlib


def write_pgm(path, rows):
    height = len(rows)
    width = len(rows[0]) if height else 0
    with open(path, "wb") as fh:
        fh.write(b"P5\n%d %d\n255\n" % (width, height))
        for row in rows:
            fh.write(bytes(int(v) & 0xFF for v in row))


def write_ppm(path, rows):
    height = len(rows)
    width = len(rows[0]) if height else 0
    with open(path, "wb") as fh:
        fh.write(b"P6\n%d %d\n255\n" % (width, height))
        for row in rows:
            fh.write(bytes(c for pixel in row for c in pixel))


def _png_chunk(tag, payload):
    chunk = tag + payload
    return struct.pack(">I", len(payload)) + chunk + struct.pack(
        ">I", zlib.crc32(chunk) & 0xFFFFFFFF)


def write_png(path, rows):
    """Write 8-bit PNG; rows of gray ints or of (r, g, b) tuples."""
    height = len(rows)
    width = len(rows[0]) if height else 0
    color = isinstance(rows[0][0], (tuple, list)) if height else False
    depth, ctype = (3, 2) if color else (1, 0)
    raw = bytearray()
    for row in rows:
        raw.append(0)
        if color:
            for pixel in row:
                raw.extend(int(c) & 0xFF for c in pixel)
        else:
            raw.extend(int(v) & 0xFF for v in row)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0)
    del depth
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_png_chunk(b"IHDR", ihdr))
        fh.write(_png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        fh.write(_png_chunk(b"IEND", b""))


def indices_to_gray_rows(index_rows, palette):
    return [[palette.colors[idx][0] for idx in row] for row in index_rows]


def indices_to_color_rows(index_rows, palette):
    return [[palette.colors[idx] for idx in row] for row in index_rows]
