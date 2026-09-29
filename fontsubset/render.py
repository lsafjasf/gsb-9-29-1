"""Minimal TrueType renderer used to verify subset correctness.

Extracts glyph outlines (resolving composite transforms), flattens
quadratic curves to polygons and rasterizes with a non-zero-winding
scanline fill (4x4 supersampling). Pure standard library; identical
input outlines always produce byte-identical bitmaps, so a subset font
can be verified by comparing its rendering against the original font.
"""

import struct
import zlib

from .ttf import (Font, i16, u16, ARG_1_AND_2_ARE_WORDS, ARGS_ARE_XY_VALUES,
                  WE_HAVE_A_SCALE, MORE_COMPONENTS, WE_HAVE_AN_X_AND_Y_SCALE,
                  WE_HAVE_A_TWO_BY_TWO)


class MissingGlyphError(Exception):
    """Raised when text contains characters the font does not cover."""

    def __init__(self, missing):
        self.missing = sorted(set(missing))
        chars = ", ".join("U+%04X(%s)" % (c, chr(c))
                          if 0x20 <= c <= 0x10FFFF else "U+%04X" % c
                          for c in self.missing[:20])
        super().__init__("font has no glyph for %d character(s): %s%s" % (
            len(self.missing), chars,
            " ..." if len(self.missing) > 20 else ""))


def _parse_simple_glyph(data, num_contours):
    """Return list of contours; each contour is a list of (x, y, on_curve)."""
    end_pts = [u16(data, 10 + 2 * i) for i in range(num_contours)]
    num_points = end_pts[-1] + 1 if end_pts else 0
    off = 10 + 2 * num_contours
    instr_len = u16(data, off)
    off += 2 + instr_len

    flags = []
    while len(flags) < num_points:
        flag = data[off]
        off += 1
        flags.append(flag)
        if flag & 0x08:  # repeat
            repeat = data[off]
            off += 1
            flags.extend([flag] * repeat)
    flags = flags[:num_points]

    xs, x = [], 0
    for flag in flags:
        if flag & 0x02:  # x-short
            delta = data[off]
            off += 1
            x += delta if (flag & 0x10) else -delta
        elif not (flag & 0x10):
            x += i16(data, off)
            off += 2
        xs.append(x)
    ys, y = [], 0
    for flag in flags:
        if flag & 0x04:  # y-short
            delta = data[off]
            off += 1
            y += delta if (flag & 0x20) else -delta
        elif not (flag & 0x20):
            y += i16(data, off)
            off += 2
        ys.append(y)

    contours = []
    start = 0
    for end in end_pts:
        contours.append([(xs[i], ys[i], bool(flags[i] & 0x01))
                         for i in range(start, end + 1)])
        start = end + 1
    return contours


def _transform(contours, xx, xy, yx, yy, dx, dy):
    return [[(xx * px + xy * py + dx, yx * px + yy * py + dy, on)
             for px, py, on in contour] for contour in contours]


class Renderer:
    def __init__(self, data):
        self.font = Font(data)

    # -- outline extraction -------------------------------------------------

    def glyph_outline(self, gid, depth=0):
        """List of contours, each a list of (x, y, on_curve) in font units."""
        if depth > 16:
            raise ValueError("composite nesting too deep")
        data = self.font.glyph_data(gid)
        if len(data) < 10:
            return []
        num_contours = i16(data, 0)
        if num_contours >= 0:
            return _parse_simple_glyph(data, num_contours)
        contours = []
        off = 10
        while True:
            flags = u16(data, off)
            comp_gid = u16(data, off + 2)
            off += 4
            if flags & ARG_1_AND_2_ARE_WORDS:
                arg1, arg2 = i16(data, off), i16(data, off + 2)
                off += 4
            else:
                arg1 = struct.unpack_from(">b", data, off)[0]
                arg2 = struct.unpack_from(">b", data, off + 1)[0]
                off += 2
            if not (flags & ARGS_ARE_XY_VALUES):
                raise NotImplementedError(
                    "point-matched composite components not supported")
            dx, dy = arg1, arg2
            xx = yy = 1.0
            xy = yx = 0.0
            if flags & WE_HAVE_A_SCALE:
                xx = yy = i16(data, off) / 16384.0
                off += 2
            elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
                xx = i16(data, off) / 16384.0
                yy = i16(data, off + 2) / 16384.0
                off += 4
            elif flags & WE_HAVE_A_TWO_BY_TWO:
                xx = i16(data, off) / 16384.0
                yx = i16(data, off + 2) / 16384.0
                xy = i16(data, off + 4) / 16384.0
                yy = i16(data, off + 6) / 16384.0
                off += 8
            sub = self.glyph_outline(comp_gid, depth + 1)
            contours.extend(_transform(sub, xx, xy, yx, yy, dx, dy))
            if not (flags & MORE_COMPONENTS):
                break
        return contours

    # -- rendering ----------------------------------------------------------

    def text_to_gids(self, text):
        """Map text to glyph ids. Raises MissingGlyphError for any
        character the font does not cover (never falls back silently)."""
        missing = [ord(c) for c in text if ord(c) not in self.font.cmap]
        if missing:
            raise MissingGlyphError(missing)
        return [self.font.cmap[ord(c)] for c in text]

    def render_text(self, text, px_size=32, supersample=4):
        """Render text to an 8-bit grayscale bitmap.
        Returns (width, height, bytes). Raises MissingGlyphError if any
        character is not covered by the font."""
        gids = self.text_to_gids(text)
        font = self.font
        scale = px_size / font.units_per_em
        pad = 2
        y_base = font.ascent * scale + pad

        polygons = []
        pen_x = float(pad)
        for gid in gids:
            for contour in self.glyph_outline(gid):
                poly = _flatten_contour(contour)
                if len(poly) >= 2:
                    polygons.append([(pen_x + x * scale, y_base - y * scale)
                                     for x, y in poly])
            pen_x += font.metrics[gid][0] * scale

        width = max(1, int(pen_x + pad + 0.9999))
        height = max(1, int((font.ascent - font.descent) * scale
                            + 2 * pad + 0.9999))
        bitmap = _rasterize(polygons, width, height, supersample)
        return width, height, bitmap


# ---------------------------------------------------------------------------
# curve flattening

def _flatten_contour(contour, steps=10):
    """Flatten a TrueType quadratic contour to a polygon point list."""
    pts = list(contour)
    # Insert implied on-curve points between consecutive off-curve points.
    expanded = []
    n = len(pts)
    for i in range(n):
        x1, y1, on1 = pts[i]
        x2, y2, on2 = pts[(i + 1) % n]
        expanded.append((x1, y1, on1))
        if not on1 and not on2:
            expanded.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, True))
    # Rotate so we start on an on-curve point.
    start = next(i for i, p in enumerate(expanded) if p[2])
    expanded = expanded[start:] + expanded[:start]

    poly = []
    i = 0
    n = len(expanded)
    while i < n:
        x, y, on = expanded[i]
        if on:
            poly.append((x, y))
            i += 1
        else:
            nx, ny, _ = expanded[(i + 1) % n]
            cx, cy = x, y
            x0, y0 = poly[-1]
            for s in range(1, steps + 1):
                t = s / steps
                mt = 1.0 - t
                qx = mt * mt * x0 + 2 * mt * t * cx + t * t * nx
                qy = mt * mt * y0 + 2 * mt * t * cy + t * t * ny
                poly.append((qx, qy))
            i += 1
    return poly


# ---------------------------------------------------------------------------
# rasterizer

def _rasterize(polygons, width, height, ss):
    """Non-zero winding scanline fill with ss x ss supersampling."""
    accum = [[0] * width for _ in range(height)]
    edges = []
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if y1 != y2:
                edges.append((x1, y1, x2, y2))
    for sy in range(height * ss):
        y = (sy + 0.5) / ss
        crossings = []
        for x1, y1, x2, y2 in edges:
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                crossings.append((x1 + t * (x2 - x1), 1 if y2 > y1 else -1))
        crossings.sort()
        winding = 0
        prev_x = 0.0
        spans = []
        for x, delta in crossings:
            if winding != 0:
                spans.append((prev_x, x))
            winding += delta
            prev_x = x
        row = sy // ss
        for x0, x1 in spans:
            sx0 = max(0, int(x0 * ss))
            sx1 = min(width * ss, int(x1 * ss + 0.9999))
            for sx in range(sx0, sx1):
                if (sx + 0.5) / ss >= x0 and (sx + 0.5) / ss < x1:
                    accum[row][sx // ss] += 1
    total = ss * ss
    out = bytearray(width * height)
    for r in range(height):
        base = r * width
        for c in range(width):
            out[base + c] = min(255, accum[r][c] * 255 // total)
    return bytes(out)


# ---------------------------------------------------------------------------
# PNG output (for eyeballing comparison results)

def save_png(path, width, height, gray):
    """Write an 8-bit grayscale PNG (uses only zlib/struct)."""
    raw = b"".join(b"\x00" + gray[r * width:(r + 1) * width]
                   for r in range(height))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(
            ">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0,
                                        0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)


def diff_bitmap(a, b):
    """Return (identical, num_different_pixels, max_abs_diff)."""
    if len(a) != len(b):
        return False, -1, -1
    diffs = 0
    max_diff = 0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d:
            diffs += 1
            max_diff = max(max_diff, d)
    return diffs == 0, diffs, max_diff
