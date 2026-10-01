"""RIF -- Row-filtered Image Format: a tiny lossless image codec (stdlib only).

Pipeline
--------
1. Per-row adaptive filtering: each row of raw pixels is transformed by one of
   five PNG-style filters (None / Sub / Up / Average / Paeth). The encoder
   tries every filter on every row and picks the best one (see "Filter
   selection criterion" below).
2. Row packing: the filtered rows are concatenated, each prefixed by a single
   filter-type byte:  [ftype][row bytes][ftype][row bytes]...
3. Entropy coding: the packed stream is compressed with zlib (DEFLATE).

Because packing is byte-oriented per row, any width works -- widths that are
not a multiple of 8 need no padding and are handled exactly like any other.

Filter selection criterion
--------------------------
For each candidate filter the filtered row is scored with

    score = sum(abs(signed_byte(b)) for b in filtered_row)

i.e. each byte is interpreted as a signed value in [-128, 127] and the sum of
absolute values (the L1 norm) is taken. This is the classic "minimum sum of
absolute differences" (MSAD) heuristic used by PNG encoders: small-magnitude
residuals correlate with low zero-order entropy, so the row that DEFLATE can
compress best is (almost always) the one with the smallest L1 score. Ties are
broken toward the lower filter number.

File layout (all integers big-endian)
-------------------------------------
    offset  size  field
    0       4     magic "RIF1"
    4       4     width   (uint32, >= 1)
    8       4     height  (uint32, >= 1)
    12      1     channels (1=gray, 2=GA, 3=RGB, 4=RGBA), 8 bits per channel
    13      1     flags (reserved, 0)
    14      ..    zlib-compressed packed row stream
"""

from __future__ import annotations

import struct
import zlib

MAGIC = b"RIF1"

NONE, SUB, UP, AVERAGE, PAETH = range(5)
FILTER_NAMES = ("None", "Sub", "Up", "Average", "Paeth")

_HEADER = struct.Struct(">4sIIBB")  # magic, width, height, channels, flags

# abs() of a byte reinterpreted as a signed value: b -> |b if b < 128 else b - 256|
_ABS_SIGNED = bytes(b if b < 128 else 256 - b for b in range(256))

VALID_CHANNELS = (1, 2, 3, 4)


class FormatError(Exception):
    """Raised when encoded data is malformed or codec parameters are invalid."""


class FilterStats:
    """Per-filter row-selection counters collected during encoding."""

    def __init__(self) -> None:
        self.counts = [0, 0, 0, 0, 0]
        self.rows = 0

    def add(self, ftype: int) -> None:
        self.counts[ftype] += 1
        self.rows += 1

    def distribution(self) -> dict:
        """{filter_name: (row_count, fraction)} for every filter."""
        return {
            FILTER_NAMES[i]: (c, (c / self.rows) if self.rows else 0.0)
            for i, c in enumerate(self.counts)
        }

    def __str__(self) -> str:
        parts = [
            f"{FILTER_NAMES[i]}={c} ({(c / self.rows * 100) if self.rows else 0:.1f}%)"
            for i, c in enumerate(self.counts)
        ]
        return f"rows={self.rows}  " + "  ".join(parts)


# ---------------------------------------------------------------------------
# Filters (forward). `cur`/`prev` are raw (unfiltered) rows, `bpp` is bytes
# per pixel. All arithmetic is mod 256, exactly like PNG.
# ---------------------------------------------------------------------------

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _filter_none(cur, prev, bpp):
    return bytes(cur[:])


def _filter_sub(cur, prev, bpp):
    return bytes(cur[:bpp]) + bytes(
        [(x - y) & 0xFF for x, y in zip(cur[bpp:], cur)]
    )


def _filter_up(cur, prev, bpp):
    return bytes([(x - y) & 0xFF for x, y in zip(cur, prev)])


def _filter_average(cur, prev, bpp):
    head = bytes([(x - (p >> 1)) & 0xFF for x, p in zip(cur[:bpp], prev[:bpp])])
    tail = bytes(
        [(x - ((a + p) >> 1)) & 0xFF for x, a, p in zip(cur[bpp:], cur, prev[bpp:])]
    )
    return head + tail


def _filter_paeth(cur, prev, bpp):
    # first pixel: a = c = 0, so the Paeth predictor reduces to b (== Up)
    head = bytes([(x - p) & 0xFF for x, p in zip(cur[:bpp], prev[:bpp])])
    tail = bytes(
        [
            (x - _paeth(a, b, c)) & 0xFF
            for x, a, b, c in zip(cur[bpp:], cur, prev[bpp:], prev)
        ]
    )
    return head + tail


_FILTERS = (_filter_none, _filter_sub, _filter_up, _filter_average, _filter_paeth)


# ---------------------------------------------------------------------------
# Reconstructors (inverse filters). `filt` is the filtered row, `prev` the
# already-reconstructed row above.
# ---------------------------------------------------------------------------

def _recon_none(filt, prev, bpp):
    return bytes(filt[:])


def _recon_sub(filt, prev, bpp):
    out = bytearray(filt)
    for i in range(bpp, len(out)):
        out[i] = (out[i] + out[i - bpp]) & 0xFF
    return bytes(out)


def _recon_up(filt, prev, bpp):
    return bytes([(x + y) & 0xFF for x, y in zip(filt, prev)])


def _recon_average(filt, prev, bpp):
    out = bytearray(filt)
    for i in range(bpp):
        out[i] = (out[i] + (prev[i] >> 1)) & 0xFF
    for i in range(bpp, len(out)):
        out[i] = (out[i] + ((out[i - bpp] + prev[i]) >> 1)) & 0xFF
    return bytes(out)


def _recon_paeth(filt, prev, bpp):
    out = bytearray(filt)
    for i in range(bpp):
        out[i] = (out[i] + prev[i]) & 0xFF
    for i in range(bpp, len(out)):
        out[i] = (out[i] + _paeth(out[i - bpp], prev[i], prev[i - bpp])) & 0xFF
    return bytes(out)


_RECONS = (_recon_none, _recon_sub, _recon_up, _recon_average, _recon_paeth)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _validate(pixels, width, height, channels) -> None:
    if not isinstance(width, int) or not isinstance(height, int):
        raise FormatError("width/height must be integers")
    if width < 1 or height < 1:
        raise FormatError("width and height must be >= 1")
    if width > 0xFFFFFFFF or height > 0xFFFFFFFF:
        raise FormatError("width/height too large (uint32)")
    if channels not in VALID_CHANNELS:
        raise FormatError(f"channels must be one of {VALID_CHANNELS}")
    expected = width * height * channels
    if len(pixels) != expected:
        raise FormatError(
            f"pixel buffer has {len(pixels)} bytes, expected {expected} "
            f"({width}x{height}x{channels})"
        )


def encode_with_stats(pixels, width, height, channels, level=9):
    """Encode raw 8-bit pixels. Returns (encoded_bytes, FilterStats)."""
    _validate(pixels, width, height, channels)
    row_len = width * channels
    prev = bytes(row_len)  # virtual all-zero row above the first row
    packed = bytearray()
    stats = FilterStats()
    for y in range(height):
        cur = pixels[y * row_len : (y + 1) * row_len]
        best_type = NONE
        best_buf = None
        best_score = None
        for ftype, fn in enumerate(_FILTERS):
            buf = fn(cur, prev, channels)
            score = sum(buf.translate(_ABS_SIGNED))  # L1 norm of signed bytes
            if best_score is None or score < best_score:
                best_type, best_buf, best_score = ftype, buf, score
        packed.append(best_type)
        packed += best_buf
        stats.add(best_type)
        prev = cur
    header = _HEADER.pack(MAGIC, width, height, channels, 0)
    return header + zlib.compress(bytes(packed), level), stats


def encode(pixels, width, height, channels, level=9) -> bytes:
    """Encode raw 8-bit pixels, returning the compressed byte stream."""
    return encode_with_stats(pixels, width, height, channels, level)[0]


def decode(data):
    """Decode a RIF stream. Returns (pixels, width, height, channels)."""
    if len(data) < _HEADER.size:
        raise FormatError("data too short for header")
    magic, width, height, channels, _flags = _HEADER.unpack_from(data)
    if magic != MAGIC:
        raise FormatError(f"bad magic {magic!r}")
    if channels not in VALID_CHANNELS:
        raise FormatError(f"invalid channel count {channels}")
    if width < 1 or height < 1:
        raise FormatError("width and height must be >= 1")
    row_len = width * channels
    try:
        packed = zlib.decompress(data[_HEADER.size :])
    except zlib.error as exc:
        raise FormatError(f"zlib payload corrupt: {exc}") from exc
    expected = height * (1 + row_len)
    if len(packed) != expected:
        raise FormatError(
            f"packed stream has {len(packed)} bytes, expected {expected}"
        )
    out = bytearray(height * row_len)
    prev = bytes(row_len)
    pos = 0
    for y in range(height):
        ftype = packed[pos]
        pos += 1
        if ftype > PAETH:
            raise FormatError(f"unknown filter type {ftype} on row {y}")
        filt = packed[pos : pos + row_len]
        pos += row_len
        row = _RECONS[ftype](filt, prev, channels)
        out[y * row_len : (y + 1) * row_len] = row
        prev = row
    return bytes(out), width, height, channels


# ---------------------------------------------------------------------------
# Minimal command-line interface for raw 8-bit pixel files.
# ---------------------------------------------------------------------------

def _main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    usage = (
        "usage:\n"
        "  python3 imgcodec.py encode WIDTH HEIGHT CHANNELS in.raw out.rif\n"
        "  python3 imgcodec.py decode in.rif out.raw\n"
        "  python3 imgcodec.py info   in.rif\n"
    )
    if len(argv) >= 1 and argv[0] == "encode" and len(argv) == 6:
        width, height, channels = (int(v) for v in argv[1:4])
        with open(argv[4], "rb") as f:
            pixels = f.read()
        data, stats = encode_with_stats(pixels, width, height, channels)
        with open(argv[5], "wb") as f:
            f.write(data)
        print(f"encoded {width}x{height}x{channels}: "
              f"{len(pixels)} B -> {len(data)} B "
              f"({len(data) / len(pixels):.4f})  {stats}")
        return 0
    if len(argv) >= 1 and argv[0] == "decode" and len(argv) == 3:
        with open(argv[1], "rb") as f:
            pixels, width, height, channels = decode(f.read())
        with open(argv[2], "wb") as f:
            f.write(pixels)
        print(f"decoded {width}x{height}x{channels}: {len(pixels)} B -> {argv[2]}")
        return 0
    if len(argv) >= 1 and argv[0] == "info" and len(argv) == 2:
        with open(argv[1], "rb") as f:
            data = f.read()
        magic, width, height, channels, flags = _HEADER.unpack_from(data)
        print(f"magic={magic.decode()} {width}x{height}x{channels} "
              f"flags={flags} file={len(data)} B")
        return 0
    print(usage)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
