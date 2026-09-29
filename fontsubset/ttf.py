"""Minimal TrueType (sfnt) parser and writer. Pure standard library.

Supports the tables needed for subsetting and rendering:
head, hhea, maxp, hmtx, loca, glyf, cmap (format 4 and 12), name, post, OS/2.
"""

import struct

# Composite glyph flags (glyf table)
ARG_1_AND_2_ARE_WORDS = 0x0001
ARGS_ARE_XY_VALUES = 0x0002
WE_HAVE_A_SCALE = 0x0008
MORE_COMPONENTS = 0x0020
WE_HAVE_AN_X_AND_Y_SCALE = 0x0040
WE_HAVE_A_TWO_BY_TWO = 0x0080


def u16(buf, off):
    return struct.unpack_from(">H", buf, off)[0]


def i16(buf, off):
    return struct.unpack_from(">h", buf, off)[0]


def u32(buf, off):
    return struct.unpack_from(">I", buf, off)[0]


def calc_checksum(data):
    """Table checksum: sum of big-endian uint32 words, zero-padded to 4."""
    pad = (-len(data)) % 4
    if pad:
        data = data + b"\0" * pad
    total = 0
    for i in range(0, len(data), 4):
        total = (total + struct.unpack_from(">I", data, i)[0]) & 0xFFFFFFFF
    return total


class FontError(Exception):
    pass


class Font:
    """Parsed TrueType font (read-only view over the original bytes)."""

    def __init__(self, data):
        self.data = data
        if len(data) < 12:
            raise FontError("file too small to be a font")
        self.sfnt_version = u32(data, 0)
        if self.sfnt_version not in (0x00010000, 0x74727565):  # 1.0 / 'true'
            raise FontError("not a TrueType font (sfnt version 0x%08X)"
                            % self.sfnt_version)
        num_tables = u16(data, 4)
        self.tables = {}
        for i in range(num_tables):
            off = 12 + 16 * i
            tag = data[off:off + 4].decode("latin1")
            offset = u32(data, off + 8)
            length = u32(data, off + 12)
            self.tables[tag] = data[offset:offset + length]
        for required in ("head", "hhea", "maxp", "hmtx", "loca", "glyf", "cmap"):
            if required not in self.tables:
                raise FontError("missing required table: %s" % required)
        self._parse()

    def _parse(self):
        head = self.tables["head"]
        self.units_per_em = u16(head, 18)
        self.index_to_loc_format = i16(head, 50)
        self.ascent = i16(self.tables["hhea"], 4)
        self.descent = i16(self.tables["hhea"], 6)
        self.num_glyphs = u16(self.tables["maxp"], 4)
        self.num_hmetrics = u16(self.tables["hhea"], 34)

        hmtx = self.tables["hmtx"]
        self.metrics = []  # per-glyph [advanceWidth, lsb]
        off = 0
        for _ in range(self.num_hmetrics):
            self.metrics.append([u16(hmtx, off), i16(hmtx, off + 2)])
            off += 4
        last_adv = self.metrics[-1][0]
        for _ in range(self.num_glyphs - self.num_hmetrics):
            self.metrics.append([last_adv, i16(hmtx, off)])
            off += 2

        loca = self.tables["loca"]
        if self.index_to_loc_format == 0:
            self.offsets = [u16(loca, 2 * i) * 2
                            for i in range(self.num_glyphs + 1)]
        else:
            self.offsets = [u32(loca, 4 * i)
                            for i in range(self.num_glyphs + 1)]
        self.glyf = self.tables["glyf"]
        self.cmap = parse_cmap(self.tables["cmap"])

    def glyph_data(self, gid):
        return self.glyf[self.offsets[gid]:self.offsets[gid + 1]]

    def glyph_count_chars(self):
        return len(self.cmap)


# ---------------------------------------------------------------------------
# cmap

def _parse_cmap_format4(data, off):
    length = u16(data, off + 2)
    seg_count = u16(data, off + 6) // 2
    p = off + 14
    end_codes = [u16(data, p + 2 * i) for i in range(seg_count)]
    p += 2 * seg_count + 2  # skip reservedPad
    start_codes = [u16(data, p + 2 * i) for i in range(seg_count)]
    p += 2 * seg_count
    id_deltas = [i16(data, p + 2 * i) for i in range(seg_count)]
    p += 2 * seg_count
    roff_base = p
    id_range_offsets = [u16(data, p + 2 * i) for i in range(seg_count)]
    mapping = {}
    for i in range(seg_count):
        start, end = start_codes[i], end_codes[i]
        if start == 0xFFFF and end == 0xFFFF:
            continue
        for c in range(start, end + 1):
            if id_range_offsets[i] == 0:
                gid = (c + id_deltas[i]) & 0xFFFF
            else:
                addr = (roff_base + 2 * i + id_range_offsets[i]
                        + 2 * (c - start))
                if addr + 2 > off + length:
                    continue
                gid = u16(data, addr)
                if gid:
                    gid = (gid + id_deltas[i]) & 0xFFFF
            if gid:
                mapping[c] = gid
    return mapping


def _parse_cmap_format12(data, off):
    n_groups = u32(data, off + 12)
    mapping = {}
    p = off + 16
    for _ in range(n_groups):
        start_char = u32(data, p)
        end_char = u32(data, p + 4)
        start_gid = u32(data, p + 8)
        p += 12
        for c in range(start_char, end_char + 1):
            mapping[c] = start_gid + (c - start_char)
    return mapping


def parse_cmap(data):
    """Return a {codepoint: glyph_id} dict from the best cmap subtable."""
    num_tables = u16(data, 2)
    best = None  # (priority, format, offset)
    for i in range(num_tables):
        platform = u16(data, 4 + 8 * i)
        encoding = u16(data, 6 + 8 * i)
        offset = u32(data, 8 + 8 * i)
        fmt = u16(data, offset)
        priority = -1
        if fmt == 12 and (platform, encoding) in ((3, 10), (0, 4), (0, 6)):
            priority = 3
        elif fmt == 4 and (platform, encoding) in ((3, 1), (0, 3), (0, 4)):
            priority = 2
        elif fmt == 4:
            priority = 1
        if priority > 0 and (best is None or priority > best[0]):
            best = (priority, fmt, offset)
    if best is None:
        raise FontError("no supported cmap subtable (format 4 or 12)")
    _, fmt, offset = best
    if fmt == 12:
        return _parse_cmap_format12(data, offset)
    return _parse_cmap_format4(data, offset)


def build_cmap_format4(mapping):
    """Build a format 4 subtable. All codepoints must be <= 0xFFFF."""
    chars = sorted(c for c in mapping if c <= 0xFFFF)
    # Merge adjacent codepoints sharing a constant (gid - char) delta.
    segs = []  # [start, end, delta]
    for c in chars:
        delta = (mapping[c] - c) & 0xFFFF
        if segs and c == segs[-1][1] + 1 and segs[-1][2] == delta:
            segs[-1][1] = c
        else:
            segs.append([c, c, delta])
    segs.append([0xFFFF, 0xFFFF, 1])  # required end sentinel
    seg_count = len(segs)
    max_pow = 1 << (seg_count.bit_length() - 1)
    search_range = max_pow * 2
    entry_selector = max_pow.bit_length() - 1
    range_shift = seg_count * 2 - search_range
    end_codes = b"".join(struct.pack(">H", s[1]) for s in segs)
    start_codes = b"".join(struct.pack(">H", s[0]) for s in segs)
    id_deltas = b"".join(struct.pack(">h", s[2] if s[2] < 0x8000
                                     else s[2] - 0x10000) for s in segs)
    id_range_offsets = b"\0\0" * seg_count
    length = 16 + 8 * seg_count
    return struct.pack(">HHHHHHH", 4, length, 0, seg_count * 2,
                       search_range, entry_selector, range_shift) \
        + end_codes + b"\0\0" + start_codes + id_deltas + id_range_offsets


def build_cmap_format12(mapping):
    """Build a format 12 subtable covering all codepoints (incl. > U+FFFF)."""
    chars = sorted(mapping)
    groups = []  # [start_char, end_char, start_gid]
    for c in chars:
        gid = mapping[c]
        if groups and c == groups[-1][1] + 1 \
                and gid == groups[-1][2] + (c - groups[-1][0]):
            groups[-1][1] = c
        else:
            groups.append([c, c, gid])
    body = b"".join(struct.pack(">III", g[0], g[1], g[2]) for g in groups)
    return struct.pack(">HHIII", 12, 0, 16 + 12 * len(groups), 0,
                       len(groups)) + body


def build_cmap(mapping):
    """Build a full cmap table for {codepoint: glyph_id}."""
    subtables = []
    has_non_bmp = any(c > 0xFFFF for c in mapping)
    fmt4 = build_cmap_format4(mapping)
    if has_non_bmp:
        fmt12 = build_cmap_format12(mapping)
        subtables = [((3, 1), fmt4), ((3, 10), fmt12)]
    else:
        subtables = [((3, 1), fmt4)]
    num = len(subtables)
    header = struct.pack(">HH", 0, num)
    offset = 4 + 8 * num
    records = b""
    body = b""
    for (platform, encoding), sub in subtables:
        records += struct.pack(">HHI", platform, encoding, offset)
        body += sub
        offset += len(sub)
    return header + records + body


# ---------------------------------------------------------------------------
# glyf helpers

def iter_composite_components(data):
    """Yield (flags, component_gid, offset_of_gid_field) for a composite
    glyph's raw bytes. Returns None for simple/empty glyphs."""
    if len(data) < 10 or i16(data, 0) >= 0:
        return None
    components = []
    off = 10
    while True:
        flags = u16(data, off)
        gid = u16(data, off + 2)
        components.append((flags, gid, off + 2))
        off += 4
        off += 4 if flags & ARG_1_AND_2_ARE_WORDS else 2
        if flags & WE_HAVE_A_SCALE:
            off += 2
        elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
            off += 4
        elif flags & WE_HAVE_A_TWO_BY_TWO:
            off += 8
        if not (flags & MORE_COMPONENTS):
            break
    return components


# ---------------------------------------------------------------------------
# font writer

def build_font(sfnt_version, tables):
    """Assemble a valid sfnt file from {tag: bytes}. Computes checksums and
    the head.checkSumAdjustment field."""
    tags = sorted(tables)
    num = len(tags)
    max_pow = 1 << (num.bit_length() - 1)
    search_range = max_pow * 16
    entry_selector = max_pow.bit_length() - 1
    range_shift = num * 16 - search_range
    header = struct.pack(">IHHHH", sfnt_version, num, search_range,
                         entry_selector, range_shift)
    offset = 12 + 16 * num
    records = b""
    body = b""
    head_offset = None
    for tag in tags:
        data = tables[tag]
        if tag == "head":
            head_offset = offset
        pad = (-len(data)) % 4
        records += tag.encode("latin1") + struct.pack(
            ">III", calc_checksum(data), offset, len(data))
        body += data + b"\0" * pad
        offset += len(data) + pad
    font = bytearray(header + records + body)
    if head_offset is not None:
        struct.pack_into(">I", font, head_offset + 8, 0)
        adjustment = (0xB1B0AFBA - calc_checksum(bytes(font))) & 0xFFFFFFFF
        struct.pack_into(">I", font, head_offset + 8, adjustment)
    return bytes(font)
