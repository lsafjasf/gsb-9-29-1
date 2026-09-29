"""Font subsetting: keep only the glyphs needed for a given character set.

- Resolves composite glyph dependencies transitively (e.g. U+00C0 'À' pulls
  in the base 'A' and the grave accent glyphs).
- Always keeps glyph 0 (.notdef).
- Missing characters are reported explicitly, never silently remapped.
- Rebuilds glyf/loca/hmtx/cmap/maxp/hhea/head; copies name/OS/2 verbatim;
  rewrites post as format 3.0 (drops glyph names); drops layout tables
  (GSUB/GPOS/kern/...) that would otherwise reference stale glyph ids.
"""

import struct
from dataclasses import dataclass, field

from .ttf import (Font, build_cmap, build_font, iter_composite_components)

COPY_TABLES = ("name", "OS/2")


@dataclass
class SubsetReport:
    requested_chars: int = 0
    missing_chars: list = field(default_factory=list)   # codepoints, sorted
    mapped_chars: int = 0
    glyphs_kept: int = 0          # including .notdef
    composite_components_pulled: int = 0
    original_size: int = 0
    subset_size: int = 0

    @property
    def compression_ratio(self):
        if not self.original_size:
            return 0.0
        return self.subset_size / self.original_size

    def summary(self):
        lines = [
            "requested chars : %d" % self.requested_chars,
            "mapped chars    : %d" % self.mapped_chars,
            "missing chars   : %d%s" % (
                len(self.missing_chars),
                "" if not self.missing_chars else " -> " + ", ".join(
                    "U+%04X" % c for c in self.missing_chars[:20])
                + (" ..." if len(self.missing_chars) > 20 else "")),
            "glyphs kept     : %d (incl. .notdef, +%d composite components)"
            % (self.glyphs_kept, self.composite_components_pulled),
            "size            : %d -> %d bytes (%.1f%%)" % (
                self.original_size, self.subset_size,
                100.0 * self.compression_ratio),
        ]
        return "\n".join(lines)


def _glyph_closure(font, gids):
    """Expand the glyph id set with transitive composite components.
    Returns (closure_set, num_pulled)."""
    closure = set(gids)
    pulled = 0
    stack = list(gids)
    while stack:
        gid = stack.pop()
        components = iter_composite_components(font.glyph_data(gid))
        if not components:
            continue
        for _flags, comp_gid, _off in components:
            if comp_gid not in closure:
                closure.add(comp_gid)
                pulled += 1
                stack.append(comp_gid)
    return closure, pulled


def _patch_composite(data, old_to_new):
    """Rewrite component glyph indices in a composite glyph to new ids."""
    buf = bytearray(data)
    for _flags, comp_gid, field_off in iter_composite_components(data):
        struct.pack_into(">H", buf, field_off, old_to_new[comp_gid])
    return bytes(buf)


def subset_font(data, chars):
    """Subset the font given as bytes to the given iterable of codepoints.

    Returns (subset_bytes, SubsetReport). Characters not present in the
    font's cmap are listed in report.missing_chars and excluded.
    """
    font = Font(data)
    chars = set(chars)
    report = SubsetReport(requested_chars=len(chars),
                          original_size=len(data))

    char_to_gid = {}
    for c in sorted(chars):
        gid = font.cmap.get(c)
        if gid is None:
            report.missing_chars.append(c)
        else:
            char_to_gid[c] = gid
    report.mapped_chars = len(char_to_gid)

    gids = set(char_to_gid.values())
    gids.add(0)  # .notdef
    closure, pulled = _glyph_closure(font, gids)
    report.composite_components_pulled = pulled
    report.glyphs_kept = len(closure)

    ordered_old = sorted(closure)  # .notdef (0) stays glyph 0
    old_to_new = {old: new for new, old in enumerate(ordered_old)}
    num_new = len(ordered_old)

    # glyf + loca (always written in long format)
    glyf_parts = []
    loca = [0]
    for old_gid in ordered_old:
        gdata = font.glyph_data(old_gid)
        if gdata and iter_composite_components(gdata):
            gdata = _patch_composite(gdata, old_to_new)
        pad = (-len(gdata)) % 4
        glyf_parts.append(gdata + b"\0" * pad)
        loca.append(loca[-1] + len(gdata) + pad)
    glyf = b"".join(glyf_parts)
    loca_table = b"".join(struct.pack(">I", o) for o in loca)

    # hmtx: one full record per glyph (numberOfHMetrics == numGlyphs)
    hmtx = b"".join(struct.pack(">Hh", font.metrics[g][0],
                                font.metrics[g][1]) for g in ordered_old)

    # head: long loca format, zeroed checkSumAdjustment (writer recomputes)
    head = bytearray(font.tables["head"])
    struct.pack_into(">I", head, 8, 0)
    struct.pack_into(">h", head, 50, 1)

    # hhea: update numberOfHMetrics
    hhea = bytearray(font.tables["hhea"])
    struct.pack_into(">H", hhea, 34, num_new)

    # maxp: update numGlyphs
    maxp = bytearray(font.tables["maxp"])
    struct.pack_into(">H", maxp, 4, num_new)

    # cmap: remap characters to new glyph ids
    new_mapping = {c: old_to_new[g] for c, g in char_to_gid.items()}
    cmap = build_cmap(new_mapping)

    # post: rewrite as format 3.0 (no glyph names)
    old_post = font.tables.get("post", b"\x00\x03\x00\x00" + b"\0" * 28)
    post = struct.pack(">I", 0x00030000) + old_post[4:32]

    tables = {
        "cmap": cmap,
        "glyf": glyf,
        "head": bytes(head),
        "hhea": bytes(hhea),
        "hmtx": hmtx,
        "loca": loca_table,
        "maxp": bytes(maxp),
        "post": post,
    }
    for tag in COPY_TABLES:
        if tag in font.tables:
            tables[tag] = font.tables[tag]

    out = build_font(font.sfnt_version, tables)
    report.subset_size = len(out)
    return out, report
