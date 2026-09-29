"""Self-tests for the font subsetter.

Covers: basic subsetting, composite-only subsets, empty subset, full
charset, very large charsets, explicit missing-character reporting,
font validity (checksums) and render equivalence. Prints a size
comparison table for all scenarios.

Run:  python3 -m unittest discover -s tests -v
"""

import glob
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fontsubset import (Font, MissingGlyphError, Renderer, diff_bitmap,
                        subset_font)
from fontsubset.ttf import calc_checksum, u16, u32

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
FONT_PATH = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)

SIZE_REPORT = []  # (scenario, original, subset, glyphs, note)


def load_font():
    with open(FONT_PATH, "rb") as fh:
        return fh.read()


def render_both(original, subset, text, px=40):
    r1, r2 = Renderer(original), Renderer(subset)
    w1, h1, b1 = r1.render_text(text, px)
    w2, h2, b2 = r2.render_text(text, px)
    return (w1, h1, b1), (w2, h2, b2)


def assert_render_identical(testcase, original, subset, text, px=40):
    (w1, h1, b1), (w2, h2, b2) = render_both(original, subset, text, px)
    testcase.assertEqual((w1, h1), (w2, h2), "bitmap dimensions differ")
    identical, ndiff, maxdiff = diff_bitmap(b1, b2)
    testcase.assertTrue(
        identical, "renderings differ: %d pixels, max diff %d"
        % (ndiff, maxdiff))


@unittest.skipUnless(FONT_PATH, "no system TrueType font found")
class SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_font()
        cls.font = Font(cls.data)

    def tearDown(self):
        # every scenario contributes to the size comparison table
        rep = getattr(self, "_size_entry", None)
        if rep:
            SIZE_REPORT.append(rep)
            self._size_entry = None

    def record_size(self, scenario, subset_data, glyphs, note=""):
        self._size_entry = (scenario, len(self.data), len(subset_data),
                            glyphs, note)

    # -- basic behaviour ----------------------------------------------------

    def test_basic_subset_render_identical(self):
        text = "Hello, World! 123"
        chars = [ord(c) for c in text]
        sub, report = subset_font(self.data, chars)
        self.assertEqual(report.missing_chars, [])
        self.assertLess(len(sub), len(self.data) // 10)
        assert_render_identical(self, self.data, sub, text)
        self.record_size("basic 'Hello, World! 123'", sub,
                         report.glyphs_kept)

    def test_subset_font_is_valid(self):
        sub, _ = subset_font(self.data, [ord(c) for c in "Valid?"])
        font = Font(sub)  # must parse
        # table checksums in the directory must match the table data
        num_tables = u16(sub, 4)
        for i in range(num_tables):
            off = 12 + 16 * i
            tag = sub[off:off + 4]
            checksum = u32(sub, off + 4)
            offset = u32(sub, off + 8)
            length = u32(sub, off + 12)
            table = sub[offset:offset + length]
            if tag == b"head":
                # per spec, the head checksum is computed with
                # checkSumAdjustment (bytes 8..12) set to zero
                table = table[:8] + b"\0\0\0\0" + table[12:]
            self.assertEqual(checksum,
                             calc_checksum(table),
                             "bad checksum for table %r" % tag)
        # whole-font checksum must equal the magic constant
        self.assertEqual(calc_checksum(sub), 0xB1B0AFBA,
                         "checkSumAdjustment is wrong")
        self.assertGreater(font.num_glyphs, 1)

    def test_metrics_preserved(self):
        text = "AVWxy"
        sub, _ = subset_font(self.data, [ord(c) for c in text])
        orig, sub_font = Font(self.data), Font(sub)
        for c in text:
            gid_o = orig.cmap[ord(c)]
            gid_s = sub_font.cmap[ord(c)]
            self.assertEqual(orig.metrics[gid_o], sub_font.metrics[gid_s],
                             "metrics changed for %r" % c)

    # -- composite glyphs ---------------------------------------------------

    def test_composite_only_subset(self):
        # Accented latin letters are composites (base letter + accent).
        text = "ÀÉÎÕÜàéîõüÄÖäöüÿ"
        chars = [ord(c) for c in text]
        sub, report = subset_font(self.data, chars)
        self.assertEqual(report.missing_chars, [])
        # composite dependencies must have been pulled in
        self.assertGreater(report.composite_components_pulled, 0)
        # the subset must contain the base glyphs too
        sub_font = Font(sub)
        self.assertGreater(sub_font.num_glyphs, len(set(chars)))
        assert_render_identical(self, self.data, sub, text)
        self.record_size("composite-only (%d accented chars)" % len(text),
                         sub, report.glyphs_kept,
                         "+%d component glyphs" %
                         report.composite_components_pulled)

    def test_composite_component_ids_remapped(self):
        # Every component reference in the subset must point at a glyph
        # that exists in the subset.
        from fontsubset.ttf import iter_composite_components
        sub, _ = subset_font(self.data, [ord(c) for c in "ÀÉÎÕÜ"])
        font = Font(sub)
        for gid in range(font.num_glyphs):
            comps = iter_composite_components(font.glyph_data(gid))
            if comps:
                for _flags, comp_gid, _off in comps:
                    self.assertLess(comp_gid, font.num_glyphs)

    # -- empty subset -------------------------------------------------------

    def test_empty_subset(self):
        sub, report = subset_font(self.data, [])
        self.assertEqual(report.missing_chars, [])
        font = Font(sub)
        self.assertEqual(font.num_glyphs, 1)  # only .notdef
        self.assertEqual(len(font.cmap), 0)
        self.assertEqual(calc_checksum(sub), 0xB1B0AFBA)
        # rendering any character must fail loudly
        with self.assertRaises(MissingGlyphError):
            Renderer(sub).render_text("A")
        # rendering empty text works and yields an empty bitmap
        w, h, bmp = Renderer(sub).render_text("")
        self.assertTrue(all(px == 0 for px in bmp))
        self.record_size("empty subset", sub, 1, "only .notdef")

    # -- full character set -------------------------------------------------

    def test_full_charset(self):
        all_chars = sorted(self.font.cmap)
        sub, report = subset_font(self.data, all_chars)
        self.assertEqual(report.missing_chars, [])
        sub_font = Font(sub)
        # every cmap-reachable glyph (plus composite deps) must survive;
        # unencoded GSUB-only glyphs (ligatures etc.) are dropped by design
        self.assertEqual(sub_font.num_glyphs, report.glyphs_kept)
        self.assertLessEqual(sub_font.num_glyphs, self.font.num_glyphs)
        # same character coverage (glyph ids are renumbered, so compare keys)
        self.assertEqual(sorted(sub_font.cmap), sorted(self.font.cmap))
        sample = "The quick brown fox ÀÉÎÕÜ 0123456789"
        sample = "".join(c for c in sample if ord(c) in self.font.cmap)
        assert_render_identical(self, self.data, sub, sample)
        self.record_size("full charset (%d chars)" % len(all_chars), sub,
                         report.glyphs_kept)

    # -- huge character set -------------------------------------------------

    def test_huge_charset(self):
        # Every BMP codepoint the font covers, plus a large tail of
        # (missing) supplementary-plane codepoints to stress reporting.
        bmp_chars = [c for c in range(0x20, 0x10000)
                     if c in self.font.cmap]
        extra = [c for c in range(0x1F600, 0x1F600 + 5000)
                 if c not in self.font.cmap]  # absent supplementary chars
        self.assertGreater(len(extra), 1000)
        sub, report = subset_font(self.data, bmp_chars + extra)
        self.assertEqual(report.mapped_chars, len(bmp_chars))
        self.assertEqual(report.missing_chars, extra)
        self.assertGreater(report.glyphs_kept, 1000)
        sample = "".join(chr(c) for c in bmp_chars[:200])
        assert_render_identical(self, self.data, sub, sample, px=24)
        self.record_size("huge charset (%d chars)" % len(bmp_chars), sub,
                         report.glyphs_kept,
                         "%d missing reported" % len(extra))

    # -- missing characters -------------------------------------------------

    def test_missing_chars_reported_not_substituted(self):
        # DejaVu has no CJK ideographs; U+10FFFE is a noncharacter.
        chars = [ord("A"), 0x4E2D, 0x6587, 0x10FFFE]
        sub, report = subset_font(self.data, chars)
        self.assertEqual(report.missing_chars, [0x4E2D, 0x6587, 0x10FFFE])
        sub_font = Font(sub)
        self.assertIn(ord("A"), sub_font.cmap)
        for c in (0x4E2D, 0x6587, 0x10FFFE):
            self.assertNotIn(c, sub_font.cmap)
        # renderer must raise, never fall back to .notdef
        with self.assertRaises(MissingGlyphError) as ctx:
            Renderer(sub).render_text("A中文")
        self.assertEqual(ctx.exception.missing, [0x4E2D, 0x6587])
        with self.assertRaises(MissingGlyphError):
            Renderer(self.data).render_text("中文")
        self.record_size("missing chars (1 of 4 present)", sub,
                         report.glyphs_kept, "3 missing reported")

    def test_missing_glyph_error_message(self):
        try:
            Renderer(self.data).render_text("中")
        except MissingGlyphError as exc:
            self.assertIn("U+4E2D", str(exc))
        else:
            self.fail("expected MissingGlyphError")

    # -- non-BMP ------------------------------------------------------------

    def test_non_bmp_chars(self):
        # Mathematical alphanumeric symbols (U+1D400+) exist in DejaVu Sans.
        plane1 = [c for c in range(0x1D400, 0x1D800)
                  if c in self.font.cmap]
        if not plane1:
            self.skipTest("font has no plane-1 characters")
        text = "".join(chr(c) for c in plane1[:20])
        sub, report = subset_font(self.data, plane1)
        self.assertEqual(report.missing_chars, [])
        font = Font(sub)
        for c in plane1:
            self.assertIn(c, font.cmap)
        assert_render_identical(self, self.data, sub, text)
        self.record_size("non-BMP (%d plane-1 chars)" % len(plane1), sub,
                         report.glyphs_kept)

    # -- idempotence --------------------------------------------------------

    def test_subset_of_subset(self):
        text = "Reentrant!"
        chars = [ord(c) for c in text]
        sub1, _ = subset_font(self.data, chars)
        sub2, report = subset_font(sub1, chars)
        self.assertEqual(report.missing_chars, [])
        assert_render_identical(self, self.data, sub2, text)


def print_size_report():
    if not SIZE_REPORT:
        return
    print("\n" + "=" * 78)
    print("SIZE COMPARISON  (font: %s)" % os.path.basename(FONT_PATH))
    print("=" * 78)
    print("%-38s %10s %10s %7s  %s" % ("scenario", "original", "subset",
                                       "glyphs", "note"))
    print("-" * 78)
    for scenario, orig, sub, glyphs, note in SIZE_REPORT:
        pct = "(%5.1f%%)" % (100.0 * sub / orig) if orig else ""
        print("%-38s %10d %7d %-8s %6d  %s"
              % (scenario, orig, sub, pct, glyphs, note))
    print("=" * 78)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    print_size_report()
