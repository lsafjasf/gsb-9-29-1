#!/usr/bin/env python3
"""Subset a TrueType font to the characters actually used.

Usage:
  python3 subset.py INPUT.ttf OUTPUT.ttf --text "需要显示的文字"
  python3 subset.py INPUT.ttf OUTPUT.ttf --text-file content.txt
  python3 subset.py INPUT.ttf OUTPUT.ttf --all          # keep full charset
  python3 subset.py INPUT.ttf OUTPUT.ttf --text "Hi" --verify --png out.png

Exit code is 2 if any requested character is missing from the font
(the subset is still written, minus the missing characters).
"""

import argparse
import sys

from fontsubset import subset_font, Renderer, MissingGlyphError, \
    diff_bitmap, save_png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="characters to keep")
    src.add_argument("--text-file", help="file whose characters to keep")
    src.add_argument("--all", action="store_true",
                     help="keep the font's full character set")
    ap.add_argument("--verify", action="store_true",
                    help="render the text with original and subset fonts and "
                         "require byte-identical bitmaps")
    ap.add_argument("--px", type=int, default=48, help="render size (px)")
    ap.add_argument("--png", help="write side-by-side render PNGs (PREFIX-orig.png / PREFIX-subset.png)")
    args = ap.parse_args(argv)

    with open(args.input, "rb") as fh:
        data = fh.read()

    if args.all:
        from fontsubset.ttf import Font
        chars = sorted(Font(data).cmap)
    elif args.text_file:
        with open(args.text_file, encoding="utf-8") as fh:
            chars = [ord(c) for c in fh.read()]
    else:
        chars = [ord(c) for c in args.text]

    subset_data, report = subset_font(data, chars)
    print(report.summary())

    if report.missing_chars:
        print("ERROR: %d requested character(s) are not in the font and "
              "were NOT substituted: %s" % (
                  len(report.missing_chars),
                  ", ".join("U+%04X" % c for c in report.missing_chars)),
              file=sys.stderr)

    with open(args.output, "wb") as fh:
        fh.write(subset_data)
    print("wrote %s" % args.output)

    if args.verify or args.png:
        text = "".join(c for c in (chr(c) for c in sorted(set(chars)))
                       if ord(c) in Renderer(data).font.cmap)
        if not text:
            print("nothing renderable to verify (empty subset)")
        else:
            orig = Renderer(data)
            sub = Renderer(subset_data)
            try:
                w1, h1, bmp1 = orig.render_text(text, args.px)
                w2, h2, bmp2 = sub.render_text(text, args.px)
            except MissingGlyphError as exc:
                print("VERIFY FAILED: %s" % exc, file=sys.stderr)
                return 2
            identical, ndiff, maxdiff = diff_bitmap(bmp1, bmp2)
            if (w1, h1) != (w2, h2) or not identical:
                print("VERIFY FAILED: renderings differ "
                      "(size %dx%d vs %dx%d, %d pixels, max diff %d)"
                      % (w1, h1, w2, h2, ndiff, maxdiff), file=sys.stderr)
                return 2
            print("verify ok: %d chars rendered at %dpx, bitmaps "
                  "byte-identical (%dx%d)" % (len(text), args.px, w1, h1))
            if args.png:
                save_png(args.png + "-orig.png", w1, h1, bmp1)
                save_png(args.png + "-subset.png", w2, h2, bmp2)
                print("wrote %s-orig.png / %s-subset.png"
                      % (args.png, args.png))
    return 2 if report.missing_chars else 0


if __name__ == "__main__":
    sys.exit(main())
