"""Command line interface.

Usage:
    python3 -m edge_detect INPUT.pgm [-o OUT.pgm]
        [--low L --high H] [--sensitivity SENS.csv] [--gt GT.pgm]
"""

import argparse
import sys

from .core import count_edges
from .pgm import edges_to_image, read_pgm, write_pgm
from .pipeline import detect_edges
from .sensitivity import sensitivity, to_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 -m edge_detect",
        description="Canny-style edge detection (stdlib only). "
                    "Thresholds are chosen automatically from image "
                    "statistics unless --low/--high are given.")
    parser.add_argument("input", help="input grayscale image (P2/P5 PGM)")
    parser.add_argument("-o", "--output",
                        help="output edge map PGM (255 = edge, 0 = background)")
    parser.add_argument("--low", type=float, default=None,
                        help="manual low hysteresis threshold")
    parser.add_argument("--high", type=float, default=None,
                        help="manual high hysteresis threshold")
    parser.add_argument("--sensitivity", metavar="CSV",
                        help="write threshold-sensitivity sweep to CSV")
    parser.add_argument("--gt", metavar="PGM",
                        help="ground-truth edge map for miss ratios "
                             "(nonzero = edge); only used with --sensitivity")
    args = parser.parse_args(argv)

    if (args.low is None) != (args.high is None):
        parser.error("--low and --high must be given together")

    rows = read_pgm(args.input)
    h, w = len(rows), len(rows[0])

    edges, (low, high) = detect_edges(rows, args.low, args.high)
    n = count_edges(edges)
    print("image: %dx%d, %d px" % (w, h, w * h))
    print("thresholds: low=%.4f high=%.4f%s"
          % (low, high, " (manual)" if args.low is not None else " (auto)"))
    print("edge pixels: %d (%.3f%%)" % (n, 100.0 * n / (w * h)))

    if args.output:
        write_pgm(args.output, edges_to_image(edges))
        print("wrote edge map: %s" % args.output)

    if args.sensitivity:
        gt = None
        if args.gt:
            gt = [[v > 0 for v in row] for row in read_pgm(args.gt)]
        report = sensitivity(rows, gt=gt)
        with open(args.sensitivity, "w") as f:
            f.write(to_csv(report))
        print("wrote sensitivity data: %s" % args.sensitivity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
