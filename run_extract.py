#!/usr/bin/env python3
"""CLI: extract terms from one or more documents.

Usage:
    python run_extract.py FILE [FILE ...] [--top K] [--min-freq N]
                          [--window W] [--candidates] [--json]
"""
import argparse
import json
import sys

from term_extract import TermExtractor


def main(argv=None):
    ap = argparse.ArgumentParser(description="Domain term extraction")
    ap.add_argument("files", nargs="+", help="UTF-8 text files")
    ap.add_argument("--top", type=int, default=30, help="keep top-k terms")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--window", type=int, default=2,
                    help="context window for boundary/completeness features")
    ap.add_argument("--candidates", action="store_true",
                    help="show pre-merge ranked candidates instead")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    docs = []
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            docs.append(fh.read())

    ex = TermExtractor(min_freq=args.min_freq, window=args.window,
                       top_k=args.top)

    if args.candidates:
        cands = ex.extract_candidates(docs)[: args.top]
        rows = [{
            "candidate": c.surface(), "freq": c.freq,
            "cvalue": round(c.cvalue, 3),
            "left_entropy": round(c.left_entropy, 3),
            "right_entropy": round(c.right_entropy, 3),
            "stickiness": round(c.stickiness, 3),
            "complete": c.complete, "score": round(c.score, 4),
        } for c in cands]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            print(f"{'candidate':<38}{'fq':>4}{'cval':>8}{'lent':>7}"
                  f"{'rent':>7}{'stick':>7}{'full':>6}{'score':>8}")
            for r in rows:
                print(f"{r['candidate']:<38}{r['freq']:>4}{r['cvalue']:>8}"
                      f"{r['left_entropy']:>7}{r['right_entropy']:>7}"
                      f"{r['stickiness']:>7}"
                      f"{'Y' if r['complete'] else 'frag':>6}{r['score']:>8}")
        return 0

    terms, merge_log = ex.extract(docs)
    if args.json:
        print(json.dumps({"terms": terms, "merge_log": merge_log},
                         ensure_ascii=False, indent=2))
    else:
        print(f"{'term':<38}{'freq':>5}{'score':>9}  variants")
        for t in terms:
            vs = ", ".join(v for v in t["variants"] if v != t["term"])
            print(f"{t['term']:<38}{t['freq']:>5}{t['score']:>9}  {vs}")
        if merge_log:
            print("\nmerge log (归并依据):")
            for line in merge_log:
                print(f"  - {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
