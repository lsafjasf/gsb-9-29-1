#!/usr/bin/env python3
"""Align every data/*.en.txt + *.zh.txt pair and score against gold.

Usage:  python3 run_demo.py [data_dir]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentalign import split_sentences, align
from sentalign.evaluate import load_gold, evaluate_with_confidence


def main(data_dir="data"):
    names = sorted({f.rsplit(".", 2)[0] for f in os.listdir(data_dir)
                    if f.endswith(".en.txt")})
    totals = {"gold": 0, "correct": 0}
    err_all = {}
    for name in names:
        with open(os.path.join(data_dir, name + ".en.txt"),
                  encoding="utf-8") as fh:
            src = split_sentences(fh.read())
        with open(os.path.join(data_dir, name + ".zh.txt"),
                  encoding="utf-8") as fh:
            tgt = split_sentences(fh.read())
        gold = load_gold(os.path.join(data_dir, name + ".gold.tsv"))

        aln = align(src, tgt)
        print("=" * 72)
        print("%s  (%d src sentences, %d tgt sentences)" % (name, len(src), len(tgt)))
        print("-" * 72)
        print(aln.path())
        res = evaluate_with_confidence(aln, gold)
        print("-" * 72)
        print("accuracy: %d/%d = %.1f%%   tentative beads: %d (hits: %d)"
              % (res["correct"], res["gold_beads"], 100 * res["accuracy"],
                 res["tentative_count"], res["tentative_hits"]))
        cf = res["confident"]
        print("confident-only accuracy: %d/%d = %.1f%%"
              % (cf["correct"], cf["gold_beads"], 100 * cf["accuracy"]))
        if res["errors"]:
            print("errors: " + ", ".join("%s=%d" % kv
                                         for kv in sorted(res["errors"].items())))
        totals["gold"] += res["gold_beads"]
        totals["correct"] += res["correct"]
        for k, v in res["errors"].items():
            err_all[k] = err_all.get(k, 0) + v
        print()

    print("=" * 72)
    print("OVERALL accuracy: %d/%d = %.1f%%"
          % (totals["correct"], totals["gold"],
             100 * totals["correct"] / max(totals["gold"], 1)))
    print("error distribution: " +
          (", ".join("%s=%d" % kv for kv in sorted(err_all.items()))
           if err_all else "none"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
