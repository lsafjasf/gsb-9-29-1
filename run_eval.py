#!/usr/bin/env python3
"""Run extraction on every scenario in data/ and compare with gold labels.

Reports precision / recall / F1 against human annotation plus the
redundancy rate of the candidate list before and after merging.

Usage: python run_eval.py
"""
import os

from term_extract import TermExtractor, evaluate
from term_extract.merging import canonical_key

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SCENARIOS = {
    "en_single (单文档/英文)": (["en_single.txt"], "en_single.gold.tsv"),
    "en_multi  (多文档/英文)": (
        [os.path.join("en_multi", f)
         for f in sorted(os.listdir(os.path.join(DATA, "en_multi")))
         if f.endswith(".txt")],
        "en_multi.gold.tsv"),
    "mixed     (中英混排)": (["mixed.txt"], "mixed.gold.tsv"),
}


def redundancy_before_merge(cands):
    """Share of ranked candidates that duplicate an already-seen concept
    (case / plural / abbreviation variants)."""
    seen, dup = set(), 0
    for c in cands:
        key = canonical_key(c.surface())
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup / len(cands) if cands else 0.0


def main():
    ex = TermExtractor(min_freq=2, window=2, top_k=40)
    hdr = (f"{'scenario':<24}{'extr':>5}{'gold':>5}{'match':>6}{'P':>7}"
           f"{'R':>7}{'F1':>7}{'raw':>5}{'red-pre':>9}{'red-merge':>11}"
           f"{'red-post':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, (files, goldf) in SCENARIOS.items():
        docs = []
        for rel in files:
            with open(os.path.join(DATA, rel), encoding="utf-8") as fh:
                docs.append(fh.read())
        terms, _ = ex.extract(docs)
        raw_cands = ex.extract_candidates(docs)
        m = evaluate(terms, os.path.join(DATA, goldf),
                     raw_candidate_count=len(raw_cands))
        pre = redundancy_before_merge(raw_cands)
        post = m["redundancy_rate_after_merge"]
        removed = m["redundancy_removed_by_merge"]
        print(f"{name:<24}{m['extracted']:>5}{m['gold_concepts']:>5}"
              f"{m['matched_concepts']:>6}{m['precision']:>7}"
              f"{m['recall']:>7}{m['f1']:>7}{len(raw_cands):>5}"
              f"{pre:>9.4f}{removed:>11.4f}{post:>10.4f}")


if __name__ == "__main__":
    main()
