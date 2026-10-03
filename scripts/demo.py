#!/usr/bin/env python3
"""Print alignment examples and dump machine-readable samples.

Run:  python3 scripts/demo.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentalign import AlignConfig, align, align_texts, evaluate, load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")

DEMO_DATASETS = [
    "basic_merge_gap",
    "fr_en_cognate",
    "repeated",
    "extreme_ratio",
    "unrelated_anchors",
]

CONF_MARK = {"high": "  ", "medium": "? ", "low": "! "}


def print_alignment(name: str, result, ev=None) -> None:
    print(f"\n=== {name}  (src={result.n_src}, tgt={result.n_tgt}, "
          f"ratio={result.length_ratio:.2f}, cost={result.total_cost:.1f}) ===")
    print(f"{'conf':4s} {'type':5s} {'post':>5s}  src -> tgt")
    for bead in result.beads:
        src_span = f"{bead.src_start}" if bead.src_end - bead.src_start == 1 else f"{bead.src_start}-{bead.src_end - 1}"
        tgt_span = f"{bead.tgt_start}" if bead.tgt_end - bead.tgt_start == 1 else f"{bead.tgt_start}-{bead.tgt_end - 1}"
        if bead.src_end == bead.src_start:
            src_span = "---"
        if bead.tgt_end == bead.tgt_start:
            tgt_span = "---"
        marker = CONF_MARK[bead.confidence]
        print(f"{marker:4s} {bead.bead_type:5s} {bead.posterior:5.2f}  S{src_span} -> T{tgt_span}")
    if ev is not None:
        print("judgement:", json.dumps(ev.counts, ensure_ascii=False))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = []
    for name in DEMO_DATASETS:
        ds = load_dataset(os.path.join(ROOT, "data", f"{name}.json"))
        result = align(ds.src, ds.tgt, AlignConfig())
        ev = evaluate(result, ds)
        print_alignment(name, result, ev)
        samples.append(
            {
                "name": name,
                "length_ratio": round(result.length_ratio, 4),
                "total_cost": round(result.total_cost, 3),
                "beads": [
                    {
                        "type": b.bead_type,
                        "src": list(range(b.src_start, b.src_end)),
                        "tgt": list(range(b.tgt_start, b.tgt_end)),
                        "posterior": round(b.posterior, 4),
                        "confidence": b.confidence,
                        "flagged": b.flagged,
                        "contradiction": b.contradiction,
                        "cost": round(b.cost, 3),
                    }
                    for b in result.beads
                ],
            }
        )

    # Raw-text API example (sentence splitting + alignment in one call).
    zh = "纯水在标准大气压下于100摄氏度时沸腾。当外界气压降低时，水的沸点也会随之降低。"
    en = ("Pure water boils at 100 degrees Celsius at standard pressure. "
          "When the ambient pressure drops, its boiling point drops as well.")
    raw_result = align_texts(zh, en)
    print("\n=== raw-text API (auto sentence splitting) ===")
    print(f"split into {raw_result.n_src} zh / {raw_result.n_tgt} en sentences")
    print_alignment("raw_texts", raw_result)
    samples.append(
        {
            "name": "raw_texts",
            "src_text": zh,
            "tgt_text": en,
            "beads": [
                {
                    "type": b.bead_type,
                    "src": list(range(b.src_start, b.src_end)),
                    "tgt": list(range(b.tgt_start, b.tgt_end)),
                    "posterior": round(b.posterior, 4),
                    "confidence": b.confidence,
                }
                for b in raw_result.beads
            ],
        }
    )

    with open(os.path.join(OUT_DIR, "alignment_samples.json"), "w", encoding="utf-8") as fh:
        json.dump(samples, fh, ensure_ascii=False, indent=2)
    print(f"\nWrote {os.path.join('outputs', 'alignment_samples.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
