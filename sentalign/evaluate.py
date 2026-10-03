"""Evaluation of predicted alignments against gold (human) annotations.

Gold format (JSON):
{
  "name": "dataset-name",
  "src": ["sentence", ...],
  "tgt": ["sentence", ...],
  "gold": [
    {"src": [0], "tgt": [0]},
    {"src": [2], "tgt": []},            # unmatched source segment
    {"src": [3], "tgt": [2, 3]},        # 1:2 merge
    {"src": [4], "tgt": [4, 5, 6], "bulk_uncertain": true}
  ]
}

`bulk_uncertain` marks beads where a human would not commit to a specific
pairing either (e.g. extreme length ratios); for those we only require the
system to *flag* its corresponding beads instead of forcing a confident match.

Error types (per gold bead):
  correct              predicted bead identical to gold bead
  split_error          gold merged bead predicted as several smaller beads
  merge_error          gold bead merged into a bigger predicted bead
  spurious_pair        gold unmatched segment got paired by the system
  missed_pair          gold matched bead left (partially) unmatched
  forced_confident_bulk  bulk-uncertain gold bead covered by confident beads
  confidence_mismatch  correct bead but flagged (calibration note, NOT an error)

Accuracy is structural: exact-bead matches (flagged or not) / total gold beads.
`confidence_mismatch` is reported for calibration analysis but does not lower
accuracy, because flagging a correct pairing is safe behavior, not an
alignment error.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .aligner import AlignmentResult

ERROR_TYPES = (
    "correct",
    "split_error",
    "merge_error",
    "spurious_pair",
    "missed_pair",
    "forced_confident_bulk",
    "confidence_mismatch",
)


@dataclass
class GoldBead:
    src: Tuple[int, ...]
    tgt: Tuple[int, ...]
    bulk_uncertain: bool = False

    @property
    def key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return (self.src, self.tgt)


@dataclass
class Dataset:
    name: str
    src: List[str]
    tgt: List[str]
    gold: List[GoldBead]


def load_dataset(path: str) -> Dataset:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    gold = [
        GoldBead(tuple(g["src"]), tuple(g["tgt"]), bool(g.get("bulk_uncertain", False)))
        for g in raw.get("gold", [])
    ]
    return Dataset(name=raw["name"], src=raw["src"], tgt=raw["tgt"], gold=gold)


def _src_partition(bead: GoldBead, pred: AlignmentResult) -> List[Tuple[int, int, int, int]]:
    """Predicted beads covering the gold bead's source span (in order)."""
    lo, hi = (min(bead.src), max(bead.src) + 1) if bead.src else (None, None)
    if lo is None:
        return []
    return [b.key for b in pred.beads if b.src_end > lo and b.src_start < hi and b.src_end > b.src_start]


def _tgt_partition(bead: GoldBead, pred: AlignmentResult) -> List[Tuple[int, int, int, int]]:
    lo, hi = (min(bead.tgt), max(bead.tgt) + 1) if bead.tgt else (None, None)
    if lo is None:
        return []
    return [b.key for b in pred.beads if b.tgt_end > lo and b.tgt_start < hi and b.tgt_end > b.tgt_start]


def _flagged(pred: AlignmentResult, key: Tuple[int, int, int, int]) -> bool:
    for b in pred.beads:
        if b.key == key:
            return b.flagged
    return False


@dataclass
class BeadJudgement:
    gold: GoldBead
    category: str
    predicted: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class EvalResult:
    dataset: str
    judgements: List[BeadJudgement]
    counts: Dict[str, int]
    accuracy: float
    link_precision: float
    link_recall: float
    link_f1: float
    flagged_total: int
    flagged_correct: int
    unflagged_total: int
    unflagged_correct: int


def _links(beads: Sequence[GoldBead]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for g in beads:
        for i in g.src:
            for j in g.tgt:
                out.add((i, j))
    return out


def _pred_links(pred: AlignmentResult) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for b in pred.beads:
        for i in range(b.src_start, b.src_end):
            for j in range(b.tgt_start, b.tgt_end):
                out.add((i, j))
    return out


def _pred_bead_keys(pred: AlignmentResult) -> Set[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    keys = set()
    for b in pred.beads:
        keys.add((tuple(range(b.src_start, b.src_end)), tuple(range(b.tgt_start, b.tgt_end))))
    return keys


def evaluate(pred: AlignmentResult, dataset: Dataset) -> EvalResult:
    pred_keys = _pred_bead_keys(pred)
    judgements: List[BeadJudgement] = []
    counts: Dict[str, int] = {t: 0 for t in ERROR_TYPES}

    for gold in dataset.gold:
        if gold.bulk_uncertain:
            covering = _src_partition(gold, pred) if gold.src else _tgt_partition(gold, pred)
            if covering and all(_flagged(pred, k) for k in covering):
                category = "correct"
            else:
                category = "forced_confident_bulk"
            judgements.append(BeadJudgement(gold, category, covering))
            counts[category] += 1
            continue

        if gold.key in pred_keys:
            key4 = None
            for b in pred.beads:
                if (tuple(range(b.src_start, b.src_end)), tuple(range(b.tgt_start, b.tgt_end))) == gold.key:
                    key4 = b.key
                    break
            flagged = _flagged(pred, key4) if key4 else False
            category = "confidence_mismatch" if flagged else "correct"
            judgements.append(BeadJudgement(gold, category, [key4] if key4 else []))
            counts[category] += 1
            continue

        src_part = _src_partition(gold, pred)
        tgt_part = _tgt_partition(gold, pred)
        covering = src_part if gold.src else tgt_part

        if not gold.src or not gold.tgt:
            # Gold says "unmatched"; any matched predicted bead is spurious.
            spurious = [k for k in covering if k[1] > k[0] and k[3] > k[2]]
            category = "spurious_pair" if spurious else "correct"
            judgements.append(BeadJudgement(gold, category, covering))
            counts[category] += 1
            continue

        has_null = any(k[1] == k[0] or k[3] == k[2] for k in covering)
        if has_null:
            category = "missed_pair"
        elif len(src_part) > 1:
            category = "split_error"
        elif len(tgt_part) > 1:
            category = "merge_error"
        else:
            # Single predicted bead on both sides but not equal to gold:
            # spans differ, so it is a merge on one side and split on other.
            category = "merge_error"
        judgements.append(BeadJudgement(gold, category, covering))
        counts[category] += 1

    total = len(judgements)
    correct = counts["correct"] + counts["confidence_mismatch"]
    accuracy = correct / total if total else 1.0

    # bulk-uncertain gold beads are excluded from link-level metrics: the
    # system is expected NOT to commit to specific links for them.
    gold_links = _links([g for g in dataset.gold if not g.bulk_uncertain])
    pred_links = _pred_links(pred)
    tp = len(gold_links & pred_links)
    link_precision = tp / len(pred_links) if pred_links else (1.0 if not gold_links else 0.0)
    link_recall = tp / len(gold_links) if gold_links else (1.0 if not pred_links else 0.0)
    denom = link_precision + link_recall
    link_f1 = 2 * link_precision * link_recall / denom if denom else 0.0

    # Confidence calibration: is flagging correlated with actual correctness?
    gold_key_set = {g.key for g in dataset.gold}
    flagged_total = flagged_correct = unflagged_total = unflagged_correct = 0
    for b in pred.beads:
        key = (tuple(range(b.src_start, b.src_end)), tuple(range(b.tgt_start, b.tgt_end)))
        is_correct = key in gold_key_set
        if b.flagged:
            flagged_total += 1
            flagged_correct += int(is_correct)
        else:
            unflagged_total += 1
            unflagged_correct += int(is_correct)

    return EvalResult(
        dataset=dataset.name,
        judgements=judgements,
        counts=counts,
        accuracy=accuracy,
        link_precision=link_precision,
        link_recall=link_recall,
        link_f1=link_f1,
        flagged_total=flagged_total,
        flagged_correct=flagged_correct,
        unflagged_total=unflagged_total,
        unflagged_correct=unflagged_correct,
    )


def aggregate(results: List[EvalResult]) -> Dict[str, object]:
    counts: Dict[str, int] = {t: 0 for t in ERROR_TYPES}
    total = 0
    for r in results:
        for t in ERROR_TYPES:
            counts[t] += r.counts[t]
        total += len(r.judgements)
    correct = counts["correct"] + counts["confidence_mismatch"]
    error_types = [t for t in ERROR_TYPES if t not in ("correct", "confidence_mismatch")]
    dist = {t: (counts[t] / total if total else 0.0) for t in error_types}
    return {
        "total_gold_beads": total,
        "accuracy": correct / total if total else 1.0,
        "error_counts": counts,
        "error_distribution": dist,
        "over_flagged": counts["confidence_mismatch"],
    }
