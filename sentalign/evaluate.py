"""Compare predicted alignments against human gold annotations.

Gold file format (TSV, one bead per line, 1-based inclusive indices):

    1-1<TAB>1-1        # sentence 1 <-> sentence 1
    2-3<TAB>2-2        # two source sentences merged into one target
    4-4<TAB>-          # source sentence deleted on target side
    -<TAB>5-6          # target-side insertion

Metrics
-------
* accuracy            : exact bead matches / gold beads
* strict accuracy     : same, but only counting *confident* predictions
* error distribution  : boundary_shift / false_merge / missed_merge /
                        missed_deletion / missed_insertion / wrong_pairing
"""

from collections import Counter


def parse_range(field):
    field = field.strip()
    if field in ("-", ""):
        return None
    a, b = field.split("-")
    return (int(a), int(b))


def load_gold(path):
    beads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            left, right = line.split("\t")
            beads.append((parse_range(left), parse_range(right)))
    return beads


def _span(bead_side):
    """(start, end) 1-based inclusive, or None for the empty side."""
    return bead_side


def _overlap(s1, s2):
    if s1 is None or s2 is None:
        return 0
    lo = max(s1[0], s2[0])
    hi = min(s1[1], s2[1])
    return max(0, hi - lo + 1)


def _size(side):
    return 0 if side is None else side[1] - side[0] + 1


def classify_error(gold_bead, predicted):
    """Classify one unmatched gold bead against the predicted bead set."""
    gs, gt = gold_bead
    # candidate predicted beads overlapping the gold bead on either side
    cands = [p for p in predicted
             if _overlap(gs, p[0]) or _overlap(gt, p[1])]
    if gs is None:                      # gold insertion (0-n)
        return "missed_insertion"
    if gt is None:                      # gold deletion (n-0)
        return "missed_deletion"
    g_multi = _size(gs) > 1 or _size(gt) > 1
    for p in cands:
        ps, pt = p
        if ps == gs or pt == gt:
            p_multi = (ps is not None and _size(ps) > 1) or \
                      (pt is not None and _size(pt) > 1)
            if g_multi and not p_multi:
                return "missed_merge"
            if p_multi and not g_multi:
                return "false_merge"
            return "boundary_shift"
    if g_multi:
        return "missed_merge"
    if any((p[0] is not None and _size(p[0]) > 1) or
           (p[1] is not None and _size(p[1]) > 1) for p in cands):
        return "false_merge"
    return "wrong_pairing"


def evaluate(pred_beads, gold_beads):
    """pred_beads / gold_beads: lists of (src_span, tgt_span) tuples.

    Returns dict with accuracy and error-type distribution.
    """
    pred_set = set(pred_beads)
    correct = sum(1 for g in gold_beads if g in pred_set)
    errors = Counter()
    for g in gold_beads:
        if g not in pred_set:
            errors[classify_error(g, pred_beads)] += 1
    total = len(gold_beads)
    return {
        "gold_beads": total,
        "correct": correct,
        "accuracy": correct / total if total else 1.0,
        "errors": dict(errors),
    }


def evaluate_with_confidence(alignment, gold_beads):
    """Evaluate an Alignment object, separating confident vs tentative."""
    confident, tentative = [], []
    for b in alignment.beads:
        span = (_span1(b.i0, b.i1), _span1(b.j0, b.j1))
        (tentative if b.tentative else confident).append(span)
    all_pred = confident + tentative
    result = evaluate(all_pred, gold_beads)
    result["confident"] = evaluate(confident, gold_beads)
    result["tentative_count"] = len(tentative)
    # how many gold beads are recovered *only* through tentative beads
    result["tentative_hits"] = sum(
        1 for g in gold_beads if g in set(tentative) and g not in set(confident))
    return result


def _span1(i0, i1):
    """0-based half-open -> 1-based inclusive; None when empty."""
    return None if i1 == i0 else (i0 + 1, i1)
