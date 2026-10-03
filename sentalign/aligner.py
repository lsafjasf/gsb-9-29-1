"""Dynamic-programming sentence aligner.

Bead types: 1:1, 1:2, 2:1, 2:2 (merges) and 1:0 / 0:1 (unmatched segments
for insertions / deletions). Cost = length-based cost (Gale & Church style
log-variance penalty) + type prior + lexical cost (hard anchors + PMI).

Confidence: forward-backward arc posteriors give a principled per-bead
probability. Beads whose posterior is below the "high" threshold are marked
as low/medium confidence instead of being presented as certain matches.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import features as F

BEAD_TYPES: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (0, 1))

# Penalty for using a bead type at all (log-prior style).
DEFAULT_TYPE_PRIOR: Dict[Tuple[int, int], float] = {
    (1, 1): 0.0,
    (1, 2): 1.2,
    (2, 1): 1.2,
    (2, 2): 2.6,
}

def length_cost(src_units: int, tgt_units: int, c: float, s2: float) -> float:
    """Gale & Church match cost: -log P(|delta|), delta = (l2 - c*l1)/sqrt(l1*s2).

    P(|delta| >= d) = erfc(d / sqrt(2)) is the two-sided normal tail, so the
    cost is 0 for a perfect length match and grows with the discrepancy.
    """
    l1 = max(src_units, 1)
    l2 = max(tgt_units, 1)
    delta = abs(l2 - c * l1) / math.sqrt(l1 * s2)
    return -math.log(max(math.erfc(delta / math.sqrt(2.0)), 1e-10))


def null_cost(units: int, c: float, s2: float, base: float = 2.0, slope: float = 0.004) -> float:
    """Cost of leaving a segment unmatched (1:0 or 0:1 bead)."""
    return base + slope * units


@dataclass
class AlignConfig:
    type_prior: Dict[Tuple[int, int], float] = field(default_factory=lambda: dict(DEFAULT_TYPE_PRIOR))
    null_base: float = 2.0
    null_slope: float = 0.004
    variance: float = 3.0          # s^2 of the length model
    ratio_prior: float = 20.0      # pseudo-count shrinking the length ratio toward 1.0
    ratio_min: float = 0.2
    ratio_max: float = 4.0
    lexical_weight: float = 1.0
    pmi_rounds: int = 2
    high_threshold: float = 0.65    # posterior >= this -> "high"
    low_threshold: float = 0.3     # posterior <  this -> "low", else "medium"


@dataclass
class Bead:
    src_start: int
    src_end: int
    tgt_start: int
    tgt_end: int
    bead_type: str
    cost: float
    posterior: float
    confidence: str            # "high" | "medium" | "low"
    contradiction: bool = False
    src_text: str = ""
    tgt_text: str = ""

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.src_start, self.src_end, self.tgt_start, self.tgt_end)

    @property
    def flagged(self) -> bool:
        return self.confidence != "high"


@dataclass
class AlignmentResult:
    beads: List[Bead]
    total_cost: float
    length_ratio: float
    n_src: int
    n_tgt: int

    @property
    def path(self) -> List[Tuple[int, int, int, int]]:
        return [b.key for b in self.beads]


def _confidence(posterior: float, contradiction: bool, cfg: AlignConfig) -> str:
    if contradiction:
        return "low"
    if posterior >= cfg.high_threshold:
        return "high"
    if posterior >= cfg.low_threshold:
        return "medium"
    return "low"


def _build_costs(
    src_feats: List[F.SentFeat],
    tgt_feats: List[F.SentFeat],
    c: float,
    cfg: AlignConfig,
    lex: Optional[F.LexicalModel],
) -> Dict[Tuple[int, int, int, int], Tuple[float, bool]]:
    """cost[(i, i+a, j, j+b)] -> (cost, contradiction) for every legal bead."""
    n, m = len(src_feats), len(tgt_feats)
    costs: Dict[Tuple[int, int, int, int], Tuple[float, bool]] = {}
    for i in range(n + 1):
        for j in range(m + 1):
            for a, b in BEAD_TYPES:
                if i + a > n or j + b > m:
                    continue
                if a == 0:
                    cost = null_cost(F.group_units(tgt_feats, j, j + b), c, cfg.variance,
                                     cfg.null_base, cfg.null_slope)
                    costs[(i, i, j, j + b)] = (cost, False)
                    continue
                if b == 0:
                    cost = null_cost(F.group_units(src_feats, i, i + a), c, cfg.variance,
                                     cfg.null_base, cfg.null_slope)
                    costs[(i, i + a, j, j)] = (cost, False)
                    continue
                lu = F.group_units(src_feats, i, i + a)
                lv = F.group_units(tgt_feats, j, j + b)
                cost = length_cost(lu, lv, c, cfg.variance)
                cost += cfg.type_prior[(a, b)]
                if cfg.lexical_weight > 0.0:
                    lc, contra = F.lexical_cost(
                        F.group_anchors(src_feats, i, i + a),
                        F.group_anchors(tgt_feats, j, j + b),
                        F.group_vocab(src_feats, i, i + a),
                        F.group_vocab(tgt_feats, j, j + b),
                        lex,
                    )
                    cost += cfg.lexical_weight * lc
                else:
                    contra = False
                costs[(i, i + a, j, j + b)] = (cost, contra)
    return costs


def _forward_backward(
    n: int, m: int, costs: Dict[Tuple[int, int, int, int], Tuple[float, bool]]
) -> Tuple[List[Tuple[int, int, int, int]], Dict[Tuple[int, int, int, int], float], float]:
    """Viterbi path + arc posteriors via log-space forward-backward."""
    neg_inf = float("-inf")
    size = (n + 1) * (m + 1)

    def idx(i: int, j: int) -> int:
        return i * (m + 1) + j

    alpha = [neg_inf] * size
    alpha[0] = 0.0
    back: Dict[int, Tuple[int, int, int, int]] = {}
    best = [neg_inf] * size
    order: List[Tuple[int, int]] = [(i, j) for i in range(n + 1) for j in range(m + 1)]

    for i, j in order:
        state = idx(i, j)
        if alpha[state] == neg_inf:
            continue
        for a, b in BEAD_TYPES:
            key = (i, i + a, j, j + b)
            entry = costs.get(key)
            if entry is None:
                continue
            score = alpha[state] - entry[0]
            nxt = idx(i + a, j + b)
            # log-add-exp for alpha
            if alpha[nxt] == neg_inf:
                alpha[nxt] = score
            else:
                hi, lo = max(alpha[nxt], score), min(alpha[nxt], score)
                alpha[nxt] = hi + math.log1p(math.exp(lo - hi))
            # Viterbi
            if score > best[nxt]:
                best[nxt] = score
                back[nxt] = key

    beta = [neg_inf] * size
    beta[idx(n, m)] = 0.0
    for i, j in reversed(order):
        state = idx(i, j)
        if beta[state] == neg_inf:
            continue
        for a, b in BEAD_TYPES:
            pi, pj = i - a, j - b
            if pi < 0 or pj < 0:
                continue
            key = (pi, i, pj, j)
            entry = costs.get(key)
            if entry is None:
                continue
            prev = idx(pi, pj)
            score = beta[state] - entry[0]
            if beta[prev] == neg_inf:
                beta[prev] = score
            else:
                hi, lo = max(beta[prev], score), min(beta[prev], score)
                beta[prev] = hi + math.log1p(math.exp(lo - hi))

    total = alpha[idx(n, m)]

    # Arc posteriors.
    posteriors: Dict[Tuple[int, int, int, int], float] = {}
    for key, (cost, _c) in costs.items():
        i, ie, j, je = key
        a_state, b_state = idx(i, j), idx(ie, je)
        if alpha[a_state] == neg_inf or beta[b_state] == neg_inf or total == neg_inf:
            posteriors[key] = 0.0
            continue
        logp = alpha[a_state] - cost + beta[b_state] - total
        posteriors[key] = math.exp(min(0.0, logp))

    # Reconstruct Viterbi path.
    path: List[Tuple[int, int, int, int]] = []
    state = idx(n, m)
    while state != 0:
        key = back[state]
        path.append(key)
        state = idx(key[0], key[2])
    path.reverse()
    return path, posteriors, total


def _run_pass(
    src_feats: List[F.SentFeat],
    tgt_feats: List[F.SentFeat],
    c: float,
    cfg: AlignConfig,
    lex: Optional[F.LexicalModel],
) -> AlignmentResult:
    n, m = len(src_feats), len(tgt_feats)
    costs = _build_costs(src_feats, tgt_feats, c, cfg, lex)
    path, posteriors, total = _forward_backward(n, m, costs)
    beads: List[Bead] = []
    for (i, ie, j, je) in path:
        cost, contra = costs[(i, ie, j, je)]
        post = posteriors.get((i, ie, j, je), 0.0)
        a, b = ie - i, je - j
        bead = Bead(
            src_start=i, src_end=ie, tgt_start=j, tgt_end=je,
            bead_type=f"{a}-{b}", cost=cost, posterior=post,
            confidence=_confidence(post, contra, cfg), contradiction=contra,
            src_text=" / ".join(f.text for f in src_feats[i:ie]),
            tgt_text=" / ".join(f.text for f in tgt_feats[j:je]),
        )
        beads.append(bead)
    return AlignmentResult(beads=beads, total_cost=-total, length_ratio=c, n_src=n, n_tgt=m)


def _estimate_ratio(src_feats: List[F.SentFeat], tgt_feats: List[F.SentFeat], cfg: AlignConfig) -> float:
    src_total = sum(f.units for f in src_feats)
    tgt_total = sum(f.units for f in tgt_feats)
    c = (tgt_total + cfg.ratio_prior) / (src_total + cfg.ratio_prior)
    return min(cfg.ratio_max, max(cfg.ratio_min, c))


def align(
    src_sentences: Sequence[str],
    tgt_sentences: Sequence[str],
    config: Optional[AlignConfig] = None,
) -> AlignmentResult:
    """Align two lists of sentences. Returns beads in order (the alignment path)."""
    cfg = config or AlignConfig()
    src_feats = F.extract(list(src_sentences))
    tgt_feats = F.extract(list(tgt_sentences))
    n, m = len(src_feats), len(tgt_feats)
    if n == 0 and m == 0:
        return AlignmentResult(beads=[], total_cost=0.0, length_ratio=1.0, n_src=0, n_tgt=0)
    c = _estimate_ratio(src_feats, tgt_feats, cfg)

    # Pass 1: length-only alignment.
    result = _run_pass(src_feats, tgt_feats, c, cfg, lex=None)
    if cfg.lexical_weight <= 0.0 or n == 0 or m == 0:
        return result

    # Pass 2..k: train PMI on confident 1:1 beads, re-align with lexical cost.
    lex = F.LexicalModel()
    for round_no in range(cfg.pmi_rounds):
        if round_no == 0:
            seeds = [b for b in result.beads if b.bead_type == "1-1"]
        else:
            seeds = [b for b in result.beads if b.bead_type == "1-1" and not b.flagged]
        pairs = [
            (F.group_vocab(src_feats, b.src_start, b.src_end),
             F.group_vocab(tgt_feats, b.tgt_start, b.tgt_end))
            for b in seeds
        ]
        if not pairs:
            break
        lex = F.LexicalModel()
        lex.train(pairs)
        if not lex.pmi:
            break
        result = _run_pass(src_feats, tgt_feats, c, cfg, lex=lex)
    return result


# ---------------------------------------------------------------------------
# Sentence splitting (standard library only; adequate for well-formed text).

_SENT_END = set("。！？!?…")
_CLOSERS = set("”’\"')）】》")


def split_sentences(text: str) -> List[str]:
    """Split mixed Chinese/English text into sentences."""
    sentences: List[str] = []
    buf: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "\r\n":
            if buf and "".join(buf).strip():
                sentences.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        if ch in _SENT_END or (ch == "." and _is_sentence_dot(buf)):
            # absorb trailing closing quotes / brackets into this sentence
            while i + 1 < n and text[i + 1] in _CLOSERS:
                buf.append(text[i + 1])
                i += 1
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(sentence)
            buf = []
        i += 1
    if buf and "".join(buf).strip():
        sentences.append("".join(buf).strip())
    return sentences


def _is_sentence_dot(buf: List[str]) -> bool:
    """A '.' ends a sentence unless it is inside a number/abbreviation."""
    if len(buf) >= 2 and buf[-2].isdigit():
        return False
    return True


def align_texts(src_text: str, tgt_text: str, config: Optional[AlignConfig] = None) -> AlignmentResult:
    """Convenience API: split raw texts into sentences, then align."""
    return align(split_sentences(src_text), split_sentences(tgt_text), config)
