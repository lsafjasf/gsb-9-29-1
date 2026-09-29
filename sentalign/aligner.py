"""Dynamic-programming sentence aligner.

Features
--------
* Length score   : Gale & Church (1991) Gaussian model on character counts.
* Lexical score  : mutual-information lexicon induced from a first-pass
                   length-only alignment (one EM-style iteration), plus
                   exact anchors (digits / Latin tokens shared verbatim).
* Search         : Viterbi DP over bead types 1-0, 0-1, 1-1, 2-1, 1-2, 2-2.
* Confidence     : forward-backward posterior of every bead; beads below
                   the threshold are reported as *tentative*, never forced.
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------- bead types
BEADS = ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (2, 1), (1, 2), (2, 2))

# Priors loosely following Gale & Church, with deletion/insertion allowed.
LOG_PRIOR = {
    (1, 1): math.log(0.72),
    (2, 1): math.log(0.06),
    (1, 2): math.log(0.06),
    (2, 2): math.log(0.03),
    (1, 0): math.log(0.04),
    (0, 1): math.log(0.04),
    (2, 0): math.log(0.02),
    (0, 2): math.log(0.02),
}

# Variance constant of the length model (chars^2 per source char).
_S2 = 6.8
# Weight of the lexical feature relative to the length feature.
_LEX_WEIGHT = 1.0
# Bonus for one exact shared anchor token (digits, latin words).
_ANCHOR_BONUS = 1.2
# Penalty for an anchor that appears on only one side of a bead.
# Numbers and names normally survive translation, so an unmatched
# anchor is strong evidence against a bead.
_ANCHOR_MISS_PENALTY = 1.2
# Plausible range for the target/source character ratio.
_C_MIN, _C_MAX = 0.25, 4.0
# Posterior below this -> tentative (low confidence) bead.
CONFIDENCE_THRESHOLD = 0.7
# Confidence multiplier for aligned beads with no lexical/anchor support.
_NO_SUPPORT_FACTOR = 0.45

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[一-鿿]")


def _tokens(sent):
    """Word tokens for Latin text, single characters for CJK."""
    return [t.lower() for t in _TOKEN_RE.findall(sent)]


def _digits(sent):
    return set(t for t in _TOKEN_RE.findall(sent)
               if re.fullmatch(r"\d+(?:[.,]\d+)*", t))


def _anchor_vocab(src, tgt):
    """Anchor vocabulary: digits, plus latin words appearing in *both*
    documents (names, abbreviations left untranslated)."""
    lat_src = set()
    lat_tgt = set()
    for s in src:
        lat_src |= set(re.findall(r"[A-Za-z]+(?:'[a-z]+)?", s.lower()))
    for s in tgt:
        lat_tgt |= set(re.findall(r"[A-Za-z]+(?:'[a-z]+)?", s.lower()))
    return lat_src & lat_tgt


def _anchors(sent, vocab):
    toks = set(_TOKEN_RE.findall(sent.lower()))
    return set(t for t in toks
               if re.fullmatch(r"\d+(?:[.,]\d+)*", t) or t in vocab)


def _nchars(sent):
    return len(re.sub(r"\s", "", sent))


# ------------------------------------------------------------------ features
class _LengthModel:
    def __init__(self, src, tgt, c=None):
        s = sum(_nchars(x) for x in src) or 1
        t = sum(_nchars(x) for x in tgt) or 1
        # target chars per source char, clamped to a plausible range so
        # that untranslated material cannot distort the ratio
        raw = c if c else t / s
        self.c = min(max(raw, _C_MIN), _C_MAX)

    def log_score(self, src_sents, tgt_sents):
        ls = sum(_nchars(x) for x in src_sents)
        lt = sum(_nchars(x) for x in tgt_sents)
        if ls == 0 and lt == 0:
            return 0.0
        if ls == 0 or lt == 0:
            # pure deletion/insertion: flat cost (Gale & Church style)
            return 0.0
        # variance scales with c^2 (target lengths are c times smaller)
        delta = (lt - ls * self.c) / math.sqrt(ls * _S2 * self.c * self.c)
        return -0.5 * delta * delta  # log of unit Gaussian, up to constant


def _induce_lexicon(src, tgt, beads, min_count=2, top_k=400):
    """Mutual-information lexicon induced from an alignment.

    Bead-level co-occurrence: for every aligned bead, each source token
    co-occurs with each target token.  Association is the pointwise
    mutual information log( co*N / (count_a * count_b) ).
    """
    src_tok = [set(_tokens(s)) for s in src]
    tgt_tok = [set(_tokens(s)) for s in tgt]
    co = Counter()
    cs = Counter()
    ct = Counter()
    for si, ti in beads:
        sa = set()
        ta = set()
        for i in si:
            sa |= src_tok[i]
        for j in ti:
            ta |= tgt_tok[j]
        for a in sa:
            cs[a] += 1
        for b in ta:
            ct[b] += 1
        for a in sa:
            for b in ta:
                co[(a, b)] += 1
    n_beads = max(len(beads), 1)
    lex = {}
    for (a, b), n in co.items():
        if n < min_count:
            continue
        # tokens occurring in nearly every bead carry no information
        if cs[a] > 0.4 * n_beads or ct[b] > 0.4 * n_beads:
            continue
        pmi = math.log(n * n_beads / (cs[a] * ct[b]))
        if pmi > 0:
            lex[(a, b)] = min(pmi, 3.0)
    # keep the strongest pairs only
    if len(lex) > top_k:
        keep = sorted(lex, key=lex.get, reverse=True)[:top_k]
        lex = {k: lex[k] for k in keep}
    return lex

def _anchor_score(src_sents, tgt_sents, vocab):
    sa = set()
    ta = set()
    for s in src_sents:
        sa |= _anchors(s, vocab)
    for s in tgt_sents:
        ta |= _anchors(s, vocab)
    shared = sa & ta
    missing = len(sa - ta) + len(ta - sa)
    return _ANCHOR_BONUS * len(shared) - _ANCHOR_MISS_PENALTY * missing


def _bead_support(src_sents, tgt_sents, lex, vocab):
    """Count of independent lexical/anchor evidences inside a bead."""
    sa = set()
    ta = set()
    for s in src_sents:
        sa |= _anchors(s, vocab)
    for s in tgt_sents:
        ta |= _anchors(s, vocab)
    support = len(sa & ta)
    if lex:
        stoks = set()
        ttoks = set()
        for s in src_sents:
            stoks |= set(_tokens(s))
        for s in tgt_sents:
            ttoks |= set(_tokens(s))
        support += sum(1 for a in stoks for b in ttoks if (a, b) in lex)
    return support


def _lex_score(src_sents, tgt_sents, lex):
    """Sum over source tokens of their best lexicon match in the bead."""
    if not lex:
        return 0.0
    stoks = set()
    ttoks = set()
    for s in src_sents:
        stoks |= set(_tokens(s))
    for s in tgt_sents:
        ttoks |= set(_tokens(s))
    score = 0.0
    for a in stoks:
        best = 0.0
        for b in ttoks:
            v = lex.get((a, b))
            if v and v > best:
                best = v
        score += best
    return _LEX_WEIGHT * score


# ------------------------------------------------------------------ DP core
def _bead_scores(src, tgt, lex, c=None, vocab=()):
    """score[a][b][i][j] = log score of bead (a,b) starting at (i,j)."""
    n, m = len(src), len(tgt)
    lm = _LengthModel(src, tgt, c)
    scores = {}
    for a, b in BEADS:
        grid = {}
        for i in range(0, n - a + 1):
            for j in range(0, m - b + 1):
                ss = src[i:i + a]
                tt = tgt[j:j + b]
                s = LOG_PRIOR[(a, b)] + lm.log_score(ss, tt)
                if a and b:
                    s += _anchor_score(ss, tt, vocab) + _lex_score(ss, tt, lex)
                grid[(i, j)] = s
        scores[(a, b)] = grid
    return scores


def _estimate_c(src, tgt, path):
    """Re-estimate chars-per-char ratio from aligned (non-deletion) beads."""
    s = t = 0
    for p, q, a, b, _ in path:
        if a and b:
            s += sum(_nchars(src[i]) for i in range(p, p + a))
            t += sum(_nchars(tgt[j]) for j in range(q, q + b))
    if not s or not t:
        return None
    return t / s


def _dp(src, tgt, scores):
    """Viterbi + forward-backward. Returns (path, posterior of each bead)."""
    n, m = len(src), len(tgt)
    NEG = float("-inf")

    def logadd(x, y):
        if x == NEG:
            return y
        if y == NEG:
            return x
        if x < y:
            x, y = y, x
        return x + math.log1p(math.exp(y - x))

    vit = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    fwd = [[NEG] * (m + 1) for _ in range(n + 1)]
    vit[0][0] = 0.0
    fwd[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if vit[i][j] == NEG:
                continue
            for a, b in BEADS:
                ni, nj = i + a, j + b
                if ni > n or nj > m:
                    continue
                s = scores[(a, b)][(i, j)]
                v = vit[i][j] + s
                if v > vit[ni][nj]:
                    vit[ni][nj] = v
                    back[ni][nj] = (i, j, a, b)
                fwd[ni][nj] = logadd(fwd[ni][nj], fwd[i][j] + s)

    # backward pass
    bwd = [[NEG] * (m + 1) for _ in range(n + 1)]
    bwd[n][m] = 0.0
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if bwd[i][j] == NEG:
                continue
            for a, b in BEADS:
                pi, pj = i - a, j - b
                if pi < 0 or pj < 0:
                    continue
                s = scores[(a, b)][(pi, pj)]
                bwd[pi][pj] = logadd(bwd[pi][pj], bwd[i][j] + s)

    total = fwd[n][m]

    # viterbi path
    path = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, a, b = back[i][j]
        path.append((pi, pj, a, b))
        i, j = pi, pj
    path.reverse()

    # posterior of each viterbi bead
    out = []
    for pi, pj, a, b in path:
        s = scores[(a, b)][(pi, pj)]
        logpost = fwd[pi][pj] + s + bwd[pi + a][pj + b] - total
        out.append((pi, pj, a, b, math.exp(min(logpost, 0.0))))
    return out


# ------------------------------------------------------------------ results
class Bead:
    """One alignment bead: src[i0:i1] <-> tgt[j0:j1]."""

    __slots__ = ("i0", "i1", "j0", "j1", "confidence", "tentative")

    def __init__(self, i0, i1, j0, j1, confidence):
        self.i0, self.i1, self.j0, self.j1 = i0, i1, j0, j1
        self.confidence = confidence
        self.tentative = confidence < CONFIDENCE_THRESHOLD

    @property
    def kind(self):
        return "%d-%d" % (self.i1 - self.i0, self.j1 - self.j0)

    def __repr__(self):
        flag = "~" if self.tentative else "="
        return "<Bead %s src[%d:%d] %s tgt[%d:%d] p=%.2f>" % (
            self.kind, self.i0, self.i1, flag, self.j0, self.j1,
            self.confidence)


class Alignment:
    def __init__(self, src, tgt, beads):
        self.src = src
        self.tgt = tgt
        self.beads = beads

    def __iter__(self):
        return iter(self.beads)

    def path(self):
        """Human-readable alignment path."""
        lines = []
        for b in self.beads:
            mark = "  (LOW CONFIDENCE)" if b.tentative else ""
            left = " / ".join(self.src[i] for i in range(b.i0, b.i1)) or "<empty>"
            right = " / ".join(self.tgt[j] for j in range(b.j0, b.j1)) or "<empty>"
            lines.append("[%s] %s  <->  %s  p=%.2f%s"
                         % (b.kind, left, right, b.confidence, mark))
        return "\n".join(lines)


# ------------------------------------------------------------------ entry
def align(src, tgt):
    """Align two lists of sentences. Returns an Alignment."""
    src = list(src)
    tgt = list(tgt)
    if not src and not tgt:
        return Alignment(src, tgt, [])
    if not src:
        return Alignment(src, tgt, [Bead(0, 0, 0, len(tgt), 1.0)] if tgt else [])
    if not tgt:
        return Alignment(src, tgt, [Bead(0, len(src), 0, 0, 1.0)])

    vocab = _anchor_vocab(src, tgt)

    # pass 1: length + anchors -> re-estimate c, induce MI lexicon
    scores1 = _bead_scores(src, tgt, {}, vocab=vocab)
    path1 = _dp(src, tgt, scores1)
    c = _estimate_c(src, tgt, path1)

    # two EM-style iterations: induce MI lexicon, re-align, re-induce
    lex = {}
    path2 = path1
    for _ in range(2):
        beads_i = [(list(range(p, p + a)), list(range(q, q + b)))
                   for p, q, a, b, _ in path2 if a and b]
        if not beads_i:
            break
        lex = _induce_lexicon(src, tgt, beads_i)
        scores2 = _bead_scores(src, tgt, lex, c=c, vocab=vocab)
        new_path = _dp(src, tgt, scores2)
        if [(p, q, a, b) for p, q, a, b, _ in new_path] == \
           [(p, q, a, b) for p, q, a, b, _ in path2]:
            path2 = new_path
            break
        path2 = new_path
    beads = []
    for p, q, a, b, post in path2:
        conf = post
        if a and b:
            support = _bead_support(src[p:p + a], tgt[q:q + b], lex, vocab)
            if support == 0:
                # no lexical evidence at all: never report as certain
                conf = min(conf, post * _NO_SUPPORT_FACTOR)
        beads.append(Bead(p, p + a, q, q + b, conf))
    return Alignment(src, tgt, beads)
