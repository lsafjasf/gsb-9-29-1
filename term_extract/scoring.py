"""Term scoring.

Combines four signals, all computable with the standard library:

1. frequency / C-value  -- discounts substrings that mostly occur nested
   inside a longer candidate (Frantzi et al.'s C-value).
2. boundary entropy     -- variety of left/right neighbours inside the
   context window; a genuine term boundary sees many different neighbours,
   a fragment sees the same token over and over.
3. stickiness (PMI)     -- collocation strength of the constituents:
   how much more often the n-gram occurs than chance under independence.
4. completeness         -- context-window test: if the candidate is almost
   always immediately extended by the same token and that longer candidate
   is nearly as frequent, the candidate is a fragment, not a full term.
"""
import math
from collections import Counter


def entropy(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total)
                for c in counter.values() if c > 0)


def _stickiness(cand, unigram_freq, total_unigrams):
    """Multiword PMI-style cohesion; 0 for unigrams."""
    n = len(cand.tokens)
    if n < 2:
        return 0.0
    denom = 1.0
    for tok in cand.tokens:
        denom *= unigram_freq.get(tok.lower(), 1)
    if denom <= 0:
        return 0.0
    p_joint = cand.freq / max(1, total_unigrams)
    p_indep = denom / (total_unigrams ** n)
    if p_indep <= 0:
        return 0.0
    return math.log2(p_joint / p_indep)


def score_candidates(cands, contains, min_freq=2, entropy_floor=0.8,
                     nest_ratio=0.8):
    """Annotate every Candidate with cvalue / entropies / stickiness /
    completeness / final score. Returns nothing (in place)."""
    unigram_freq = Counter()
    for cand in cands.values():
        if len(cand.tokens) == 1:
            unigram_freq[cand.tokens[0].lower()] += cand.freq
    total_unigrams = max(1, sum(unigram_freq.values()))

    from .candidates import STOPWORDS_EN

    def _container_ok(k):
        # A container with an internal English closed-class word
        # ("knowledge graphs are hot") is not a plausible term and must
        # not absorb the frequency of its sub-terms.
        toks = cands[k].tokens
        return not any(t.lower() in STOPWORDS_EN for t in toks[1:-1])

    for key, cand in cands.items():
        # --- 1. C-value ------------------------------------------------
        longer = [k for k in contains.get(key, []) if _container_ok(k)]
        # Total-containment discount: occurrences accounted for by longer
        # candidates are removed from the candidate's own mass. Only
        # containers that reach min_freq count -- freq-1 junk n-grams
        # (e.g. "word embeddings remain") must not eat real terms, while
        # "然语言处理" is still absorbed by "自然语言处理".
        nested = min(cand.freq,
                     sum(cands[k].freq for k in longer
                         if cands[k].freq >= min_freq))
        length_bonus = math.log2(len(cand.tokens) + 1)
        cand.cvalue = length_bonus * (cand.freq - nested)

        # --- 2. boundary entropy --------------------------------------
        cand.left_entropy = entropy(cand.left)
        cand.right_entropy = entropy(cand.right)

        # --- 3. stickiness --------------------------------------------
        cand.stickiness = _stickiness(cand, unigram_freq, total_unigrams)

        # --- 4. completeness via context window -----------------------
        # A fragment has (a) low neighbour entropy on at least one side
        # and (b) a dominant longer candidate absorbing most of its mass.
        # Sides with no observations (e.g. always sentence-initial) carry
        # no evidence and are excluded from the minimum.
        observed = [e for e, c in ((cand.left_entropy, cand.left),
                                   (cand.right_entropy, cand.right))
                    if sum(c.values()) > 0]
        min_ent = min(observed) if observed else 0.0
        if longer and min_ent < entropy_floor:
            top_longer = max(cands[k].freq for k in longer)
            if top_longer >= nest_ratio * cand.freq:
                cand.complete = False

        # --- final score ----------------------------------------------
        freq_score = math.log2(1 + max(0.0, cand.cvalue))
        boundary_score = min_ent / (1 + min_ent)            # in [0, 1)
        stick = max(0.0, cand.stickiness)
        stick_score = stick / (1 + stick) if len(cand.tokens) > 1 else 0.5
        score = freq_score * (0.5 + boundary_score) * (0.5 + stick_score)
        if not cand.complete:
            score *= 0.25                                    # fragment penalty
        cand.score = score


def _english_unigram_ok(cand, unigram_min_freq):
    """Single English words are the noisiest candidate class; keep them
    only if they are frequent, acronym-shaped, or capitalized inside a
    sentence (proper-noun-like terms such as 'Transformer')."""
    if len(cand.tokens) > 1:
        return True
    from .tokenizer import is_cjk
    if is_cjk(cand.tokens[0]):
        return False  # single CJK char: never a term (filtered earlier)
    if cand.freq >= unigram_min_freq:
        return True
    surface = cand.surface()
    # acronym-shaped: all caps or caps-with-plural ('SVMs'), 'NoSQL', ...
    if sum(ch.isupper() for ch in surface) >= 2:
        return True
    return cand.caps_mid > 0


def _zh_bigram_ok(cand, zh2_min_freq):
    """Two-character Chinese candidates are extremely ambiguous (they
    include many generic words like '表现'/'出现'), so require stronger
    frequency evidence than for longer terms."""
    from .tokenizer import is_cjk
    if (len(cand.tokens) == 2
            and all(is_cjk(t) for t in cand.tokens)):
        return cand.freq >= zh2_min_freq
    return True


def rank(cands, min_freq=2, top_k=None, min_score=0.0, unigram_min_freq=3,
         zh2_min_freq=3):
    """Filter + sort candidates; returns list of Candidate."""
    out = [c for c in cands.values()
           if c.freq >= min_freq and c.score >= min_score
           and _english_unigram_ok(c, unigram_min_freq)
           and _zh_bigram_ok(c, zh2_min_freq)]
    out.sort(key=lambda c: (-c.score, -c.freq, c.key))
    if top_k:
        out = out[:top_k]
    return out
