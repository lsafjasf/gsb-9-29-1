"""Candidate term generation with boundary-feature collection.

For every n-gram that passes the boundary filters we record:
  - frequency and observed surface forms (original casing)
  - left / right neighbour distribution within a context window
    (used by the boundary-entropy and completeness tests)
"""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .tokenizer import is_cjk

# Closed-class English words that may not begin/end a term.
STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "not", "no", "nor", "so", "too", "very", "just", "as",
    "it", "its", "this", "that", "these", "those", "there", "here", "we",
    "our", "you", "your", "they", "their", "he", "she", "his", "her", "i",
    "me", "my", "us", "them", "him", "such", "than", "when", "where", "which",
    "who", "whom", "how", "what", "all", "each", "both", "few", "more",
    "most", "other", "some", "any", "only", "own", "same", "also", "via",
    "per", "etc", "eg", "ie", "using", "used", "use", "based", "like",
}

# High-frequency Chinese function characters that may not begin/end a term.
STOPCHARS_ZH = set(
    "的了和是在我有就都一上也到说要去你会着没看好这那它们他她我们个"
    "于与及或而把被让向从为以等很更最不只又都还就才并且但若因所"
    "由于对于根据通过进行相关其中之其此该各每某种样时后前中内外"
    "吗呢吧啊嘛么啦呀哪谁怎什"
)

_NUM_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


@dataclass
class Candidate:
    key: str                       # normalized join of tokens (lowercase)
    tokens: tuple
    freq: int = 0
    surfaces: Counter = field(default_factory=Counter)
    left: Counter = field(default_factory=Counter)
    right: Counter = field(default_factory=Counter)
    caps_mid: int = 0   # occurrences capitalized in non-sentence-initial pos
    # filled in by scoring
    cvalue: float = 0.0
    left_entropy: float = 0.0
    right_entropy: float = 0.0
    stickiness: float = 0.0
    complete: bool = True
    score: float = 0.0

    @property
    def text(self):
        return self.surface()

    def surface(self):
        if self.surfaces:
            return self.surfaces.most_common(1)[0][0]
        return " ".join(self.tokens)


def _is_stopword(tok):
    if is_cjk(tok):
        return tok in STOPCHARS_ZH
    return tok.lower() in STOPWORDS_EN


def _is_content(tok):
    if _NUM_RE.match(tok):
        return False
    if is_cjk(tok):
        return tok not in STOPCHARS_ZH
    return tok.lower() not in STOPWORDS_EN and any(c.isalpha() for c in tok)


def _join_surface(tokens):
    """Human-readable surface: no space between adjacent CJK chars or
    between a CJK char and its neighbour."""
    out = []
    for i, tok in enumerate(tokens):
        if i and not (is_cjk(tokens[i - 1][-1]) or is_cjk(tok[0])):
            out.append(" ")
        out.append(tok)
    return "".join(out)


def _boundary_ok(tokens):
    """A term may not start or end with a stopword / bare number."""
    if _is_stopword(tokens[0]) or _is_stopword(tokens[-1]):
        return False
    if _NUM_RE.match(tokens[0]) or _NUM_RE.match(tokens[-1]):
        return False
    return True


def generate_candidates(docs_tokens, max_n=5, max_n_zh=8, window=2):
    """docs_tokens: list of documents, each a list of sentences (token lists).

    Returns dict key -> Candidate.
    """
    cands = {}
    for sentences in docs_tokens:
        for sent in sentences:
            n_tokens = len(sent)
            for i in range(n_tokens):
                for n in range(1, max(max_n, max_n_zh) + 1):
                    j = i + n
                    if j > n_tokens:
                        break
                    toks = tuple(sent[i:j])
                    n_cjk = sum(1 for t in toks if is_cjk(t))
                    # length limits differ: CJK chars are single tokens,
                    # so a 6-char Chinese term needs n=6 while English
                    # phrases rarely exceed 5 tokens.
                    if n_cjk and n > max_n_zh:
                        break
                    if n_cjk < n and n > max_n:
                        break
                    # single CJK char is never a standalone term
                    if n == 1 and is_cjk(toks[0]):
                        continue
                    # a Chinese term never contains a function character
                    if any(is_cjk(t) and t in STOPCHARS_ZH for t in toks):
                        continue
                    # all-CJK n-grams need length >= 2 chars (already n>=2 here)
                    if not any(_is_content(t) for t in toks):
                        continue
                    if not _boundary_ok(toks):
                        continue
                    key = " ".join(t.lower() for t in toks)
                    cand = cands.get(key)
                    if cand is None:
                        cand = cands[key] = Candidate(key=key, tokens=toks)
                    cand.freq += 1
                    cand.surfaces[_join_surface(toks)] += 1
                    if n == 1 and i > 0 and toks[0][0].isupper():
                        cand.caps_mid += 1
                    # context window neighbours (distance-weighted)
                    for d in range(1, window + 1):
                        w = 1.0 / d
                        if i - d >= 0:
                            cand.left[sent[i - d].lower()] += w
                        if j - 1 + d < n_tokens:
                            cand.right[sent[j - 1 + d].lower()] += w
    return cands


def build_containment(cands):
    """Map candidate key -> list of strictly-longer candidate keys that
    contain it as a contiguous token subsequence."""
    contains = defaultdict(list)
    items = [(c.key, c.tokens) for c in cands.values()]
    by_len = defaultdict(list)
    for key, toks in items:
        by_len[len(toks)].append((key, toks))
    for key, toks in items:
        for ln in range(len(toks) + 1, max(by_len) + 1):
            for okey, other in by_len.get(ln, []):
                for k in range(len(other) - len(toks) + 1):
                    if other[k:k + len(toks)] == toks:
                        contains[key].append(okey)
                        break
    return contains
