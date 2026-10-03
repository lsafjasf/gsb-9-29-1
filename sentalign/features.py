"""Feature extraction for sentence alignment.

Tokenization is deliberately simple (standard library only):
  * Latin scripts: word tokens (letters + optional internal apostrophe/hyphen),
    accents folded (e.g. "conférence" -> "conference") for lexical matching.
  * Alphanumeric marks (Q1, X5, B200) stay intact as one token.
  * CJK scripts: single characters (no external segmenter available).
  * Numerals: contiguous digit groups.

Length is measured in non-whitespace characters, which is a stable proxy
across scripts (Gale & Church style length-based alignment).

Hard anchors are tokens containing digits: numbers and model/order IDs tend
to be preserved verbatim even between unrelated scripts. Ordinary words go
through the self-trained PMI model instead, so that e.g. French "le" versus
English "the" is not treated as a contradiction.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

_LATIN = "A-Za-z\u00c0-\u00ff"
_TOKEN_RE = re.compile(
    rf"\d+(?:[.,:/-]\d+)*|[{_LATIN}0-9]+(?:['’\-][{_LATIN}0-9]+)*|[一-鿿぀-ヿ가-힯]"
)
_DIGIT_RE = re.compile(r"\d")

# Very small stopword list; only applied to PMI candidate collection so that
# function words do not dominate the co-occurrence statistics.
STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its this "
    "that these those as at by from not no le la les des un une de du et en "
    "au aux ce cette ces il elle ils elles on ne pas plus est sont dans par "
    "pour qui que".split()
)


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, fold: bool = True) -> List[str]:
    """Mixed-script tokenizer: numbers, latin/alnum words, CJK chars.

    Elisions and hyphenated words ("l'intelligence", "real-time") are split
    into their parts; numeric separators ("3,000", "2024-03") are kept whole.
    """
    if fold:
        text = fold_accents(text)
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(0).lower()
        if _DIGIT_RE.search(tok):
            tokens.append(tok)
        elif "'" in tok or "’" in tok or "-" in tok:
            tokens.extend(p for p in re.split(r"['’\-]", tok) if p)
        else:
            tokens.append(tok)
    return tokens


def hard_anchors(text: str) -> Set[str]:
    """Digit-bearing tokens (numbers, IDs), preserved verbatim across languages."""
    return {tok for tok in tokenize(text) if _DIGIT_RE.search(tok)}


def pmi_vocab(text: str) -> Set[str]:
    """Alphabetic content tokens eligible for PMI statistics."""
    return {
        tok
        for tok in tokenize(text)
        if not _DIGIT_RE.search(tok) and len(tok) >= 3 and tok not in STOPWORDS
    }


def unit_count(text: str) -> int:
    """Length of a sentence in non-whitespace characters."""
    return sum(1 for ch in text if not ch.isspace())


@dataclass
class SentFeat:
    text: str
    units: int
    anchors: Set[str] = field(default_factory=set)
    pmi_vocab: Set[str] = field(default_factory=set)


def extract(sentences: List[str]) -> List[SentFeat]:
    return [
        SentFeat(
            text=s,
            units=unit_count(s),
            anchors=hard_anchors(s),
            pmi_vocab=pmi_vocab(s),
        )
        for s in sentences
    ]


def group_units(feats: List[SentFeat], start: int, end: int) -> int:
    return sum(f.units for f in feats[start:end])


def group_anchors(feats: List[SentFeat], start: int, end: int) -> Set[str]:
    out: Set[str] = set()
    for f in feats[start:end]:
        out |= f.anchors
    return out


def group_vocab(feats: List[SentFeat], start: int, end: int) -> Set[str]:
    out: Set[str] = set()
    for f in feats[start:end]:
        out |= f.pmi_vocab
    return out


class LexicalModel:
    """Word-translation model estimated from the data itself.

    Candidate translation pairs come from co-occurrences inside confident
    1:1 beads. Probabilities are document-level:
        P(a) = df(a)/n,  P(b) = df(b)/n,  P(a,b) = cooc(a,b)/n
    so that a token appearing in every sentence (e.g. a ubiquitous word)
    gets PMI ~= 0 and is filtered out, while stable correspondences get
    positive PMI. Only positive-PMI pairs are kept.
    """

    def __init__(self, min_pmi: float = 0.0):
        self.min_pmi = min_pmi
        self.pmi: Dict[Tuple[str, str], float] = {}
        self.n_docs: int = 0

    def train(self, pairs: List[Tuple[Set[str], Set[str]]]) -> None:
        cooc: Dict[Tuple[str, str], int] = {}
        src_df: Dict[str, int] = {}
        tgt_df: Dict[str, int] = {}
        n = len(pairs)
        self.n_docs = n
        if n == 0:
            self.pmi = {}
            return
        for src_vocab, tgt_vocab in pairs:
            for a in src_vocab:
                src_df[a] = src_df.get(a, 0) + 1
            for b in tgt_vocab:
                tgt_df[b] = tgt_df.get(b, 0) + 1
            for a in src_vocab:
                for b in tgt_vocab:
                    key = (a, b)
                    cooc[key] = cooc.get(key, 0) + 1
        self.pmi = {}
        for (a, b), c_ab in cooc.items():
            pmi = math.log((c_ab * n) / (src_df[a] * tgt_df[b]))
            if pmi > self.min_pmi:
                self.pmi[(a, b)] = pmi

    def pair_pmi(self, a: str, b: str) -> Optional[float]:
        return self.pmi.get((a, b))

    def similarity(self, src_vocab: Set[str], tgt_vocab: Set[str]) -> Tuple[float, int]:
        """Mean PMI over the co-occurring candidate pairs, plus pair count."""
        vals = [self.pmi[(a, b)] for a in src_vocab for b in tgt_vocab if (a, b) in self.pmi]
        if not vals:
            return 0.0, 0
        vals.sort(reverse=True)
        return sum(vals) / len(vals), len(vals)


def lexical_cost(
    src_anchors: Set[str],
    tgt_anchors: Set[str],
    src_vocab: Set[str],
    tgt_vocab: Set[str],
    lex: Optional[LexicalModel],
    contradiction_penalty: float = 3.0,
    positive_bonus: float = 0.8,
    no_evidence_cost: float = 0.4,
    pmi_norm: float = 4.0,
) -> Tuple[float, bool]:
    """Lexical cost for a matched bead; lower is better.

    A contradiction means both sides carry hard anchors but share none of
    them (e.g. "order A100" vs "invoice C900").
    """
    cost = 0.0
    contradiction = False
    if src_anchors and tgt_anchors:
        dice = 2.0 * len(src_anchors & tgt_anchors) / (len(src_anchors) + len(tgt_anchors))
        if dice == 0.0:
            contradiction = True
            cost += contradiction_penalty
        else:
            cost -= positive_bonus * dice
    if lex is not None and lex.pmi and src_vocab and tgt_vocab:
        sim, _n = lex.similarity(src_vocab, tgt_vocab)
        if sim > 0.0:
            cost -= min(1.5, sim / pmi_norm)
        elif not contradiction and not (src_anchors and tgt_anchors):
            cost += no_evidence_cost
    return cost, contradiction
