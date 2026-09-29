"""Tokenizer for mixed Chinese/English text (stdlib only).

Produces a flat token stream where:
  - English words (incl. hyphenated / apostrophe forms) are single tokens
  - numbers are single tokens
  - each CJK ideograph is its own token (no external segmenter allowed)
  - punctuation is a token and acts as a hard boundary
"""
import re

_WORD = r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*"
_NUM = r"\d+(?:[.,]\d+)*"
_CJK = r"[一-鿿㐀-䶿]"
_PUNCT = r"[^\sA-Za-z0-9一-鿿㐀-䶿]"

TOKEN_RE = re.compile(f"{_WORD}|{_NUM}|{_CJK}|{_PUNCT}")
_CJK_RE = re.compile(_CJK)
_WORD_RE = re.compile(_WORD + r"$")

SENT_END = set(".!?。！？；;…\n")


def tokenize(text):
    """Return list of (token, start, end) triples."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def is_cjk(tok):
    return bool(_CJK_RE.fullmatch(tok))


def is_wordish(tok):
    """Tokens that may participate in a term (words, numbers, CJK chars)."""
    return bool(_WORD_RE.fullmatch(tok)) or is_cjk(tok) or tok[:1].isdigit()


def sentences(tokens):
    """Split a token list into sentence-level token lists.

    Punctuation never appears inside a sentence; sentence-final punctuation
    acts as a separator so candidates never cross sentence boundaries.
    """
    sents, cur = [], []
    for tok, _, _ in tokens:
        if tok in SENT_END:
            if cur:
                sents.append(cur)
                cur = []
        elif is_wordish(tok):
            cur.append(tok)
        else:
            # other punctuation (commas, parens, ...) is a soft barrier:
            # candidates may not cross it either
            if cur:
                sents.append(cur)
                cur = []
    if cur:
        sents.append(cur)
    return sents
