"""Sentence splitting for English / Chinese text (standard library only)."""

import re

# Sentence-final punctuation for EN and ZH.
_FINAL = ".!?。！？…;；"
# CJK finals never need trailing whitespace to end a sentence.
_CJK_FINAL = "。！？…；"
# Closing quotes / brackets that may follow final punctuation.
_CLOSERS = "\"'”’』」）)]}"

# Titles etc. never end a sentence.
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|Fig|No|Vol|pp|Inc|Ltd|Co|[A-Z])\.$"
)
# These end a sentence only when the next word is capitalised.
_ABBREV_LOWER = re.compile(r"\b(?:etc|e\.g|i\.e|a\.m|p\.m)\.$")


def split_sentences(text):
    """Split *text* into a list of sentence strings.

    Rules (deliberately simple and deterministic):
    - A sentence ends at one of the final punctuation marks, optionally
      followed by closing quotes/brackets, when followed by whitespace,
      a newline, or end of text.
    - Blank lines also act as hard sentence boundaries.
    - Common English abbreviations do not end a sentence.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sentences = []
    buf = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            # blank line -> hard boundary
            j = i
            while j < n and text[j] in " \t\n":
                j += 1
            if "\n" in text[i:j] and "".join(buf).strip():
                _flush(buf, sentences)
            i = j
            continue
        buf.append(ch)
        if ch in _FINAL:
            # absorb consecutive final marks (e.g. "..." or "?!")
            j = i + 1
            while j < n and text[j] in _FINAL:
                buf.append(text[j])
                j += 1
            # absorb closing quotes/brackets
            while j < n and text[j] in _CLOSERS:
                buf.append(text[j])
                j += 1
            tail = "".join(buf).strip()
            if tail and not _ABBREV.search(tail):
                if _ABBREV_LOWER.search(tail):
                    # continue only before a lowercase word
                    k = j
                    while k < n and text[k] in " \t\n":
                        k += 1
                    if k < n and text[k].islower():
                        i = j
                        continue
                if ch in _CJK_FINAL or j >= n or text[j] in " \t\n":
                    _flush(buf, sentences)
                    i = j
                    continue
            i = j
            continue
        i += 1
    _flush(buf, sentences)
    return sentences


def _flush(buf, out):
    s = "".join(buf).strip()
    if s:
        out.append(re.sub(r"\s+", " ", s))
    del buf[:]
