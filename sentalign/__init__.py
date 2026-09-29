"""Bilingual sentence alignment (standard library only).

Public API:
    split_sentences(text)           -> list[str]
    align(src_sents, tgt_sents)     -> Alignment
"""

from .splitter import split_sentences
from .aligner import align, Alignment, Bead

__all__ = ["split_sentences", "align", "Alignment", "Bead"]
__version__ = "1.0.0"
