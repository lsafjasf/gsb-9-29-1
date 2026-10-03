"""sentalign: sentence alignment for bilingual parallel-ish texts.

Standard library only. See README.md for usage.
"""

from .aligner import (
    AlignConfig,
    AlignmentResult,
    Bead,
    align,
    align_texts,
    split_sentences,
)
from .evaluate import Dataset, EvalResult, GoldBead, aggregate, evaluate, load_dataset

__all__ = [
    "AlignConfig",
    "AlignmentResult",
    "Bead",
    "align",
    "align_texts",
    "split_sentences",
    "Dataset",
    "EvalResult",
    "GoldBead",
    "aggregate",
    "evaluate",
    "load_dataset",
]

__version__ = "0.1.0"
