"""term_extract -- unsupervised domain term extraction (stdlib only)."""
from .extractor import TermExtractor
from .evaluate import evaluate, load_gold

__all__ = ["TermExtractor", "evaluate", "load_gold"]
__version__ = "0.1.0"
