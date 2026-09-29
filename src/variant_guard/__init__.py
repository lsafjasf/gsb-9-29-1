"""变体还原与敏感词匹配库（仅标准库）。"""
from .config import Config, load_config
from .normalizer import Normalizer, Normalized
from .matcher import Match, ContentModerator

__all__ = [
    "Config",
    "load_config",
    "Normalizer",
    "Normalized",
    "Match",
    "ContentModerator",
]
