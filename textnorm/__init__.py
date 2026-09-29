"""textnorm — 文档文本规范化库（统一标点 / 空白 / 引号，支持区域豁免与回滚）。"""
from .core import (
    DEFAULT_PUNCT_MAP,
    DEFAULT_QUOTE_MAP,
    EXEMPTION_REASONS,
    Change,
    ExemptConfig,
    NormConfig,
    NormResult,
    Span,
    build_manifest,
    find_exempt_spans,
    normalize,
    rollback,
)

__all__ = [
    "DEFAULT_PUNCT_MAP",
    "DEFAULT_QUOTE_MAP",
    "EXEMPTION_REASONS",
    "Change",
    "ExemptConfig",
    "NormConfig",
    "NormResult",
    "Span",
    "build_manifest",
    "find_exempt_spans",
    "normalize",
    "rollback",
]
