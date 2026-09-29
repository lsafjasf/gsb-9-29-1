"""敏感词变体检出库：降噪 + 规则侧变体扩展 + 证据输出。"""

from .matcher import Hit, Matcher, ScanResult
from .normalizer import normalize

__all__ = ["Hit", "Matcher", "ScanResult", "normalize"]
