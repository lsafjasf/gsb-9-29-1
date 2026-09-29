"""zhconv —— 简繁转换库（仅标准库）。

特性：
- 映射表由 JSON 配置文件提供，新增条目无需改代码；
- 支持按词/上下文消解一对多映射（如「后面→後面」与「皇后→皇后」）；
- 无法消解时保留原字并标记为待确认；
- 加载时检测表内冲突；
- 与标注语料对拍，输出逐句准确率与错误分类。
"""

from .errors import MappingError, MappingConflictError
from .mapping import MappingTable
from .converter import Converter, ConversionResult, PendingItem

__all__ = [
    "MappingError",
    "MappingConflictError",
    "MappingTable",
    "Converter",
    "ConversionResult",
    "PendingItem",
]

__version__ = "0.1.0"
