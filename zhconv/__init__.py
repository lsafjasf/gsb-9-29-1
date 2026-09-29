"""zhconv：基于可扩展映射表的简繁转换库（仅标准库）。"""

from .converter import ConversionResult, Converter, PendingItem, make_converters
from .mapping import DirectionTable, Mapping, MappingError, load_mapping, load_mapping_from_dict

__all__ = [
    "ConversionResult",
    "Converter",
    "DirectionTable",
    "Mapping",
    "MappingError",
    "PendingItem",
    "load_mapping",
    "load_mapping_from_dict",
    "make_converters",
]
