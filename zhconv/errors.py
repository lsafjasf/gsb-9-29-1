"""异常定义。"""


class MappingError(Exception):
    """映射表加载/校验失败。"""


class MappingConflictError(MappingError):
    """映射表内部存在冲突（重复键、一词多义、规则不一致等）。"""
