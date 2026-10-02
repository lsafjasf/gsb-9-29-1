"""writingaid：写作辅助纠错建议库（仅标准库）。

用法：
    from writingaid import WritingAid, load_config
    aid = WritingAid(load_config())
    for s in aid.analyze("文本"):
        print(s.to_dict())
"""

from .config import load_config
from .engine import Suggestion, WritingAid

__all__ = ["WritingAid", "Suggestion", "load_config"]
__version__ = "0.1.0"
