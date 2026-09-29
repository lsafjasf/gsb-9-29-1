"""转换器：最长词匹配 + 字级映射 + 待确认标记。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mapping import MappingTable

PENDING_FMT = "⟦{char}|{candidates}⟧"  # 待确认的内联标记格式


@dataclass
class PendingItem:
    """一处无法消解、被保留原字的待确认位置。"""

    position: int           # 在源文本中的下标
    char: str               # 原字
    candidates: list[str]   # 候选目标字
    context: str            # 上下文片段，便于人工确认


@dataclass
class ConversionResult:
    text: str
    pending: list[PendingItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.pending


class Converter:
    """单向（s2t 或 t2s）转换器。

    算法：对源文本从左到右扫描，每一步先做词规则的最长匹配；
    未命中词规则时按单字映射（一对一直接映射；一对多有 default 用
    default，否则保留原字并记录为待确认）；不在表中的字原样通过
    （因此简繁混排文本中已属于目标端的字不会被误改）。
    """

    def __init__(self, table: MappingTable):
        self.table = table
        self._word_rules = table.word_rules
        self._max_word_len = max((len(w) for w in self._word_rules), default=0)

    @property
    def direction(self) -> str:
        return self.table.direction

    def convert(self, text: str, mark_pending: bool = False) -> ConversionResult:
        out: list[str] = []
        pending: list[PendingItem] = []
        i, n = 0, len(text)
        while i < n:
            word, target = self._match_word(text, i)
            if word is not None:
                out.append(target)
                i += len(word)
                continue
            char = text[i]
            mapped, is_pending = self.table.lookup_char(char)
            if is_pending:
                entry = self.table.ambiguous[char]
                pending.append(PendingItem(
                    position=i,
                    char=char,
                    candidates=list(entry.candidates),
                    context=text[max(0, i - 5):i + 6],
                ))
                if mark_pending:
                    mapped = PENDING_FMT.format(
                        char=char, candidates="/".join(entry.candidates))
            out.append(mapped)
            i += 1
        return ConversionResult(text="".join(out), pending=pending)

    def _match_word(self, text: str, pos: int):
        """在 pos 处做最长词匹配，返回 (源词, 目标词) 或 (None, None)。"""
        upper = min(self._max_word_len, len(text) - pos)
        for length in range(upper, 1, -1):
            word = text[pos:pos + length]
            target = self._word_rules.get(word)
            if target is not None:
                return word, target
        return None, None
