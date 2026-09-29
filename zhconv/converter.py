"""简繁转换器：词级最长匹配消解一对多映射，无法消解时保留原字并标记。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mapping import DirectionTable, Mapping

DEFAULT_MARK = "⟦{}⟧"


@dataclass
class PendingItem:
    """一处待确认：源文中的位置、原字与候选。"""

    index: int
    char: str
    candidates: list[str]


@dataclass
class ConversionResult:
    source: str
    text: str                 # 带待确认标记的输出
    plain_text: str           # 去掉标记的输出（原字保留）
    pending: list[PendingItem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.pending


class Converter:
    """单向转换器。用 Mapping.s2t / Mapping.t2t 构造，或直接用 DirectionTable。"""

    def __init__(self, table: DirectionTable, mark: str = DEFAULT_MARK):
        if "{}" not in mark:
            raise ValueError("mark 必须包含 {} 占位符")
        self._chars = table.char_map
        self._words = table.all_words
        self._max_word_len = max((len(w) for w in self._words), default=0)
        self.mark = mark

    def convert(self, text: str) -> ConversionResult:
        out_marked: list[str] = []
        out_plain: list[str] = []
        pending: list[PendingItem] = []
        i = 0
        n = len(text)
        while i < n:
            word_target = self._match_word(text, i)
            if word_target is not None:
                out_marked.append(word_target)
                out_plain.append(word_target)
                i += len(word_target)
                continue
            char = text[i]
            candidates = self._chars.get(char)
            if candidates is None or len(candidates) == 1:
                replacement = candidates[0] if candidates else char
                out_marked.append(replacement)
                out_plain.append(replacement)
            else:
                pending.append(PendingItem(i, char, list(candidates)))
                out_marked.append(self.mark.format(char))
                out_plain.append(char)
            i += 1
        return ConversionResult(text, "".join(out_marked), "".join(out_plain), pending)

    def _match_word(self, text: str, pos: int) -> str | None:
        """从左向右最长匹配词表（含专名）。返回目标词或 None。"""
        upper = min(self._max_word_len, len(text) - pos)
        for length in range(upper, 1, -1):
            segment = text[pos : pos + length]
            target = self._words.get(segment)
            if target is not None:
                return target
        return None


def make_converters(mapping: Mapping, mark: str = DEFAULT_MARK) -> dict[str, Converter]:
    """一次构造双向转换器：{'s2t': ..., 't2s': ...}。"""
    return {
        "s2t": Converter(mapping.s2t, mark),
        "t2s": Converter(mapping.t2s, mark),
    }
