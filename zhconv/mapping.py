"""映射表：加载、校验与冲突检测。

配置文件格式（JSON）::

    {
      "direction": "s2t",
      "one_to_one": { "发": "發", ... },          // 一对一字映射
      "one_to_many": {
        "后": {                                   // 一对多字映射
          "candidates": ["後", "后"],             // 候选目标字
          "default": "後",                        // 可选：无词规则命中时的兜底
          "words": {                              // 词级消解规则：整词 -> 整词
            "皇后": "皇后",
            "后面": "後面"
          }
        }
      }
    }

约束（加载时校验，违反即抛 MappingConflictError / MappingError）：
- 同一个字不得同时出现在 one_to_one 与 one_to_many；
- JSON 对象内不允许重复键；
- 词规则中的词必须包含该多义字，且目标词与源词等长（转换保持逐字等长）；
- 词在源词中对应多义字的位置，目标字必须属于 candidates；
- 同一个词不得在多个多义字条目下给出不同目标；
- default 必须属于 candidates。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import MappingConflictError, MappingError


def _unique_object(pairs):
    """object_pairs_hook：检出 JSON 对象内的重复键。"""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise MappingConflictError(f"JSON 对象中存在重复键: {key!r}")
        obj[key] = value
    return obj


@dataclass
class AmbiguousEntry:
    """一个一对多映射条目。"""

    char: str
    candidates: list[str]
    default: str | None = None
    words: dict[str, str] = field(default_factory=dict)


class MappingTable:
    """一个方向的映射表。"""

    def __init__(self, direction, one_to_one=None, one_to_many=None):
        self.direction = direction
        self.one_to_one: dict[str, str] = dict(one_to_one or {})
        self.ambiguous: dict[str, AmbiguousEntry] = {}
        self.word_rules: dict[str, str] = {}
        for char, spec in (one_to_many or {}).items():
            self.ambiguous[char] = AmbiguousEntry(
                char=char,
                candidates=list(spec.get("candidates", [])),
                default=spec.get("default"),
                words=dict(spec.get("words", {})),
            )
        self._validate()
        for entry in self.ambiguous.values():
            for word, target in entry.words.items():
                self.word_rules[word] = target

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dict(cls, data: dict) -> "MappingTable":
        if not isinstance(data, dict):
            raise MappingError("映射表顶层必须是 JSON 对象")
        direction = data.get("direction")
        if direction not in ("s2t", "t2s"):
            raise MappingError(f"direction 必须是 's2t' 或 't2s'，得到: {direction!r}")
        return cls(
            direction=direction,
            one_to_one=data.get("one_to_one", {}),
            one_to_many=data.get("one_to_many", {}),
        )

    @classmethod
    def from_file(cls, path) -> "MappingTable":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise MappingError(f"{path}: JSON 解析失败: {exc}") from exc
        try:
            return cls.from_dict(data)
        except MappingError as exc:
            raise type(exc)(f"{path}: {exc}") from exc

    # -------------------------------------------------------------- validate
    def _validate(self) -> None:
        for char, target in self.one_to_one.items():
            if len(char) != 1 or len(target) != 1:
                raise MappingError(
                    f"one_to_one 的键和值都必须是单字: {char!r} -> {target!r}")

        seen_words: dict[str, str] = {}  # word -> 所属多义字
        for char, entry in self.ambiguous.items():
            if len(char) != 1:
                raise MappingError(f"one_to_many 的键必须是单字: {char!r}")
            if char in self.one_to_one:
                raise MappingConflictError(
                    f"字 {char!r} 同时出现在 one_to_one 与 one_to_many 中")
            if not entry.candidates:
                raise MappingError(f"字 {char!r} 的 candidates 不能为空")
            for cand in entry.candidates:
                if len(cand) != 1:
                    raise MappingError(
                        f"字 {char!r} 的候选必须是单字: {cand!r}")
            if len(set(entry.candidates)) != len(entry.candidates):
                raise MappingConflictError(
                    f"字 {char!r} 的 candidates 存在重复: {entry.candidates}")
            if entry.default is not None and entry.default not in entry.candidates:
                raise MappingConflictError(
                    f"字 {char!r} 的 default {entry.default!r} 不在 candidates 中")

            for word, target in entry.words.items():
                if char not in word:
                    raise MappingConflictError(
                        f"词规则 {word!r} 不包含其所属多义字 {char!r}")
                if len(word) != len(target):
                    raise MappingConflictError(
                        f"词规则 {word!r} -> {target!r} 源词与目标词长度不一致")
                for src_ch, tgt_ch in zip(word, target):
                    if src_ch == char and tgt_ch not in entry.candidates:
                        raise MappingConflictError(
                            f"词规则 {word!r} -> {target!r} 把 {char!r} 映射为 "
                            f"{tgt_ch!r}，不在 candidates {entry.candidates} 中")
                if word in seen_words:
                    raise MappingConflictError(
                        f"词 {word!r} 在字 {seen_words[word]!r} 与字 {char!r} "
                        f"的条目下被重复定义")
                seen_words[word] = char

    # -------------------------------------------------------------- query
    def lookup_char(self, char: str):
        """返回 (目标字, 是否待确认)。查不到返回 (原字, False)。"""
        if char in self.one_to_one:
            return self.one_to_one[char], False
        entry = self.ambiguous.get(char)
        if entry is None:
            return char, False
        if entry.default is not None:
            return entry.default, False
        return char, True  # 无法消解：保留原字，待确认
