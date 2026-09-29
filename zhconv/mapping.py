"""映射表加载与校验。

映射表是一个 JSON 文件，结构如下::

    {
      "version": 1,
      "s2t": {
        "char_map":     {"后": ["後", "后"], "门": "門"},
        "word_map":     {"后面": "後面", "皇后": "皇后"},
        "proper_nouns": {"于谦": "于謙"}
      },
      "t2s": { ... 同上 ... }
    }

- char_map: 单字 -> 目标字（字符串）或候选字列表（一对多）。
- word_map / proper_nouns: 词 -> 词，用于消解一对多字映射；proper_nouns 优先级最高。
- 加载时做严格校验，任何表内冲突都会抛出 MappingError。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class MappingError(ValueError):
    """映射表内容不合法（冲突、格式错误等）。"""


def _reject_duplicate_keys(pairs):
    """object_pairs_hook：检出 JSON 对象中的重复键。"""
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise MappingError(f"映射表中存在重复键: {key!r}")
        obj[key] = value
    return obj


@dataclass
class DirectionTable:
    """单向（s2t 或 t2s）映射。"""

    char_map: dict[str, list[str]]
    word_map: dict[str, str] = field(default_factory=dict)
    proper_nouns: dict[str, str] = field(default_factory=dict)

    @property
    def all_words(self) -> dict[str, str]:
        """词表合并视图：专名优先于普通词。"""
        merged = dict(self.word_map)
        merged.update(self.proper_nouns)
        return merged


@dataclass
class Mapping:
    s2t: DirectionTable
    t2s: DirectionTable
    warnings: list[str] = field(default_factory=list)


def _validate_char_map(section: str, raw) -> dict[str, list[str]]:
    if not isinstance(raw, dict):
        raise MappingError(f"{section}.char_map 必须是对象")
    char_map: dict[str, list[str]] = {}
    for src, value in raw.items():
        if len(src) != 1:
            raise MappingError(f"{section}.char_map 的键必须是单字: {src!r}")
        candidates = [value] if isinstance(value, str) else list(value)
        if not candidates or not all(isinstance(c, str) and len(c) == 1 for c in candidates):
            raise MappingError(f"{section}.char_map[{src!r}] 的候选必须是单字或单字列表")
        if len(set(candidates)) != len(candidates):
            raise MappingError(f"{section}.char_map[{src!r}] 的候选存在重复: {candidates!r}")
        char_map[src] = candidates
    return char_map


def _validate_word_map(section: str, kind: str, raw, char_map) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise MappingError(f"{section}.{kind} 必须是对象")
    word_map: dict[str, str] = {}
    for src, tgt in raw.items():
        where = f"{section}.{kind}[{src!r}]"
        if not isinstance(tgt, str):
            raise MappingError(f"{where} 的目标必须是字符串")
        if len(src) < 2:
            raise MappingError(f"{where} 长度小于 2，单字映射请放入 char_map")
        if len(src) != len(tgt):
            raise MappingError(f"{where} 源与目标长度不一致: {src!r} -> {tgt!r}")
        for s_char, t_char in zip(src, tgt):
            if s_char in char_map:
                if t_char not in char_map[s_char]:
                    raise MappingError(
                        f"{where} 与 char_map 冲突: {s_char!r} 的候选为 "
                        f"{char_map[s_char]!r}，词表却映射为 {t_char!r}"
                    )
            elif t_char != s_char:
                raise MappingError(
                    f"{where} 引用了 char_map 中不存在的映射: {s_char!r} -> {t_char!r}"
                )
        word_map[src] = tgt
    return word_map


def _build_direction(name: str, raw) -> tuple[DirectionTable, list[str]]:
    if not isinstance(raw, dict):
        raise MappingError(f"{name} 必须是对象")
    known = {"char_map", "word_map", "proper_nouns"}
    unknown = set(raw) - known
    if unknown:
        raise MappingError(f"{name} 存在未知小节: {sorted(unknown)!r}")
    char_map = _validate_char_map(name, raw.get("char_map", {}))
    word_map = _validate_word_map(name, "word_map", raw.get("word_map", {}), char_map)
    proper_nouns = _validate_word_map(name, "proper_nouns", raw.get("proper_nouns", {}), char_map)

    for word, target in proper_nouns.items():
        if word in word_map and word_map[word] != target:
            raise MappingError(
                f"{name}: 词 {word!r} 在 word_map 与 proper_nouns 中冲突: "
                f"{word_map[word]!r} != {target!r}"
            )

    warnings = []
    covered = set()
    for word in list(word_map) + list(proper_nouns):
        covered.update(word)
    for char, candidates in char_map.items():
        if len(candidates) > 1 and char not in covered:
            warnings.append(
                f"{name}: 字 {char!r} 是一对多映射 {candidates!r}，"
                f"但词表中没有包含它的词，转换时一律标记为待确认"
            )
    return DirectionTable(char_map, word_map, proper_nouns), warnings


def load_mapping_from_dict(raw: dict) -> Mapping:
    """从已解析的 dict 构建 Mapping（便于测试与嵌入式使用）。"""
    if not isinstance(raw, dict):
        raise MappingError("映射表顶层必须是对象")
    warnings: list[str] = []
    tables = {}
    for name in ("s2t", "t2s"):
        table, w = _build_direction(name, raw.get(name, {}))
        tables[name] = table
        warnings.extend(w)
    return Mapping(tables["s2t"], tables["t2s"], warnings)


def load_mapping(path: str | Path) -> Mapping:
    """从 JSON 文件加载映射表，重复键与冲突会在此时抛出 MappingError。"""
    text = Path(path).read_text(encoding="utf-8")
    raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    return load_mapping_from_dict(raw)
