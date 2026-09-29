"""配置加载与校验。映射表全部来自 JSON 配置，便于运营维护。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    lexicon: list[dict[str, str]]
    allowlist: list[str]
    nfkc_casefold: bool
    strip_categories: set[str]
    extra_strip: set[str]
    extra_keep: set[str]
    keep_space_inside_latin: bool
    char_map: dict[str, list[str]]
    sequence_map: list[dict[str, str]]
    max_consecutive: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        noise = data.get("noise", {})
        repeat = data.get("repeat_collapse", {})
        return cls(
            lexicon=list(data.get("lexicon", [])),
            allowlist=list(data.get("allowlist", [])),
            nfkc_casefold=bool(data.get("normalize", {}).get("nfkc_casefold", True)),
            strip_categories=set(noise.get("strip_categories", [])),
            extra_strip=set(noise.get("extra_strip", [])),
            extra_keep=set(noise.get("extra_keep", [])),
            keep_space_inside_latin=bool(noise.get("keep_space_inside_latin", True)),
            char_map={k: list(v) for k, v in data.get("char_map", {}).items()},
            sequence_map=list(data.get("sequence_map", [])),
            max_consecutive=int(repeat.get("max_consecutive", 1)),
        )


def load_config(path: str | Path) -> Config:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = Config.from_dict(data)
    for entry in cfg.lexicon:
        if not entry.get("id") or not entry.get("text"):
            raise ValueError("lexicon 条目必须包含 id 与 text")
    for seq in cfg.sequence_map:
        if not seq.get("key") or "to" not in seq:
            raise ValueError("sequence_map 条目必须包含 key 与 to")
    return cfg
