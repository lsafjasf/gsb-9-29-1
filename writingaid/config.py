"""配置加载：规则与词典全部由 JSON 配置提供。"""

from __future__ import annotations

import json
import os

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "rules.sample.json",
)


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    ids = set()
    for rule in config.get("rules", []):
        rid = rule.get("id")
        if not rid:
            raise ValueError("规则缺少 id")
        if rid in ids:
            raise ValueError(f"规则 id 重复: {rid}")
        ids.add(rid)
        if "pattern" not in rule or "replacement" not in rule:
            raise ValueError(f"规则 {rid} 缺少 pattern/replacement")
        kind = rule.get("kind", "literal")
        if kind not in ("literal", "regex"):
            raise ValueError(f"规则 {rid} 的 kind 非法: {kind}")
        conf = float(rule.get("confidence", 0.9))
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"规则 {rid} 的 confidence 越界: {conf}")
