"""匹配器：还原结果上跑 AC 自动机，做白名单抑制、置信度分级与证据组装。"""
from __future__ import annotations

from dataclasses import dataclass

from .aho_corasick import AhoCorasick, PatternHit
from .config import Config
from .normalizer import Change, Normalized, Normalizer


@dataclass
class Match:
    entry_id: str
    category: str
    word: str
    start: int          # 原文起始偏移（code point）
    end: int            # 原文结束偏移（半开）
    matched_text: str   # 原文片段
    normalized_text: str
    confidence: str     # exact / high / medium / low
    rules: list[str]

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "category": self.category,
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "matched_text": self.matched_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "rules": self.rules,
        }


@dataclass
class ScanResult:
    text: str
    normalized: Normalized
    matches: list[Match]


class ContentModerator:
    def __init__(self, config: Config):
        self.cfg = config
        self.normalizer = Normalizer(config)
        lex_patterns: list[tuple[str, str]] = []
        self.entries: dict[str, dict[str, str]] = {}
        for entry in config.lexicon:
            canon = self.normalizer.normalize(entry["text"]).text
            lex_patterns.append((canon, entry["id"]))
            self.entries[entry["id"]] = {
                "word": canon,
                "category": entry.get("category", ""),
            }
        self.lex_ac = AhoCorasick(lex_patterns)

        allow_patterns = [
            (self.normalizer.normalize(phrase).text, f"allowlist:{phrase}")
            for phrase in config.allowlist
        ]
        allow_patterns = [(t, l) for t, l in allow_patterns if t]
        self.allow_ac = AhoCorasick(allow_patterns)

    def scan(self, text: str) -> ScanResult:
        normalized = self.normalizer.normalize(text)
        intervals = self._merge_intervals(
            [(h.start, h.end) for h in self.allow_ac.search(normalized.text)]
        )
        matches: list[Match] = []
        for hit in self.lex_ac.search(normalized.text):
            if self._contained(hit, intervals):
                continue
            match = self._build_match(text, normalized, hit)
            if match.confidence == "exact" and not self._word_boundary_ok(
                normalized.text, hit.start, hit.end
            ):
                # 仅精确 ASCII 命中要求词边界，避免 drugs 命中 drugstore；
                # 经 leet/符号还原的变体不受此限制（drug5 本就是刻意混淆）。
                continue
            matches.append(match)
        return ScanResult(text=text, normalized=normalized, matches=matches)

    @staticmethod
    def _word_boundary_ok(text: str, start: int, end: int) -> bool:
        chunk = text[start:end]
        if not chunk.isascii() or not any(ch.isalpha() for ch in chunk):
            return True
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        for neighbor in (left, right):
            if neighbor and (neighbor.isalnum() or neighbor == "_"):
                return False
        return True

    @staticmethod
    def _merge_intervals(raw: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not raw:
            return []
        raw.sort()
        merged: list[list[int]] = [list(raw[0])]
        for start, end in raw[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(a, b) for a, b in merged]

    @staticmethod
    def _contained(hit: PatternHit, intervals: list[tuple[int, int]]) -> bool:
        for start, end in intervals:
            if start <= hit.start and hit.end <= end:
                return True
        return False

    def _build_match(
        self, text: str, normalized: Normalized, hit: PatternHit
    ) -> Match:
        entry = self.entries[hit.label]
        orig_start = normalized.spans[hit.start][0]
        orig_end = normalized.spans[hit.end - 1][1]
        changes = [
            c
            for c in normalized.changes
            if c.start < orig_end and c.end > orig_start
        ]
        rules = self._rules(hit.label, entry["category"], changes)
        return Match(
            entry_id=hit.label,
            category=entry["category"],
            word=entry["word"],
            start=orig_start,
            end=orig_end,
            matched_text=text[orig_start:orig_end],
            normalized_text=normalized.text[hit.start:hit.end],
            confidence=self._confidence(changes),
            rules=rules,
        )

    @staticmethod
    def _rules(entry_id: str, category: str, changes: list[Change]) -> list[str]:
        prefix = {"fold": "shape", "sequence": "sequence", "char": "char"}
        rules = [f"lexicon:{entry_id}({category})"]
        seen = {rules[0]}
        for ch in changes:
            name = f"{prefix[ch.kind]}:{ch.rule}" if ch.kind in prefix else ch.rule
            if name not in seen:
                seen.add(name)
                rules.append(name)
        return rules

    @staticmethod
    def _confidence(changes: list[Change]) -> str:
        kinds = {
            c.kind
            for c in changes
            if not c.dropped and c.kind in {"char", "sequence"}
        }
        if not changes:
            return "exact"
        if not kinds:
            return "high"
        if "sequence" in kinds:
            return "low"
        return "medium"
