"""纠错引擎：统一的全文 / 增量入口。

一致性保证：analyze(text) 与 分块 feed(...) + flush() 走同一个
_analyze_lines(lines, base_offset)，增量只是按「完整行」切分后逐段调用，
偏移量用已消费字符数累加，因此两条路径产出完全一致（有自测断言）。

排序（可复现）：
  1. 命中强度 strength 降序：规则命中=1.0；近似命中=(max_len-距离)/max_len
  2. 规则置信度 confidence 降序
  3. 上下文长度 context_len 降序（左右字面量上下文总长）
  4. 并列打破：start 升序 -> end 升序 -> rule_id 字典序 -> 建议文本字典序
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .approximate import ApproximateMatcher
from .rules import LiteralTrie, Rule, literal_hits, regex_hits
import re

from .scan import (
    cjk_runs,
    is_cjk,
    latin_tokens,
    normalize,
    protected_intervals,
)

_FENCE_RE = re.compile(r"(```+|~~~+)")

_SORT_KEY = lambda s: (  # noqa: E731
    -s.strength,
    -s.confidence,
    -s.context_len,
    s.start,
    s.end,
    s.rule_id,
    s.suggestion,
)


@dataclass(frozen=True)
class Suggestion:
    start: int
    end: int
    original: str
    suggestion: str
    reason: str
    confidence: float
    rule_id: str
    kind: str  # rule / approx
    strength: float = 1.0
    context_len: int = 0

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "original": self.original,
            "suggestion": self.suggestion,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "rule_id": self.rule_id,
            "kind": self.kind,
            "strength": round(self.strength, 4),
            "context_len": self.context_len,
        }


class WritingAid:
    def __init__(self, config: dict):
        self.config = config
        self.rules: list[Rule] = []
        for raw in config.get("rules", []):
            rule = Rule(
                id=raw["id"],
                pattern=raw["pattern"],
                replacement=raw["replacement"],
                reason=raw.get("reason", ""),
                confidence=float(raw.get("confidence", 0.9)),
                kind=raw.get("kind", "literal"),
                left_context=raw.get("left_context"),
                right_context=raw.get("right_context"),
                ascii_word=raw.get("ascii_word"),
            )
            if rule.kind == "literal" and "\n" in rule.pattern:
                raise ValueError(f"规则 {rule.id} 的 pattern 不得跨行")
            self.rules.append(rule)
        self.literal_rules = [r for r in self.rules if r.kind == "literal"]
        self.regex_rules = [r for r in self.rules if r.kind == "regex"]

        approx_cfg = config.get("approximate", {})
        self.max_distance = int(approx_cfg.get("max_distance", 1))
        self.prune = approx_cfg.get("prune", "index")
        self.min_latin_len = int(approx_cfg.get("min_latin_len", 4))
        self.min_cjk_len = int(approx_cfg.get("min_cjk_len", 4))
        self.approx_base_confidence = float(approx_cfg.get("base_confidence", 0.6))
        self.approx_enabled = bool(approx_cfg.get("enabled", True))

        cjk_dict = config.get("cjk_dictionary", [])
        en_dict = config.get("en_dictionary", [])
        self.cjk_matcher = ApproximateMatcher(cjk_dict, self.max_distance, prune=self.prune)
        self.en_matcher = ApproximateMatcher(en_dict, self.max_distance, prune=self.prune)
        self._cjk_lengths = sorted({len(w) for w in cjk_dict}, reverse=True)

        whitelist = list(config.get("whitelist", [])) + list(config.get("proper_nouns", []))
        self._whitelist_trie = LiteralTrie((w, w) for w in whitelist)
        self._whitelist_trie.build()
        self._dict_trie = LiteralTrie((w, w) for w in cjk_dict)
        self._dict_trie.build()

        self.reset()

    # ---------- 增量状态 ----------
    def reset(self):
        self._buffer = ""
        self._offset = 0
        self._pending_cr = False
        self._in_fence = False
        self._fence_mark = "```"

    # ---------- 公共入口 ----------
    def analyze(self, text: str) -> list[Suggestion]:
        """整体一次处理。"""
        self.reset()
        text = normalize(text)
        lines = [ln + "\n" for ln in text.split("\n")]
        if lines:
            lines[-1] = lines[-1][:-1]  # 最后一行不带换行
        out = self._analyze_lines(lines, 0)
        return sorted(out, key=_SORT_KEY)

    def feed(self, chunk: str) -> list[Suggestion]:
        """增量送入一块文本；返回本次新完成的行产生的建议（可能为空）。"""
        if self._pending_cr:
            chunk = "\r" + chunk
            self._pending_cr = False
        if chunk.endswith("\r"):
            self._pending_cr = True
            chunk = chunk[:-1]
        self._buffer += normalize(chunk)
        parts = self._buffer.split("\n")
        complete = [ln + "\n" for ln in parts[:-1]]
        self._buffer = parts[-1]
        if not complete:
            return []
        out = self._analyze_lines(complete, self._offset)
        self._offset += sum(len(ln) for ln in complete)
        return sorted(out, key=_SORT_KEY)

    def flush(self) -> list[Suggestion]:
        """收尾：处理缓冲区中不足一行的剩余文本。"""
        tail, self._buffer = self._buffer, ""
        if self._pending_cr:
            tail += "\r"
            self._pending_cr = False
        tail = normalize(tail)
        if not tail:
            return []
        out = self._analyze_lines([tail], self._offset)
        self._offset += len(tail)
        return sorted(out, key=_SORT_KEY)

    def analyze_chunked(self, text: str, chunk_size: int = 64) -> list[Suggestion]:
        """便捷方法：分块处理并汇总，结果与 analyze(text) 完全一致。"""
        self.reset()
        out: list[Suggestion] = []
        for i in range(0, len(text), chunk_size):
            out.extend(self.feed(text[i:i + chunk_size]))
        out.extend(self.flush())
        return sorted(out, key=_SORT_KEY)

    # ---------- 核心：逐行分析 ----------
    def _analyze_lines(self, lines: list[str], base: int) -> list[Suggestion]:
        # 围栏代码块是跨行结构，状态机放在这里维护；围栏内整行跳过分析。
        # 增量模式按完整行送入，状态随 _analyze_lines 的调用自然延续。
        suggestions: list[Suggestion] = []
        offset = base
        for line in lines:
            text = line[:-1] if line.endswith("\n") else line
            stripped = text.lstrip(" \t")
            if self._in_fence:
                if stripped.startswith(self._fence_mark):
                    self._in_fence = False
                offset += len(line)
                continue
            m = _FENCE_RE.match(stripped)
            if m:
                mark = m.group(1)
                self._fence_mark = "```" if mark.startswith("`") else "~~~"
                self._in_fence = True
                offset += len(line)
                continue
            suggestions.extend(self._analyze_line(text, offset))
            offset += len(line)
        return suggestions

    def _analyze_line(self, text: str, base: int) -> list[Suggestion]:
        if not text:
            return []
        protected = protected_intervals(text)
        protected = _merge_intervals(
            protected + [(s, e) for s, e, _k, _p in self._whitelist_trie.scan(text)]
        )
        gaps = _gaps(len(text), protected)

        out: list[Suggestion] = []
        for rule, start, end in literal_hits(text, self.literal_rules, protected):
            out.append(self._rule_suggestion(rule, text, start, end, base))
        for rule, start, end in regex_hits(text, self.regex_rules, protected):
            out.append(self._rule_suggestion(rule, text, start, end, base))
        if self.approx_enabled:
            rule_spans = [(s.start - base, s.end - base) for s in out]
            out.extend(self._approx_line(text, base, gaps, rule_spans))
        return out

    def _rule_suggestion(self, rule: Rule, text: str, start: int, end: int, base: int) -> Suggestion:
        original = text[start:end]
        if rule.kind == "regex":
            import re

            replaced = re.sub(rule.pattern, rule.replacement, original, count=1)
        else:
            left = rule.left_context or ""
            right = rule.right_context or ""
            core = original[len(left): len(original) - len(right) if right else len(original)]
            replaced = left + rule.replacement + right if (left or right) else rule.replacement
            _ = core
        return Suggestion(
            start=base + start,
            end=base + end,
            original=original,
            suggestion=replaced,
            reason=rule.reason or f"规则 {rule.id} 命中",
            confidence=rule.confidence,
            rule_id=rule.id,
            kind="rule",
            strength=1.0,
            context_len=rule.context_len,
        )

    # ---------- 近似匹配 ----------
    def _approx_line(self, text: str, base: int, gaps, rule_spans=()) -> list[Suggestion]:
        candidates = []  # (start, end, original, suggestion, distance)
        for lo, hi in gaps:
            seg = text[lo:hi]
            # 拉丁 token
            for rel, token in latin_tokens(seg):
                hit = self._approx_latin(token)
                if hit:
                    candidates.append((lo + rel, lo + rel + len(token), token, hit[0], hit[1]))
            # CJK 滑窗（长度分桶：窗口长度取自词典长度集合）
            for rel, run in cjk_runs(seg):
                occupied = [(s, e) for s, e, _k, _p in self._dict_trie.scan(run)]
                for length in self._cjk_lengths:
                    if length < self.min_cjk_len or length > len(run):
                        continue
                    for i in range(0, len(run) - length + 1):
                        if _inside_any(i, i + length, occupied):
                            continue
                        window = run[i:i + length]
                        hit = self.cjk_matcher.correct(window)
                        if hit:
                            candidates.append(
                                (lo + rel + i, lo + rel + i + length, window, hit[0], hit[1])
                            )
        # 非极大值抑制：重叠的近似命中只保留最优；与规则命中重叠时规则优先，近似不报
        picked = []
        for cand in sorted(candidates, key=lambda c: (-self._approx_conf(c[2], c[3], c[4]), c[0], c[1], c[3])):
            if any(cand[0] < r[1] and cand[1] > r[0] for r in rule_spans):
                continue
            if any(cand[0] < p[1] and cand[1] > p[0] for p in picked):
                continue
            picked.append(cand)
        out = []
        for start, end, original, suggestion, dist in picked:
            conf = self._approx_conf(original, suggestion, dist)
            max_len = max(len(original), len(suggestion))
            out.append(
                Suggestion(
                    start=base + start,
                    end=base + end,
                    original=original,
                    suggestion=suggestion,
                    reason=f"疑似错别字：'{original}' 与词典词 '{suggestion}' 的编辑距离为 {dist}",
                    confidence=conf,
                    rule_id="approx:" + ("cjk" if is_cjk(original[0]) else "latin"),
                    kind="approx",
                    strength=(max_len - dist) / max_len,
                    context_len=0,
                )
            )
        return out

    def _approx_latin(self, token: str):
        if len(token) < self.min_latin_len:
            return None
        if any(ch.isdigit() for ch in token):
            return None  # 含数字的标识符不做近似
        if len(token) > 1 and token.isupper():
            return None  # 全大写多为缩写/常量
        if self.en_matcher.is_known(token.lower()) or self.en_matcher.is_known(token):
            return None
        query = token.lower()
        hit = self.en_matcher.correct(query)
        if not hit:
            return None
        suggestion, dist = hit
        if token[0].isupper():
            suggestion = suggestion.capitalize()
        return suggestion, dist

    def _approx_conf(self, original: str, suggestion: str, dist: int) -> float:
        # 基础置信度随词长提升（长词单字误写更可信），距离每增加 1 扣 0.15，封顶 0.95
        length_factor = min(1.0, max(len(original), len(suggestion)) / 8.0)
        conf = self.approx_base_confidence + 0.30 * length_factor - 0.15 * (dist - 1)
        return round(min(0.95, max(0.0, conf)), 4)


def _merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _gaps(length: int, protected):
    gaps = []
    pos = 0
    for lo, hi in protected:
        if lo > pos:
            gaps.append((pos, lo))
        pos = max(pos, hi)
    if pos < length:
        gaps.append((pos, length))
    return gaps


def _inside_any(start: int, end: int, intervals) -> bool:
    for lo, hi in intervals:
        if start < hi and end > lo:
            return True
    return False
