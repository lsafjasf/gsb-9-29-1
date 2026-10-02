"""规则匹配路径：字面量规则（Aho-Corasick 自动机）与正则规则。

每条规则：
  id            稳定标识（并列打破时参与排序）
  kind          literal / regex
  pattern       错误写法（literal）或正则（regex，不跨行）
  replacement   建议文本；regex 时可使用反向引用 \\1
  reason        命中原因（用词/搭配/标点）
  confidence    规则置信度 0..1
  left_context  可选，要求紧邻命中左侧出现的字面量（计入命中范围与替换结果）
  right_context 可选，要求紧邻命中右侧出现的字面量
  ascii_word    字面量规则是否要求拉丁词边界（默认按 pattern 是否含 CJK 推断）

字面量扫描使用 Aho-Corasick，复杂度 O(n + 命中数)，与规则数量无关；
白名单表达也挂在同一类自动机上（最长匹配）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .scan import is_cjk, is_latin_word_char


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: str
    replacement: str
    reason: str = ""
    confidence: float = 0.9
    kind: str = "literal"
    left_context: str | None = None
    right_context: str | None = None
    ascii_word: bool | None = None

    @property
    def context_len(self) -> int:
        return len(self.left_context or "") + len(self.right_context or "")

    @property
    def require_ascii_word_boundary(self) -> bool:
        if self.ascii_word is not None:
            return self.ascii_word
        return not any(is_cjk(c) for c in self.pattern)


class LiteralTrie:
    """Aho-Corasick 自动机；items 为 (key, payload)，payload 可重复。"""

    def __init__(self, items=()):
        # goto: [dict]；每个节点 outputs: list[(length, key, payload)]
        self.goto = [{}]
        self.fail = [0]
        self.outputs: list[list[tuple[int, str, object]]] = [[]]
        self._built = True
        for key, payload in items:
            self.add(key, payload)

    def add(self, key: str, payload):
        node = 0
        for ch in key:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto[node][ch] = nxt
                self.goto.append({})
                self.fail.append(0)
                self.outputs.append([])
            node = nxt
        self.outputs[node].append((len(key), key, payload))

    def build(self):
        from collections import deque

        queue = deque()
        for ch, child in self.goto[0].items():
            self.fail[child] = 0
            queue.append(child)
        while queue:
            node = queue.popleft()
            for ch, child in self.goto[node].items():
                queue.append(child)
                f = self.fail[node]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[child] = self.goto[f].get(ch, 0)
                self.outputs[child] = self.outputs[child] + self.outputs[self.fail[child]]

    def scan(self, text: str):
        """产出 (start, end, key, payload)，含所有模式（不做最长选择，调用方自决）。"""
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for length, key, payload in self.outputs[node]:
                yield i - length + 1, i + 1, key, payload


def _ascii_boundary_ok(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before and is_latin_word_char(before):
        return False
    if after and (is_latin_word_char(after) or after in "-_"):
        return False
    return True


def literal_hits(text: str, rules: list[Rule], protected):
    """对一组 literal 规则做 AC 扫描，按规则逐个返回不重叠左优先命中。"""
    trie = LiteralTrie((r.pattern, r) for r in rules)
    trie.build()
    # 同一规则可能在同一位置被多次报告（重复模式），用 (rule_id, start) 去重
    by_rule: dict[str, list[tuple[int, int]]] = {}
    for start, end, _key, rule in trie.scan(text):
        if rule.require_ascii_word_boundary and not _ascii_boundary_ok(text, start, end):
            continue
        if _overlaps_protected(start, end, protected):
            continue
        left = rule.left_context or ""
        right = rule.right_context or ""
        lstart = start - len(left)
        if left and (lstart < 0 or text[lstart:start] != left):
            continue
        rend = end + len(right)
        if right and (rend > len(text) or text[end:rend] != right):
            continue
        span = (lstart, rend)
        lst = by_rule.setdefault(rule.id, [])
        if lst and span[0] < lst[-1][1]:  # AC 输出按结束位置有序；丢弃重叠
            continue
        lst.append(span)
        yield rule, span[0], span[1]


def regex_hits(text: str, rules: list[Rule], protected):
    """逐条执行正则规则（数量少；每条 O(n)）。"""
    for rule in rules:
        for m in re.finditer(rule.pattern, text):
            start, end = m.span()
            if "\n" in text[start:end]:
                continue
            if _overlaps_protected(start, end, protected):
                continue
            yield rule, start, end


def _overlaps_protected(start: int, end: int, protected) -> bool:
    for lo, hi in protected:
        if end <= lo:
            break
        if start < hi and end > lo:
            return True
    return False
