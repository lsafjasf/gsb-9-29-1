"""Aho-Corasick 自动机，保证超长文本下匹配为 O(n + 命中数)。仅用标准库。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternHit:
    start: int
    end: int  # 半开区间
    label: str


class AhoCorasick:
    def __init__(self, patterns: list[tuple[str, str]]):
        # patterns: [(pattern_text, label)]
        self.goto: list[dict[str, int]] = [{}]
        self.out: list[list[tuple[str, str]]] = [[]]
        self.fail: list[int] = [0]
        for text, label in patterns:
            if not text:
                continue
            self._add(text, label)
        self._build_failure()

    def _add(self, text: str, label: str) -> None:
        node = 0
        for ch in text:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                nxt = len(self.goto)
                self.goto[node][ch] = nxt
                self.goto.append({})
                self.out.append([])
                self.fail.append(0)
            node = nxt
        self.out[node].append((text, label))

    def _build_failure(self) -> None:
        queue: deque[int] = deque()
        for child in self.goto[0].values():
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
                self.out[child].extend(self.out[self.fail[child]])

    def search(self, text: str) -> list[PatternHit]:
        hits: list[PatternHit] = []
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pat, label in self.out[node]:
                start = i - len(pat) + 1
                hits.append(PatternHit(start, i + 1, label))
        return hits
