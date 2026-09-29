"""Aho-Corasick 多模式串匹配自动机（纯标准库实现）。

扫描复杂度 O(文本长度 + 命中数)，与模式数量无关，适合超长文本。
"""

from collections import deque


class Automaton:
    def __init__(self):
        self._trans = [{}]   # state -> {char: next_state}
        self._fail = [0]     # state -> fail state
        self._out = [[]]     # state -> [pattern_id, ...]
        self.patterns = []   # pattern_id -> pattern string

    def add(self, pattern):
        """添加模式串，返回 pattern_id。空模式返回 None。"""
        if not pattern:
            return None
        pid = len(self.patterns)
        self.patterns.append(pattern)
        state = 0
        for ch in pattern:
            nxt = self._trans[state].get(ch)
            if nxt is None:
                nxt = len(self._trans)
                self._trans[state][ch] = nxt
                self._trans.append({})
                self._fail.append(0)
                self._out.append([])
            state = nxt
        self._out[state].append(pid)
        return pid

    def build(self):
        """构建失败指针，所有模式添加完后调用一次。"""
        queue = deque()
        for st in self._trans[0].values():
            self._fail[st] = 0
            queue.append(st)
        while queue:
            r = queue.popleft()
            for ch, s in self._trans[r].items():
                queue.append(s)
                f = self._fail[r]
                while f and ch not in self._trans[f]:
                    f = self._fail[f]
                self._fail[s] = self._trans[f].get(ch, 0)
                self._out[s].extend(self._out[self._fail[s]])

    def search(self, text):
        """扫描文本，产出 (start, end, pattern_id)，end 为开区间。"""
        state = 0
        trans, fail, out, patterns = self._trans, self._fail, self._out, self.patterns
        for i, ch in enumerate(text):
            while state and ch not in trans[state]:
                state = fail[state]
            state = trans[state].get(ch, 0)
            for pid in out[state]:
                yield i - len(patterns[pid]) + 1, i + 1, pid
