"""近似匹配路径：编辑距离纠错。

剪枝结构（两层）：
1. 长度分桶：编辑距离 <= max_distance 的两个字符串，长度差必然 <= max_distance，
   因此按长度分桶后只需比较 |len(a)-len(b)| <= max_distance 的桶。
2. 删除索引（SymSpell 思路，仅标准库 dict）：对词典中每个词 w，
   预计算 w 及其删除 1 个字符后的所有变体作为键，键 -> 原词列表。
   对查询串 q，用 q 及其删除变体查键，即可在 O(len(q)) 次哈希查找内
   取回所有可能与 q 编辑距离 <= 1 的候选，再逐一精确验证。

完备性（d=1）：若 lev(q, w) <= 1，则
  - 替换：q 与 w 等长且恰一位不同，删除该位后二者相等 -> 命中删除索引；
  - 插入（w 比 q 长 1）：w 的某个删除变体 == q -> 命中；
  - 删除（w 比 q 短 1）：q 的某个删除变体 == w -> 命中。
故索引不会漏掉任何 d<=1 的候选，候选集合是真实命中集的超集，再精确验证保证精确性。

复杂度：
  建索引：O(Σ|w|^2) 时间，O(Σ|w|^2) 空间（键总长）。
  单次查询：O(L) 次哈希查找 + O(C·L) 验证，C 为候选数，L 为查询长度。
  典型文本 C 为个位数；最坏情况（如全 'a' 文本对全 'a' 词典）C=O(N)，
  退化为 O(N·L)，与朴素全词典扫描同阶，但仍有正确性保证。
"""

from __future__ import annotations


def levenshtein_at_most(a: str, b: str, max_dist: int) -> int:
    """返回 lev(a, b)，若超过 max_dist 则返回 max_dist + 1（提前剪枝）。"""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, 1):
        curr = [j]
        row_min = j
        for i, ca in enumerate(a, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
            curr.append(val)
            if val < row_min:
                row_min = val
        if row_min > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]


def levenshtein(a: str, b: str) -> int:
    """精确编辑距离（无截断），用于基准对照。"""
    return levenshtein_at_most(a, b, max(len(a), len(b)))


class DeletionIndex:
    """删除变体索引：键 = 原词或其删除 max_distance 个字符后的变体。"""

    def __init__(self, words, max_distance: int = 1):
        if max_distance != 1:
            raise ValueError("DeletionIndex 目前仅支持 max_distance=1")
        self.max_distance = max_distance
        self._index: dict[str, list[str]] = {}
        for w in words:
            variants = {w}
            for i in range(len(w)):
                variants.add(w[:i] + w[i + 1:])
            for key in variants:
                bucket = self._index.get(key)
                if bucket is None:
                    self._index[key] = [w]
                elif w not in bucket:
                    bucket.append(w)

    def candidates(self, query: str) -> list[str]:
        """返回可能与 query 编辑距离 <= 1 的候选词（去重、排序保证确定性）。"""
        found: set[str] = set()
        hit = self._index.get(query)
        if hit:
            found.update(hit)
        for i in range(len(query)):
            hit = self._index.get(query[:i] + query[i + 1:])
            if hit:
                found.update(hit)
        return sorted(found)


class ApproximateMatcher:
    """近似匹配器：长度分桶 + 删除索引 + 精确验证。"""

    def __init__(self, words, max_distance: int = 1, prune: str = "index"):
        """prune:
          index  长度分桶 + 删除索引（生产默认）
          bucket 仅长度分桶
          naive  不剪枝，全词典逐个比较（基准对照/最坏情况）
        """
        self.max_distance = max_distance
        self.prune = prune
        self.use_index = prune == "index"
        self.words = sorted(set(words))
        self._word_set = set(self.words)
        self._index = DeletionIndex(self.words, max_distance) if self.use_index else None
        # 长度分桶（无论是否启用删除索引都保留，作为第一层剪枝）
        self._buckets: dict[int, list[str]] = {}
        for w in self.words:
            self._buckets.setdefault(len(w), []).append(w)

    def is_known(self, token: str) -> bool:
        return token in self._word_set

    def _raw_candidates(self, token: str) -> list[str]:
        if self.prune == "index":
            return self._index.candidates(token)
        if self.prune == "naive":
            return list(self.words)
        # 仅长度分桶：|len diff| <= max_distance 的桶内全部词
        out: list[str] = []
        for length in range(len(token) - self.max_distance,
                            len(token) + self.max_distance + 1):
            out.extend(self._buckets.get(length, ()))
        return out

    def correct(self, token: str):
        """对 token 找最佳纠正。

        返回 (建议词, 距离)；无候选或 token 本身在词典中返回 None。
        多个候选距离相同时按 (距离, 词典序) 确定性选择。
        """
        if not token or token in self._word_set:
            return None
        best = None
        for cand in self._raw_candidates(token):
            if cand == token:
                continue
            d = levenshtein_at_most(token, cand, self.max_distance)
            if d > self.max_distance:
                continue
            key = (d, cand)
            if best is None or key < best[0]:
                best = (key, cand, d)
        if best is None:
            return None
        return best[1], best[2]
