"""抽取式摘要库（仅使用 Python 标准库）。

打分模型：词频得分 + 位置得分 + 标题相似度，线性加权（权重可配置）。
选择策略：
  - "greedy"：最大边际相关（MMR）贪心，冗余惩罚强度可配置；
  - "dp"    ：0/1 背包动态规划，在预算内最大化句子总分（不含成对冗余项）。
预算严格：按词数或句数限制，超预算的句子不会被选中；
预算过紧时保证非空（返回单句并置 overflow 标记）。
输出顺序与原文一致；同一输入结果完全确定。
支持分块（流式）处理超长文档，结果与整体一次处理一致。
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "summarize",
    "summarize_chunks",
    "split_sentences",
    "tokenize",
    "content_words",
    "SummaryResult",
    "SentenceScore",
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
_SENT_RE = re.compile(r"[^。！？!?；;.\n]+[。！？!?；;.]*")
_TERMINATORS = frozenset("。！？!?；;.")
_EPS = 1e-12

_EN_STOPWORDS = frozenset(
    "a an the and or but if then else of in on at to for with by from as is are "
    "was were be been being it its this that these those i you he she we they "
    "not no do does did have has had will would can could should may might must".split()
)


def tokenize(text: str) -> List[str]:
    """小写化分词：英文/数字按词，中文按单字（标准库无分词器）。"""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_words(tokens: Sequence[str]) -> List[str]:
    """过滤停用词与无信息单字符（中文单字保留）。"""
    out = []
    for t in tokens:
        if t in _EN_STOPWORDS:
            continue
        if len(t) == 1 and not "一" <= t <= "鿿":
            continue
        out.append(t)
    return out


def split_sentences(text: str) -> List[str]:
    """按中英文句末标点与换行切句（不含任何有效词的纯标点段丢弃）。"""
    out = []
    for m in _SENT_RE.finditer(text):
        s = m.group(0).strip()
        if s and _TOKEN_RE.search(s):
            out.append(s)
    return out


def _iter_chunk_sentences(chunks: Iterable[str]) -> Iterator[str]:
    """流式切句：跨块边界的句子会被正确拼接，结果与 split_sentences 一致。"""
    buf = ""
    for chunk in chunks:
        buf += chunk
        cut = 0
        for m in _SENT_RE.finditer(buf):
            nxt = buf[m.end():m.end() + 1]
            if nxt and nxt not in _TERMINATORS:
                # 后续字符不可能是该句的延续，可以安全产出
                s = m.group(0).strip()
                if s and _TOKEN_RE.search(s):
                    yield s
                cut = m.end()
            else:
                # 到达缓冲区末尾，句子可能未完，留到下一块
                break
        buf = buf[cut:]
    for m in _SENT_RE.finditer(buf):
        s = m.group(0).strip()
        if s and _TOKEN_RE.search(s):
            yield s


def _cosine_sets(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) * len(b)) ** 0.5


@dataclass
class SentenceScore:
    """单句得分构成（每句都会输出）。"""

    index: int
    text: str
    n_words: int
    freq: float
    position: float
    title: float
    total: float
    mmr_penalty: float = 0.0
    mmr_value: float = 0.0
    selected: bool = False
    words: frozenset = field(default_factory=frozenset, repr=False)


@dataclass
class SummaryResult:
    sentences: List[str]
    indices: List[int]
    scores: List[SentenceScore]
    keywords: List[str]
    coverage: float
    redundancy: float
    budget: int
    budget_unit: str
    used: int
    overflow: bool
    strategy: str
    n_sentences: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def selected(self) -> List[SentenceScore]:
        return [s for s in self.scores if s.selected]


def _keywords(df: Dict[str, int], k: int, n_sent: int = 0) -> List[str]:
    """文档关键词：按文档频率取 top-k，但优先排除出现在 >50% 句子里的通用词。"""
    items = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))
    if n_sent:
        core = [w for w, c in items if c * 2 <= n_sent]
        rest = [w for w, c in items if c * 2 > n_sent]
        return (core + rest)[:k]
    return [w for w, _ in items[:k]]


def _score_one(index: int, text: str, words: frozenset, n_words: int,
               df: Dict[str, int], n_sent: int, title_words: frozenset,
               weights: Tuple[float, float, float]) -> SentenceScore:
    if words and n_sent:
        freq = sum(df.get(w, 0) / n_sent for w in words) / len(words)
    else:
        freq = 0.0
    position = 1.0 / (1.0 + index)
    title = _cosine_sets(words, title_words)
    total = weights[0] * freq + weights[1] * position + weights[2] * title
    return SentenceScore(index=index, text=text, n_words=n_words, freq=freq,
                         position=position, title=title, total=total, words=words)


def _cost(sc: SentenceScore, unit: str) -> int:
    return sc.n_words if unit == "words" else 1


def _select_greedy(cands: List[SentenceScore], budget: int, unit: str,
                   penalty: float) -> List[SentenceScore]:
    """MMR 贪心：mmr = (1-penalty)*total - penalty*max_sim(已选)。"""
    remaining = list(cands)
    selected: List[SentenceScore] = []
    used = 0
    while True:
        best: Optional[SentenceScore] = None
        best_key: Optional[Tuple[float, int]] = None
        for sc in remaining:
            if used + _cost(sc, unit) > budget:
                continue  # 放不下的句子本轮跳过
            sim = max((_cosine_sets(sc.words, s.words) for s in selected), default=0.0)
            val = (1.0 - penalty) * sc.total - penalty * sim
            sc.mmr_penalty = sim
            sc.mmr_value = val
            key = (val, -sc.index)  # 同分取原文靠前者，保证确定性
            if best_key is None or key > best_key:
                best_key = key
                best = sc
        if best is None:
            break  # 剩余句子都放不下
        remaining.remove(best)
        selected.append(best)
        used += _cost(best, unit)
    return selected


def _select_dp(cands: List[SentenceScore], budget: int, unit: str) -> List[SentenceScore]:
    """0/1 背包：在预算内最大化句子总分之和（O(n*budget)，不含成对冗余项）。"""
    dp = [0.0] * (budget + 1)
    take = [bytearray(budget + 1) for _ in cands]
    for i, sc in enumerate(cands):
        c = _cost(sc, unit)
        if c > budget:
            continue
        row = take[i]
        for w in range(budget, c - 1, -1):
            nv = dp[w - c] + sc.total
            if nv > dp[w] + _EPS:
                dp[w] = nv
                row[w] = 1
    best_w = max(range(budget + 1), key=lambda w: (dp[w], -w))
    chosen: List[SentenceScore] = []
    w = best_w
    for i in range(len(cands) - 1, -1, -1):
        if take[i][w]:
            chosen.append(cands[i])
            w -= _cost(cands[i], unit)
    chosen.reverse()
    return chosen


def _fallback_nonempty(cands: List[SentenceScore], budget: int, unit: str,
                       ) -> Tuple[List[SentenceScore], bool]:
    """预算过紧时保证非空：能放下则取最优单句，否则取全局最优单句并标记溢出。"""
    fits = [sc for sc in cands if _cost(sc, unit) <= budget]
    if fits:
        return [max(fits, key=lambda sc: (sc.total, -sc.index))], False
    return [max(cands, key=lambda sc: (sc.total, -sc.index))], True


def _finalize(all_scores: List[SentenceScore], chosen: List[SentenceScore],
              keywords: List[str], budget: int, unit: str,
              strategy: str, overflow: bool) -> SummaryResult:
    chosen = sorted(chosen, key=lambda sc: sc.index)  # 输出顺序与原文一致
    for sc in chosen:
        sc.selected = True
    used = sum(_cost(sc, unit) for sc in chosen)
    if keywords:
        have = set().union(*(sc.words for sc in chosen)) if chosen else set()
        cov = len(set(keywords) & have) / len(keywords)
    else:
        cov = 1.0
    sims = []
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            sims.append(_cosine_sets(chosen[i].words, chosen[j].words))
    red = sum(sims) / len(sims) if sims else 0.0
    return SummaryResult(
        sentences=[sc.text for sc in chosen],
        indices=[sc.index for sc in chosen],
        scores=all_scores,
        keywords=keywords,
        coverage=cov,
        redundancy=red,
        budget=budget,
        budget_unit=unit,
        used=used,
        overflow=overflow,
        strategy=strategy,
        n_sentences=len(all_scores),
    )


def _df_pass(sentences: Iterable[str]) -> Tuple[Dict[str, int], int]:
    """第一遍：只累计全局文档频率与句数，内存与词表大小成正比。"""
    df: Dict[str, int] = {}
    n = 0
    for s in sentences:
        for w in frozenset(content_words(tokenize(s))):
            df[w] = df.get(w, 0) + 1
        n += 1
    return df, n


def _score_pass(sentences: Iterable[str], df: Dict[str, int], n_sent: int,
                title_words: frozenset, weights: Tuple[float, float, float],
                candidate_limit: Optional[int],
                ) -> Tuple[List[SentenceScore], List[SentenceScore]]:
    """第二遍：逐句打分。

    candidate_limit 为 None 时保留全部句子的词集合；
    否则用堆只保留总分最高的 candidate_limit 句的词集合（其余句子的
    词集合立即释放，选择阶段内存有界），非候选句的得分构成仍完整输出。
    """
    all_scores: List[SentenceScore] = []
    if candidate_limit is None:
        for i, s in enumerate(sentences):
            tokens = tokenize(s)
            ws = frozenset(content_words(tokens))
            all_scores.append(_score_one(i, s, ws, len(tokens), df, n_sent,
                                         title_words, weights))
        return all_scores, all_scores
    heap: List[Tuple[float, int]] = []
    for i, s in enumerate(sentences):
        tokens = tokenize(s)
        ws = frozenset(content_words(tokens))
        sc = _score_one(i, s, ws, len(tokens), df, n_sent, title_words, weights)
        all_scores.append(sc)
        key = (sc.total, -i)
        if len(heap) < candidate_limit:
            heapq.heappush(heap, key)
        elif key > heap[0]:
            old = heapq.heapreplace(heap, key)
            all_scores[-old[1]].words = frozenset()  # 淘汰者释放词集合
        else:
            sc.words = frozenset()
    cands = sorted((all_scores[-k[1]] for k in heap), key=lambda sc: sc.index)
    return all_scores, cands


def _run(all_scores, cands, keywords, budget, budget_unit, strategy,
         redundancy_penalty) -> SummaryResult:
    if not cands:
        return _finalize(all_scores, [], keywords, budget, budget_unit,
                         strategy, overflow=False)
    if strategy == "greedy":
        chosen = _select_greedy(cands, budget, budget_unit, redundancy_penalty)
    else:
        chosen = _select_dp(cands, budget, budget_unit)
    overflow = False
    if not chosen:
        chosen, overflow = _fallback_nonempty(cands, budget, budget_unit)
    return _finalize(all_scores, chosen, keywords, budget, budget_unit,
                     strategy, overflow)


def _validate(budget: int, budget_unit: str, strategy: str,
              redundancy_penalty: float) -> None:
    if budget_unit not in ("words", "sentences"):
        raise ValueError("budget_unit 必须是 'words' 或 'sentences'")
    if strategy not in ("greedy", "dp"):
        raise ValueError("strategy 必须是 'greedy' 或 'dp'")
    if not 0.0 <= redundancy_penalty <= 1.0:
        raise ValueError("redundancy_penalty 必须在 [0,1]")
    if budget < 0:
        raise ValueError("budget 不能为负")


def summarize(text: str, title: str = "", budget: int = 100,
              budget_unit: str = "words", strategy: str = "greedy",
              redundancy_penalty: float = 0.5,
              weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
              keyword_k: int = 20, chunk_size: Optional[int] = None,
              candidate_limit: Optional[int] = None) -> SummaryResult:
    """生成抽取式摘要。

    参数：
        budget / budget_unit: 长度预算，"words" 按词数、"sentences" 按句数。
        strategy: "greedy"（MMR）或 "dp"（背包动态规划）。
        redundancy_penalty: MMR 冗余惩罚强度，[0,1]，越大越惩罚重复。
        weights: (词频, 位置, 标题) 三项得分权重。
        chunk_size: 按固定字符数分块流式处理，结果与整体处理一致。
        candidate_limit: 仅保留总分最高的 N 句进入选择阶段（超长文档省内存）。
    """
    _validate(budget, budget_unit, strategy, redundancy_penalty)
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit 必须为正整数")
    if chunk_size:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return _summarize_stream(chunks, title, budget, budget_unit, strategy,
                                 redundancy_penalty, weights, keyword_k,
                                 candidate_limit)
    sentences = split_sentences(text)
    df, n_sent = _df_pass(sentences)
    title_words = frozenset(content_words(tokenize(title)))
    all_scores, cands = _score_pass(sentences, df, n_sent, title_words,
                                    weights, candidate_limit)
    keywords = _keywords(df, keyword_k, n_sent)
    return _run(all_scores, cands, keywords, budget, budget_unit, strategy,
                redundancy_penalty)


def summarize_chunks(chunks: Sequence[str], title: str = "", budget: int = 100,
                     budget_unit: str = "words", strategy: str = "greedy",
                     redundancy_penalty: float = 0.5,
                     weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
                     keyword_k: int = 20,
                     candidate_limit: Optional[int] = None) -> SummaryResult:
    """分块输入接口（如按块读取的大文件）。结果与 summarize(完整文本) 一致。"""
    _validate(budget, budget_unit, strategy, redundancy_penalty)
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit 必须为正整数")
    return _summarize_stream(chunks, title, budget, budget_unit, strategy,
                             redundancy_penalty, weights, keyword_k,
                             candidate_limit)


def _summarize_stream(chunks: Sequence[str], title: str, budget: int,
                      budget_unit: str, strategy: str, redundancy_penalty: float,
                      weights: Tuple[float, float, float], keyword_k: int,
                      candidate_limit: Optional[int]) -> SummaryResult:
    """流式两遍处理：第一遍词频、第二遍打分，不一次性物化全部句子。"""
    df, n_sent = _df_pass(_iter_chunk_sentences(chunks))
    title_words = frozenset(content_words(tokenize(title)))
    all_scores, cands = _score_pass(_iter_chunk_sentences(chunks), df, n_sent,
                                    title_words, weights, candidate_limit)
    keywords = _keywords(df, keyword_k, n_sent)
    return _run(all_scores, cands, keywords, budget, budget_unit, strategy,
                redundancy_penalty)
