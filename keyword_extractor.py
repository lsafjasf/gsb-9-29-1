"""关键词抽取库：TF-IDF + 位置权重 + 短语合并（仅依赖 Python 标准库）。

加权口径（详细说明见 README.md）：

    score(t, d) = (1 + ln(wtf(t, d))) * idf(t) * penalty(t)

    wtf(t, d) : t 在 d 中各次出现的位置权重之和（不是原始次数）
    位置权重  : 标题 3.0 / 每段首句 2.0（首句按句末标点界定，
                覆盖其内部各分句）/ 末段 1.5 / 其余正文 1.0
                同一次出现命中多条位置规则时取最大值，不叠乘
    idf(t)    : ln((N + 1) / (df(t) + 1)) + 1
                N 为语料文档数，df(t) 为语料中包含 t 的文档数；
                单文档模式（未提供语料）下 idf 恒为 1，退化为位置加权词频
    penalty   : 单个汉字构成的词乘 0.6，避免单字淹没短语；其余为 1

短语合并判据（相邻高频 token 组成候选短语）：
    1. 频次  : 候选词在本文档出现 >= 2 次（超短文档 < 30 token 时放宽为 1）
    2. 凝聚度: min_split  f(短语) * T / (f(左部) * f(右部)) >= tau(T)
               T 为文档 token 总数，split 遍历所有二分点取最小值；
               tau(T) = min(6.0, 2.0 + T/22.5)，阈值随文档长度自适应：
               超短文档约 2.0，百字级长文封顶 6.0，防止长文 PMI 整体膨胀
    3. 边界  : 候选词内部不得含停用词、不得为纯数字、不得整词命中
               停用短语表（新闻套话动词，如“表示/发布/指出”）
    4. 去冗余: 两个候选存在包含关系且出现次数完全相等（短词从未脱离
               长词独立出现）时，只保留得分更高者
"""

import math
import re
from collections import Counter

__all__ = ["KeywordExtractor", "extract_keywords"]

# ---- 位置权重 ----
W_TITLE = 3.0
W_PARA_START = 2.0
W_END = 1.5
W_BODY = 1.0

# ---- 短语合并参数 ----
MAX_NGRAM = 8           # 候选短语最大 token 数（覆盖“高考综合改革方案”等长术语）
PHRASE_MIN_FREQ = 2     # 候选词文档内最低出现次数
PHRASE_MIN_COHESION = 6.0
PHRASE_MIN_COHESION_SHORT = 2.0
COHESION_RAMP_TOKENS = 90
UNIGRAM_PENALTY = 0.5
SHORT_DOC_TOKENS = 30   # 低于此 token 数视为超短文档，频次阈值放宽为 1

_CJK = "\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#%-]*|[" + _CJK + "]")
_CJK_CHAR_RE = re.compile(r"[" + _CJK + r"]$")
_SENT_SPLIT_RE = re.compile(
    r"[，。！？；：、…—·「」『』“”‘’《》（）【】〈〉,.!?;:()\[\]{}<>\"'`~@$^*|\\/+=]+"
)
_TERMINAL_RE = re.compile(r"[。！？!?…；;]+")

# 仅收录虚词级单字与英文功能词；实义语素（中/人/大/高/并/过…）一律不收，
# 否则会误伤“中国男篮”“人工智能”“并发症”这类合法短语。
DEFAULT_STOPWORDS = frozenset(
    "的 了 是 在 和 与 及 或 而 之 其 这 那 么 吗 呢 吧 啊 呀 嘛 "
    "由 以 于 被 把 让 从 向 对 为 且 但 却 再 又 只 将 曾 已 "
    "更 最 太 都 还 也 就 不 有 没 很 着 我 你 他 她 它 "
    "the a an and or of to in on for with as at by is are was were be been "
    "it its this that these those from not no but if then so such more most "
    "other some any all each into about we you he she they them his her our "
    "your their will would can could has have had do does did".split()
)

# 停用短语：整词命中即排除（新闻套话与通用副词，通用语料下几乎不可能是关键词）。
DEFAULT_STOP_PHRASES = frozenset(
    "表示 指出 强调 宣布 认为 介绍 据悉 记者 发布 显示 透露 "
    "快速 不断 持续 目前 近日".split()
)


def _tokenize(sentence):
    """连续文本直接切 token：CJK 逐字、拉丁/数字整词，无需分词器。"""
    return _TOKEN_RE.findall(sentence.lower())


def _sentences(text):
    return [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _weighted_token_streams(title, text):
    """产出 (token 序列, 位置权重)，权重取命中规则的最大值。"""
    streams = []
    for sentence in _TERMINAL_RE.split(title):
        for clause in _SENT_SPLIT_RE.split(sentence):
            if clause.strip():
                streams.append((_tokenize(clause), W_TITLE))
    paragraphs = [p for p in re.split(r"\n+", text) if p.strip()]
    for para_idx, para in enumerate(paragraphs):
        sentences = [s for s in _TERMINAL_RE.split(para) if s.strip()]
        for sent_idx, sentence in enumerate(sentences):
            weight = W_BODY
            if sent_idx == 0:
                weight = max(weight, W_PARA_START)
            if para_idx == len(paragraphs) - 1:
                weight = max(weight, W_END)
            for clause in _SENT_SPLIT_RE.split(sentence):
                if clause.strip():
                    streams.append((_tokenize(clause), weight))
    return streams


def _count_candidates(streams, stopwords, stop_phrases=()):
    """统计候选词的加权词频 wtf、原始频次 freq 与 token 总数。"""
    wtf = Counter()
    freq = Counter()
    n_tokens = 0
    for tokens, weight in streams:
        n_tokens += len(tokens)
        for n in range(1, min(MAX_NGRAM, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                if any(t in stopwords for t in gram):
                    continue
                if all(t.isdigit() for t in gram):
                    continue
                if len(gram) > 1 and "".join(gram) in stop_phrases:
                    continue
                wtf[gram] += weight
                freq[gram] += 1
    return wtf, freq, n_tokens


def _cohesion(gram, freq, n_tokens):
    """凝聚度：所有二分点上 f(整体)*T/(f(左)*f(右)) 的最小值。"""
    worst = math.inf
    for cut in range(1, len(gram)):
        left = freq.get(gram[:cut], 0) or 1
        right = freq.get(gram[cut:], 0) or 1
        worst = min(worst, freq[gram] * n_tokens / (left * right))
    return worst


def _cohesion_threshold(n_tokens):
    """凝聚度阈值：短文档 2.0 起步，随长度线性升至 8.0 封顶。"""
    if n_tokens >= COHESION_RAMP_TOKENS:
        return PHRASE_MIN_COHESION
    scale = n_tokens / COHESION_RAMP_TOKENS
    return PHRASE_MIN_COHESION_SHORT + (PHRASE_MIN_COHESION - PHRASE_MIN_COHESION_SHORT) * scale


def _is_sub(small, big):
    """small 是否为 big 的连续子序列。"""
    if len(small) > len(big):
        return False
    span = len(small)
    return any(big[i:i + span] == small for i in range(len(big) - span + 1))


def _join(gram):
    """展示用拼接：CJK 字之间不加空格，拉丁词之间加空格。"""
    out = gram[0]
    for tok in gram[1:]:
        if not (_CJK_CHAR_RE.match(out[-1]) and _CJK_CHAR_RE.match(tok)):
            out += " "
        out += tok
    return out


class KeywordExtractor:
    """关键词抽取器。

    corpus: 可选，字符串或 (title, text) 元组的可迭代对象，用于估计 IDF；
            缺省为单文档模式，idf 恒为 1。
    """

    def __init__(self, corpus=None, stopwords=None, stop_phrases=None):
        self.stopwords = set(stopwords) if stopwords else set(DEFAULT_STOPWORDS)
        self.stop_phrases = (
            set(stop_phrases) if stop_phrases else set(DEFAULT_STOP_PHRASES)
        )
        self._df = Counter()
        self._n_docs = 0
        for doc in corpus or ():
            title, text = doc if isinstance(doc, tuple) else ("", doc)
            streams = _weighted_token_streams(title, text)
            _, freq, _ = _count_candidates(streams, self.stopwords, self.stop_phrases)
            self._n_docs += 1
            for gram in freq:
                self._df[gram] += 1

    def _idf(self, gram):
        if not self._n_docs:
            return 1.0
        return math.log((self._n_docs + 1) / (self._df.get(gram, 0) + 1)) + 1.0

    def extract(self, text, title="", topk=10, with_scores=False):
        """抽取 topk 个关键词；with_scores=True 时返回 (词, 得分) 列表。"""
        streams = _weighted_token_streams(title, text)
        if not streams:
            return []
        wtf, freq, n_tokens = _count_candidates(streams, self.stopwords, self.stop_phrases)
        min_freq = 1 if n_tokens < SHORT_DOC_TOKENS else PHRASE_MIN_FREQ

        candidates = []
        for gram in wtf:
            if freq[gram] < min_freq:
                continue
            if len(gram) >= 2 and _cohesion(gram, freq, n_tokens) < _cohesion_threshold(n_tokens):
                continue
            score = (1.0 + math.log(wtf[gram])) * self._idf(gram)
            if len(gram) == 1 and _CJK_CHAR_RE.match(gram[0]):
                score *= UNIGRAM_PENALTY
            candidates.append((gram, score))
        candidates.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))

        selected = []
        for gram, score in candidates:
            if any(
                (_is_sub(gram, kept) or _is_sub(kept, gram))
                and freq[gram] == freq[kept]
                for kept, _ in selected
            ):
                continue
            selected.append((gram, score))
            if len(selected) >= topk:
                break

        if with_scores:
            return [(_join(gram), score) for gram, score in selected]
        return [_join(gram) for gram, _ in selected]


def extract_keywords(text, title="", topk=10, corpus=None, with_scores=False):
    """便捷函数：单文档抽取。corpus 语义同 KeywordExtractor。"""
    return KeywordExtractor(corpus).extract(
        text, title=title, topk=topk, with_scores=with_scores
    )


def main(argv=None):
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="关键词抽取：TF-IDF + 位置权重 + 短语合并")
    parser.add_argument("file", help="待抽取文本文件（默认首行为标题）")
    parser.add_argument("--corpus", help="语料目录，用其中的 .txt 估计 IDF；缺省为单文档模式")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--no-title", action="store_true", help="不把首行当标题")
    args = parser.parse_args(argv)

    def read_doc(path):
        raw = Path(path).read_text(encoding="utf-8")
        head, _, body = raw.partition("\n")
        return head.strip(), body

    corpus = None
    if args.corpus:
        corpus = [read_doc(p) for p in sorted(Path(args.corpus).glob("*.txt"))]

    if args.no_title:
        title, text = "", Path(args.file).read_text(encoding="utf-8")
    else:
        title, text = read_doc(args.file)

    for term, score in KeywordExtractor(corpus).extract(
        text, title=title, topk=args.topk, with_scores=True
    ):
        print(f"{term}\t{score:.4f}")


if __name__ == "__main__":
    main()
