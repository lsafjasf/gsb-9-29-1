"""关键词抽取库（仅标准库）。

加权口径（详见 docs/WEIGHTING.md）：
    score(t, d) = tf(t, d) * idf(t) * pos_weight(t, d) [* phrase_bonus]

- tf          : 词在文档内的原始出现次数（未取对数）。
- idf         : log((N + 1) / (df + 1)) + 1，N 为文档数，df 为文档频率。
                单文档语料时 idf 恒为 1，退化为 TF x 位置权重。
- pos_weight  : 取该词所有出现位置中的最大区域权重：
                标题 2.0 / 段首句 1.5 / 结尾段 1.3 / 正文 1.0。
- phrase_bonus: 多词短语（英文 bigram、中文 >=3 字 n-gram）x 1.5。

短语合并判据：
- 英文：同一句子内严格相邻（中间无标点/停用词）的两个实词，
  共现次数 >= 2（超短文档 >= 1），且包含度
  count(pair) / min(count(w1), count(w2)) >= 0.5。
- 中文：无分词，直接在去停用字后的连续汉字段内取 2~5 字 n-gram 作为候选；
  短 n-gram 若被更长候选"吸收"则舍弃（见 _prune_cjk_ngrams）。
- 被短语吸收的单词若 count(phrase) >= 0.5 * count(word)，该单词不再单独输出。
"""

import json
import math
import re
from collections import Counter

# ---------------------------------------------------------------- 常量

ZONE_TITLE = "title"
ZONE_PARA_START = "para_start"
ZONE_ENDING = "ending"
ZONE_BODY = "body"

DEFAULT_ZONE_WEIGHTS = {
    ZONE_TITLE: 2.0,
    ZONE_PARA_START: 1.5,
    ZONE_ENDING: 1.3,
    ZONE_BODY: 1.0,
}

PHRASE_BONUS = 1.5
MIN_PAIR_CONTAINMENT = 0.5   # 短语合并：共现 / min(各自词频)
MIN_ABSORB_RATIO = 0.5       # 单词被短语吸收（不再单独输出）的比例
CJK_MAX_N = 6                # 中文候选 n-gram 最大长度
CJK_LONG_MIN_TF = 3          # >=5 字的长 n-gram 极易是碎片，要求 tf >= 3
CJK_ABSORB_SUM = 0.8         # 中文吸收判据(1)：覆盖率阈值
CJK_ABSORB_SINGLE = 0.6      # 中文吸收判据(2)：单一强吸收阈值
SHORT_DOC_TOKENS = 50        # 低于此 token 数视为超短文档，min_tf 降为 1
MIN_TF_NORMAL = 2
MIN_TF_SHORT = 1

EN_STOPWORDS = frozenset(
    "a an the and or but if then else when while of at by for with about "
    "into through during before after above below to from up down in out "
    "on off over under again further once here there all any both each "
    "few more most other some such no nor not only own same so than too "
    "very can will just should now is are was were be been being have "
    "has had having do does did doing would could ought i you he she it "
    "we they them his her its our their this that these those as us".split()
)

# 中文停用字（按字切分连续汉字段）。
# 口径：只收几乎不会出现在关键词内部的虚字；「过/正/据/通/行」等虽常作
# 虚字，但也是「过拟合/正则/数据/通胀/银行」的词内字，故不收录（已知局限，
# 见 docs/WEIGHTING.md）。
ZH_STOPCHARS = frozenset(
    "的了和与及或但因为所如虽于在又也都就还要会能可已将着之其"
    "这那我你他她它们个等上下中内前后以为由对把被向从很更最没不有无是地得且而却"
)

_SENT_SPLIT_RE = re.compile(r"[.!?。!?！？；;…‥]+")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[㐀-䶿一-鿿豈-﫿]+"
)
_ZH_STOPCHAR_CLASS = "[" + "".join(ZH_STOPCHARS) + "]"

ZONE_WEIGHTS_REF = DEFAULT_ZONE_WEIGHTS


# ---------------------------------------------------------------- 文档结构解析

def _split_paragraphs(text):
    """按空行分段，返回非空段落列表。"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sentences_with_zones(text):
    """把原始文本解析为 [(sentence, zone)]。

    口径：首个非空行视为标题；每段第一句为段首；最后一段为结尾段。
    同一句子命中多个区域时取权重较高者。
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    title, body = lines[0], "\n".join(lines[1:])
    result = [(title, ZONE_TITLE)]
    paragraphs = _split_paragraphs(body)
    for p_idx, para in enumerate(paragraphs):
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(para) if s.strip()]
        for s_idx, sent in enumerate(sentences):
            zone = ZONE_BODY
            if s_idx == 0:
                zone = ZONE_PARA_START
            if p_idx == len(paragraphs) - 1 and (
                ZONE_WEIGHTS_REF[zone] < ZONE_WEIGHTS_REF[ZONE_ENDING]
            ):
                zone = ZONE_ENDING
            result.append((sent, zone))
    return result


# ---------------------------------------------------------------- 候选生成

def _content_word(token):
    return (
        len(token) >= 2
        and any(ch.isalpha() for ch in token)
        and token not in EN_STOPWORDS
    )


def _iter_sentence_units(sentence):
    """产出句子内的单元序列：('word', token) 或 ('cjk', 去停用字后的汉字段)。"""
    units = []
    for match in _TOKEN_RE.finditer(sentence):
        chunk = match.group(0)
        if _CJK_RE.match(chunk):
            for run in re.split(_ZH_STOPCHAR_CLASS, chunk):
                if run:
                    units.append(("cjk", run))
        else:
            units.append(("word", chunk.lower()))
    return units


def _cjk_ngrams(run, max_n=CJK_MAX_N):
    """对一段连续汉字生成 2..max_n 的全部 n-gram；单字不成候选。"""
    n_max = min(max_n, len(run))
    for n in range(2, n_max + 1):
        for i in range(len(run) - n + 1):
            yield run[i : i + n]


class _DocCandidates:
    def __init__(self):
        self.tf = Counter()          # term -> 文档内词频
        self.zones = {}              # term -> 最高区域权重
        self.word_tokens = 0         # 用于超短文档判定
        self.pair_tf = Counter()     # 英文相邻实词对词频
        self.pair_zones = {}


def _collect_doc(text, zone_weights):
    """解析单篇文档，统计单词/n-gram/短语候选的词频与位置。"""
    doc = _DocCandidates()
    for sentence, zone in _sentences_with_zones(text):
        weight = zone_weights[zone]
        units = _iter_sentence_units(sentence)
        doc.word_tokens += sum(len(u[1]) if u[0] == "cjk" else 1 for u in units)
        raw_words = [u[1] if u[0] == "word" else None for u in units]
        for i, tok in enumerate(raw_words):
            if tok is None or not _content_word(tok):
                continue
            doc.tf[tok] += 1
            doc.zones[tok] = max(doc.zones.get(tok, 0.0), weight)
            if i + 1 < len(raw_words):
                nxt = raw_words[i + 1]
                if nxt is not None and _content_word(nxt):
                    pair = tok + " " + nxt
                    doc.pair_tf[pair] += 1
                    doc.pair_zones[pair] = max(
                        doc.pair_zones.get(pair, 0.0), weight
                    )
        for kind, run in units:
            if kind != "cjk":
                continue
            for gram in _cjk_ngrams(run):
                doc.tf[gram] += 1
                doc.zones[gram] = max(doc.zones.get(gram, 0.0), weight)
    return doc


# ---------------------------------------------------------------- 剪枝与合并

def _prune_cjk_ngrams(tf):
    """中文吸收判据：满足任一条件则舍弃短 n-gram A（B 为包含 A 的更长候选）：

    1) 所有 tf(B) >= 2 的 B 的词频之和 >= 0.8 * tf(A)（A 大多长在长词里）；
    2) 存在某个 tf(B) >= 2 的 B 使 tf(B) >= 0.6 * tf(A)（单一强吸收）。
    """
    terms = [t for t in tf if _CJK_RE.match(t)]
    dropped = set()
    for a in terms:
        cover_sum = 0
        strong = False
        for b in terms:
            if len(b) <= len(a) or a not in b:
                continue
            if tf[b] < MIN_TF_NORMAL:
                continue
            cover_sum += tf[b]
            if tf[b] >= CJK_ABSORB_SINGLE * tf[a]:
                strong = True
        if strong or cover_sum >= CJK_ABSORB_SUM * tf[a]:
            dropped.add(a)
    return dropped


def _merge_en_phrases(doc, min_count):
    """英文短语合并：返回合法短语集合，以及被短语吸收、不再单独输出的单词。"""
    phrases = set()
    for pair, cnt in doc.pair_tf.items():
        if cnt < min_count:
            continue
        w1, w2 = pair.split(" ")
        base = min(doc.tf.get(w1, 0), doc.tf.get(w2, 0))
        if base > 0 and cnt / base >= MIN_PAIR_CONTAINMENT:
            phrases.add(pair)
    absorbed = set()
    for pair in phrases:
        for w in pair.split(" "):
            if doc.tf.get(w, 0) and doc.pair_tf[pair] >= MIN_ABSORB_RATIO * doc.tf[w]:
                absorbed.add(w)
    return phrases, absorbed


# ---------------------------------------------------------------- 打分

def _idf(n_docs, df):
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def _is_phrase(term):
    if " " in term:
        return True
    # 中文 3 字词（过拟合/褪黑素）不加短语加成，>=4 字才视为短语；
    # 否则「胀预期」一类 3 字碎片会被加成放大
    return bool(_CJK_RE.match(term)) and len(term) >= 4


def extract_keywords(documents, top_k=10, zone_weights=None):
    """对语料中的每篇文档抽取关键词。

    documents : list[str]，每篇文档的首个非空行视为标题。
    top_k     : 每篇返回的最大关键词数。
    返回      : list[list[(term, score)]]，与输入顺序一致，按分数降序。
    """
    weights = dict(DEFAULT_ZONE_WEIGHTS)
    if zone_weights:
        weights.update(zone_weights)
    global ZONE_WEIGHTS_REF
    ZONE_WEIGHTS_REF = weights

    docs = [_collect_doc(text, weights) for text in documents]
    n_docs = len(docs)

    df = Counter()
    for doc in docs:
        for term in set(doc.tf) | set(doc.pair_tf):
            df[term] += 1

    results = []
    for doc in docs:
        short = doc.word_tokens < SHORT_DOC_TOKENS
        min_tf = MIN_TF_SHORT if short else MIN_TF_NORMAL
        min_pair = MIN_TF_SHORT if short else MIN_TF_NORMAL

        dropped = _prune_cjk_ngrams(doc.tf)
        phrases, absorbed = _merge_en_phrases(doc, min_pair)
        if short:
            # 超短文档中 tf=1 的短语也会"合法"，此时不做单词吸收，
            # 避免「migration」被 tf=1 的「system migration」吞掉
            absorbed = set()

        scored = []
        for term, count in doc.tf.items():
            if count < min_tf or term in dropped or term in absorbed:
                continue
            # 5 字以上的中文长 n-gram 极易是碎片，任何文档都要求 tf >= 3
            if (
                _CJK_RE.match(term)
                and len(term) >= 5
                and count < CJK_LONG_MIN_TF
            ):
                continue
            score = count * _idf(n_docs, df[term]) * doc.zones[term]
            if _is_phrase(term):
                score *= PHRASE_BONUS
            scored.append((term, score))
        for pair in phrases:
            score = (
                doc.pair_tf[pair]
                * _idf(n_docs, df[pair])
                * doc.pair_zones[pair]
                * PHRASE_BONUS
            )
            scored.append((pair, score))

        # 排序：分数降序；同分优先词频高、再优先更长（更具体），最后按字典序
        scored.sort(
            key=lambda x: (
                -x[1],
                -doc.tf.get(x[0], doc.pair_tf.get(x[0], 0)),
                -len(x[0]),
                x[0],
            )
        )
        results.append(scored[:top_k])
    return results


def extract_single(text, top_k=10, zone_weights=None):
    """单文档便捷接口（语料仅一篇，idf 退化为 1）。"""
    return extract_keywords([text], top_k=top_k, zone_weights=zone_weights)[0]


# ---------------------------------------------------------------- 语料加载

def load_corpus(corpus_dir, annotations_path):
    """读取 data/corpus/*.txt 与 annotations.json，返回 (doc_ids, texts, gold)。"""
    import os

    with open(annotations_path, encoding="utf-8") as fh:
        gold = json.load(fh)
    doc_ids, texts = [], []
    for doc_id in sorted(gold):
        path = os.path.join(corpus_dir, doc_id + ".txt")
        with open(path, encoding="utf-8") as fh:
            texts.append(fh.read())
        doc_ids.append(doc_id)
    return doc_ids, texts, gold
