"""敏感词变体匹配器。

核心流程：
  1. 构建期：对每条敏感词规则，用配置里的谐音/形近/拆字/leet 映射做
     "规则侧变体扩展"，把所有变体编入 Aho-Corasick 自动机；
  2. 扫描期：文本只做无损降噪（去符号/零宽/重复折叠/全半角/大小写），
     然后用自动机匹配，并通过位置映射把命中还原到原文坐标。

误伤控制手段：
  - 文本侧不做谐音替换，正常词不会被"还原成"敏感词；
  - 置信度分级：exact/explicit=high，形近/拆字/leet=medium，谐音=low；
  - 白名单短语：命中完全落在白名单短语内则抑制（如"有威信"之于"微信"）；
  - 纯拉丁模式要求词边界，避免 "vx" 命中 "pvxz"；
  - 变体数量与同时替换字数上限，防止规则爆炸。
"""

import itertools
import json
from dataclasses import dataclass, field

from .automaton import Automaton
from .normalizer import normalize

_CONF_ORDER = ("low", "medium", "high")
_CONF_RANK = {c: i for i, c in enumerate(_CONF_ORDER)}

# 各类映射替换后得到的变体的置信度
_SUB_CONFIDENCE = {
    "homophone": "low",
    "shape": "medium",
    "split": "medium",
    "leet": "medium",
}
_MAP_TYPES = ("homophone", "shape", "split", "leet")


def _min_conf(a, b):
    return _CONF_ORDER[min(_CONF_RANK[a], _CONF_RANK[b])]


def _latin_boundary_ok(text, start, end):
    """纯拉丁模式要求两侧不是字母/数字，防止子串误伤。"""
    if start > 0 and text[start - 1].isascii() and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isascii() and text[end].isalnum():
        return False
    return True


@dataclass
class Hit:
    """一次命中的完整证据。"""
    rule_id: str
    word: str            # 命中的敏感词（规则原文）
    variant: str         # 实际命中的变体
    confidence: str      # high / medium / low
    vtype: str           # exact / explicit / homophone / shape / split / leet（可组合）
    start: int           # 原文下标，闭区间起点
    end: int             # 原文下标，开区间终点
    snippet: str         # 原文片段（含噪声符号，作为证据）
    norm_start: int      # 归一化文本中的位置
    norm_end: int
    normalized: str      # 整段归一化后的文本
    suppressed: bool = False
    suppress_reason: str = ""

    def to_dict(self):
        return {
            "rule_id": self.rule_id,
            "word": self.word,
            "variant": self.variant,
            "confidence": self.confidence,
            "vtype": self.vtype,
            "span": [self.start, self.end],
            "snippet": self.snippet,
            "norm_span": [self.norm_start, self.norm_end],
            "normalized": self.normalized,
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
        }


@dataclass
class ScanResult:
    text: str
    normalized: str
    hits: list = field(default_factory=list)        # 有效命中（未被白名单抑制）
    suppressed: list = field(default_factory=list)  # 被白名单抑制的命中

    @property
    def all_hits(self):
        return self.hits + self.suppressed

    def evidence(self):
        return [h.to_dict() for h in self.hits]


class Matcher:
    def __init__(self, config):
        settings = config.get("settings", {})
        self.collapse_repeats = settings.get("collapse_repeats", True)
        self.max_substitutions = settings.get("max_substitutions", 3)
        self.max_variants = settings.get("max_variants_per_word", 512)
        self.extra_noise = config.get("extra_noise", [])
        self.keep_chars = config.get("keep_chars", [])
        self.maps = config.get("maps", {})
        self.rules = config.get("rules", [])
        self.whitelist = config.get("whitelist", [])
        self._build()

    @classmethod
    def from_files(cls, rules_path, maps_path):
        with open(rules_path, encoding="utf-8") as f:
            config = json.load(f)
        with open(maps_path, encoding="utf-8") as f:
            config.update(json.load(f))
        return cls(config)

    # ---------- 构建期 ----------

    def _normalize_pattern(self, s):
        norm, _ = normalize(
            s,
            collapse_repeats=self.collapse_repeats,
            extra_noise=self.extra_noise,
            keep_chars=self.keep_chars,
        )
        return norm

    def _variants_for(self, word):
        """规则侧变体扩展：raw_variant -> (confidence, vtype)。

        按"同时替换字数 k"逐层生成，保证连续谐音（多处替换）被覆盖，
        同时用 max_variants 限制组合爆炸。
        """
        variants = {word: ("high", "exact")}
        n = len(word)
        tables = [(t, self.maps.get(t, {})) for t in _MAP_TYPES]
        for k in range(1, self.max_substitutions + 1):
            if len(variants) >= self.max_variants:
                break
            for positions in itertools.combinations(range(n), k):
                option_lists = []
                for p in positions:
                    opts = [
                        (rep, mtype)
                        for mtype, table in tables
                        for rep in table.get(word[p], [])
                    ]
                    if not opts:
                        break
                    option_lists.append(opts)
                else:
                    for combo in itertools.product(*option_lists):
                        chars = list(word)
                        conf, types = "high", []
                        for p, (rep, mtype) in zip(positions, combo):
                            chars[p] = rep  # rep 可以是多字符（拆字）
                            conf = _min_conf(conf, _SUB_CONFIDENCE[mtype])
                            types.append(mtype)
                        variant = "".join(chars)
                        variants.setdefault(variant, (conf, "+".join(sorted(set(types)))))
                        if len(variants) >= self.max_variants:
                            return variants
        return variants

    def _build(self):
        self._automaton = Automaton()
        self._pattern_meta = []  # pattern_id -> [meta, ...]
        by_norm = {}
        for rule in self.rules:
            rid = rule.get("id") or rule["word"]
            word = rule["word"]
            variants = self._variants_for(word)
            for raw in rule.get("variants", []):  # 配置的显式变体（如 "vx"）
                variants.setdefault(raw, ("high", "explicit"))
            for raw_variant, (conf, vtype) in variants.items():
                norm = self._normalize_pattern(raw_variant)
                if not norm:
                    continue
                if norm not in by_norm:
                    pid = self._automaton.add(norm)
                    by_norm[norm] = pid
                    self._pattern_meta.append([])
                self._pattern_meta[by_norm[norm]].append({
                    "rule_id": rid,
                    "word": word,
                    "variant": raw_variant,
                    "confidence": conf,
                    "vtype": vtype,
                    "needs_boundary": norm.isascii() and norm.isalnum(),
                })
        self._automaton.build()

        self._wl_automaton = Automaton()
        self._wl_phrases = []
        seen_wl = set()
        for phrase in self.whitelist:
            norm = self._normalize_pattern(phrase)
            if not norm or norm in seen_wl:
                continue
            seen_wl.add(norm)
            self._wl_automaton.add(norm)
            self._wl_phrases.append(phrase)
        self._wl_automaton.build()

    # ---------- 扫描期 ----------

    def scan(self, text):
        norm_text, index_map = normalize(
            text,
            collapse_repeats=self.collapse_repeats,
            extra_noise=self.extra_noise,
            keep_chars=self.keep_chars,
        )

        candidates = []
        for start, end, pid in self._automaton.search(norm_text):
            for meta in self._pattern_meta[pid]:
                if meta["needs_boundary"] and not _latin_boundary_ok(norm_text, start, end):
                    continue
                candidates.append((start, end, meta))

        # 同一规则在同一位置可能由多个变体命中，保留置信度最高的
        best = {}
        for start, end, meta in candidates:
            key = (meta["rule_id"], start, end)
            cur = best.get(key)
            if cur is None or _CONF_RANK[meta["confidence"]] > _CONF_RANK[cur[2]["confidence"]]:
                best[key] = (start, end, meta)

        wl_spans = [
            (s, e, self._wl_phrases[pid])
            for s, e, pid in self._wl_automaton.search(norm_text)
        ]

        hits, suppressed = [], []
        for start, end, meta in sorted(best.values(), key=lambda t: (t[0], t[1])):
            o_start = index_map[start]
            o_end = index_map[end - 1] + 1
            hit = Hit(
                rule_id=meta["rule_id"],
                word=meta["word"],
                variant=meta["variant"],
                confidence=meta["confidence"],
                vtype=meta["vtype"],
                start=o_start,
                end=o_end,
                snippet=text[o_start:o_end],
                norm_start=start,
                norm_end=end,
                normalized=norm_text,
            )
            wl = next((p for ws, we, p in wl_spans if ws <= start and end <= we), None)
            if wl is not None:
                hit.suppressed = True
                hit.suppress_reason = "whitelist:" + wl
                suppressed.append(hit)
            else:
                hits.append(hit)

        return ScanResult(text=text, normalized=norm_text, hits=hits, suppressed=suppressed)
