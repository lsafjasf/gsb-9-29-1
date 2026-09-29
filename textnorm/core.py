"""textnorm — 文档文本规范化（统一标点 / 空白 / 引号风格），仅依赖标准库。

设计要点
--------
1. 按区域豁免：围栏代码块、行内代码、URL、引用段。豁免范围在 *原始文本* 上
   判定一次（见 find_exempt_spans），随后所有改写都跳过这些区间；每个区间
   附带判定依据（EXEMPTION_REASONS）。
2. 幂等：每条规则都是不动点变换（输出字符集不再被任何规则命中），因此
   normalize(normalize(x).text).text == normalize(x).text，且第二次执行
   不产生任何改动。
3. 可定位 / 可回滚：每条改动记录其在 *最终文本* 中的偏移、旧文本、新文本、
   行号列号与前后文；rollback() 按清单把新文本逆替换回旧文本，且校验
   清单位置与文本一致，可安全回滚。

规则管线（顺序执行，各规则字符域互不相交，保证改动清单不重叠）：
    fullwidth_alnum -> punctuation -> quotes -> whitespace -> final_newline
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------- 豁免区域

EXEMPTION_REASONS = {
    "fenced_code": "围栏代码块(``` 或 ~~~)：内容是源码，空白与标点属于语法，改写会破坏可运行性",
    "inline_code": "行内代码(`...`)：逐字内容，其中的标点/空白可能有语义",
    "url": "URL：字符序列本身是定位符，任何改写都会破坏链接可访问性",
    "blockquote": "引用段(> 开头)：引文须忠实于来源，不应被风格化改写",
}


@dataclass
class ExemptConfig:
    """豁免开关；全部可独立关闭（例如 ExemptConfig(url=False)）。"""

    fenced_code: bool = True
    inline_code: bool = True
    url: bool = True
    blockquote: bool = True


@dataclass
class Span:
    start: int
    end: int
    kind: str
    reason: str


_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BLOCKQUOTE = re.compile(r"^ {0,3}>")
_BACKTICK_RUN = re.compile(r"`+")
# URL 字符集：排除空白与常见中英文句读；句读视为 URL 的边界而非其内容。
_URL = re.compile(
    r"(?:https?://|ftp://|www\.)"
    r"[^\s<>\"'`“”‘’「」『』【】《》〈〉，。；：？！、（）…—]+"
)
_URL_TRAIL = ".,;:!?'\""
_URL_TRAIL_BRACKETS = {")": "(", "]": "[", "}": "{"}


def _overlaps(s: int, e: int, spans: list[Span]) -> bool:
    return any(sp.start < e and s < sp.end for sp in spans)


def find_exempt_spans(text: str, config: Optional[ExemptConfig] = None) -> list[Span]:
    """在原始文本上判定豁免区间，返回按起点排序、附判定依据的区间列表。"""
    cfg = config or ExemptConfig()
    spans: list[Span] = []
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))

    fenced_line_ranges: list[tuple[int, int]] = []
    if cfg.fenced_code:
        i = 0
        while i < len(lines):
            m = _FENCE_OPEN.match(lines[i])
            if not m:
                i += 1
                continue
            fence = m.group(1)
            close = re.compile(
                r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*\r?\n?$"
            )
            j = i + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            last = j if j < len(lines) else len(lines) - 1
            spans.append(
                Span(offsets[i], offsets[last + 1], "fenced_code", EXEMPTION_REASONS["fenced_code"])
            )
            fenced_line_ranges.append((i, last))
            i = last + 1

    if cfg.blockquote:
        for idx, ln in enumerate(lines):
            if any(a <= idx <= b for a, b in fenced_line_ranges):
                continue
            if _BLOCKQUOTE.match(ln):
                spans.append(
                    Span(offsets[idx], offsets[idx + 1], "blockquote", EXEMPTION_REASONS["blockquote"])
                )

    spans.sort(key=lambda sp: sp.start)

    if cfg.inline_code:
        runs = [
            (m.start(), m.end())
            for m in _BACKTICK_RUN.finditer(text)
            if not _overlaps(m.start(), m.end(), spans)
        ]
        used = [False] * len(runs)
        for i, (s, e) in enumerate(runs):
            if used[i]:
                continue
            width = e - s
            for j in range(i + 1, len(runs)):
                if not used[j] and runs[j][1] - runs[j][0] == width:
                    spans.append(
                        Span(s, runs[j][1], "inline_code", EXEMPTION_REASONS["inline_code"])
                    )
                    used[i] = used[j] = True
                    break
        spans.sort(key=lambda sp: sp.start)

    if cfg.url:
        for m in _URL.finditer(text):
            url = m.group().rstrip(_URL_TRAIL)
            while url and url[-1] in _URL_TRAIL_BRACKETS:
                opener = _URL_TRAIL_BRACKETS[url[-1]]
                if url.count(url[-1]) > url.count(opener):
                    url = url[:-1]
                else:
                    break
            if not url:
                continue
            s, e = m.start(), m.start() + len(url)
            if _overlaps(s, e, spans):
                continue
            spans.append(Span(s, e, "url", EXEMPTION_REASONS["url"]))
        spans.sort(key=lambda sp: sp.start)

    return spans


# ---------------------------------------------------------------- 规则配置

DEFAULT_PUNCT_MAP = {
    "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
    "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
    "、": ",", "…": "...", "—": "--", "–": "-", "～": "~",
}
DEFAULT_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'"}


@dataclass
class NormConfig:
    exempt: ExemptConfig = field(default_factory=ExemptConfig)
    fullwidth_alnum: bool = True          # 全角英数字 -> 半角
    punctuation: bool = True              # 标点映射（punctuation_map）
    punctuation_map: dict = field(default_factory=lambda: dict(DEFAULT_PUNCT_MAP))
    quotes: bool = True                   # 弯引号 -> 直引号（quote_map）
    quote_map: dict = field(default_factory=lambda: dict(DEFAULT_QUOTE_MAP))
    whitespace: bool = True               # 空白统一（见 _pass_whitespace）
    final_newline: bool = False           # 文末补换行（空文本不补）


@dataclass
class Change:
    rule: str
    start: int   # 在 *最终文本* 中的偏移
    end: int
    old: str
    new: str


@dataclass
class NormResult:
    text: str
    changes: list[Change]
    exempt_spans: list[Span]


# ---------------------------------------------------------------- 位置映射

class _EditMapper:
    """把改写前的偏移映射到改写后（同一遍内各编辑互不重叠）。"""

    def __init__(self, edits: list[tuple[int, int, int]]):
        self.starts = [e[0] for e in edits]
        self.ends = [e[1] for e in edits]
        self.new_lens = [e[2] for e in edits]
        self.prefix = [0]
        for s, e, nl in edits:
            self.prefix.append(self.prefix[-1] + nl - (e - s))

    def map(self, pos: int, is_end: bool) -> int:
        done = bisect.bisect_right(self.ends, pos)  # end <= pos 的编辑完整生效
        delta = self.prefix[done]
        j = bisect.bisect_left(self.starts, pos)
        if j > 0 and self.starts[j - 1] < pos < self.ends[j - 1]:
            k = j - 1  # pos 落在被替换区间内部：钳到替换结果的起点/终点
            return self.starts[k] + self.prefix[k] + (self.new_lens[k] if is_end else 0)
        return pos + delta


def _apply_edits(text: str, edits: list[tuple[int, int, str, str]]) -> str:
    parts, pos = [], 0
    for s, e, new, _rule in edits:
        parts.append(text[pos:s])
        parts.append(new)
        pos = e
    parts.append(text[pos:])
    return "".join(parts)


# ---------------------------------------------------------------- 各规则

def _hit(s: int, e: int, spans: list[Span], starts: list[int]) -> bool:
    i = bisect.bisect_right(starts, s) - 1
    if i >= 0 and spans[i].end > s:
        return True
    i += 1
    return i < len(spans) and spans[i].start < e


def _map_pass(mapping: dict, rule: str):
    cls = "[" + "".join(re.escape(k) for k in mapping) + "]"
    pattern = re.compile(cls)

    def run(text, spans, starts, _cfg):
        edits = []
        for m in pattern.finditer(text):
            if _hit(m.start(), m.end(), spans, starts):
                continue
            edits.append((m.start(), m.end(), mapping[m.group()], rule))
        return edits

    return run


def _fullwidth_mapping(cfg) -> dict:
    """全角 ASCII（FF01-FF5E）-> 半角；标点/引号映射表覆盖的字符除外，
    以保证各规则字符域互不相交（全角空格 U+3000 由 whitespace 规则处理）。"""
    reserved = set(cfg.punctuation_map) | set(cfg.quote_map)
    return {
        chr(c): chr(c - 0xFEE0)
        for c in range(0xFF01, 0xFF5F)
        if chr(c) not in reserved
    }

_WS_RUN = re.compile(r"[ \t　]+")


def _pass_whitespace(text, spans, starts, _cfg):
    """空白统一（单遍完成，编辑互不重叠）：

    - 行尾空白串          -> 删除
    - 行首缩进串          -> 仅把全角空格换成半角，宽度/Tab 不动（保留缩进结构）
    - 行内空白串          -> 折叠为单个半角空格（含 Tab、全角空格）
    """
    edits = []
    for m in _WS_RUN.finditer(text):
        s, e = m.start(), m.end()
        if _hit(s, e, spans, starts):
            continue
        run = m.group()
        at_line_start = s == 0 or text[s - 1] == "\n"
        at_line_end = e == len(text) or text[e] == "\n"
        if at_line_end:
            new = ""
        elif at_line_start:
            new = run.replace("　", " ")
        else:
            new = " "
        if new != run:
            edits.append((s, e, new, "whitespace"))
    return edits


def _pass_final_newline(text, spans, starts, _cfg):
    if text and not text.endswith("\n"):
        return [(len(text), len(text), "\n", "final_newline")]
    return []


# ---------------------------------------------------------------- 主流程

def normalize(text: str, config: Optional[NormConfig] = None) -> NormResult:
    cfg = config or NormConfig()
    spans = find_exempt_spans(text, cfg.exempt)

    passes = []
    if cfg.fullwidth_alnum:
        passes.append(_map_pass(_fullwidth_mapping(cfg), "fullwidth_alnum"))
    if cfg.punctuation and cfg.punctuation_map:
        passes.append(_map_pass(cfg.punctuation_map, "punctuation"))
    if cfg.quotes and cfg.quote_map:
        passes.append(_map_pass(cfg.quote_map, "quotes"))
    if cfg.whitespace:
        passes.append(_pass_whitespace)
    if cfg.final_newline:
        passes.append(_pass_final_newline)

    current = text
    changes: list[Change] = []
    for fn in passes:
        starts = [sp.start for sp in spans]
        edits = fn(current, spans, starts, cfg)
        if not edits:
            continue
        olds = [current[s:e] for s, e, _n, _r in edits]
        mapper = _EditMapper([(s, e, len(n)) for s, e, n, _r in edits])
        current = _apply_edits(current, edits)
        for ch in changes:  # 既有改动平移到新坐标系
            ch.start = mapper.map(ch.start, False)
            ch.end = mapper.map(ch.end, True)
        spans = [
            Span(mapper.map(sp.start, False), mapper.map(sp.end, True), sp.kind, sp.reason)
            for sp in spans
        ]
        for (s, e, new, rule), old in zip(edits, olds):
            changes.append(
                Change(rule, mapper.map(s, False), mapper.map(e, True), old, new)
            )
    return NormResult(current, changes, spans)


# ---------------------------------------------------------------- 清单与回滚

def build_manifest(result: NormResult, config: Optional[NormConfig] = None) -> dict:
    """把 NormResult 转成可 JSON 序列化的改动清单（含位置、行号列号、前后文）。"""
    line_starts = [0] + [m.end() for m in re.finditer("\n", result.text)]
    items = []
    for idx, ch in enumerate(result.changes):
        line_no = bisect.bisect_right(line_starts, ch.start)
        col = ch.start - line_starts[line_no - 1] + 1
        items.append(
            {
                "id": idx,
                "rule": ch.rule,
                "start": ch.start,
                "end": ch.end,
                "line": line_no,
                "column": col,
                "old": ch.old,
                "new": ch.new,
                "context_before": result.text[max(0, ch.start - 24):ch.start],
                "context_after": result.text[ch.end:ch.end + 24],
            }
        )
    manifest = {"version": 1, "change_count": len(items), "changes": items}
    if config is not None:
        manifest["config"] = {
            "exempt": asdict(config.exempt),
            "fullwidth_alnum": config.fullwidth_alnum,
            "punctuation": config.punctuation,
            "quotes": config.quotes,
            "whitespace": config.whitespace,
            "final_newline": config.final_newline,
        }
    return manifest


def rollback(text: str, manifest: dict) -> str:
    """按清单把已改写文本还原为原文；清单位置与文本不符时抛 ValueError。"""
    changes = sorted(manifest["changes"], key=lambda c: (c["start"], c["end"]))
    parts, pos = [], 0
    for c in changes:
        if c["start"] < pos:
            raise ValueError(f"改动清单区间重叠: id={c['id']}")
        if text[c["start"]:c["end"]] != c["new"]:
            raise ValueError(
                f"清单与文本不匹配: id={c['id']} 期望 {c['new']!r} "
                f"实际 {text[c['start']:c['end']]!r}"
            )
        parts.append(text[pos:c["start"]])
        parts.append(c["old"])
        pos = c["end"]
    parts.append(text[pos:])
    return "".join(parts)
