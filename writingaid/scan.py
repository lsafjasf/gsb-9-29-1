"""文本扫描：保护区识别与 token 切分。

保护区（不得改写）：
- 围栏代码块：``` 或 ~~~ 之间的所有行（含围栏行）；
- 引文段：行首 ">" 标记的块引用；
- 成对引号：中文引号 “”、『』、英文双引号 "（跨行配对，无法配对的开引号保护到文末）；
- 行内代码：单反引号 `...`；
- URL / 邮箱：http(s)://、www.、ftp://、邮件地址；
- 白名单表达：引擎层用 trie 补充（不在本模块）。

所有偏移量均基于「归一化文本」字符下标：引擎在入口把 CRLF/CR 归一化为 \\n，
全文与增量共用同一个归一化器，因此两条路径的偏移完全一致。
"""

from __future__ import annotations

import re

_URL_RE = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s，。！？；：、）)】」』”’>…" r"\"']+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*?`+")

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF))


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def is_latin_word_char(ch: str) -> bool:
    return ch.isalnum() and not is_cjk(ch)


def normalize(text: str) -> str:
    """统一换行：CRLF / CR -> LF。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _quote_pairs(text: str, opening: str, closing: str, intervals):
    """按出现顺序配对（栈），未闭合的开引号保护到文末。"""
    stack = []
    for i, ch in enumerate(text):
        if ch == opening:
            stack.append(i)
        elif ch == closing:
            if stack:
                lo = stack.pop()
                intervals.append((lo, i + 1))
            # 没有配对开引号的闭引号作为单字符保护，避免在其上误改
            else:
                intervals.append((i, i + 1))
    end = len(text)
    while stack:
        lo = stack.pop()
        intervals.append((lo, end))


def protected_intervals(text: str) -> list[tuple[int, int]]:
    """计算全文保护区（半开区间）。增量模式按行调用本函数。"""
    intervals: list[tuple[int, int]] = []
    in_fence = False
    fence_mark = ""
    for line in text.split("\n"):
        # split 后需要带偏移，下面用 finditer 重写围栏部分
        pass

    # 围栏代码块（行首允许空白；同类型围栏配对，未闭合则保护到文末）
    pos = 0
    lines = text.split("\n")
    offset = 0
    n = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.lstrip(" \t")
        if in_fence:
            if stripped.startswith(fence_mark):
                end = offset + len(line) + (1 if idx < n - 1 else 0)
                intervals.append((fence_start, end))
                in_fence = False
            offset += len(line) + 1
            continue
        m = re.match(r"[ \t]*(```+|~~~+)", stripped)
        if m:
            mark = m.group(1)[:3]
            fence_mark = "`" * 3 if mark.startswith("`") else "~" * 3
            in_fence = True
            fence_start = offset
        offset += len(line) + 1
    if in_fence:
        intervals.append((fence_start, len(text)))

    # 块引用：行首 >（在围栏外的行会被合并区间覆盖）
    offset = 0
    for idx, line in enumerate(lines):
        if re.match(r"[ \t]*>", line):
            intervals.append((offset, offset + len(line)))
        offset += len(line) + 1

    # 成对引号
    _quote_pairs(text, "\u201c", "\u201d", intervals)  # “ ”
    _quote_pairs(text, "\u300e", "\u300f", intervals)  # 『 』
    _quote_pairs(text, '"', '"', intervals)

    # 行内代码（跨行不生效，避免吞掉整段）
    for m in _INLINE_CODE_RE.finditer(text):
        intervals.append((m.start(), m.end()))

    # URL / 邮箱
    for m in _URL_RE.finditer(text):
        intervals.append((m.start(), m.end()))

    return _merge(intervals)


def latin_tokens(text: str):
    """产出拉丁/数字 token：(起始偏移, token 文本)，纯数字 token 也产出由调用方过滤。"""
    i = 0
    n = len(text)
    while i < n:
        if is_latin_word_char(text[i]):
            j = i + 1
            while j < n and (is_latin_word_char(text[j]) or text[j] in "-_"):
                j += 1
            yield i, text[i:j]
            i = j
        else:
            i += 1


def cjk_runs(text: str):
    """产出连续 CJK 串：(起始偏移, 串文本)。"""
    i = 0
    n = len(text)
    while i < n:
        if is_cjk(text[i]):
            j = i + 1
            while j < n and is_cjk(text[j]):
                j += 1
            yield i, text[i:j]
            i = j
        else:
            i += 1


def is_inside(pos: int, intervals) -> bool:
    for lo, hi in intervals:
        if lo <= pos < hi:
            return True
        if pos < lo:
            break
    return False
