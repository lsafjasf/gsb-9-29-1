"""变体还原管线。

管线顺序：
  1. 归一化折叠（NFKC + casefold，逐字符）
  2. 去噪（符号 / 零宽 / 不可见字符；拉丁字母之间的空格保留）
  3. 折叠连续重复字符
  4. 多字序列映射（拆字 / 拼音）
  5. 单字映射（谐音 / 形近 / leet）

每个还原后的字符都带原文半开区间 [start, end)，所有变换记录在 changes 中，
可用于后续取证；索引统一为原文 code point 偏移。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from .config import Config


@dataclass
class Change:
    start: int
    end: int
    kind: str          # fold / noise / repeat / sequence / char
    rule: str          # 命中的规则名
    dropped: bool = False  # 该 span 是否为被删除的原文片段（噪声、重复中被丢弃的部分）


BOUNDARY_CHAR = "\ue000"  # 私用区码位：去噪位置留下的虚拟边界，阻断 AC 匹配但不显示


@dataclass
class _Tok:
    text: str
    start: int
    end: int
    was_space: bool = False
    suspicious: bool = False

    @property
    def is_boundary(self) -> bool:
        return self.text == BOUNDARY_CHAR


@dataclass
class Normalized:
    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)


class Normalizer:
    def __init__(self, config: Config):
        self.cfg = config
        self._seq_trie: dict = {}
        self._seq_max = 0
        for seq in config.sequence_map:
            key = self.fold(seq["key"])
            if not key:
                continue
            node = self._seq_trie
            for ch in key:
                node = node.setdefault(ch, {})
            node["\0"] = seq
            self._seq_max = max(self._seq_max, len(key))
        self._char_keys = {self.fold(k): k for k in config.char_map}

    def fold(self, text: str) -> str:
        if not self.cfg.nfkc_casefold:
            return text
        return unicodedata.normalize("NFKC", text).casefold()

    def normalize(self, text: str) -> Normalized:
        changes: list[Change] = []
        tokens = self._tokenize(text, changes)
        tokens = self._drop_noise(text, tokens, changes)
        tokens = self._collapse_repeats(tokens, changes)
        tokens = self._apply_sequences(tokens, changes)
        tokens = self._apply_chars(tokens, changes)

        spans: list[tuple[int, int]] = []
        for tok in tokens:
            spans.extend([(tok.start, tok.end)] * len(tok.text))
        changes.sort(key=lambda c: (c.start, c.end))
        return Normalized("".join(t.text for t in tokens), spans, changes)

    def _tokenize(self, text: str, changes: list[Change]) -> list[_Tok]:
        tokens: list[_Tok] = []
        for i, ch in enumerate(text):
            folded = self.fold(ch)
            if folded != ch:
                changes.append(Change(i, i + 1, "fold", f"fold:{ch}->{folded}"))
            tokens.append(_Tok(folded, i, i + 1))
        return tokens

    def _drop_noise(
        self, text: str, tokens: list[_Tok], changes: list[Change]
    ) -> list[_Tok]:
        # kind 标记：keep / strip_noise / strip_space / boundary / none
        decision: list[str] = ["none"] * len(tokens)
        for idx, tok in enumerate(tokens):
            ch = tok.text
            if len(ch) == 1 and ch in self._char_keys:
                decision[idx] = "keep"  # 显式字符映射优先于去噪（如 @ -> a）
            elif self._is_space(ch):
                decision[idx] = "strip_space"  # 默认去空格，拉丁邻居则保留
            elif self._is_noise(ch):
                decision[idx] = "strip_noise"
            else:
                decision[idx] = "keep"

        # 拉丁字母之间的空格标记为 was_space 边界，是否真正保留留到
        # 序列/字符映射之后再判定（那时才知道是不是拼音还原）。
        # 注意只看直接相邻的字符，不能跨越其他噪声位置。
        if self.cfg.keep_space_inside_latin:
            for idx, tok in enumerate(tokens):
                if decision[idx] != "strip_space":
                    continue
                left_tok = tokens[idx - 1] if idx > 0 else None
                right_tok = tokens[idx + 1] if idx + 1 < len(tokens) else None
                if (
                    left_tok is not None
                    and right_tok is not None
                    and self._is_ascii_letter(left_tok.text)
                    and self._is_ascii_letter(right_tok.text)
                ):
                    decision[idx] = "latin_space"

        out: list[_Tok] = []
        for idx, tok in enumerate(tokens):
            if decision[idx] == "keep":
                out.append(tok)
            else:
                if decision[idx] == "strip_noise":
                    changes.append(
                        Change(tok.start, tok.end, "noise",
                               f"strip:{self._show(tok.text)}", dropped=True)
                    )
                else:
                    changes.append(
                        Change(tok.start, tok.end, "noise", "strip:space", dropped=True)
                    )
                left_idx = self._nearest_decision(tokens, decision, idx, -1)
                right_idx = self._nearest_decision(tokens, decision, idx, 1)
                if left_idx is not None and right_idx is not None:
                    out.append(_Tok(
                        BOUNDARY_CHAR, tok.start, tok.end,
                        was_space=(decision[idx] in {"strip_space", "latin_space"}),
                    ))
        return out

    @staticmethod
    def _is_ascii_alnum(ch: str) -> bool:
        return bool(ch) and ch[0].isascii() and ch[0].isalnum()

    def _nearest_decision(
        self, tokens: list[_Tok], decision: list[str], idx: int, step: int
    ) -> int | None:
        j = idx + step
        while 0 <= j < len(decision):
            if decision[j] == "keep":
                return j
            j += step
        return None

    @staticmethod
    def _is_space(ch: str) -> bool:
        return len(ch) == 1 and ch.isspace()

    @staticmethod
    def _is_ascii_letter(ch: str) -> bool:
        return bool(ch) and ch[0].isascii() and ch[0].isalpha()

    def _is_noise(self, ch: str) -> bool:
        if ch in self.cfg.extra_keep:
            return False
        if ch in self.cfg.extra_strip:
            return True
        if len(ch) == 1:
            cat = unicodedata.category(ch)
            if cat in self.cfg.strip_categories:
                return True
            if cat in {"Cc", "Cf"} and ch not in {"\n", "\t"}:
                return True
        return False

    def _collapse_repeats(
        self, tokens: list[_Tok], changes: list[Change]
    ) -> list[_Tok]:
        if self.cfg.max_consecutive < 1:
            return tokens
        out: list[_Tok] = []
        run: list[_Tok] = []

        def flush() -> None:
            if not run:
                return
            out.append(_Tok(run[0].text, run[0].start, run[-1].end))
            if len(run) > self.cfg.max_consecutive:
                changes.append(
                    Change(
                        run[self.cfg.max_consecutive].start,
                        run[-1].end,
                        "repeat",
                        f"collapse:{self._show(run[0].text)} x{len(run)}",
                        dropped=True,
                    )
                )

        for tok in tokens:
            if tok.is_boundary:
                flush()
                out.append(tok)
                run = []
            elif run and tok.text == run[-1].text:
                run.append(tok)
            else:
                flush()
                run = [tok]
        flush()
        return out

    def _apply_sequences(
        self, tokens: list[_Tok], changes: list[Change]
    ) -> list[_Tok]:
        out: list[_Tok] = []
        i = 0
        n = len(tokens)
        while i < n:
            tok_i = tokens[i]
            if tok_i.is_boundary:
                out.append(tok_i)
                i += 1
                continue
            node = self._seq_trie
            best: tuple[int, dict, int] | None = None  # (token_end_idx, seq, key_chars)
            j = i
            key_chars = 0
            while j < n and key_chars < self._seq_max:
                tok = tokens[j]
                if tok.is_boundary:
                    j += 1
                    continue  # ASCII 键允许跨越噪声边界，键长不计数
                ch = tok.text
                if len(ch) != 1 or ch not in node:
                    break
                node = node[ch]
                key_chars += 1
                if "\0" in node:
                    best = (j, node["\0"], key_chars)
                j += 1
            if best is not None and self._seq_boundary_ok(tokens, i, best[0]):
                j, seq, key_chars = best
                # 还原键跨越的边界符原样输出在替换字之后，分隔相邻两个拼音键
                out.append(_Tok(seq["to"], tokens[i].start, tokens[j].end, suspicious=True))
                changes.append(
                    Change(
                        tokens[i].start,
                        tokens[j].end,
                        "sequence",
                        seq.get("rule", f"seq:{seq['key']}={seq['to']}"),
                    )
                )
                i = j + 1
            else:
                out.append(tok_i)
                i += 1
        return out

    def _seq_boundary_ok(self, tokens: list[_Tok], i: int, j: int) -> bool:
        """ASCII 序列（拼音）两侧必须不是拉丁字母/数字，避免 during/bought 误伤。"""
        if not self._seq_is_ascii(tokens, i, j):
            return True
        # 直接相邻（无噪声边界）的 ASCII 字母数字才算粘连；
        # 边界符另一侧的字符不算，故 du-bo 可命中、during 不命中。
        prev_idx = i - 1
        if prev_idx >= 0 and not tokens[prev_idx].is_boundary:
            ch = tokens[prev_idx].text[:1]
            if ch.isascii() and ch.isalnum():
                return False
        next_idx = j + 1
        if next_idx < len(tokens) and not tokens[next_idx].is_boundary:
            ch = tokens[next_idx].text[:1]
            if ch.isascii() and ch.isalnum():
                return False
        return True

    @staticmethod
    def _seq_is_ascii(tokens: list[_Tok], i: int, j: int) -> bool:
        for k in range(i, j + 1):
            tok = tokens[k]
            if tok.is_boundary or not tok.text or not tok.text[0].isascii():
                return False
        return True

    def _apply_chars(self, tokens: list[_Tok], changes: list[Change]) -> list[_Tok]:
        mapped: list[_Tok] = []
        for tok in tokens:
            if tok.is_boundary:
                mapped.append(tok)
            elif len(tok.text) == 1 and tok.text in self._char_keys:
                orig_key = self._char_keys[tok.text]
                canon = self.cfg.char_map[orig_key][0]
                folded_canon = self.fold(canon)
                suspicious = tok.text != folded_canon
                if suspicious:
                    changes.append(
                        Change(
                            tok.start,
                            tok.end,
                            "char",
                            f"map:{self._show(orig_key)}->{folded_canon}",
                        )
                    )
                # @/$ 等 leet 字虽属于噪声字符表，但显式映射优先，
                # 直接作为替换字并入拉丁词（dr0gs@home -> drogsahome）。
                mapped.append(_Tok(folded_canon, tok.start, tok.end, suspicious=suspicious))
            else:
                mapped.append(tok)

        out: list[_Tok] = []
        for idx, tok in enumerate(mapped):
            if not tok.is_boundary:
                out.append(tok)
                continue
            prev_tok = self._prev_real(mapped, idx)
            next_tok = self._next_real(mapped, idx)
            if prev_tok is None or next_tok is None:
                continue  # 边界在串首/串尾无粘连风险，直接丢弃
            # 优先级最高：同一混淆拉丁词内的符号（词内含 leet 还原字符），
            # 无论另一侧是否中文都连通（dr0g#s渠道、dr0gs@home）。
            if not tok.was_space and self._run_has_leet(mapped, idx):
                continue
            non_ascii_side = (
                not self._is_ascii_alnum(prev_tok.text)
                or not self._is_ascii_alnum(next_tok.text)
            )
            if non_ascii_side:
                both_non_ascii = (
                    not self._is_ascii_alnum(prev_tok.text)
                    and not self._is_ascii_alnum(next_tok.text)
                )
                if both_non_ascii:
                    continue  # 两个汉字（含拼音/拆字产物）之间：连通
                if self._adjacent_suspicious(mapped, idx, prev_tok, next_tok):
                    continue  # 紧邻语义还原字：连通（如 堵 du 博）
                out.append(_Tok(" ", tok.start, tok.end))
                continue
            # 两侧都是 ASCII 的符号：普通英文/拼音片段之间保留边界，
            # 防止 during#boring 被打通。
            if not tok.was_space:
                out.append(tok)
                continue
            # 空格边界：仅当紧邻 token 本身是语义还原产物（du bo 两个拼音）
            # 才压缩；与普通英文词相邻时一律保留空格。
            if not self._adjacent_suspicious(mapped, idx, prev_tok, next_tok):
                out.append(_Tok(" ", tok.start, tok.end))
        return out

    @staticmethod
    def _run_has_leet(tokens: list[_Tok], idx: int) -> bool:
        # 该方法只处理非空格的符号边界。
        # 向两侧扫描，遇到真实空格或非 ASCII 即停止；期间跨越其他符号边界。
        # 片段任意位置出现 leet 还原字符即判定为同一混淆词（dr0g#s）。
        for direction in (-1, 1):
            j = idx + direction
            while 0 <= j < len(tokens):
                tok = tokens[j]
                if tok.text == " " or (tok.is_boundary and tok.was_space):
                    break
                if tok.is_boundary:
                    j += direction
                    continue
                if not tok.text[:1].isascii():
                    break
                if tok.suspicious:
                    return True
                j += direction
        return False

    @staticmethod
    def _prev_real(tokens: list[_Tok], idx: int) -> _Tok | None:
        j = idx - 1
        while j >= 0 and tokens[j].is_boundary:
            j -= 1
        return tokens[j] if j >= 0 else None

    @staticmethod
    def _next_real(tokens: list[_Tok], idx: int) -> _Tok | None:
        j = idx + 1
        while j < len(tokens) and tokens[j].is_boundary:
            j += 1
        return tokens[j] if j < len(tokens) else None

    @staticmethod
    def _adjacent_suspicious(tokens, idx, prev_tok, next_tok) -> bool:
        for tok in (prev_tok, next_tok):
            if tok is not None and tok.suspicious:
                return True
        return False

    @staticmethod
    def _show(ch: str) -> str:
        named = {
            "\u200b": "ZWSP", "\u200c": "ZWNJ", "\u200d": "ZWJ",
            "\ufeff": "BOM", "\u2060": "WJ", "\ufe0f": "VS16",
            " ": "SPACE", "\t": "TAB", "\n": "NEWLINE", "\r": "CR",
        }
        return named.get(ch, ch)
