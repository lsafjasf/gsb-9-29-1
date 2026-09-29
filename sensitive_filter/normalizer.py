"""文本降噪与归一化。

设计原则：对文本只做"无损"变换（删除噪声、形态归一），绝不做谐音/形近替换。
谐音、形近、拆字等变体在规则侧扩展（见 matcher.py），这样正常文本不会
被"还原成"敏感词，从机制上控制误伤。

normalize() 返回 (归一化文本, 位置映射)，位置映射把归一化后的每个字符
映射回原文下标，用于命中证据定位。
"""

import unicodedata

# 零宽 / 不可见字符（常见绕过手段）
ZERO_WIDTH = frozenset([
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero width no-break space
    "\u180e",  # mongolian vowel separator
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # bidi controls
])

# 保留的 Unicode 大类：L*=字母, N*=数字；P(标点)/S(符号)/Z(空白)/C(控制)/M(组合符) 一律视为噪声
_KEPT_MAJOR_CATEGORIES = {"L", "N"}


def normalize(text, collapse_repeats=True, extra_noise=(), keep_chars=()):
    """归一化文本，返回 (normalized_text, index_map)。

    index_map[i] 是 normalized_text[i] 在原文中的下标。

    处理流水线（逐字符）：
      1. keep_chars 白名单原样保留；
      2. NFKC 兼容分解：全角->半角、圈字->普通字、合字->拆分等；
      3. casefold 大小写折叠；
      4. 丢弃零宽字符、配置的额外噪声字符、以及非字母非数字的字符；
      5. 可选：折叠连续重复字符（"赌赌赌博" -> "赌博"）。
    """
    extra_noise = frozenset(extra_noise)
    keep_chars = frozenset(keep_chars)

    chars = []
    index_map = []
    for orig_idx, raw in enumerate(text):
        if raw in keep_chars:
            candidates = (raw,)
        else:
            candidates = unicodedata.normalize("NFKC", raw)
        for ch in candidates:
            for folded in ch.casefold():
                if folded in keep_chars:
                    pass  # 显式保留
                elif folded in ZERO_WIDTH or folded in extra_noise:
                    continue
                elif unicodedata.category(folded)[0] not in _KEPT_MAJOR_CATEGORIES:
                    continue
                chars.append(folded)
                index_map.append(orig_idx)

    if collapse_repeats:
        collapsed_chars = []
        collapsed_map = []
        prev = None
        for ch, idx in zip(chars, index_map):
            if ch == prev:
                continue
            collapsed_chars.append(ch)
            collapsed_map.append(idx)
            prev = ch
        chars, index_map = collapsed_chars, collapsed_map

    return "".join(chars), index_map
