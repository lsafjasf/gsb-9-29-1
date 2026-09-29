# -*- coding: utf-8 -*-
"""textnorm 自测：幂等断言、区域豁免、边界用例、改动清单与回滚。"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textnorm import (
    ExemptConfig,
    NormConfig,
    build_manifest,
    find_exempt_spans,
    normalize,
    rollback,
)

MESSY = """# 标题　Ｄｅｍｏ

这是　一段  混排文本：全角ＡＢＣ１２３，标点。；！？以及“弯引号”和‘单引号’……

```python
def f():  
    return "，。　Ａ"   # 代码块内：逐字保留
```

行内代码 `a，b　c` 不动；链接 https://example.com/ＡＢＣ?x=１ 也不动。

> 引用段：，。　“保持原样”
> 第二行　　同样不动

  - 缩进　保留，行内　　多个空格压缩
行尾有空格   
"""


def norm(text, **kw):
    return normalize(text, NormConfig(**kw))


class IdempotencyTest(unittest.TestCase):
    """幂等：对已规范化文本再执行，结果不变且无新改动。"""

    def test_normalize_twice_stable(self):
        first = norm(MESSY)
        second = norm(first.text)
        self.assertEqual(first.text, second.text)

    def test_second_run_yields_no_changes(self):
        first = norm(MESSY)
        second = norm(first.text)
        self.assertEqual(second.changes, [])

    def test_idempotent_with_final_newline(self):
        cfg = NormConfig(final_newline=True)
        first = normalize(MESSY, cfg)
        second = normalize(first.text, cfg)
        self.assertEqual(first.text, second.text)
        self.assertEqual(second.changes, [])


class ExemptTest(unittest.TestCase):
    def test_fenced_code_untouched(self):
        out = norm(MESSY).text
        self.assertIn('return "，。　Ａ"', out)
        self.assertIn("def f():  \n", out)  # 代码块内行尾空格保留

    def test_inline_code_untouched(self):
        self.assertIn("`a，b　c`", norm(MESSY).text)

    def test_url_untouched(self):
        self.assertIn("https://example.com/ＡＢＣ?x=１", norm(MESSY).text)

    def test_blockquote_untouched(self):
        out = norm(MESSY).text
        self.assertIn("> 引用段：，。　“保持原样”", out)
        self.assertIn("> 第二行　　同样不动", out)

    def test_all_exempt_no_changes(self):
        text = "```\n全角ＡＢＣ，。　“引号”   \n```\n"
        result = norm(text)
        self.assertEqual(result.text, text)
        self.assertEqual(result.changes, [])

    def test_exemption_is_configurable(self):
        text = "见 https://example.com/ＡＢＣ 。\n"
        kept = norm(text)
        self.assertIn("ＡＢＣ", kept.text)
        dropped = normalize(text, NormConfig(exempt=ExemptConfig(url=False)))
        self.assertIn("https://example.com/ABC", dropped.text)

    def test_spans_carry_reasons(self):
        spans = find_exempt_spans(MESSY)
        kinds = {sp.kind for sp in spans}
        self.assertEqual(kinds, {"fenced_code", "inline_code", "url", "blockquote"})
        for sp in spans:
            self.assertTrue(sp.reason, "每个豁免区间都必须给出判定依据")

    def test_url_trailing_punctuation_not_swallowed(self):
        out = norm("见（https://example.com/a）。\n").text
        self.assertEqual(out, "见(https://example.com/a).\n")


class RulesTest(unittest.TestCase):
    def test_fullwidth_alnum(self):
        self.assertEqual(norm("ＡＢＣａｂｃ０１２\n").text, "ABCabc012\n")

    def test_punctuation(self):
        self.assertEqual(norm("你好，世界。是的！\n").text, "你好,世界.是的!\n")

    def test_quotes(self):
        self.assertEqual(norm("“双引号”和‘单引号’\n").text, '"双引号"和\'单引号\'\n')

    def test_whitespace_collapse_keeps_indent(self):
        out = norm("  - 缩进　保留，行内　　压缩\n").text
        self.assertEqual(out, "  - 缩进 保留,行内 压缩\n")

    def test_trailing_whitespace_trimmed(self):
        self.assertEqual(norm("行尾空格   \n下一行\t\n").text, "行尾空格\n下一行\n")

    def test_tab_midline_collapsed(self):
        self.assertEqual(norm("a\t\tb\n").text, "a b\n")

    def test_final_newline_optional(self):
        self.assertEqual(norm("abc").text, "abc")
        self.assertEqual(normalize("abc", NormConfig(final_newline=True)).text, "abc\n")

    def test_custom_punctuation_map(self):
        cfg = NormConfig(punctuation_map={"。": "."})
        # "，"不在自定义映射中，落入全角 ASCII 规则被转为半角
        self.assertEqual(normalize("好。坏，", cfg).text, "好.坏,")
        cfg2 = NormConfig(punctuation_map={"。": "."}, fullwidth_alnum=False)
        self.assertEqual(normalize("好。坏，", cfg2).text, "好.坏，")


class EdgeCaseTest(unittest.TestCase):
    def test_empty_text(self):
        result = norm("")
        self.assertEqual(result.text, "")
        self.assertEqual(result.changes, [])
        self.assertEqual(normalize("", NormConfig(final_newline=True)).text, "")

    def test_only_whitespace(self):
        self.assertEqual(norm("   \n\t\n").text, "\n\n")

    def test_mixed_full_half(self):
        out = norm("ＡaＢb１2，,。．\n").text
        self.assertEqual(out, "AaBb12,,..\n")

    def test_long_single_line(self):
        unit = "ab１２ ，。“x”"
        text = unit * 10000  # 约 11 万字符的单行
        first = norm(text)
        self.assertNotEqual(first.text, text)
        self.assertEqual(first.text.count("ab12"), 10000)
        second = norm(first.text)
        self.assertEqual(second.text, first.text)   # 幂等
        self.assertEqual(second.changes, [])
        manifest = build_manifest(first)
        self.assertEqual(rollback(first.text, manifest), text)  # 可回滚

    def test_long_line_with_url(self):
        text = "前缀Ａ " + "https://example.com/ＡＢＣ" + " 后缀，" * 5000 + "\n"
        out = norm(text).text
        self.assertIn("https://example.com/ＡＢＣ", out)
        self.assertIn("前缀A", out)


class ManifestTest(unittest.TestCase):
    def test_change_positions_match_final_text(self):
        result = norm(MESSY)
        for ch in result.changes:
            self.assertEqual(result.text[ch.start:ch.end], ch.new)

    def test_manifest_fields(self):
        manifest = build_manifest(norm(MESSY))
        self.assertEqual(manifest["change_count"], len(manifest["changes"]))
        for item in manifest["changes"]:
            for key in ("id", "rule", "start", "end", "line", "column",
                        "old", "new", "context_before", "context_after"):
                self.assertIn(key, item)

    def test_rollback_restores_original(self):
        result = norm(MESSY)
        manifest = build_manifest(result)
        self.assertEqual(rollback(result.text, manifest), MESSY)

    def test_manifest_json_roundtrip(self):
        result = norm(MESSY)
        manifest = json.loads(json.dumps(build_manifest(result), ensure_ascii=False))
        self.assertEqual(rollback(result.text, manifest), MESSY)

    def test_rollback_rejects_mismatched_text(self):
        result = norm(MESSY)
        manifest = build_manifest(result)
        first = manifest["changes"][0]
        width = first["end"] - first["start"]
        tampered = result.text[:first["start"]] + "篡" * width + result.text[first["end"]:]
        with self.assertRaises(ValueError):
            rollback(tampered, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
