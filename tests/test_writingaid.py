"""writingaid 自测：python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writingaid import WritingAid, load_config
from writingaid.approximate import ApproximateMatcher, levenshtein_at_most

AID = None


def aid() -> WritingAid:
    global AID
    if AID is None:
        AID = WritingAid(load_config())
    return AID


def analyze(text: str):
    return [s.to_dict() for s in aid().analyze(text)]


class TestBasicHits(unittest.TestCase):
    def test_rule_hit_fields(self):
        out = analyze("我们要千方白计完成任务。")
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual((s["start"], s["end"]), (3, 7))
        self.assertEqual(s["original"], "千方白计")
        self.assertEqual(s["suggestion"], "千方百计")
        self.assertTrue(s["reason"])
        self.assertGreater(s["confidence"], 0.9)
        self.assertEqual(s["kind"], "rule")

    def test_approx_hit_fields(self):
        out = analyze("这真是千方百记的好办法。")
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual(s["kind"], "approx")
        self.assertEqual(s["suggestion"], "千方百计")
        self.assertIn("编辑距离", s["reason"])
        self.assertGreater(s["confidence"], 0.5)

    def test_latin_approx(self):
        out = analyze("The qick brown fox.")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["suggestion"], "quick")

    def test_capitalized_approx(self):
        out = analyze("Qick brown fox")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["suggestion"], "Quick")

    def test_punctuation_rules(self):
        out = analyze("你好,世界。真的吗?太好了!")
        kinds = {s["rule_id"] for s in out}
        self.assertIn("punct-comma", kinds)
        self.assertIn("punct-question", kinds)
        self.assertIn("punct-exclaim", kinds)


class TestEdgeCases(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(analyze(""), [])
        self.assertEqual(analyze("\n\n\n"), [])

    def test_no_hit_clean_text(self):
        self.assertEqual(analyze("今天天气很好，我们一起去公园散步。"), [])

    def test_all_caps_skipped(self):
        self.assertEqual(analyze("PLEASE CHECK THE SYSTEM NOW"), [])

    def test_digits_skipped(self):
        self.assertEqual(analyze("在 2024 年完成 qick2 部署。"), [])

    def test_mixed_width(self):
        out = analyze("Hello world, 你好,世界. 我们认真的学习。")
        suggestions = {s["suggestion"] for s in out}
        self.assertIn("好，", suggestions)
        self.assertIn("界。", suggestions)
        self.assertIn("认真地学习", suggestions)
        # 英文部分不得被改写
        self.assertFalse(any(s["original"].startswith("Hello") for s in out))

    def test_long_paragraph(self):
        text = "科学技术不断进步，给日常生活带来许多便利。" * 500
        out = analyze(text)
        self.assertEqual(out, [])
        bad = text[:100] + "按步就班" + text[100:]
        out = analyze(bad)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start"], 100)

    def test_crlf_normalized(self):
        a = analyze("你好,世界\r\n他跑的很快。")
        b = analyze("你好,世界\n他跑的很快。")
        self.assertEqual(a, b)


class TestProtection(unittest.TestCase):
    def assert_untouched(self, text: str):
        self.assertEqual(analyze(text), [], f"保护区被改写: {text!r}")

    def test_closed_quote(self):
        self.assert_untouched("他提醒我：“按步就班是错误的写法”，不要改。")

    def test_unclosed_quote(self):
        self.assert_untouched("他说：“按步就班没有引号收尾")

    def test_blockquote(self):
        self.assert_untouched("> 按步就班 也不要改")

    def test_inline_code(self):
        self.assert_untouched("变量名 `按步就班` 不会被改写。")

    def test_fenced_code(self):
        self.assert_untouched("正常文字。\n```\n按步就班 = 1\n千方白计 = 2\n```\n结尾文字。")

    def test_fenced_code_unclosed(self):
        self.assert_untouched("开头。\n```\n按步就班 = 1\n千方白计 = 2")

    def test_url(self):
        self.assert_untouched("请访问 https://example.com/p/千方白记 查看。")

    def test_whitelist(self):
        self.assert_untouched("Taka 是新平台。")
        self.assert_untouched("我们使用 OpenAI 和 GitHub。")

    def test_protection_does_not_leak_outside(self):
        out = analyze("“按步就班”是引文，但这里的按步就班要改。")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["original"], "按步就班")
        self.assertGreater(out[0]["start"], 8)


class TestSorting(unittest.TestCase):
    def test_sort_reproducible(self):
        text = "他一如继往努力，增加水平。你好,世界。这真是千方百记。"
        first = analyze(text)
        for _ in range(5):
            self.assertEqual(analyze(text), first)

    def test_rule_before_approx(self):
        text = "千方白计和千方百记。"
        out = analyze(text)
        self.assertEqual(out[0]["kind"], "rule")
        self.assertEqual(out[1]["kind"], "approx")

    def test_tie_break_by_position(self):
        text = "按步就班，按步就班。"
        out = analyze(text)
        self.assertEqual(len(out), 2)
        self.assertLess(out[0]["start"], out[1]["start"])


class TestIncrementalConsistency(unittest.TestCase):
    CASES = [
        "我们要千方白计完成任务。",
        "多行文本。\n他一如继往努力。\n```\n按步就班 = 1\n```\n结尾按步就班。",
        "他说：“引文按步就班”\n第二行,标点。",
        "带\r\nCRLF\r\n换行,测试。",
        "末尾没有换行的按步就班",
        "",
        "\n\n\n",
        "超长" + "段落" * 3000 + "按步就班。",
    ]

    def test_chunked_equals_whole(self):
        for text in self.CASES:
            whole = [s.to_dict() for s in aid().analyze(text)]
            for size in (1, 2, 3, 7, 64, 4096):
                chunked = [s.to_dict() for s in aid().analyze_chunked(text, chunk_size=size)]
                self.assertEqual(whole, chunked, f"chunk_size={size} 不一致: {text[:30]!r}")

    def test_feed_flush_equals_whole(self):
        for text in self.CASES:
            whole = [s.to_dict() for s in aid().analyze(text)]
            engine = aid()
            engine.reset()
            got = []
            for ch in text:  # 逐字符送入是最严苛的切分
                got.extend(s.to_dict() for s in engine.feed(ch))
            got.extend(s.to_dict() for s in engine.flush())
            self.assertEqual(whole, sorted(got, key=lambda s: (
                -s["strength"], -s["confidence"], -s["context_len"],
                s["start"], s["end"], s["rule_id"], s["suggestion"],
            )), f"逐字符增量不一致: {text[:30]!r}")


class TestPruning(unittest.TestCase):
    WORDS = ["千方百计", "一如既往", "按部就班", "制订计划", "quick", "world", "take"]

    def test_deletion_index_matches_naive(self):
        """删除索引候选集必须覆盖朴素全扫描的所有 d<=1 命中（完备性断言）。"""
        indexed = ApproximateMatcher(self.WORDS, max_distance=1, prune="index")
        queries = ["千方白计", "千方百计", "qick", "quik", "tae", "wortd", "ab", "一"]
        for q in queries:
            naive = {w for w in self.WORDS if levenshtein_at_most(q, w, 1) <= 1}
            via_index = set(indexed._raw_candidates(q))
            via_index = {w for w in via_index if levenshtein_at_most(q, w, 1) <= 1}
            self.assertEqual(naive - {""}, via_index, f"查询 {q!r} 索引结果与朴素扫描不一致")

    def test_bucket_only_equals_index(self):
        a = ApproximateMatcher(self.WORDS, prune="index")
        b = ApproximateMatcher(self.WORDS, prune="bucket")
        for q in ["千方白记", "qick", "quik", "tek"]:
            self.assertEqual(a.correct(q), b.correct(q))

    def test_levenshtein_cutoff(self):
        self.assertEqual(levenshtein_at_most("abc", "abd", 1), 1)
        self.assertEqual(levenshtein_at_most("abc", "xyz", 1), 2)
        self.assertEqual(levenshtein_at_most("abc", "abcde", 1), 2)


if __name__ == "__main__":
    unittest.main()
