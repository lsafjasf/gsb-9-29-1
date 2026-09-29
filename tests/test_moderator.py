"""变体还原匹配的单元测试（python3 -m unittest -v）。"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_guard import ContentModerator, load_config  # noqa: E402


class ModeratorTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.moderator = ContentModerator(load_config(ROOT / "config" / "mappings.json"))

    def ids(self, text: str) -> list[str]:
        return [m.entry_id for m in self.moderator.scan(text).matches]


class TestDetection(ModeratorTestBase):
    def test_exact(self) -> None:
        self.assertEqual(self.ids("网络赌博害人"), ["DU_BO"])

    def test_symbol_insertion(self) -> None:
        self.assertEqual(self.ids("赌#博"), ["DU_BO"])
        self.assertEqual(self.ids("赌★☆※博"), ["DU_BO"])

    def test_pure_symbol_gibberish(self) -> None:
        self.assertEqual(self.ids("★☆※！@#"), [])

    def test_zero_width(self) -> None:
        self.assertEqual(self.ids("赌\u200b博"), ["DU_BO"])
        self.assertEqual(self.ids("赌\u200b\u200c\u200d博"), ["DU_BO"])

    def test_emoji_insertion(self) -> None:
        self.assertEqual(self.ids("赌🎲博"), ["DU_BO"])

    def test_homophone(self) -> None:
        self.assertEqual(self.ids("堵搏"), ["DU_BO"])
        self.assertEqual(self.ids("海洛茵"), ["HAILUOYIN"])

    def test_repeated_collapse(self) -> None:
        self.assertEqual(self.ids("堵堵堵博"), ["DU_BO"])
        self.assertEqual(self.ids("赌！！！！博"), ["DU_BO"])

    def test_split_chars(self) -> None:
        self.assertEqual(self.ids("贝者十专"), ["DU_BO"])
        self.assertEqual(self.ids("木仓十又"), ["QIANG_ZHI"])
        self.assertEqual(self.ids("氵每氵各囗大"), ["HAILUOYIN"])

    def test_pinyin_mixed(self) -> None:
        self.assertEqual(self.ids("du博"), ["DU_BO"])
        self.assertEqual(self.ids("du-bo"), ["DU_BO"])
        self.assertEqual(self.ids("du bo"), ["DU_BO"])
        self.assertEqual(self.ids("zha pian"), ["ZHA_PIAN"])

    def test_pinyin_not_glued_inside_words(self) -> None:
        self.assertEqual(self.ids("during the day"), [])
        self.assertEqual(self.ids("bought ten books"), [])
        self.assertEqual(self.ids("abound in buyers"), [])

    def test_english_leet(self) -> None:
        self.assertEqual(self.ids("drug5"), ["DRUGS"])
        self.assertEqual(self.ids("drug#5渠道"), ["DRUGS"])
        self.assertEqual(self.ids("buy drug5@home"), ["DRUGS"])

    def test_english_word_boundary(self) -> None:
        self.assertEqual(self.ids("drugstore"), [])
        self.assertEqual(self.ids("druggist"), [])
        self.assertEqual(self.ids("xdrugsy"), [])
        self.assertEqual(self.ids("buy drugs online"), ["DRUGS"])

    def test_half_and_full_width(self) -> None:
        self.assertEqual(self.ids("ＤＲＵＧＳ"), ["DRUGS"])
        self.assertEqual(self.ids("DrUgS"), ["DRUGS"])

    def test_multiple_hits(self) -> None:
        ids = self.ids("一边堵搏一边卖drug5@qq")
        self.assertIn("DU_BO", ids)
        self.assertIn("DRUGS", ids)

    def test_combined_obfuscation(self) -> None:
        self.assertEqual(self.ids("d\u200bu-博"), ["DU_BO"])

    def test_whitelist_suppresses(self) -> None:
        self.assertEqual(self.ids("柚木仓支柱承重很强"), [])

    def test_whitelist_does_not_mask_other_hits(self) -> None:
        self.assertEqual(self.ids("柚木仓支柱旁边赌#博"), ["DU_BO"])

    def test_split_not_a_word(self) -> None:
        self.assertEqual(self.ids("这个木仓库存放木材"), [])
        self.assertEqual(self.ids("贝者害人"), [])


class TestPositionMapping(ModeratorTestBase):
    def test_spans_point_to_original(self) -> None:
        text = "xx堵#搏yy"
        result = self.moderator.scan(text)
        match = result.matches[0]
        self.assertEqual(match.entry_id, "DU_BO")
        self.assertEqual((match.start, match.end), (2, 5))
        self.assertEqual(text[match.start:match.end], "堵#搏")
        self.assertEqual(match.normalized_text, "赌博")

    def test_zero_width_span(self) -> None:
        text = "a赌\u200b博b"
        match = self.moderator.scan(text).matches[0]
        self.assertEqual(text[match.start:match.end], "赌\u200b博")

    def test_split_span_covers_all_components(self) -> None:
        text = "木仓十又"
        match = self.moderator.scan(text).matches[0]
        self.assertEqual(match.entry_id, "QIANG_ZHI")
        self.assertEqual(text[match.start:match.end], "木仓十又")

    def test_evidence_rules_nonempty(self) -> None:
        match = self.moderator.scan("堵#搏").matches[0]
        self.assertTrue(any("map:堵->赌" in rule for rule in match.rules))
        self.assertTrue(any("strip:#" == rule for rule in match.rules))

    def test_confidence_levels(self) -> None:
        self.assertEqual(self.moderator.scan("赌博").matches[0].confidence, "exact")
        self.assertEqual(self.moderator.scan("赌#博").matches[0].confidence, "high")
        self.assertEqual(self.moderator.scan("堵搏").matches[0].confidence, "medium")
        self.assertEqual(self.moderator.scan("贝者十专").matches[0].confidence, "low")


class TestFalsePositiveGuards(ModeratorTestBase):
    def test_normal_corpus_has_zero_fp(self) -> None:
        sentences = json.loads(
            (ROOT / "data" / "benign.json").read_text(encoding="utf-8")
        )["sentences"]
        hits = []
        for sentence in sentences:
            hits.extend((sentence, m.entry_id) for m in self.moderator.scan(sentence).matches)
        self.assertEqual(hits, [])

    def test_joined_corpus_has_zero_fp(self) -> None:
        sentences = json.loads(
            (ROOT / "data" / "benign.json").read_text(encoding="utf-8")
        )["sentences"]
        self.assertEqual(self.moderator.scan("\n".join(sentences)).matches, [])

    def test_normal_numbers_and_punctuation(self) -> None:
        for text in ["价格只要5元", "订单号 13579", "call 911", "满300减50",
                     "验证码 482916", "admin@example.com"]:
            self.assertEqual(self.ids(text), [], text)

    def test_latin_spaces_preserved(self) -> None:
        self.assertEqual(self.ids("d r u g s are letters"), [])
        self.assertEqual(self.ids("during the boring meeting"), [])


class TestLongText(ModeratorTestBase):
    def test_linear_scan_200k(self) -> None:
        text = "日常流水内容，平安无事。" * 18000
        start = time.perf_counter()
        result = self.moderator.scan(text)
        baseline_ms = (time.perf_counter() - start) * 1000
        self.assertEqual(result.matches, [])

        text2 = text + "结尾赌\u200b博"
        start = time.perf_counter()
        result2 = self.moderator.scan(text2)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.assertEqual([m.entry_id for m in result2.matches], ["DU_BO"])
        self.assertLess(elapsed_ms, 3000)
        self.assertGreater(len(text2), 200_000)
        # 命中位置必须指向末尾原文
        match = result2.matches[0]
        self.assertEqual(text2[match.start:match.end], "赌\u200b博")


if __name__ == "__main__":
    unittest.main(verbosity=2)
