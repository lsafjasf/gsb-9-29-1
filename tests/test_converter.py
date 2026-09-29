"""转换器测试：词级消解、待确认标记、混排、专名、边界用例。"""

import unittest
from pathlib import Path

from zhconv import Converter, MappingTable

ROOT = Path(__file__).resolve().parent.parent


def make_converter(direction):
    return Converter(MappingTable.from_file(ROOT / "mapping" / f"{direction}.json"))


class TestS2T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conv = make_converter("s2t")

    def convert(self, text):
        return self.conv.convert(text).text

    def test_headline_example(self):
        """逐字替换会把「后面/皇后」里的 后 转错，词级消解不会。"""
        self.assertEqual(self.convert("后面"), "後面")
        self.assertEqual(self.convert("皇后"), "皇后")

    def test_one_to_one(self):
        self.assertEqual(self.convert("汉语"), "漢語")

    def test_word_disambiguation_fa(self):
        self.assertEqual(self.convert("头发"), "頭髮")
        self.assertEqual(self.convert("理发"), "理髮")
        self.assertEqual(self.convert("发展"), "發展")

    def test_default_used_when_no_word_matches(self):
        self.assertEqual(self.convert("后宫"), "後宮")  # 后 默认 -> 後

    def test_longest_match_wins(self):
        # 「人云亦云」整体命中词规则，而不是逐字默认
        self.assertEqual(self.convert("人云亦云"), "人云亦云")

    def test_unresolvable_char_kept_and_marked(self):
        result = self.conv.convert("他什么都会干。")
        self.assertEqual(result.text, "他什麼都會干。")  # 原字保留
        self.assertEqual(len(result.pending), 1)
        item = result.pending[0]
        self.assertEqual(item.char, "干")
        self.assertEqual(item.candidates, ["幹", "乾", "干"])
        self.assertFalse(result.ok)

    def test_mark_pending_inline(self):
        result = self.conv.convert("他什么都会干。", mark_pending=True)
        self.assertEqual(result.text, "他什麼都會⟦干|幹/乾/干⟧。")

    def test_mixed_script_passthrough(self):
        """已是繁体的字原样保留，简繁混排不出错。"""
        self.assertEqual(self.convert("他說后面有猫。"), "他說後面有貓。")

    def test_proper_nouns(self):
        self.assertEqual(self.convert("姜子牙"), "姜子牙")
        self.assertEqual(self.convert("范冰冰是演员。"), "范冰冰是演員。")
        self.assertEqual(self.convert("沈阳"), "瀋陽")
        self.assertEqual(self.convert("乾隆皇帝"), "乾隆皇帝")

    def test_empty_and_non_cjk(self):
        self.assertEqual(self.convert(""), "")
        self.assertEqual(self.convert("Hello, 世界 123!"), "Hello, 世界 123!")

    def test_length_preserved(self):
        for text in ("他站在门的后面。", "皇后住在皇宫里。", "他什么都会干。"):
            self.assertEqual(len(self.convert(text)), len(text))


class TestT2S(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conv = make_converter("t2s")

    def convert(self, text):
        return self.conv.convert(text).text

    def test_basic(self):
        self.assertEqual(self.convert("漢語"), "汉语")

    def test_many_to_one(self):
        self.assertEqual(self.convert("頭髮"), "头发")
        self.assertEqual(self.convert("發展"), "发展")

    def test_qian_disambiguation(self):
        self.assertEqual(self.convert("乾燥"), "干燥")
        self.assertEqual(self.convert("乾隆"), "乾隆")
        self.assertEqual(self.convert("乾坤"), "乾坤")

    def test_jie_disambiguation(self):
        self.assertEqual(self.convert("藉口"), "借口")
        self.assertEqual(self.convert("慰藉"), "慰藉")

    def test_mixed_script_passthrough(self):
        self.assertEqual(self.convert("这个頭髮很長。"), "这个头发很长。")


if __name__ == "__main__":
    unittest.main()
