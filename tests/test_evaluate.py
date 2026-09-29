"""对拍与错误分类的测试。"""

import unittest
from pathlib import Path

from zhconv import Converter, MappingTable
from zhconv.evaluate import (OVER_CONVERSION, UNDER_CONVERSION, UNRESOLVED,
                             WRONG_TARGET, evaluate, load_corpus,
                             load_converters)

ROOT = Path(__file__).resolve().parent.parent


def toy_converter():
    table = MappingTable("s2t",
                         one_to_one={"汉": "漢"},
                         one_to_many={"后": {"candidates": ["後", "后"],
                                             "default": "後",
                                             "words": {"皇后": "皇后"}},
                                      "干": {"candidates": ["幹", "乾", "干"]}})
    return Converter(table)


def rec(source, target, rid="t"):
    return {"id": rid, "direction": "s2t", "category": "test",
            "source": source, "target": target}


class TestErrorClassification(unittest.TestCase):
    def setUp(self):
        self.converters = {"s2t": toy_converter()}

    def classes(self, source, target):
        report = evaluate([rec(source, target)], self.converters).sentences[0]
        return [e.cls for e in report.errors]

    def test_correct_conversion(self):
        report = evaluate([rec("皇后", "皇后")], self.converters).sentences[0]
        self.assertTrue(report.exact)
        self.assertEqual(report.char_accuracy, 1.0)

    def test_unresolved(self):
        # 干 无词规则无默认 -> 保留原字并标记待确认
        self.assertEqual(self.classes("干", "幹"), [UNRESOLVED])

    def test_under_conversion(self):
        # 字不在表中，原样通过，但标注要求转换
        self.assertEqual(self.classes("国", "國"), [UNDER_CONVERSION])

    def test_over_conversion(self):
        # 后 默认转 後，但此处标注要求保留
        self.assertEqual(self.classes("后", "后"), [OVER_CONVERSION])

    def test_wrong_target(self):
        # 汉->漢，但标注期望别的字
        self.assertEqual(self.classes("汉", "韓"), [WRONG_TARGET])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([rec("汉", "漢字")], self.converters)


class TestAnnotatedCorpus(unittest.TestCase):
    """与仓库自带标注语料对拍：除故意保留的待确认用例外全部命中。"""

    @classmethod
    def setUpClass(cls):
        cls.converters = load_converters(ROOT / "mapping")
        cls.records = load_corpus(ROOT / "corpus" / "annotated.jsonl")
        cls.report = evaluate(cls.records, cls.converters)

    def test_categories_covered(self):
        cats = {r["category"] for r in self.records}
        self.assertEqual(cats, {"pure_simplified", "pure_traditional",
                                "mixed", "proper_noun"})

    def test_only_pending_case_fails(self):
        failed = [s for s in self.report.sentences if not s.exact]
        self.assertEqual([s.id for s in failed], ["s2t-012"])
        self.assertEqual(failed[0].errors[0].cls, UNRESOLVED)
        self.assertEqual(failed[0].pending, 1)

    def test_summary_numbers(self):
        summary = self.report.summary()
        self.assertEqual(summary["total_sentences"], len(self.records))
        self.assertEqual(summary["exact_match"], len(self.records) - 1)
        self.assertEqual(summary["error_classes"][UNRESOLVED], 1)
        self.assertEqual(summary["error_classes"][UNDER_CONVERSION], 0)
        self.assertEqual(summary["error_classes"][OVER_CONVERSION], 0)
        self.assertEqual(summary["error_classes"][WRONG_TARGET], 0)


if __name__ == "__main__":
    unittest.main()
