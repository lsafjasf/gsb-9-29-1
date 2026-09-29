import json
import tempfile
import unittest
from pathlib import Path

from zhconv import Converter, MappingError, load_mapping, load_mapping_from_dict, make_converters

ROOT = Path(__file__).resolve().parent.parent
MAPPING = ROOT / "data" / "mapping.json"
EDGE_CASES = ROOT / "data" / "edge_cases.jsonl"


def mini_mapping():
    return load_mapping_from_dict({
        "s2t": {
            "char_map": {"后": ["後", "后"], "发": ["發", "髮"], "门": "門", "头": "頭", "润": "潤", "现": "現"},
            "word_map": {"后面": "後面", "皇后": "皇后", "头发": "頭髮", "发现": "發現"},
            "proper_nouns": {"周润发": "周潤發"},
        },
        "t2s": {
            "char_map": {"後": "后", "乾": ["干", "乾"]},
            "word_map": {"乾隆": "乾隆", "乾燥": "干燥"},
        },
    })


class TestMappingValidation(unittest.TestCase):
    def test_duplicate_json_key_detected(self):
        text = '{"s2t": {"char_map": {"门": "門", "门": "門"}}}'
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            path = fh.name
        with self.assertRaises(MappingError):
            load_mapping(path)

    def test_word_conflicts_with_char_map(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {"char_map": {"后": ["後", "后"]}, "word_map": {"后面": "后面面"}},
                "t2s": {},
            })

    def test_word_length_mismatch(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {"char_map": {"后": ["後", "后"]}, "word_map": {"后面": "後"}},
                "t2s": {},
            })

    def test_word_references_unknown_char_mapping(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {"char_map": {}, "word_map": {"头发": "頭髮"}},
                "t2s": {},
            })

    def test_single_char_word_rejected(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {"char_map": {"后": ["後", "后"]}, "word_map": {"后": "後"}},
                "t2s": {},
            })

    def test_proper_noun_conflicts_with_word_map(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {
                    "char_map": {"后": ["後", "后"]},
                    "word_map": {"皇后": "皇后"},
                    "proper_nouns": {"皇后": "後后"},
                },
                "t2s": {},
            })

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(MappingError):
            load_mapping_from_dict({
                "s2t": {"char_map": {"后": ["後", "後"]}},
                "t2s": {},
            })

    def test_ambiguous_char_without_word_coverage_warns(self):
        mapping = load_mapping_from_dict({
            "s2t": {"char_map": {"后": ["後", "后"]}},
            "t2s": {},
        })
        self.assertTrue(any("后" in w for w in mapping.warnings))

    def test_sample_mapping_loads_clean(self):
        mapping = load_mapping(MAPPING)
        self.assertEqual(mapping.warnings, [])


class TestConversion(unittest.TestCase):
    def setUp(self):
        converters = make_converters(mini_mapping())
        self.s2t = converters["s2t"]
        self.t2s = converters["t2s"]

    def test_word_disambiguation_minimal_pair(self):
        self.assertEqual(self.s2t.convert("后面").plain_text, "後面")
        self.assertEqual(self.s2t.convert("皇后").plain_text, "皇后")

    def test_one_to_many_by_word(self):
        self.assertEqual(self.s2t.convert("头发").plain_text, "頭髮")
        self.assertEqual(self.s2t.convert("发现").plain_text, "發現")

    def test_unresolvable_kept_and_marked(self):
        result = self.s2t.convert("后")
        self.assertEqual(result.plain_text, "后")
        self.assertEqual(result.text, "⟦后⟧")
        self.assertFalse(result.ok)
        self.assertEqual(result.pending[0].char, "后")
        self.assertEqual(result.pending[0].candidates, ["後", "后"])

    def test_proper_noun_priority(self):
        self.assertEqual(self.s2t.convert("周润发").plain_text, "周潤發")

    def test_longest_match(self):
        self.assertEqual(self.s2t.convert("头发和发现").plain_text, "頭髮和發現")

    def test_t2s_ambiguity(self):
        self.assertEqual(self.t2s.convert("乾隆").plain_text, "乾隆")
        self.assertEqual(self.t2s.convert("乾燥").plain_text, "干燥")
        self.assertTrue(self.t2s.convert("乾").pending)

    def test_custom_mark(self):
        converter = Converter(mini_mapping().s2t, mark="【{}?】")
        self.assertEqual(converter.convert("后").text, "【后?】")

    def test_invalid_mark_rejected(self):
        with self.assertRaises(ValueError):
            Converter(mini_mapping().s2t, mark="no-placeholder")


class TestEdgeCases(unittest.TestCase):
    """逐条跑 data/edge_cases.jsonl。"""

    @classmethod
    def setUpClass(cls):
        cls.converters = make_converters(load_mapping(MAPPING))

    def test_edge_cases(self):
        with open(EDGE_CASES, encoding="utf-8") as fh:
            cases = [json.loads(line) for line in fh if line.strip()]
        self.assertGreaterEqual(len(cases), 10)
        for case in cases:
            with self.subTest(id=case["id"], note=case.get("note")):
                result = self.converters[case["direction"]].convert(case["source"])
                self.assertEqual(result.plain_text, case["expected"])
                self.assertEqual([p.char for p in result.pending], case["expect_pending"])


class TestErrorClassification(unittest.TestCase):
    def test_classify_all_types(self):
        from evaluate import (
            ERR_OVER_CONVERSION, ERR_UNDER_CONVERSION, ERR_UNRESOLVED_PENDING,
            ERR_WRONG_CONVERSION, classify_errors,
        )
        # source=后干发 expected=後乾發
        errors = classify_errors("后干发", "后干髮", "後乾發", {0})
        types = {e["index"]: e["type"] for e in errors}
        self.assertEqual(types[0], ERR_UNRESOLVED_PENDING)  # 待确认未消解
        self.assertEqual(types[1], ERR_UNDER_CONVERSION)    # 该转未转
        self.assertEqual(types[2], ERR_WRONG_CONVERSION)    # 选错候选

        errors = classify_errors("皇后", "皇後", "皇后", set())
        self.assertEqual(errors[0]["type"], ERR_OVER_CONVERSION)  # 不该转却转了

    def test_length_mismatch(self):
        from evaluate import ERR_LENGTH_MISMATCH, classify_errors
        errors = classify_errors("后面", "後面", "後面啊", set())
        self.assertEqual(errors[0]["type"], ERR_LENGTH_MISMATCH)


class TestEvaluateOnCorpus(unittest.TestCase):
    def test_corpus_all_exact(self):
        from evaluate import evaluate
        report = evaluate(str(MAPPING), str(ROOT / "data" / "corpus.jsonl"))
        failed = [s for s in report["sentences"] if not s["exact_match"]]
        self.assertEqual(failed, [])
        self.assertEqual(report["summary"]["sentence_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
