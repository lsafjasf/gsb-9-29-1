"""映射表加载与冲突检测的测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from zhconv import MappingConflictError, MappingError, MappingTable


def make_table(one_to_one=None, one_to_many=None):
    return MappingTable("s2t", one_to_one or {}, one_to_many or {})


class TestValidation(unittest.TestCase):
    def test_valid_table_loads(self):
        table = make_table(
            one_to_one={"汉": "漢"},
            one_to_many={"后": {"candidates": ["後", "后"], "default": "後",
                                "words": {"皇后": "皇后"}}},
        )
        self.assertEqual(table.one_to_one["汉"], "漢")
        self.assertEqual(table.word_rules["皇后"], "皇后")

    def test_char_in_both_maps_is_conflict(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_one={"后": "後"},
                       one_to_many={"后": {"candidates": ["後", "后"]}})

    def test_word_must_contain_char(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={"后": {"candidates": ["後", "后"],
                                           "words": {"乾隆": "乾隆"}}})

    def test_word_target_length_mismatch(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={"后": {"candidates": ["後", "后"],
                                           "words": {"皇后": "皇后大人"}}})

    def test_target_char_must_be_candidate(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={"后": {"candidates": ["後", "后"],
                                           "words": {"王后": "王候"}}})

    def test_default_must_be_candidate(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={"后": {"candidates": ["後", "后"],
                                           "default": "後面"}})

    def test_duplicate_candidates(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={"后": {"candidates": ["後", "後"]}})

    def test_empty_candidates(self):
        with self.assertRaises(MappingError):
            make_table(one_to_many={"后": {"candidates": []}})

    def test_same_word_defined_in_two_entries(self):
        with self.assertRaises(MappingConflictError):
            make_table(one_to_many={
                "后": {"candidates": ["後", "后"], "words": {"王后": "王后"}},
                "王": {"candidates": ["王"], "words": {"王后": "王后"}},
            })

    def test_one_to_one_requires_single_chars(self):
        with self.assertRaises(MappingError):
            make_table(one_to_one={"中国": "中國"})

    def test_duplicate_json_key_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(
                '{"direction": "s2t", "one_to_one": {"汉": "漢", "汉": "漢"}}',
                encoding="utf-8")
            with self.assertRaises(MappingConflictError):
                MappingTable.from_file(path)

    def test_bad_direction_rejected(self):
        with self.assertRaises(MappingError):
            MappingTable.from_dict({"direction": "s2s"})

    def test_bad_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(MappingError):
                MappingTable.from_file(path)

    def test_extension_without_code_change(self):
        """新增条目只需改配置文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s2t.json"
            path.write_text(json.dumps({
                "direction": "s2t",
                "one_to_one": {"汉": "漢"},
                "one_to_many": {"后": {"candidates": ["後", "后"],
                                       "default": "後",
                                       "words": {"皇后": "皇后"}}},
            }, ensure_ascii=False), encoding="utf-8")
            table = MappingTable.from_file(path)
            self.assertIn("皇后", table.word_rules)


if __name__ == "__main__":
    unittest.main()
