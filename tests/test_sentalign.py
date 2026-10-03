"""Self tests for sentalign: core behavior + all boundary cases + gold checks."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentalign import AlignConfig, align, align_texts, evaluate, load_dataset, split_sentences
from sentalign import features as F

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def align_dataset(name, cfg=None):
    ds = load_dataset(os.path.join(DATA_DIR, f"{name}.json"))
    result = align(ds.src, ds.tgt, cfg or AlignConfig())
    return ds, result, evaluate(result, ds)


class FeatureTests(unittest.TestCase):
    def test_tokenizer_mixed_script(self):
        toks = F.tokenize("2024年 Q1 产品 launch")
        self.assertIn("2024", toks)
        self.assertIn("年", toks)
        self.assertIn("q1", toks)
        self.assertIn("launch", toks)

    def test_tokenizer_accents_folded(self):
        self.assertIn("conference", F.tokenize("La conférence annuelle"))
        self.assertIn("intelligence", F.tokenize("l'intelligence"))

    def test_hard_anchors_numbers(self):
        anchors = F.hard_anchors("销售了300000台, 模型 X5, 2024年")
        self.assertIn("300000", anchors)
        self.assertIn("x5", anchors)
        self.assertIn("2024", anchors)

    def test_ordinary_words_are_not_anchors(self):
        anchors = F.hard_anchors("The conference launches phones")
        self.assertEqual(anchors, set())

    def test_unit_count_ignores_whitespace(self):
        self.assertEqual(F.unit_count("a b\tc"), 3)
        self.assertEqual(F.unit_count("中文 测试"), 4)


class SplitTests(unittest.TestCase):
    def test_chinese_split(self):
        text = "第一句。第二句！第三句？"
        self.assertEqual(split_sentences(text), ["第一句。", "第二句！", "第三句？"])

    def test_english_split_with_dot(self):
        sents = split_sentences("Hello world. It works!")
        self.assertEqual(sents, ["Hello world.", "It works!"])

    def test_number_dot_not_split(self):
        sents = split_sentences("Version 3.14 is stable.")
        self.assertEqual(sents, ["Version 3.14 is stable."])

    def test_closing_quote_absorbed(self):
        sents = split_sentences('He said "hi." She left.')
        self.assertEqual(sents, ['He said "hi."', "She left."])


class BoundaryTests(unittest.TestCase):
    def test_empty_both(self):
        result = align([], [])
        self.assertEqual(result.beads, [])
        self.assertEqual(result.path, [])

    def test_empty_src(self):
        result = align([], ["a", "b"])
        self.assertEqual([b.bead_type for b in result.beads], ["0-1", "0-1"])
        self.assertTrue(all(not b.flagged for b in result.beads))

    def test_empty_tgt(self):
        result = align(["a"], [])
        self.assertEqual([b.bead_type for b in result.beads], ["1-0"])

    def test_single(self):
        result = align(["水在100摄氏度时沸腾。"], ["Water boils at 100 degrees Celsius."])
        self.assertEqual(len(result.beads), 1)
        self.assertEqual(result.beads[0].bead_type, "1-1")

    def test_path_covers_all_indices(self):
        src = [f"第{i}句很长一些的内容描述。" for i in range(5)]
        tgt = [f"Sentence number {i} with enough length to align." for i in range(5)]
        result = align(src, tgt)
        si = ti = 0
        for b in result.beads:
            self.assertEqual(b.src_start, si)
            self.assertEqual(b.tgt_start, ti)
            si, ti = b.src_end, b.tgt_end
        self.assertEqual((si, ti), (5, 5))


class MergeGapTests(unittest.TestCase):
    def test_basic_dataset_perfect(self):
        _ds, result, ev = align_dataset("basic_merge_gap")
        # Structural accuracy: every gold bead recovered exactly.
        self.assertEqual(ev.accuracy, 1.0, ev.counts)
        self.assertEqual(ev.counts["split_error"], 0)
        self.assertEqual(ev.counts["merge_error"], 0)
        self.assertEqual(ev.counts["spurious_pair"], 0)
        self.assertEqual(ev.counts["missed_pair"], 0)
        types = [b.bead_type for b in result.beads]
        self.assertIn("2-1", types)
        self.assertIn("1-2", types)
        self.assertIn("1-0", types)
        # Beads adjacent to the missing segment may be flagged: safe, not an error.
        self.assertTrue(any(b.flagged for b in result.beads))

    def test_skew_4_8_all_one_to_two(self):
        _ds, result, ev = align_dataset("skew_4_8")
        self.assertEqual(ev.accuracy, 1.0, ev.counts)
        self.assertTrue(all(b.bead_type == "1-2" for b in result.beads))

    def test_extreme_ratio_is_flagged_not_forced(self):
        _ds, result, ev = align_dataset("extreme_ratio")
        # Bulk-uncertain: every covering bead must be flagged.
        self.assertEqual(ev.counts["forced_confident_bulk"], 0, ev.counts)
        self.assertEqual(ev.counts["correct"], 1, ev.counts)

    def test_repeated_sentences(self):
        _ds, result, ev = align_dataset("repeated")
        # Every gold link recovered (order monotonic).
        self.assertEqual(ev.link_recall, 1.0)
        self.assertEqual(ev.link_precision, 1.0)

    def test_fr_en_cognate_merge(self):
        _ds, result, ev = align_dataset("fr_en_cognate")
        self.assertEqual(ev.accuracy, 1.0, ev.counts)
        self.assertIn("2-1", [b.bead_type for b in result.beads])

    def test_unrelated_anchors_get_flagged(self):
        _ds, result, ev = align_dataset("unrelated_anchors")
        # None of the matched beads should be asserted with high confidence.
        confident_matches = [
            b for b in result.beads
            if b.src_end > b.src_start and b.tgt_end > b.tgt_start and not b.flagged
        ]
        self.assertEqual(confident_matches, [])


class FlaggingTests(unittest.TestCase):
    def test_contradiction_forces_low_confidence(self):
        src = ["步骤3需要大约十分钟完成操作。"]
        tgt = ["Step 7 takes about ten minutes."]
        result = align(src, tgt)
        bead = result.beads[0]
        self.assertTrue(bead.contradiction)
        self.assertEqual(bead.confidence, "low")

    def test_high_confidence_when_anchors_agree(self):
        src = ["编号5000的订单在2024年发货。"]
        tgt = ["Order 5000 was shipped in 2024."]
        result = align(src, tgt)
        self.assertEqual(result.beads[0].confidence, "high")


class RawTextTests(unittest.TestCase):
    def test_align_texts(self):
        zh = "纯水在标准大气压下于100摄氏度时沸腾。当外界气压降低时，沸点也会随之降低。"
        en = ("Pure water boils at 100 degrees Celsius at standard pressure. "
              "When the ambient pressure drops, its boiling point drops as well.")
        result = align_texts(zh, en)
        self.assertEqual(len(result.beads), 2)
        self.assertTrue(all(b.bead_type == "1-1" for b in result.beads))


class PMITests(unittest.TestCase):
    def test_pmi_finds_association(self):
        pairs = [({"chat"}, {"cat"})] * 4 + [({"chien"}, {"dog"})] * 2
        lex = F.LexicalModel()
        lex.train(pairs)
        sim, n = lex.similarity({"chat"}, {"cat"})
        self.assertGreater(n, 0)
        self.assertGreater(sim, 0.0)

    def test_pmi_unrelated_not_kept(self):
        pairs = [
            ({"a1"}, {"b1", "b2", "b3"}),
            ({"a2"}, {"b1", "b4", "b5"}),
            ({"a3"}, {"b1", "b6", "b7"}),
        ]
        lex = F.LexicalModel()
        lex.train(pairs)
        self.assertIsNone(lex.pair_pmi("a1", "b1"))  # b1 appears in every doc -> PMI 0

    def test_pmi_stable_pair_kept(self):
        pairs = [
            ({"chat", "rouge"}, {"cat", "red"}),
            ({"chat", "bleu"}, {"cat", "blue"}),
            ({"chat", "noir"}, {"cat", "black"}),
            ({"chien"}, {"dog"}),
        ]
        lex = F.LexicalModel()
        lex.train(pairs)
        self.assertGreater(lex.pair_pmi("chat", "cat"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
