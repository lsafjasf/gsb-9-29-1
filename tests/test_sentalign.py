"""Edge-case and unit tests for sentalign. Run: python3 -m unittest discover -s tests -v"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentalign import split_sentences, align
from sentalign.evaluate import evaluate, classify_error


class TestSplitter(unittest.TestCase):
    def test_english(self):
        s = split_sentences("Dr. Smith left. He came back at 7 p.m. Was he late?")
        self.assertEqual(len(s), 3)

    def test_chinese(self):
        s = split_sentences("他来了。她走了！真的吗？")
        self.assertEqual(s, ["他来了。", "她走了！", "真的吗？"])

    def test_blank_line_boundary(self):
        s = split_sentences("第一段没有句号\n\n第二段开始了。")
        self.assertEqual(len(s), 2)

    def test_empty(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n\n "), [])


class TestEdgeCases(unittest.TestCase):
    def test_both_empty(self):
        aln = align([], [])
        self.assertEqual(list(aln), [])

    def test_one_side_empty(self):
        aln = align(["Only source."], [])
        self.assertEqual(len(aln.beads), 1)
        self.assertEqual(aln.beads[0].kind, "1-0")
        aln2 = align([], ["只有目标句。"])
        self.assertEqual(aln2.beads[0].kind, "0-1")

    def test_single_sentence_pair(self):
        aln = align(["The cat sleeps."], ["猫在睡觉。"])
        self.assertEqual(len(aln.beads), 1)
        self.assertEqual(aln.beads[0].kind, "1-1")

    def test_skewed_ratio(self):
        # 1 translated sentence + 8 source-only sentences (ratio 9:1)
        src = ["The report was published in 2024."] + \
              ["Filler sentence number %d with no counterpart." % i
               for i in range(8)]
        tgt = ["该报告于2024年发布。"]
        aln = align(src, tgt)
        # every source sentence must be covered exactly once
        covered = sorted(i for b in aln.beads for i in range(b.i0, b.i1))
        self.assertEqual(covered, list(range(len(src))))
        # the real pair must be found
        self.assertTrue(any(b.i0 == 0 and b.i1 == 1 and b.j0 == 0 and b.j1 == 1
                            for b in aln.beads))

    def test_repeated_sentences(self):
        src = ["He smiled.", "She left.", "He smiled."]
        tgt = ["他笑了。", "她离开了。", "他笑了。"]
        aln = align(src, tgt)
        kinds = [b.kind for b in aln.beads]
        self.assertEqual(kinds, ["1-1", "1-1", "1-1"])
        for k, b in enumerate(aln.beads):
            self.assertEqual((b.i0, b.j0), (k, k))

    def test_merge_and_deletion(self):
        src = ["Chapter 1 begins here.", "Section 2 follows.",
               "Section 3 continues.", "Chapter 4 ends.",
               "Appendix 9 has no translation at all."]
        tgt = ["第1章从这里开始。", "第2节紧随其后，第3节继续。", "第4章结束。"]
        aln = align(src, tgt)
        kinds = [b.kind for b in aln.beads]
        self.assertIn("2-1", kinds)   # merge detected
        self.assertIn("1-0", kinds)   # deletion detected
        # the deletion must be the last (untranslated) sentence
        deletions = [b for b in aln.beads if b.j1 == b.j0]
        self.assertTrue(any(b.i0 == 4 for b in deletions))

    def test_low_confidence_flagged(self):
        # ambiguous content: similar lengths, no lexical signal at all
        src = ["Xylophone quark jabber.", "Wobble flibberty gibbet.",
               "Zanzibar hoopla nimbus."]
        tgt = ["完全无关的内容。", "没有任何对应关系。", "随机的一句话。"]
        aln = align(src, tgt)
        self.assertTrue(any(b.tentative for b in aln.beads),
                        "unrelated text must not be paired with confidence")

    def test_confidence_range(self):
        aln = align(["The cat sleeps.", "Dogs bark loudly."],
                    ["猫在睡觉。", "狗叫得很响。"])
        for b in aln.beads:
            self.assertGreater(b.confidence, 0.0)
            self.assertLessEqual(b.confidence, 1.0)


class TestEvaluate(unittest.TestCase):
    def test_perfect(self):
        gold = [((1, 1), (1, 1)), ((2, 3), (2, 2)), ((4, 4), None)]
        res = evaluate(gold, gold)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["errors"], {})

    def test_error_types(self):
        gold = [((1, 1), (1, 1)),      # ok
                ((2, 3), (2, 2)),      # pred splits -> missed_merge
                ((4, 4), None),        # pred aligns it -> missed_deletion
                ((5, 5), (3, 3))]      # pred shifts -> boundary_shift
        pred = [((1, 1), (1, 1)),
                ((2, 2), (2, 2)),
                ((3, 3), None),
                ((4, 4), (3, 3)),
                ((5, 5), (4, 4))]
        res = evaluate(pred, gold)
        self.assertEqual(res["correct"], 1)
        self.assertEqual(res["errors"].get("missed_merge"), 1)
        self.assertEqual(res["errors"].get("missed_deletion"), 1)
        self.assertEqual(res["errors"].get("boundary_shift"), 1)


if __name__ == "__main__":
    unittest.main()
