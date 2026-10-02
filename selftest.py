"""summarizer 库的自测（标准库 unittest）。

运行：python3 selftest.py [-v]
覆盖：空文档、单句、全重复句、超长句、无标题、严格预算、
      顺序保持、确定性、分块一致性、得分构成、冗余惩罚效果。
"""

import random
import unittest

from summarizer import (
    summarize,
    summarize_chunks,
    split_sentences,
    tokenize,
    _iter_chunk_sentences,
)

SAMPLE_TITLE = "太阳能发电的发展"
SAMPLE_DOC = (
    "太阳能是一种清洁的可再生能源。近年来光伏发电成本大幅下降。"
    "太阳能电池板的效率不断提升。风力发电也是重要的清洁能源。"
    "光伏发电成本下降主要得益于技术进步。储能技术是可再生能源发展的关键。"
    "许多国家制定了碳中和目标。太阳能发电在全球能源结构中的占比持续上升。"
    "电池储能成本也在快速下降。太阳能是一种清洁的可再生能源。"
    "未来太阳能有望成为主要电力来源。"
)


def make_doc(n_sent, seed=42):
    rng = random.Random(seed)
    topics = ["太阳能", "储能", "电网", "碳中和", "风电"]
    fillers = ["发展", "成本", "效率", "政策", "市场", "技术"]
    sents = []
    for i in range(n_sent):
        t = topics[i % len(topics)]
        words = " ".join(rng.choice(fillers) for _ in range(rng.randint(3, 8)))
        sents.append(f"{t}领域的{words}持续推进，编号{i}。")
    return "".join(sents)


class TestEdgeCases(unittest.TestCase):
    def test_empty_document(self):
        r = summarize("")
        self.assertEqual(r.sentences, [])
        self.assertEqual(r.n_sentences, 0)
        self.assertFalse(r.overflow)

    def test_whitespace_only(self):
        r = summarize("   \n\t 。")
        self.assertEqual(r.sentences, [])

    def test_single_sentence_fits(self):
        r = summarize("机器学习是人工智能的重要分支。", budget=100)
        self.assertEqual(len(r.sentences), 1)
        self.assertFalse(r.overflow)

    def test_single_sentence_tight_budget_non_empty(self):
        # 预算小于唯一句子长度：不得返回空，标记 overflow
        r = summarize("机器学习是人工智能的重要分支。", budget=3)
        self.assertEqual(len(r.sentences), 1)
        self.assertTrue(r.overflow)

    def test_all_duplicate_sentences(self):
        doc = "重复的内容出现了。" * 30
        r = summarize(doc, budget=10, budget_unit="sentences")
        self.assertTrue(r.sentences)
        self.assertGreater(r.redundancy, 0.99)  # 全重复 -> 冗余度接近 1
        self.assertLessEqual(len(r.sentences), 10)

    def test_super_long_sentence(self):
        doc = " ".join(f"词{i}" for i in range(5000)) + "。"
        r = summarize(doc, budget=50)
        self.assertEqual(len(r.sentences), 1)  # 不为空
        self.assertTrue(r.overflow)

    def test_no_title(self):
        r = summarize(SAMPLE_DOC, budget=5, budget_unit="sentences")
        self.assertTrue(r.sentences)
        self.assertTrue(all(s.title == 0.0 for s in r.scores))

    def test_zero_budget_still_non_empty(self):
        r = summarize(SAMPLE_DOC, budget=0)
        self.assertEqual(len(r.sentences), 1)
        self.assertTrue(r.overflow)


class TestBudget(unittest.TestCase):
    def test_word_budget_strict(self):
        doc = make_doc(200)
        for budget in (20, 50, 100, 300):
            r = summarize(doc, budget=budget)
            self.assertLessEqual(r.used, budget)
            self.assertTrue(r.sentences)

    def test_sentence_budget_strict(self):
        doc = make_doc(100)
        for k in (1, 3, 7):
            r = summarize(doc, budget=k, budget_unit="sentences")
            self.assertLessEqual(len(r.sentences), k)
            self.assertEqual(len(r.sentences), k)

    def test_dp_budget_strict(self):
        doc = make_doc(150)
        for budget in (30, 80, 200):
            r = summarize(doc, budget=budget, strategy="dp")
            self.assertLessEqual(r.used, budget)
            self.assertTrue(r.sentences)


class TestDeterminismAndOrder(unittest.TestCase):
    def test_deterministic(self):
        doc = make_doc(120)
        r1 = summarize(doc, title="能源", budget=80)
        r2 = summarize(doc, title="能源", budget=80)
        self.assertEqual(r1.sentences, r2.sentences)
        self.assertEqual(r1.indices, r2.indices)
        self.assertEqual([s.total for s in r1.scores],
                         [s.total for s in r2.scores])

    def test_order_preserved(self):
        doc = make_doc(150)
        for strategy in ("greedy", "dp"):
            r = summarize(doc, budget=60, strategy=strategy)
            self.assertEqual(r.indices, sorted(r.indices))

    def test_chunked_equals_whole(self):
        doc = make_doc(300)
        whole = summarize(doc, title="能源", budget=100)
        for cs in (1, 3, 17, 64, 1024):
            chunked = summarize(doc, title="能源", budget=100, chunk_size=cs)
            self.assertEqual(whole.sentences, chunked.sentences, f"chunk_size={cs}")
            self.assertEqual(whole.indices, chunked.indices)

    def test_streaming_split_equals_whole_split(self):
        # 含跨块标点、连续标点、换行的恶意文本
        text = "第一句... 还是第一句？不，第二句!!第三句。\n\n第四句。。。第五句"
        expected = split_sentences(text)
        for cs in range(1, 10):
            chunks = [text[i:i + cs] for i in range(0, len(text), cs)]
            self.assertEqual(list(_iter_chunk_sentences(chunks)), expected,
                             f"chunk_size={cs}")

    def test_summarize_chunks_api(self):
        doc = make_doc(200)
        whole = summarize(doc, budget=80)
        chunks = [doc[i:i + 50] for i in range(0, len(doc), 50)]
        r = summarize_chunks(chunks, budget=80)
        self.assertEqual(whole.sentences, r.sentences)


class TestScoring(unittest.TestCase):
    def test_score_breakdown_complete(self):
        r = summarize(SAMPLE_DOC, title=SAMPLE_TITLE, budget=60)
        w = (1.0, 0.5, 0.5)
        for s in r.scores:
            self.assertAlmostEqual(
                s.total, w[0] * s.freq + w[1] * s.position + w[2] * s.title,
                places=9)
        self.assertEqual(len(r.scores), r.n_sentences)

    def test_weights_configurable(self):
        r = summarize(SAMPLE_DOC, title=SAMPLE_TITLE, budget=60,
                      weights=(0.0, 0.0, 1.0))
        for s in r.scores:
            self.assertAlmostEqual(s.total, s.title, places=9)

    def test_redundancy_penalty_configurable(self):
        # 构造含近重复句的文档：惩罚越强，冗余度应越低
        base = "太阳能发电技术不断进步，成本持续下降。"
        noise = [f"第{i}条完全不同的电网政策新闻。" for i in range(10)]
        doc = base + base + base + "".join(noise)
        r_low = summarize(doc, budget=4, budget_unit="sentences",
                          redundancy_penalty=0.0)
        r_high = summarize(doc, budget=4, budget_unit="sentences",
                           redundancy_penalty=0.9)
        self.assertLess(r_high.redundancy, r_low.redundancy)

    def test_coverage_and_redundancy_ranges(self):
        r = summarize(make_doc(100), budget=80)
        self.assertGreaterEqual(r.coverage, 0.0)
        self.assertLessEqual(r.coverage, 1.0)
        self.assertGreaterEqual(r.redundancy, 0.0)
        self.assertLessEqual(r.redundancy, 1.0)
        self.assertTrue(r.keywords)


class TestCandidateLimit(unittest.TestCase):
    def test_candidate_limit_consistent_when_large(self):
        doc = make_doc(300)
        r_full = summarize(doc, budget=80)
        r_lim = summarize(doc, budget=80, candidate_limit=300)
        self.assertEqual(r_full.sentences, r_lim.sentences)

    def test_candidate_limit_bounds_selection(self):
        doc = make_doc(500)
        r = summarize(doc, budget=80, candidate_limit=50)
        self.assertTrue(r.sentences)
        self.assertLessEqual(r.used, 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
