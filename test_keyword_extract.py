"""边界用例自测：python3 test_keyword_extract.py [-v]

覆盖：单文档、无分词语言（中文）、超短文档、领域词集中、
停用词过滤、短语合并、位置权重、空输入。
"""

import unittest

from keyword_extract import (
    EN_STOPWORDS,
    ZH_STOPCHARS,
    extract_keywords,
    extract_single,
)


def terms(result):
    return [t for t, _ in result]


class TestStopwords(unittest.TestCase):
    def test_stopwords_never_extracted(self):
        doc = "结论\n\n的是以及和在出现了多次，的是以及和在又出现了，的是以及和在还出现。"
        out = terms(extract_single(doc, top_k=10))
        for t in out:
            self.assertNotIn(t, EN_STOPWORDS)
            self.assertNotIn(t, ZH_STOPCHARS)
            for ch in t:
                self.assertNotIn(ch, ZH_STOPCHARS)


class TestSingleDocument(unittest.TestCase):
    """单文档语料：idf 退化为 1，不应报错且仍有合理输出。"""

    def test_single_doc_idf_degenerate(self):
        doc = (
            "石墨烯导电性研究\n\n"
            "石墨烯是单层碳原子材料，石墨烯的载流子迁移率极高。"
            "实验表明石墨烯在室温下仍保持优异导电性。"
        )
        out = extract_single(doc, top_k=5)
        self.assertTrue(out)
        self.assertEqual(out[0][0], "石墨烯")
        for _, score in out:
            self.assertGreater(score, 0)


class TestNoSegmentationLanguage(unittest.TestCase):
    """无分词语言：中文整句无空格，仍应抽出多字词。"""

    def test_cjk_without_spaces(self):
        doc = (
            "区块链共识机制\n\n"
            "区块链依赖共识机制保证账本一致，共识机制中工作量证明最为经典，"
            "区块链网络借此抵御篡改。"
        )
        out = terms(extract_single(doc, top_k=5))
        self.assertIn("区块链", out)
        self.assertIn("共识机制", out)

    def test_cjk_absorption_prefers_longer_term(self):
        # 「量子」的出现几乎都被「量子计算/量子比特」覆盖，短词应被吸收
        doc = "量子计算\n\n量子计算依赖量子比特，量子比特数决定量子计算规模。"
        out = terms(extract_single(doc, top_k=5))
        self.assertIn("量子计算", out)
        self.assertNotIn("量子", out)


class TestUltraShortDocument(unittest.TestCase):
    """超短文档：min_tf 降为 1，不崩溃且有输出。"""

    def test_title_only(self):
        out = extract_single("年度总结", top_k=5)
        self.assertIsInstance(out, list)  # 单字不成候选，允许为空，但不许崩溃

    def test_one_sentence(self):
        out = terms(extract_single("项目启动\n\n迁移冲刺周一开始。"))
        self.assertTrue(len(out) >= 1)

    def test_empty_and_whitespace(self):
        self.assertEqual(extract_single("", top_k=5), [])
        self.assertEqual(extract_single("   \n\n  ", top_k=5), [])
        self.assertEqual(extract_keywords([], top_k=5), [])


class TestDomainTermConcentration(unittest.TestCase):
    """领域词集中：高频领域词必须排在最前，通用词不得压过它。"""

    def test_domain_term_ranks_first(self):
        doc = (
            "Transformer 与注意力\n\n"
            "The transformer relies on attention. Attention lets the "
            "transformer model long-range dependencies. Each transformer "
            "layer applies attention again, so the transformer trains fast."
        )
        out = terms(extract_single(doc, top_k=5))
        self.assertEqual(out[0], "transformer")
        self.assertIn("attention", out[:2])
        self.assertNotIn("the", out)


class TestPhraseMerging(unittest.TestCase):
    """短语合并：相邻高频实词组成短语，被吸收的单词不重复输出。"""

    def test_bigram_merged_and_components_absorbed(self):
        # 正常长度文档（>50 token），短语合并与吸收规则生效
        doc = (
            "Climate Change\n\n"
            "Climate change accelerates as emissions accumulate in the "
            "atmosphere. This climate change report tracks how warming "
            "reshapes coastlines, agriculture, and freshwater supplies "
            "around the world. Scientists warn that climate change also "
            "intensifies storms, droughts, and heatwaves on every "
            "continent. When climate change is discussed at summits, "
            "negotiators debate targets, timelines, and funding."
        )
        out = terms(extract_single(doc, top_k=5))
        self.assertIn("climate change", out)
        self.assertNotIn("climate", out)
        self.assertNotIn("change", out)

    def test_low_containment_pair_not_merged(self):
        # red 与很多词相邻，red car 共现占比低，不应合成短语
        doc = (
            "Colors\n\n"
            "red car stopped. red apple fell. red sky glowed. "
            "red door opened. car horns sounded. car engines roared."
        )
        out = terms(extract_single(doc, top_k=8))
        self.assertNotIn("red car", out)


class TestPositionWeight(unittest.TestCase):
    """位置权重：词频相同时，标题词 > 段首词 > 正文词。"""

    def test_title_beats_body_at_same_tf(self):
        doc = "Zeppelin\n\nBalloon stories are old."
        # zeppelin 仅标题 1 次（权重 2.0）；balloon 段首 1 次（权重 1.5）
        kws = dict(extract_single(doc, top_k=10))
        self.assertGreater(kws["zeppelin"], kws["balloon"])

    def test_para_start_beats_body_at_same_tf(self):
        doc = (
            "Notes\n\n"
            "Anchor opens the first part. Filler text follows here.\n\n"
            "Beacon is mentioned mid paragraph only. More filler text."
        )
        # anchor 段首 1 次，beacon 正文 1 次；超短文档 min_tf=1
        kws = dict(extract_single(doc, top_k=30))
        self.assertGreater(kws["anchor"], kws["beacon"])


class TestCorpusMode(unittest.TestCase):
    """多篇语料：idf 应让独有词压过跨文档通用词。"""

    def test_idf_penalizes_common_term(self):
        docs = [
            "Report\n\nalpha one alpha two alpha three. "
            "shared four shared five shared six shared seven.",
            "Notes\n\nbeta one beta two beta three. "
            "shared four shared five shared six shared seven.",
        ]
        out = extract_keywords(docs, top_k=3)
        # shared 词频更高但 df=2，idf 更低；alpha/beta 应排第一
        self.assertEqual(out[0][0][0], "alpha")
        self.assertEqual(out[1][0][0], "beta")


if __name__ == "__main__":
    unittest.main()
