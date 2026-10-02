#!/usr/bin/env python3
"""边界用例自测：python3 -m unittest -v"""

import unittest

from keyword_extractor import KeywordExtractor, extract_keywords


class TestEdgeCases(unittest.TestCase):
    def test_empty_and_punct_only(self):
        self.assertEqual(extract_keywords(""), [])
        self.assertEqual(extract_keywords("！！！。。。？？？"), [])

    def test_single_document_mode(self):
        # 单文档：不提供语料时 idf 恒为 1，退化为位置加权词频，仍能正常抽取
        text = "梯田是山地农业的重要形式。梯田能够保持水土，梯田景观也吸引游客。"
        result = extract_keywords(text, title="梯田农业", topk=5)
        self.assertTrue(result)
        self.assertIn("梯田", result)

    def test_unsegmented_language(self):
        # 无分词语言（中文/日文，词与词之间无空格）：不依赖任何词典或分词器
        text = "自然言語処理は面白い。自然言語処理を学ぶ学生が増えている。"
        result = extract_keywords(text, topk=5)
        self.assertIn("自然言語処理", result)

    def test_ultra_short_document(self):
        # 超短文档：频次阈值自动放宽为 1，不报错、有产出
        result = extract_keywords("今天天气不错。", topk=5)
        self.assertTrue(result)
        for term in result:
            self.assertNotIn("的", term)

    def test_domain_terms(self):
        # 领域词集中且含未登录新词：光刻胶 / 国产化 / 制程，无词典也能抽出
        text = (
            "光刻胶是半导体制造的关键材料。光刻胶国产化正在加速，"
            "国产光刻胶企业通过制程验证，光刻胶产品实现小批量供货。"
        )
        result = extract_keywords(text, title="光刻胶国产化", topk=5)
        self.assertIn("光刻胶", result)

    def test_stopwords_filtered(self):
        text = "生命的意义以及生活的价值是哲学的核心问题。哲学探讨生命的意义。"
        result = extract_keywords(text, topk=10)
        for word in ("的", "是", "以及", "以", "及"):
            self.assertNotIn(word, result)

    def test_phrase_merging(self):
        # 相邻高频字合并为短语；“机器”从未脱离“机器学习”独立出现，应被合并掉
        text = "机器学习很有趣。机器学习很有用。机器学习应用广泛。"
        result = extract_keywords(text, topk=5)
        self.assertIn("机器学习", result)
        self.assertNotIn("机器", result)
        self.assertNotIn("学习", result)

    def test_subterm_survives_when_independent(self):
        # “水稻”在“杂交水稻”之外独立出现时，短词与短语应同时保留
        text = "杂交水稻亩产再创新高。水稻育种持续推进，杂交水稻推广面积扩大。"
        result = extract_keywords(text, title="杂交水稻", topk=8)
        self.assertIn("杂交水稻", result)
        self.assertIn("水稻", result)

    def test_position_weight_title_beats_body(self):
        # 苹果仅在标题出现 1 次（3.0），香蕉仅在段首出现 1 次（2.0）
        result = extract_keywords(
            "香蕉是一种常见水果。", title="苹果", topk=2, with_scores=True
        )
        self.assertEqual(result[0][0], "苹果")
        self.assertGreater(result[0][1], result[1][1])

    def test_english_phrase(self):
        text = "Machine learning is fun. Machine learning is useful. Machine learning matters."
        result = extract_keywords(text, topk=5)
        self.assertIn("machine learning", result)
        self.assertNotIn("machine", result)

    def test_idf_discriminates_corpus_terms(self):
        # “苹果”在语料每篇都出现（idf 低），“芯片”只出现一次（idf 高）
        corpus = [
            "苹果 香蕉 水果",
            "苹果 手机 芯片",
            "苹果 果汁 水果",
        ]
        extractor = KeywordExtractor(corpus)
        result = extractor.extract(
            "苹果、芯片。苹果、芯片。苹果、芯片。", topk=2, with_scores=True
        )
        self.assertEqual(result[0][0], "芯片")


if __name__ == "__main__":
    unittest.main()
