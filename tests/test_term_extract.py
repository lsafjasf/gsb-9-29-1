"""Self tests for term_extract, including boundary cases.

Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from term_extract import TermExtractor, evaluate, load_gold
from term_extract.tokenizer import tokenize, sentences, is_cjk
from term_extract.candidates import generate_candidates, STOPWORDS_EN
from term_extract.merging import (canonical_key, singularize,
                                  find_abbrev_pairs, merge_candidates)
from term_extract.scoring import entropy


def extract_terms(text, **kw):
    ex = TermExtractor(**kw)
    terms, log = ex.extract(text)
    return {t["term"].lower(): t for t in terms}, terms, log


class TestTokenizer(unittest.TestCase):
    def test_english_words_and_numbers(self):
        toks = [t for t, _, _ in tokenize("BERT uses 3.5 layers.")]
        self.assertIn("BERT", toks)
        self.assertIn("3.5", toks)

    def test_cjk_chars_are_single_tokens(self):
        toks = [t for t, _, _ in tokenize("深度学习")]
        self.assertEqual(toks, ["深", "度", "学", "习"])

    def test_hyphenated_word_is_one_token(self):
        toks = [t for t, _, _ in tokenize("write-ahead logging")]
        self.assertIn("write-ahead", toks)

    def test_sentence_split_on_cjk_period(self):
        sents = sentences(tokenize("机器学习很好。深度学习更强。"))
        self.assertEqual(len(sents), 2)

    def test_comma_is_hard_barrier(self):
        # candidates must not span a comma
        sents = sentences(tokenize("neural networks, and deep learning"))
        self.assertEqual(len(sents), 2)


class TestCandidateBoundaries(unittest.TestCase):
    def _cands(self, text, **kw):
        return generate_candidates([sentences(tokenize(text))], **kw)

    def test_no_stopword_at_boundary(self):
        cands = self._cands("the neural network and the deep neural network "
                            "the neural network")
        for key, c in cands.items():
            self.assertNotIn(c.tokens[0].lower(), STOPWORDS_EN)
            self.assertNotIn(c.tokens[-1].lower(), STOPWORDS_EN)

    def test_term_at_sentence_edges(self):
        # term appears at both sentence start and sentence end
        cands = self._cands("Knowledge graphs are useful. We study "
                            "knowledge graphs. Knowledge graphs win.")
        self.assertIn("knowledge graphs", cands)
        self.assertEqual(cands["knowledge graphs"].freq, 3)

    def test_term_adjacent_to_punctuation(self):
        cands = self._cands("(graph neural networks) -- graph neural "
                            "networks; graph neural networks!")
        self.assertEqual(cands["graph neural networks"].freq, 3)

    def test_single_cjk_char_not_a_term(self):
        cands = self._cands("学习学习学习学习")
        self.assertNotIn("学", cands)
        self.assertNotIn("习", cands)

    def test_pure_number_not_a_term(self):
        cands = self._cands("version 3 3 3 3 of the 2024 2024 2024 model")
        self.assertNotIn("3", cands)
        self.assertNotIn("2024", cands)

    def test_zh_stopchar_boundary(self):
        cands = self._cands("神经网络的神经网络的神经网络")
        self.assertNotIn("神经网络的", cands)


class TestScoring(unittest.TestCase):
    def test_entropy(self):
        self.assertAlmostEqual(entropy({"a": 1, "b": 1}), 1.0)
        self.assertAlmostEqual(entropy({"a": 4}), 0.0)

    def test_fragment_marked_incomplete(self):
        # "deep" is almost always followed by "neural network(s)"
        text = ("Deep neural networks learn fast. "
                "Deep neural networks need data. "
                "Deep neural networks scale well. "
                "Deep neural networks win benchmarks. "
                "We study neural networks. Neural networks adapt. "
                "These neural networks converge. ")
        ex = TermExtractor(min_freq=2, window=1, min_score=0.0)
        cands = {c.key: c for c in ex.extract_candidates(text)}
        self.assertIn("deep neural networks", cands)
        self.assertTrue(cands["deep neural networks"].complete)
        self.assertIn("deep", cands)
        self.assertFalse(cands["deep"].complete)

    def test_window_parameter_changes_neighbours(self):
        text = "the big red graph kernel method and a graph kernel method"
        ex1 = TermExtractor(min_freq=1, window=1, min_score=0.0)
        ex2 = TermExtractor(min_freq=1, window=3, min_score=0.0)
        c1 = {c.key: c for c in ex1.extract_candidates(text)}
        c2 = {c.key: c for c in ex2.extract_candidates(text)}
        self.assertLessEqual(sum(c1["graph kernel"].left.values()),
                             sum(c2["graph kernel"].left.values()))


class TestMerging(unittest.TestCase):
    def test_singularize(self):
        self.assertEqual(singularize("networks"), "network")
        self.assertEqual(singularize("companies"), "company")
        self.assertEqual(singularize("classes"), "class")
        self.assertEqual(singularize("analysis"), "analysis")  # -is kept
        self.assertEqual(singularize("bus"), "bus")

    def test_canonical_key_case_and_plural(self):
        self.assertEqual(canonical_key("Neural Networks"),
                         canonical_key("neural network"))
        self.assertEqual(canonical_key("BERT"), canonical_key("bert"))

    def test_hyphen_space_equivalence(self):
        self.assertEqual(canonical_key("fine-tuning"),
                         canonical_key("fine tuning"))

    def test_explicit_abbrev_pattern(self):
        pairs = find_abbrev_pairs(
            "Natural Language Processing (NLP) is fun.")
        self.assertIn(("Natural Language Processing", "NLP",
                       pairs[0][2]), pairs)

    def test_zh_english_paren_pattern(self):
        pairs = find_abbrev_pairs(
            "卷积神经网络（Convolutional Neural Network, CNN）很好用")
        longs = {(a, b) for a, b, _ in pairs}
        self.assertIn(("卷积神经网络", "Convolutional Neural Network"), longs)
        self.assertIn(("Convolutional Neural Network", "CNN"), longs)

    def test_case_variants_merge_with_reason(self):
        text = "Knowledge Graphs are hot. knowledge graphs are hot. " \
               "Knowledge graphs are hot."
        _, terms, _ = extract_terms(text, min_freq=1)
        kg = [t for t in terms if t["term"].lower() == "knowledge graphs"]
        self.assertEqual(len(kg), 1)
        self.assertTrue(any("case_variant" in r for r in kg[0]["merge_reasons"]))

    def test_plural_merges_into_singular_group(self):
        text = ("A graph kernel compares structures. "
                "Graph kernels are expressive. "
                "The graph kernel matrix is positive definite. "
                "Many graph kernels run in polynomial time. "
                "One graph kernel suffices here. "
                "These graph kernels differ.")
        _, terms, _ = extract_terms(text, min_freq=2)
        gk = [t for t in terms
              if canonical_key(t["term"]) == canonical_key("graph kernel")]
        self.assertEqual(len(gk), 1)
        self.assertTrue(any("inflection_variant" in r
                            for r in gk[0]["merge_reasons"]))

    def test_abbrev_fullform_merge_with_reason(self):
        text = ("Support Vector Machines (SVMs) are classic. "
                "SVMs are still used. Support vector machines rock. "
                "support vector machines win.")
        _, terms, log = extract_terms(text, min_freq=1)
        svm = [t for t in terms if canonical_key(t["term"])
               == canonical_key("support vector machines")]
        self.assertEqual(len(svm), 1)
        self.assertTrue(any("abbreviation" in r
                            for r in svm[0]["merge_reasons"]))
        self.assertTrue(any("abbreviation" in entry for entry in log))

    def test_no_duplicate_concepts_after_merge(self):
        text = "Neural networks learn. A neural network learns. " \
               "NEURAL NETWORKS rule. neural network!"
        _, terms, _ = extract_terms(text, min_freq=1)
        keys = [canonical_key(t["term"]) for t in terms]
        self.assertEqual(len(keys), len(set(keys)))


class TestEndToEnd(unittest.TestCase):
    DATA = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")

    def _run(self, *files, gold):
        docs = []
        for f in files:
            with open(os.path.join(self.DATA, f), encoding="utf-8") as fh:
                docs.append(fh.read())
        ex = TermExtractor(min_freq=2, window=2, top_k=40)
        terms, log = ex.extract(docs)
        return evaluate(terms, os.path.join(self.DATA, gold),
                        raw_candidate_count=len(ex.extract_candidates(docs)))

    def test_en_single_doc(self):
        m = self._run("en_single.txt", gold="en_single.gold.tsv")
        self.assertGreaterEqual(m["precision"], 0.6)
        self.assertGreaterEqual(m["recall"], 0.6)
        self.assertEqual(m["redundancy_rate_after_merge"], 0.0)

    def test_en_multi_doc(self):
        m = self._run("en_multi/doc1.txt", "en_multi/doc2.txt",
                      "en_multi/doc3.txt", gold="en_multi.gold.tsv")
        self.assertGreaterEqual(m["precision"], 0.55)
        self.assertGreaterEqual(m["recall"], 0.6)

    def test_mixed_zh_en(self):
        m = self._run("mixed.txt", gold="mixed.gold.tsv")
        self.assertGreaterEqual(m["precision"], 0.6)
        self.assertGreaterEqual(m["recall"], 0.5)

    def test_zh_abbrev_merged(self):
        with open(os.path.join(self.DATA, "mixed.txt"),
                  encoding="utf-8") as fh:
            text = fh.read()
        _, terms, _ = extract_terms(text, min_freq=2)
        cnn = [t for t in terms if "卷积神经网络" in t["term"]]
        self.assertEqual(len(cnn), 1)
        self.assertIn("CNN", cnn[0]["variants"])


if __name__ == "__main__":
    unittest.main()
