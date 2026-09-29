# -*- coding: utf-8 -*-
"""敏感词变体检出自测：覆盖符号干扰、连续谐音、中英混排、拆字、
重复字符、位置映射、白名单、拉丁边界与超长文本。"""

import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sensitive_filter import Matcher, normalize  # noqa: E402

RULES = ROOT / "config" / "rules.json"
MAPS = ROOT / "config" / "maps.json"
ATTACK = ROOT / "data" / "attack_corpus.json"
TRICKY = ROOT / "data" / "normal_tricky.txt"


def make_matcher():
    return Matcher.from_files(RULES, MAPS)


class TestNormalize(unittest.TestCase):
    def test_zero_width_and_symbols_removed(self):
        norm, idx = normalize("微​信※号")
        self.assertEqual(norm, "微信号")
        self.assertEqual(idx, [0, 2, 4])  # 位置映射指回原文

    def test_fullwidth_and_casefold(self):
        norm, _ = normalize("ＶＸ　Ｃａｓｉｎｏ")
        self.assertEqual(norm, "vxcasino")

    def test_repeat_collapsed(self):
        norm, idx = normalize("赌赌赌博")
        self.assertEqual(norm, "赌博")
        self.assertEqual(idx, [0, 3])

    def test_only_noise(self):
        norm, idx = normalize("※★　​")
        self.assertEqual(norm, "")
        self.assertEqual(idx, [])


class TestEvasionDetection(unittest.TestCase):
    """攻击语料全部应命中（按类别覆盖）。"""

    @classmethod
    def setUpClass(cls):
        cls.matcher = make_matcher()
        cls.corpus = json.loads(ATTACK.read_text(encoding="utf-8"))

    def test_all_attack_samples_detected(self):
        missed = []
        for item in self.corpus:
            res = self.matcher.scan(item["text"])
            got = {h.rule_id for h in res.hits}
            if not set(item["expect"]) <= got:
                missed.append((item["text"], item["expect"], sorted(got)))
        self.assertEqual(missed, [])

    def test_categories_covered(self):
        cats = {item["category"] for item in self.corpus}
        for expected in ["symbol", "repeat", "homophone", "homophone_chain",
                         "shape", "split", "mixed_script", "combo"]:
            self.assertIn(expected, cats)


class TestEvidenceAndPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = make_matcher()

    def test_position_maps_back_to_original(self):
        text = "快来赌※博吧"
        hit = self.matcher.scan(text).hits[0]
        self.assertEqual((hit.start, hit.end), (2, 5))
        self.assertEqual(text[hit.start:hit.end], "赌※博")
        self.assertEqual(hit.snippet, "赌※博")

    def test_zero_width_span(self):
        text = "微​信"
        hit = self.matcher.scan(text).hits[0]
        self.assertEqual((hit.start, hit.end), (0, 3))
        self.assertEqual(hit.snippet, "微​信")

    def test_evidence_fields(self):
        hit = self.matcher.scan("加我唯★欣").hits[0]
        d = hit.to_dict()
        for key in ["rule_id", "word", "variant", "confidence", "span",
                    "snippet", "normalized"]:
            self.assertIn(key, d)
        self.assertEqual(d["word"], "微信")
        self.assertEqual(d["variant"], "唯欣")
        self.assertEqual(d["confidence"], "low")  # 连续谐音 -> 低置信度

    def test_confidence_levels(self):
        m = self.matcher
        self.assertEqual(m.scan("赌博").hits[0].confidence, "high")     # 原文
        self.assertEqual(m.scan("赌愽").hits[0].confidence, "medium")   # 形近
        self.assertEqual(m.scan("贝者博").hits[0].confidence, "medium") # 拆字
        self.assertEqual(m.scan("堵博").hits[0].confidence, "low")      # 谐音


class TestFalsePositiveControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = make_matcher()
        cls.lines = [l.strip() for l in TRICKY.read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_tricky_normal_texts_no_hit(self):
        fp = []
        for line in self.lines:
            for h in self.matcher.scan(line).hits:
                fp.append((line, h.rule_id, h.variant))
        self.assertEqual(fp, [])

    def test_whitelist_suppression_records_evidence(self):
        res = self.matcher.scan("他很有威信，大家都服他。")
        self.assertEqual(res.hits, [])
        self.assertEqual(len(res.suppressed), 1)
        self.assertEqual(res.suppressed[0].suppress_reason, "whitelist:有威信")

    def test_latin_boundary(self):
        m = self.matcher
        self.assertEqual(m.scan("pvxz").hits, [])          # 子串不命中
        self.assertEqual(m.scan("casinox").hits, [])
        self.assertEqual(len(m.scan("vx").hits), 1)        # 独立出现才命中
        self.assertEqual(len(m.scan("加vx哦").hits), 1)    # 中文语境边界
        self.assertEqual(m.scan("The casinos are closed.").hits, [])  # 复数边界

    def test_no_hit_cases(self):
        m = self.matcher
        for text in ["", "※★　", "哈哈哈哈", "今天天气不错", "hello world"]:
            self.assertEqual(m.scan(text).hits, [], text)


class TestEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = make_matcher()

    def test_overlapping_rules(self):
        res = self.matcher.scan("赌博博彩")
        self.assertEqual({h.rule_id for h in res.hits}, {"R-DB", "R-BC"})

    def test_hit_at_boundaries(self):
        res = self.matcher.scan("赌博")
        self.assertEqual((res.hits[0].start, res.hits[0].end), (0, 2))

    def test_long_repeat_run(self):
        res = self.matcher.scan("赌" * 10000 + "博")
        self.assertEqual(len(res.hits), 1)

    def test_long_text_performance(self):
        base = "今天天气不错，我们一起去公园散步，顺便买了点水果。"
        text = (base * 20000) + "贝★者 博" + (base * 20000)  # 约 100 万字符
        t0 = time.perf_counter()
        res = self.matcher.scan(text)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(res.hits), 1)
        self.assertEqual(res.hits[0].rule_id, "R-DB")
        print(f"\n[perf] {len(text)} chars scanned in {elapsed:.3f}s "
              f"({len(text)/elapsed/1e6:.1f} MB/s)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
