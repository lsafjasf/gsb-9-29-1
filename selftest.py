"""自测：边界用例 + 跨截断自洽性断言 + 汇总口径校验 + 自助区间合理性。

运行：python3 selftest.py
"""

import math

import ranking_metrics as M
from diff_test import random_dataset
import random


def approx_equal(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_edge_cases():
    # 1) 空排序：所有指标为 0，不报错
    labels = {"a": 1, "b": 2}
    for k in (0, 1, 5):
        res = M.evaluate_query([], labels, k, qid="empty")
        assert res["ndcg"] == res["ap"] == res["mrr"] == res["hit"] == 0.0
        assert res["idcg"] == (M.idcg_at(labels, k) if k else 0.0)

    # 2) 无相关项（零相关查询）：全部 0，nDCG 不除零
    res = M.evaluate_query([("a", 0.9), ("b", 0.5)], {"a": 0, "b": 0}, 5, qid="zero")
    assert res["ndcg"] == res["ap"] == res["mrr"] == res["hit"] == 0.0
    assert res["zero_relevance"] is True

    # 3) 全相关且排序覆盖全部文档：nDCG=AP=MRR=Hit=1
    ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    labels = {"a": 1, "b": 1, "c": 1}
    res = M.evaluate_query(ranking, labels, 5, qid="all")
    assert res["ndcg"] == 1.0 and res["ap"] == 1.0
    assert res["mrr"] == 1.0 and res["hit"] == 1.0

    # 4) 标注缺失：排序中的 d 无标注，按不相关处理
    ranking = [("x", 0.9), ("y", 0.5)]
    labels = {"z": 3}  # 唯一相关文档不在排序中
    res = M.evaluate_query(ranking, labels, 2, qid="missing")
    assert res["ndcg"] == 0.0 and res["ap"] == 0.0
    assert res["mrr"] == 0.0 and res["hit"] == 0.0
    # 相关文档在排序外：AP 分母 R=1，top1 未命中时 AP@1=0、AP@2=0
    ranking2 = [("x", 0.9), ("z", 0.5)]
    assert M.average_precision_at(ranking2, labels, 1) == 0.0
    assert M.average_precision_at(ranking2, labels, 2) == 0.5

    # 5) 只有一条结果
    assert M.evaluate_query([("a", 1.0)], {"a": 1}, 1)["ndcg"] == 1.0
    assert M.evaluate_query([("a", 1.0)], {"a": 0}, 1)["mrr"] == 0.0

    # 6) k=0：全部为 0
    res = M.evaluate_query([("a", 0.9)], {"a": 3}, 0)
    assert res["dcg"] == res["idcg"] == 0.0
    assert res["ndcg"] == res["ap"] == res["mrr"] == res["hit"] == 0.0

    # 7) 并列得分确定性：打乱输入顺序结果不变，且 doc_id 小者在前
    labels = {"b": 1, "a": 1, "c": 0}
    base = [("c", 0.5), ("a", 0.5), ("b", 0.5)]
    rng = random.Random(7)
    first = M.evaluate_query(base, labels, 3, qid="tie")
    for _ in range(20):
        perm = list(base)
        rng.shuffle(perm)
        other = M.evaluate_query(perm, labels, 3)
        for key in ("ndcg", "ap", "mrr", "hit", "dcg"):
            assert first[key] == other[key], f"并列不确定: {key}"
    # 并列顺序：a, b 相关且 doc_id 升序，RR = 1，DCG 与显式顺序 [a,b,c] 一致
    explicit = M.evaluate_query([("a", 0.5), ("b", 0.5), ("c", 0.5)], labels, 3)
    assert first["dcg"] == explicit["dcg"]

    # 8) 多级标注的 nDCG 与手工值
    ranking = [("d1", 0.9), ("d2", 0.8)]
    labels = {"d1": 3, "d2": 0, "d3": 2}
    k = 2
    dcg = (2.0 ** 3 - 1.0) / math.log2(2.0)
    idcg = (2.0 ** 3 - 1.0) / math.log2(2.0) + (2.0 ** 2 - 1.0) / math.log2(3.0)
    assert M.ndcg_at(ranking, labels, k) == dcg / idcg

    print("边界用例全部通过（空排序/无相关/全相关/标注缺失/单条结果/k=0/并列确定性/多级标注）。")


def test_truncation_monotonicity():
    """跨截断自洽：k 增大时 DCG/AP/MRR/Hit（逐查询与宏平均）单调不减。"""
    rng = random.Random(42)
    for trial in range(300):
        dataset = random_dataset(rng)
        max_len = max(
            (len(item["ranking"]) for item in dataset), default=1
        )
        prev = None
        for k in range(0, max_len + 3):
            cur = M.evaluate(dataset, k)
            if prev is not None:
                for idx, item in enumerate(cur["per_query"]):
                    old = prev["per_query"][idx]
                    for metric in ("dcg", "ap", "mrr", "hit"):
                        assert item[metric] >= old[metric] - 0.0, (
                            f"trial={trial} q={item['qid']} {metric}@k={k} "
                            f"下降: {old[metric]} -> {item[metric]}"
                        )
                for metric in ("ap", "mrr", "hit"):
                    assert cur["macro"][metric] >= prev["macro"][metric], (
                        f"trial={trial} macro {metric}@k={k} 下降"
                    )
                # DCG 宏平均同样单调
                old_dcg = sum(q["dcg"] for q in prev["per_query"]) / max(1, len(prev["per_query"]))
                new_dcg = sum(q["dcg"] for q in cur["per_query"]) / max(1, len(cur["per_query"]))
                assert new_dcg >= old_dcg
            prev = cur

    # 显式验证一个曾经会出错的 min(R,k) 反例：
    # 只有 1 个相关文档且排在 top1，AP@1 必须 == AP@2（按 R 归一化时）
    ranking = [("rel", 0.9), ("non", 0.8)]
    labels = {"rel": 1}
    assert M.average_precision_at(ranking, labels, 1) == 1.0
    assert M.average_precision_at(ranking, labels, 2) == 1.0  # 不允许降到 0.5

    print("跨截断自洽断言通过：DCG/AP/MRR/Hit 随 k 单调不减（300 个随机数据集）。")


def test_aggregation():
    # 等规模查询：微平均 AP 必须等于宏平均 AP
    dataset = [
        {"qid": "q1", "ranking": [("a", 0.9), ("b", 0.1)], "labels": {"a": 1, "b": 1}},
        {"qid": "q2", "ranking": [("c", 0.9), ("d", 0.1)], "labels": {"c": 1, "d": 1}},
    ]
    res = M.evaluate(dataset, 2)
    assert approx_equal(res["macro"]["ap"], res["micro"]["ap"])
    assert approx_equal(res["macro"]["ndcg"], res["micro"]["ndcg"])

    # 不等规模：微平均被相关文档多的查询主导，构造出宏 != 微
    dataset = [
        # q1: 1 个相关文档且没召回 -> AP=0，R=1
        {"qid": "small", "ranking": [("x", 0.9)], "labels": {"a": 1}},
        # q2: 4 个相关文档全部排在前 4 -> AP=1，R=4
        {"qid": "big", "ranking": [("b1", 0.9), ("b2", 0.8), ("b3", 0.7), ("b4", 0.6)],
         "labels": {"b1": 1, "b2": 1, "b3": 1, "b4": 1}},
    ]
    res = M.evaluate(dataset, 10)
    assert res["macro"]["ap"] == 0.5          # (0 + 1) / 2
    assert res["micro"]["ap"] == 4.0 / 5.0    # 4 个精度命中 / 5 个相关文档
    assert res["micro"]["ap"] > res["macro"]["ap"]
    # MRR / Hit 的两种口径恒等
    assert res["macro"]["mrr"] == res["micro"]["mrr"]
    assert res["macro"]["hit"] == res["micro"]["hit"]
    attribution = M.explain_aggregation(res)
    assert "big" in attribution and "微权重" in attribution

    # 零相关查询以 0 计入宏平均（不剔除）
    dataset = [
        {"qid": "good", "ranking": [("a", 0.9)], "labels": {"a": 1}},
        {"qid": "none", "ranking": [("b", 0.9)], "labels": {"b": 0}},
    ]
    res = M.evaluate(dataset, 1)
    assert res["num_zero_relevance"] == 1
    assert res["macro"]["ap"] == 0.5
    assert res["micro"]["ap"] == 1.0  # 零相关查询不占 R 分母
    print("汇总口径校验通过（宏/微一致条件、不一致归因、零相关查询计数）。")


def test_bootstrap():
    # 正常数据：区间包住点估计，且随查询增多趋于收窄
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    low, high, info = M.bootstrap_ci(values, n_boot=4000, seed=1)
    point = sum(values) / len(values)
    assert low <= point <= high and low < high
    assert info["warning"] and "n=10" in info["warning"]

    wide = values * 6
    low2, high2, _ = M.bootstrap_ci(wide, n_boot=4000, seed=1)
    assert (high2 - low2) < (high - low)

    # 退化：所有查询取值相同 -> 单点区间并带说明
    low, high, info = M.bootstrap_ci([1.0, 1.0, 1.0], seed=1)
    assert low == high == 1.0 and info["degenerate"]

    # 微平均自助区间
    pairs = [(0.0, 1.0), (2.5, 3.0), (1.0, 2.0)]
    low, high, info = M.bootstrap_ci_micro(pairs, n_boot=2000, seed=1)
    point = sum(n for n, _ in pairs) / sum(d for _, d in pairs)
    assert low <= point <= high and info["warning"]

    print("自助区间校验通过（覆盖点估计、收窄趋势、退化情形、微平均、小样本警告）。")


if __name__ == "__main__":
    test_edge_cases()
    test_truncation_monotonicity()
    test_aggregation()
    test_bootstrap()
    print("\n全部自测通过。")
