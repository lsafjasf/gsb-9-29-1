"""排序指标库（Python 3，仅标准库）。

指标
----
- DCG / nDCG：折损累计增益（指数增益 (2**rel - 1)，log2 折损，rank 从 1 起）。
- AP@k：平均精度，按相关文档总数 R 归一化（不是 min(R, k)），保证随 k 单调不减。
- MRR@k：平均倒数排名，取 top-k 内第一个相关文档位置的倒数。
- HitRate@k：top-k 内是否命中至少一个相关文档（0/1）。

确定性规则（所有实现与对拍共用）
--------------------------------
1. 并列得分：先按 score 降序；score 相同时按 doc_id 升序。结果与输入顺序无关。
   doc_id 必须可比较且唯一（典型为 str 或 int）。
2. 缺失标注：出现在排序中但 labels 里没有的文档，相关性按 0.0 处理（视为不相关，
   而非抛错或忽略该位置——忽略位置会改变后续文档的折损）。
3. 零相关查询（labels 中不存在 rel > 0 的文档）：
   DCG/AP/MRR/HitRate 均为 0.0；IDCG=0 时 nDCG 定义为 0.0（而非除零）；
   该查询作为 0 值样本参与宏平均，并在汇总中单独计数。
4. 截断 k：k <= 0 视为空截断（全为 0）；k 大于排序长度时等价于全长。
5. 相关判定：rel > 0 即为相关（同时支持二值与多级标注）。
"""

from __future__ import annotations

import math
import random

METRICS = ("ndcg", "ap", "mrr", "hit")


# ---------- 确定性排序 ----------

def order_ranking(ranking):
    """返回确定性的 (doc_id, score) 列表：score 降序，并列时 doc_id 升序。"""
    return sorted(ranking, key=lambda item: (-item[1], item[0]))


def _ordered_pairs(ranking, k):
    return order_ranking(ranking)[: max(k, 0)]


# ---------- 单查询指标 ----------

def dcg_of_gains(gains):
    """给定已按位置排好的相关性序列，计算 DCG（rank 从 1 开始，log2 折损）。"""
    total = 0.0
    for index, gain in enumerate(gains):
        total += (2.0 ** gain - 1.0) / math.log2(index + 2.0)
    return total


def _ranked_gains(ranking, labels, k):
    return [float(labels.get(doc_id, 0.0)) for doc_id, _ in _ordered_pairs(ranking, k)]


def dcg_at(ranking, labels, k):
    return dcg_of_gains(_ranked_gains(ranking, labels, k))


def idcg_at(labels, k):
    """理想排序（按 labels 中所有标注降序）在截断 k 下的 DCG。

    包含“不在排序结果里但有标注”的文档——它们决定 AP 的分母与 nDCG 的理想值。
    """
    ideal_gains = sorted((float(g) for g in labels.values()), reverse=True)[: max(k, 0)]
    return dcg_of_gains(ideal_gains)


def ndcg_at(ranking, labels, k):
    ideal = idcg_at(labels, k)
    if ideal == 0.0:
        return 0.0  # 零相关查询：0/0 定义为 0
    return dcg_at(ranking, labels, k) / ideal


def total_relevant(labels):
    return sum(1 for g in labels.values() if float(g) > 0.0)


def _ap_numerator_at(ranking, labels, k):
    """AP 的分子：top-k 中相关位置 i 上的精度 P@i 之和（位置升序累加）。"""
    hits = 0
    precision_sum = 0.0
    for index, (doc_id, _score) in enumerate(_ordered_pairs(ranking, k)):
        if float(labels.get(doc_id, 0.0)) > 0.0:
            hits += 1
            precision_sum += hits / (index + 1.0)
    return precision_sum


def average_precision_at(ranking, labels, k):
    """AP@k = sum(P@i * [doc i 相关], i<=k) / R，R 为该查询相关文档总数。

    分母固定为 R 而非 min(R, k)：这样在 k 增大（只多看到无关文档）时指标不会下降。
    """
    relevant = total_relevant(labels)
    if relevant == 0:
        return 0.0
    return _ap_numerator_at(ranking, labels, k) / relevant


def reciprocal_rank_at(ranking, labels, k):
    for index, (doc_id, _score) in enumerate(_ordered_pairs(ranking, k)):
        if float(labels.get(doc_id, 0.0)) > 0.0:
            return 1.0 / (index + 1.0)
    return 0.0


def hit_rate_at(ranking, labels, k):
    for doc_id, _score in _ordered_pairs(ranking, k):
        if float(labels.get(doc_id, 0.0)) > 0.0:
            return 1.0
    return 0.0


# ---------- 汇总 ----------

def evaluate_query(ranking, labels, k, qid=None):
    """计算单查询在截断 k 下的全部指标及微平均所需的中间量。"""
    relevant = total_relevant(labels)
    return {
        "qid": qid,
        "ndcg": ndcg_at(ranking, labels, k),
        "ap": average_precision_at(ranking, labels, k),
        "mrr": reciprocal_rank_at(ranking, labels, k),
        "hit": hit_rate_at(ranking, labels, k),
        "dcg": dcg_at(ranking, labels, k),
        "idcg": idcg_at(labels, k),
        "ap_numerator": _ap_numerator_at(ranking, labels, k),
        "relevant": relevant,
        "zero_relevance": relevant == 0,
    }


def evaluate(dataset, k):
    """按查询集合汇总。

    dataset: [{"qid": ..., "ranking": [(doc_id, score), ...], "labels": {doc_id: rel}}, ...]

    返回 per_query（逐条）、macro（按查询等权平均）、micro（按规模加权）。
      - micro nDCG = sum(DCG) / sum(IDCG)
      - micro AP   = sum(AP 分子) / sum(R)
      - MRR / HitRate 每个查询本身只有一个 0/1（或倒数值），微平均与宏平均定义相同。
    """
    per_query = [
        evaluate_query(item["ranking"], item["labels"], k, item.get("qid"))
        for item in dataset
    ]
    count = len(per_query)
    macro = {
        metric: (sum(item[metric] for item in per_query) / count if count else 0.0)
        for metric in METRICS
    }
    total_relevant_docs = sum(item["relevant"] for item in per_query)
    total_ideal = sum(item["idcg"] for item in per_query)
    micro = {
        "ndcg": (
            sum(item["dcg"] for item in per_query) / total_ideal
            if total_ideal > 0.0
            else 0.0
        ),
        "ap": (
            sum(item["ap_numerator"] for item in per_query) / total_relevant_docs
            if total_relevant_docs > 0
            else 0.0
        ),
        "mrr": macro["mrr"],
        "hit": macro["hit"],
    }
    return {
        "k": k,
        "num_queries": count,
        "num_zero_relevance": sum(1 for item in per_query if item["zero_relevance"]),
        "per_query": per_query,
        "macro": macro,
        "micro": micro,
    }


def explain_aggregation(result, top_n=3):
    """归因说明：宏平均（每查询等权 1/Q）与微平均（按规模加权）为何不一致。"""
    per_query = result["per_query"]
    count = len(per_query)
    if count == 0:
        return "无查询，宏/微平均均未定义（按 0 处理）。"
    total_rel = sum(item["relevant"] for item in per_query)
    total_ideal = sum(item["idcg"] for item in per_query)

    lines = []
    weight_specs = (
        ("ap", "相关文档数 R", lambda item: item["relevant"], total_rel),
        ("ndcg", "理想折损增益 IDCG", lambda item: item["idcg"], total_ideal),
    )
    for metric, weight_name, weight_of, weight_total in weight_specs:
        macro_value = result["macro"][metric]
        micro_value = result["micro"][metric]
        gap = micro_value - macro_value
        if abs(gap) < 1e-12:
            lines.append(f"{metric}: 宏平均={macro_value:.6f}，微平均={micro_value:.6f}，一致。")
            continue
        rows = []
        for item in per_query:
            macro_weight = 1.0 / count
            micro_weight = weight_of(item) / weight_total if weight_total > 0 else 0.0
            rows.append(
                (abs(micro_weight - macro_weight), item["qid"],
                 macro_weight, micro_weight, item[metric])
            )
        rows.sort(reverse=True)
        lines.append(
            f"{metric}: 宏平均={macro_value:.6f}，微平均={micro_value:.6f}，"
            f"差={gap:+.6f}。宏平均给每查询 1/{count} 的等权；微平均按{weight_name}加权，"
            f"规模大的查询占比高。权重差异最大的查询："
        )
        for _, qid, macro_weight, micro_weight, value in rows[:top_n]:
            lines.append(
                f"  qid={qid!r}: 宏权重={macro_weight:.4f} -> 微权重={micro_weight:.4f}，"
                f"单查询 {metric}={value:.6f}"
            )
    lines.append(
        "mrr / hit: 每查询只贡献一个值（倒数排名或 0/1），微平均与宏平均恒等，"
        "不存在两种口径。"
    )
    lines.append(
        f"零相关查询 {result['num_zero_relevance']} 个：以 0 值计入两种口径；"
        "它只在 AP 的微分母（R）中不贡献，不参与 nDCG 微分母（IDCG=0）。"
    )
    return "\n".join(lines)


# ---------- 自助法置信区间 ----------

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=20260101):
    """宏平均的百分位自助法置信区间。

    以查询为重采样单位，有放回抽取 Q 个查询 -> 计算均值 -> 取 alpha/2 与
    1-alpha/2 分位点。随机数用固定种子，结果可复现。

    查询很少时的稳健性处理：
      - n < 30：返回 warning，区间覆盖率可能明显低于标称值，只能作参考下界，
        报告时必须同时给出 n；优先增加查询数。
      - 所有查询取值相同：区间退化为单点（degenerate=True），这不是“估计很精确”，
        而是自助法在该数据上无法估计波动，应改用跨查询/时间切片的数据或直接报告
        点估计与样本数。
      - n 很小（如 < 8）时自助法重采样组合数极少，分位点本身不稳定；
        建议同时看全部逐查询值而不是只报区间。
    """
    values = list(values)
    n = len(values)
    info = {"n": n, "n_boot": n_boot, "alpha": alpha, "degenerate": False, "warning": None}
    if n == 0:
        info["warning"] = "无查询，置信区间未定义。"
        return 0.0, 0.0, info
    point = sum(values) / n
    if len(set(values)) == 1:
        info["degenerate"] = True
        info["warning"] = (
            "所有查询取值相同，自助区间退化为单点；这不表示无方差，"
            "应扩大数据来源后再估计不确定性。"
        )
        return point, point, info
    if n < 30:
        info["warning"] = (
            f"查询数 n={n} 偏少（<30），百分位自助法覆盖率可能不足，"
            "区间只能作参考；应同时报告逐查询分布与 n，并优先补充查询。"
        )
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int((alpha / 2.0) * n_boot)]
    high = means[min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))]
    return low, high, info


def bootstrap_ci_micro(pairs, n_boot=2000, alpha=0.05, seed=20260101):
    """微平均的自助法区间：重采样查询后再汇总 sum(numerator)/sum(denominator)。

    pairs: [(numerator, denominator), ...]，例如 AP 的 (ap_numerator, R)
    或 nDCG 的 (dcg, idcg)。denominator 总和为 0 时该次重采样记为 0.0。
    """
    pairs = list(pairs)
    n = len(pairs)
    info = {"n": n, "n_boot": n_boot, "alpha": alpha, "degenerate": False, "warning": None}
    if n == 0:
        info["warning"] = "无查询，置信区间未定义。"
        return 0.0, 0.0, info
    if n < 30:
        info["warning"] = (
            f"查询数 n={n} 偏少（<30），微平均自助区间同样可能偏窄，需谨慎解读。"
        )
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        num_total = 0.0
        den_total = 0.0
        for _ in range(n):
            num, den = pairs[rng.randrange(n)]
            num_total += num
            den_total += den
        stats.append(num_total / den_total if den_total > 0.0 else 0.0)
    if len(set(stats)) == 1:
        info["degenerate"] = True
        info["warning"] = (info["warning"] or "") + " 自助统计量全部相同，区间退化为单点。"
        return stats[0], stats[0], info
    stats.sort()
    low = stats[int((alpha / 2.0) * n_boot)]
    high = stats[min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))]
    return low, high, info
