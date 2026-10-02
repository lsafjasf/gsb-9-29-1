"""基准与对比数据生成脚本（仅标准库）。

运行：python3 benchmark.py
产出：
  1. 摘要样例 + 每句得分构成
  2. 贪心(MMR) vs 动态规划(背包) 策略差异数据
  3. 与前几句 / 等间隔 / 随机三种基线的覆盖度、冗余度对比
  4. 冗余惩罚强度扫描
  5. 超长文档：分块 vs 整体的一致性断言 + 内存与耗时
  6. 确定性断言
"""

import random
import time
import tracemalloc

from summarizer import summarize, _cosine_sets

# ---------------------------------------------------------------- 样例文档
TITLE = "太阳能发电的发展与挑战"
ARTICLE = (
    "太阳能是一种清洁的可再生能源，近年来受到各国重视。"
    "过去十年间，光伏发电的成本下降了超过八成。"
    "光伏成本下降主要得益于电池板效率提升和规模化生产。"
    "中国已经成为全球最大的太阳能组件生产国和装机市场。"
    "然而太阳能发电具有间歇性，阴天和夜晚无法稳定供电。"
    "储能技术因此成为可再生能源大规模并网的关键。"
    "锂电池储能的成本也在快速下降，但安全性仍需改进。"
    "一些地区出现了弃光现象，电网消纳能力跟不上装机速度。"
    "特高压输电线路可以把西部沙漠的电力送往东部负荷中心。"
    "分布式光伏让普通家庭也能在屋顶安装太阳能板。"
    "政府补贴退坡后，行业正逐步走向平价上网。"
    "研究人员正在开发钙钛矿电池，实验室效率已超过百分之二十五。"
    "钙钛矿电池的稳定性问题尚未完全解决，商业化仍需时间。"
    "国际能源署预测，太阳能将在未来十年成为最大的新增电源。"
    "但供应链集中和原材料价格波动仍是行业面临的风险。"
    "总体来看，太阳能发电前景广阔，储能与电网建设必须同步推进。"
)


def make_section_doc():
    """三段式文档：三个主题各占一段且词表基本不相交，前几句无法覆盖结论段。"""
    secs = [["光伏", "硅片", "组件", "屋顶", "沙漠", "逆变器"],
            ["锂电池", "抽水蓄能", "调峰", "电站", "容量", "充放电"],
            ["电解槽", "绿氢", "储运", "加氢站", "催化剂", "重卡"]]
    templates = ["{a}与{b}取得关键突破", "{a}推动{b}快速普及",
                 "{b}带动{a}成本下探", "{a}和{b}进入规模化阶段",
                 "{b}为{a}打开新市场"]
    sents = []
    for kws in secs:
        for j in range(20):
            a = kws[j % len(kws)]
            b = kws[(2 * j + 1) % len(kws)]
            if a == b:
                b = kws[(2 * j + 2) % len(kws)]
            sents.append(templates[j % len(templates)].format(a=a, b=b) + "。")
    return "".join(sents)


def make_duplicate_doc():
    """含高词频近重复句 + 若干不重复句的文档，用于惩罚强度扫描。"""
    dups = [
        "太阳能发电成本持续下降，技术进步十分显著。",
        "太阳能发电成本不断下降，技术进步非常明显。",
        "太阳能发电的成本持续走低，电池技术进步显著。",
        "太阳能发电成本明显下降，组件技术持续进步。",
    ]
    others = [
        "风电在北方草原地区增长迅速。",
        "电网调度需要应对新能源的波动。",
        "绿氢制取依赖廉价的可再生电力。",
        "分布式光伏正在改变居民用电方式。",
        "碳市场为减排项目提供经济激励。",
        "储能电站可以提供调峰与备用容量。",
    ]
    return "".join(dups + others)


def make_long_doc(n_sent, seed=7):
    """生成含主题关键词与近重复句的超长文档。"""
    rng = random.Random(seed)
    topics = ["太阳能", "储能", "电网", "碳中和", "风电", "氢能"]
    verbs = ["持续推进", "成本下降", "效率提升", "规模扩大", "政策加码"]
    sents = []
    for i in range(n_sent):
        t = topics[i % len(topics)]
        v = rng.choice(verbs)
        extra = " ".join(rng.choice(topics) for _ in range(rng.randint(0, 4)))
        sents.append(f"{t}领域{v}，{extra}相关指标编号{i}。")
        if i % 50 == 49:  # 周期性插入近重复句，制造冗余
            sents.append(f"{t}领域{v}，{extra}相关指标编号{i}，再次确认。")
    return "".join(sents)


# ---------------------------------------------------------------- 基线
def pick_by_order(order, scores, budget):
    chosen, used = [], 0
    for i in order:
        sc = scores[i]
        if used + sc.n_words <= budget:
            chosen.append(sc)
            used += sc.n_words
    chosen.sort(key=lambda s: s.index)
    return chosen


def baseline_lead(scores, budget):
    return pick_by_order(range(len(scores)), scores, budget)


def baseline_spaced(scores, budget):
    n = len(scores)
    k = max(1, min(n, budget // max(1, min(s.n_words for s in scores))))
    step = n / k
    order = [int(i * step) for i in range(k)]
    return pick_by_order(order, scores, budget)


def baseline_random(scores, budget, seed=0):
    order = list(range(len(scores)))
    random.Random(seed).shuffle(order)
    return pick_by_order(order, scores, budget)


def metrics(chosen, keywords):
    if keywords:
        have = set().union(*(s.words for s in chosen)) if chosen else set()
        cov = len(set(keywords) & have) / len(keywords)
    else:
        cov = 1.0
    sims = [_cosine_sets(a.words, b.words)
            for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    red = sum(sims) / len(sims) if sims else 0.0
    return cov, red


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- 1. 摘要样例
def demo_sample():
    hr("1. 摘要样例（标题：%s，预算 60 词，greedy，penalty=0.5）" % TITLE)
    r = summarize(ARTICLE, title=TITLE, budget=60)
    print("【摘要】", r.text)
    print("【句序】", r.indices, "【用词】%d/%d 词" % (r.used, r.budget))
    print("\n每句得分构成（freq=词频 pos=位置 title=标题相似度 total=加权总分）：")
    print("%4s %6s %6s %6s %6s %4s  %s" %
          ("句号", "freq", "pos", "title", "total", "选中", "句子"))
    for s in r.scores:
        print("%4d %6.3f %6.3f %6.3f %6.3f %4s  %s" %
              (s.index, s.freq, s.position, s.title, s.total,
               "√" if s.selected else "", s.text[:34]))
    print("覆盖度 %.3f 冗余度 %.3f" % (r.coverage, r.redundancy))
    return r


# ---------------------------------------------------------------- 2. 策略差异
def demo_strategies():
    hr("2. 贪心(MMR) vs 动态规划(背包) 策略差异（同一文档，不同预算）")
    print("%6s %8s %10s %8s %8s %8s  %s" %
          ("预算", "策略", "总分", "覆盖度", "冗余度", "用词", "选中句号"))
    for budget in (40, 60, 90):
        for strategy in ("greedy", "dp"):
            r = summarize(ARTICLE, title=TITLE, budget=budget, strategy=strategy)
            total = sum(s.total for s in r.selected)
            print("%6d %8s %10.3f %8.3f %8.3f %8d  %s" %
                  (budget, strategy, total, r.coverage, r.redundancy,
                   r.used, r.indices))
    print("说明：DP 最大化总分之和但不含成对冗余项 -> 总分更高、冗余可能更高；")
    print("      MMR 贪心显式惩罚重复 -> 冗余更低、覆盖通常更好。")


# ---------------------------------------------------------------- 3. 基线对比
def demo_baselines():
    hr("3. 与前几句 / 等间隔 / 随机基线的覆盖度、冗余度对比（预算 60 词）")
    doc = make_section_doc()
    ref = summarize(doc, title=TITLE, budget=60)
    scores, keywords = ref.scores, ref.keywords
    rows = []
    for name, chosen in [
        ("本文-greedy", ref.selected),
        ("本文-dp", summarize(doc, title=TITLE, budget=60,
                             strategy="dp").selected),
        ("前几句", baseline_lead(scores, 60)),
        ("等间隔", baseline_spaced(scores, 60)),
        ("随机", baseline_random(scores, 60)),
    ]:
        cov, red = metrics(chosen, keywords)
        rows.append((name, cov, red, sum(s.n_words for s in chosen)))
    print("文档：60 句三段式（太阳能 20 句 / 储能 20 句 / 氢能 20 句）")
    print("%14s %10s %10s %8s" % ("方法", "覆盖度", "冗余度", "用词"))
    for name, cov, red, used in rows:
        print("%14s %10.3f %10.3f %8d" % (name, cov, red, used))
    print("覆盖度 = 摘要命中的文档关键词比例（关键词取文档频率 top-20）")
    print("冗余度 = 摘要句两两余弦相似度均值（越低越好）")


# ---------------------------------------------------------------- 4. 惩罚强度扫描
def demo_penalty_sweep():
    hr("4. 冗余惩罚强度扫描（penalty 0.0 -> 1.0，预算 60 词）")
    doc = make_duplicate_doc()
    print("文档：4 句高词频近重复句 + 6 句不重复句（预算 4 句）")
    print("%8s %10s %10s  %s" % ("penalty", "覆盖度", "冗余度", "选中句号"))
    for pen in (0.0, 0.3, 0.5, 0.7, 1.0):
        r = summarize(doc, title=TITLE, budget=4, budget_unit="sentences",
                      redundancy_penalty=pen)
        print("%8.1f %10.3f %10.3f  %s" % (pen, r.coverage, r.redundancy, r.indices))


# ---------------------------------------------------------------- 5. 超长文档
def demo_scaling():
    hr("5. 超长文档：分块 vs 整体（一致性断言 + 内存/耗时）")
    print("%8s %9s %10s %10s %10s %6s" %
          ("句数", "整体(s)", "分块(s)", "整体峰值", "分块+裁剪", "一致"))
    for n in (2000, 8000, 20000):
        doc = make_long_doc(n)

        t0 = time.perf_counter()
        tracemalloc.start()
        r_whole = summarize(doc, title="能源", budget=100)
        peak_whole = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        t_whole = time.perf_counter() - t0

        t0 = time.perf_counter()
        tracemalloc.start()
        r_chunk = summarize(doc, title="能源", budget=100, chunk_size=65536)
        tracemalloc.stop()
        t_chunk = time.perf_counter() - t0

        tracemalloc.start()
        r_lim = summarize(doc, title="能源", budget=100, chunk_size=65536,
                          candidate_limit=200)
        peak_lim = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        same = r_whole.sentences == r_chunk.sentences
        assert same, f"分块结果与整体不一致 (n={n})"
        print("%8d %9.3f %10.3f %8.1fMB %10.1fMB %6s" %
              (len(r_whole.scores), t_whole, t_chunk,
               peak_whole / 1e6, peak_lim / 1e6, "✓" if same else "✗"))

    print("\n说明：分块（chunk_size=64KB）做两遍流式扫描，结果与整体处理逐句一致；")
    print("      再叠加 candidate_limit=200 后，只保留 top-200 候选的词集合，")
    print("      选择阶段内存从随文档线性增长变为有界（裁剪结果为近似，非裁剪为精确）。")
    print("      20000 句裁剪摘要：", r_lim.text[:80], "...")


# ---------------------------------------------------------------- 6. 确定性
def demo_determinism():
    hr("6. 确定性断言（同一输入运行两次，结果逐字节一致）")
    doc = make_long_doc(500)
    r1 = summarize(doc, title="能源", budget=80)
    r2 = summarize(doc, title="能源", budget=80)
    assert r1.text == r2.text and r1.indices == r2.indices
    assert [s.total for s in r1.scores] == [s.total for s in r2.scores]
    print("通过：摘要文本、句序、每句得分完全一致。")


if __name__ == "__main__":
    demo_sample()
    demo_strategies()
    demo_baselines()
    demo_penalty_sweep()
    demo_scaling()
    demo_determinism()
