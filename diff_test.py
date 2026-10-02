"""对拍脚本：在随机数据上逐位比较 ranking_metrics 与 reference_metrics。

随机数据刻意覆盖：
  - 大量并列得分（score 只取 0..3）与随机输入顺序（验证排序确定性）；
  - 缺失标注（约 30% 排序文档无标注）；
  - 排序外的标注文档（影响 IDCG 与 AP 分母 R）；
  - 空排序、零相关查询、全相关查询。

比较口径：浮点结果必须 ==（逐位一致），而非近似相等。
"""

import random

import ranking_metrics as lib
import reference_metrics as ref

SINGLE_METRICS = ("ndcg", "ap", "mrr", "hit", "dcg", "idcg", "ap_numerator", "relevant")
AGG_METRICS = ("ndcg", "ap", "mrr", "hit")


def random_dataset(rng):
    dataset = []
    num_queries = rng.randint(1, 8)
    for qi in range(num_queries):
        kind = rng.random()
        if kind < 0.12:
            ranking = []  # 空排序
            docs = []
        elif kind < 0.2:
            n_docs = 1  # 只有一条结果
            docs = [f"d{qi}_0"]
            ranking = list(zip(docs, [rng.randint(0, 3)]))
        else:
            n_docs = rng.randint(2, 14)
            docs = [f"d{qi}_{j}" for j in range(n_docs)]
            ranking = list(zip(docs, (rng.randint(0, 3) for _ in docs)))
        rng.shuffle(ranking)  # 输入顺序随机，结果必须不变

        if kind < 0.28 and kind >= 0.2:
            # 零相关查询：所有已标注文档相关性为 0（也允许完全无标注）
            labels = {d: 0 for d in docs if rng.random() < 0.6}
        elif kind < 0.36 and kind >= 0.28:
            # 全相关
            labels = {d: rng.choice([1, 2, 3]) for d in docs}
        else:
            labels = {}
            for doc_id in docs:
                if rng.random() < 0.7:  # 约 30% 缺失标注
                    labels[doc_id] = rng.choice([0, 0, 0, 1, 1, 2, 3])

        # 排序外的标注文档：不出现在 ranking 中，但有相关性
        if rng.random() < 0.4:
            labels[f"ghost_{qi}"] = rng.randint(1, 3)
            if rng.random() < 0.3:
                labels[f"ghost_{qi}b"] = rng.randint(1, 3)

        dataset.append({"qid": f"q{qi}", "ranking": ranking, "labels": labels})
    return dataset


def assert_bitwise_equal(a, b, where):
    if a != b:
        raise AssertionError(f"逐位不一致 @ {where}: library={a!r} reference={b!r}")
    if isinstance(a, float) and (a != a or b != b):
        raise AssertionError(f"出现 NaN @ {where}")


def run(iterations=2000, seed_base=1000):
    for trial in range(iterations):
        rng = random.Random(seed_base + trial)
        dataset = random_dataset(rng)
        max_len = max((len(item["ranking"]) for item in dataset), default=0)
        ks = [0, 1] + [rng.randint(1, max(2, max_len + 2)) for _ in range(3)]

        # 输入顺序扰动：再次打乱每个 ranking，结果必须完全相同（验证并列确定性）
        shuffled = []
        for item in dataset:
            ranking = list(item["ranking"])
            rng.shuffle(ranking)
            shuffled.append({**item, "ranking": ranking})

        for k in sorted(set(ks)):
            got = lib.evaluate(dataset, k)
            want = ref.evaluate(dataset, k)
            got_shuf = lib.evaluate(shuffled, k)

            for idx, item in enumerate(dataset):
                a, b = got["per_query"][idx], want["per_query"][idx]
                c = got_shuf["per_query"][idx]
                for metric in SINGLE_METRICS:
                    assert_bitwise_equal(a[metric], b[metric], f"trial={trial} q={item['qid']} k={k} {metric}")
                    assert_bitwise_equal(a[metric], c[metric], f"trial={trial} q={item['qid']} k={k} shuffled {metric}")

            for metric in AGG_METRICS:
                assert_bitwise_equal(got["macro"][metric], want["macro"][metric],
                                     f"trial={trial} k={k} macro {metric}")
                assert_bitwise_equal(got["micro"][metric], want["micro"][metric],
                                     f"trial={trial} k={k} micro {metric}")
    print(f"对拍通过：{iterations} 个随机数据集 x 多个截断点，逐位一致。")


if __name__ == "__main__":
    run()
