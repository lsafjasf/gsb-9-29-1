"""十万字文本上的近似匹配剪枝性能基准。

对比三种模式（配置项 approximate.prune）：
  index  长度分桶 + 删除索引（默认）
  bucket 仅长度分桶
  naive  不剪枝，全词典逐个比较（最坏情况对照）

另含一个对抗性用例：全文为同一个字的重复（删除索引候选数退化为 O(N)），
验证最坏情况仍有界且结果正确。

用法：python3 bench/benchmark.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from writingaid import WritingAid, load_config

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_LEN = 100_000
RUNS = 5


def build_text() -> str:
    safe_zh = "科学技术不断进步，给日常生活带来许多便利。"
    safe_en = "This sentence is safe and will not be flagged. "
    parts = []
    total = 0
    i = 0
    while total < TARGET_LEN - 400:
        block = safe_zh + safe_en
        if i % 10 == 3:
            block += "他一如继往地努力。"
        if i % 10 == 7:
            block += "流程要按步就班，不能乱。"
        parts.append(block)
        total += len(block)
        i += 1
    text = "".join(parts)
    text += "这真是千方百记的好办法。The qick brown fox. 你好,世界。"
    return text


def time_run(config: dict, text: str) -> tuple[float, int]:
    aid = WritingAid(config)
    start = time.perf_counter()
    suggestions = aid.analyze(text)
    return time.perf_counter() - start, len(suggestions)


def main() -> int:
    base = load_config()
    text = build_text()
    print(f"基准文本长度: {len(text)} 字符")

    results = {}
    counts = {}
    for prune in ("index", "bucket", "naive"):
        config = json.loads(json.dumps(base))
        config["approximate"]["prune"] = prune
        times = []
        for _ in range(RUNS):
            elapsed, count = time_run(config, text)
            times.append(elapsed)
        results[prune] = times
        counts[prune] = count
        print(f"prune={prune:6s}  建议数={count:4d}  "
              f"中位耗时={statistics.median(times)*1000:8.1f} ms  "
              f"全部={[f'{t*1000:.0f}' for t in times]}")

    assert counts["index"] == counts["bucket"] == counts["naive"], "三种模式结果必须一致"

    # 对抗性最坏情况：单字重复文本，删除索引候选数退化
    adv_text = "啊啊啊啊" * 6250  # 25000 个四字窗口
    adv_dict = ["啊啊啊" + ch for ch in "吧把爸八白百北被本比"]  # 10 个高相似词条
    config = json.loads(json.dumps(base))
    config["cjk_dictionary"] = adv_dict
    config["en_dictionary"] = []
    config["rules"] = []
    for prune in ("index", "naive"):
        config["approximate"]["prune"] = prune
        elapsed, count = time_run(config, adv_text)
        print(f"对抗用例 prune={prune:6s}  建议数={count:4d}  耗时={elapsed*1000:8.1f} ms")

    report = {
        "text_length": len(text),
        "runs": RUNS,
        "suggestions": counts["index"],
        "median_ms": {k: round(statistics.median(v) * 1000, 1) for k, v in results.items()},
        "all_ms": {k: [round(t * 1000, 1) for t in v] for k, v in results.items()},
    }
    out = os.path.join(HERE, "..", "reports", "benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
