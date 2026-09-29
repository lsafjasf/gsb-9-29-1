# -*- coding: utf-8 -*-
"""评估脚本：变体检出率 + 误伤率 + 性能，输出 report/metrics.json。

用法: python3 run_eval.py
"""

import json
import time
from pathlib import Path

from sensitive_filter import Matcher

ROOT = Path(__file__).resolve().parent
ATTACK = ROOT / "data" / "attack_corpus.json"
TRICKY = ROOT / "data" / "normal_tricky.txt"
REPORT = ROOT / "report" / "metrics.json"


def build_normal_corpus():
    """确定性生成正常语料：棘手样例 + 模板批量句（中/英/混排）。"""
    sentences = [l.strip() for l in TRICKY.read_text(encoding="utf-8").splitlines() if l.strip()]

    subjects = ["我", "我们", "他", "她", "老师", "同事", "孩子", "妈妈", "邻居老王", "朋友"]
    times = ["昨天", "今天早上", "上周", "周末", "刚才", "去年", "下班后", "假期里"]
    places = ["在公园", "在图书馆", "在公司", "在学校", "在超市", "在地铁站", "在家里", "在咖啡馆"]
    actions = ["看书", "散步", "开会", "买菜", "锻炼身体", "写作业", "喝咖啡", "聊天", "做饭", "听音乐"]
    feels = ["很开心", "很充实", "特别放松", "收获很大", "感觉不错"]
    i = 0
    for s in subjects:
        for t in times:
            for p in places:
                a = actions[i % len(actions)]
                f = feels[i % len(feels)]
                sentences.append(f"{s}{t}{p}{a}，{f}。")
                i += 1

    en_subj = ["I", "We", "They", "My friend", "The team"]
    en_verb = ["went to", "stayed at", "visited", "walked around"]
    en_obj = ["the park", "school", "the museum", "downtown", "the library"]
    for s in en_subj:
        for v in en_verb:
            for o in en_obj:
                sentences.append(f"{s} {v} {o} yesterday.")

    apps = ["晨曦笔记", "山丘阅读", "白噪音助手", "番茄时钟", "每日英语"]
    for app in apps:
        for who in ["我", "同事", "朋友"]:
            sentences.append(f"{who}最近在用{app}这个App，体验不错。")
    return sentences


def main():
    matcher = Matcher.from_files(ROOT / "config" / "rules.json", ROOT / "config" / "maps.json")

    # ---- 1. 变体检出率 ----
    attack = json.loads(ATTACK.read_text(encoding="utf-8"))
    by_cat, missed = {}, []
    for item in attack:
        res = matcher.scan(item["text"])
        got = {h.rule_id for h in res.hits}
        cat = by_cat.setdefault(item["category"], [0, 0])
        cat[1] += 1
        if set(item["expect"]) <= got:
            cat[0] += 1
        else:
            missed.append({"text": item["text"], "expect": item["expect"], "got": sorted(got)})
    detected = sum(v[0] for v in by_cat.values())
    attack_report = {
        "total": len(attack),
        "detected": detected,
        "recall": round(detected / len(attack), 4),
        "by_category": {k: {"detected": v[0], "total": v[1]} for k, v in sorted(by_cat.items())},
        "missed": missed,
    }

    # ---- 2. 误伤率 ----
    normal = build_normal_corpus()
    fp, suppressed = [], []
    for text in normal:
        res = matcher.scan(text)
        for h in res.hits:
            fp.append({"text": text, "rule_id": h.rule_id, "variant": h.variant,
                       "confidence": h.confidence, "snippet": h.snippet})
        for h in res.suppressed:
            suppressed.append({"text": text, "rule_id": h.rule_id, "variant": h.variant,
                               "reason": h.suppress_reason})
    normal_report = {
        "sentences": len(normal),
        "false_positives": len(fp),
        "fp_rate": round(len(fp) / len(normal), 6),
        "fp_samples": fp[:20],
        "whitelist_suppressed": len(suppressed),
        "suppressed_samples": suppressed,
    }

    # ---- 3. 超长文本性能 ----
    long_text = "。".join(normal)
    while len(long_text) < 1_000_000:
        long_text += "。" + long_text
    t0 = time.perf_counter()
    res = matcher.scan(long_text)
    elapsed = time.perf_counter() - t0
    perf = {
        "chars": len(long_text),
        "seconds": round(elapsed, 3),
        "chars_per_sec": int(len(long_text) / elapsed),
        "hits_in_long_text": len(res.hits),
    }

    metrics = {
        "patterns_in_automaton": len(matcher._automaton.patterns),
        "attack": attack_report,
        "normal": normal_report,
        "performance": perf,
    }
    REPORT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 变体检出 ===")
    print(f"检出 {detected}/{len(attack)}  召回 {attack_report['recall']:.1%}")
    for cat, v in attack_report["by_category"].items():
        print(f"  {cat:<16} {v['detected']}/{v['total']}")
    print("=== 误伤 ===")
    print(f"正常语料 {len(normal)} 句，误伤 {len(fp)} 句，误伤率 {normal_report['fp_rate']:.4%}")
    print(f"白名单拦截（潜在误伤样例）{len(suppressed)} 条")
    print("=== 性能 ===")
    print(f"{perf['chars']:,} 字符扫描 {perf['seconds']}s（{perf['chars_per_sec']:,} 字符/秒）")
    print(f"\n报告已写入 {REPORT}")


if __name__ == "__main__":
    main()
