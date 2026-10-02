"""同一目标函数下对比三种调度器，并产出诊断数据。

运行：python3 compare_schedulers.py
输出：控制台表格 + data/comparison.md + data/comparison.json + data/diagnostics.md
"""

import json
import os

from lrdiag import (
    CosineDecay,
    GDTrainer,
    LossDiagnoser,
    PlateauDecay,
    Quadratic,
    StepDecay,
)
from lrdiag.trainer import tail_stability

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SCENARIOS = {
    "A_正常收敛": dict(
        objective=lambda: Quadratic(axes=(1.0, 0.1), seed=7),
        base_lr=0.05, max_epochs=2000, tol=1e-6,
        step=dict(step_size=200, gamma=0.5),
        cosine=dict(total_epochs=2000),
        plateau=dict(patience=30, factor=0.5, threshold=1e-4),
    ),
    "B_一开始就发散": dict(
        objective=lambda: Quadratic(axes=(1.0, 0.1), seed=7),
        base_lr=3.0, max_epochs=3000, tol=1e-6,
        step=dict(step_size=1000, gamma=0.5),
        cosine=dict(total_epochs=3000),
        plateau=dict(patience=30, factor=0.5, threshold=1e-4),
    ),
    "C_噪声极大": dict(
        objective=lambda: Quadratic(axes=(1.0, 1.0), noise_std=0.5, seed=11),
        base_lr=0.05, max_epochs=600, tol=1e-8,
        step=dict(step_size=150, gamma=0.5),
        cosine=dict(total_epochs=600),
        plateau=dict(patience=50, factor=0.5, threshold=1e-3),
    ),
    "D_长时间不下降": dict(
        objective=lambda: Quadratic(axes=(1.0, 1.0), seed=1),
        base_lr=1e-6, max_epochs=300, tol=1e-12,
        step=dict(step_size=100, gamma=0.5),
        cosine=dict(total_epochs=300),
        plateau=dict(patience=10, factor=0.5, threshold=1e-4),
    ),
}

MIN_LR = 1e-8


def make_scheduler(kind, base_lr, params):
    if kind == "step":
        return StepDecay(base_lr, min_lr=MIN_LR, **params)
    if kind == "cosine":
        return CosineDecay(base_lr, min_lr=MIN_LR, **params)
    return PlateauDecay(base_lr, min_lr=MIN_LR, **params)


def run_scenario(name, cfg):
    rows = {}
    for kind in ("step", "cosine", "plateau"):
        objective = cfg["objective"]()
        sched = make_scheduler(kind, cfg["base_lr"], cfg[kind])
        trainer = GDTrainer(objective, sched, tol=cfg["tol"])
        result = trainer.run(max_epochs=cfg["max_epochs"])
        stab = tail_stability(result, tail=50)
        rows[kind] = dict(
            converged=result.converged,
            unstable=result.unstable,
            epochs=result.epochs,
            final_true_loss=result.final_true_loss,
            best_true_loss=result.best_true_loss,
            final_lr=result.final_lr,
            rollbacks=result.rollback_count,
            plateau_events=result.plateau_count,
            oscillating_events=result.oscillating_count,
            tail_mean=stab["tail_mean"],
            tail_std=stab["tail_std"],
            tail_range=stab["tail_range"],
            rollback_events=[
                dict(epoch=e.epoch, pre_lr=e.pre_lr, post_lr=e.post_lr,
                     bad_loss=e.bad_loss, best_true_loss=e.best_true_loss,
                     reason=e.reason)
                for e in result.rollback_events
            ],
        )
    return rows


def fmt(x, digits=3):
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, int):
        return str(x)
    if x != x:  # NaN
        return "nan"
    if abs(x) >= 1e4 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{digits}g}"


def render_table(all_rows):
    header = (
        "| 场景 | 调度器 | 收敛 | 收敛轮数 | 最终真实损失 | 历史最优 | "
        "回滚次数 | 平台期事件 | 震荡事件 | 末态步长 | 尾部均值 | 尾部标准差 | 尾部极差 |"
    )
    sep = "|" + "---|" * 13
    lines = [header, sep]
    names = {"step": "阶梯衰减", "cosine": "余弦衰减", "plateau": "按平台期衰减"}
    for scenario, rows in all_rows.items():
        for kind, r in rows.items():
            lines.append(
                "| {sc} | {k} | {c} | {e} | {f} | {b} | {rb} | {pl} | {osc} | "
                "{lr} | {tm} | {ts} | {tr} |".format(
                    sc=scenario, k=names[kind], c=fmt(r["converged"]),
                    e=r["epochs"], f=fmt(r["final_true_loss"]),
                    b=fmt(r["best_true_loss"]), rb=r["rollbacks"],
                    pl=r["plateau_events"], osc=r["oscillating_events"],
                    lr=fmt(r["final_lr"]), tm=fmt(r["tail_mean"]),
                    ts=fmt(r["tail_std"]), tr=fmt(r["tail_range"]),
                )
            )
    return "\n".join(lines)


def diagnosis_probes():
    """用典型合成曲线演示诊断器的判定依据与建议动作。"""
    probes = {}
    cases = {
        "发散曲线(指数爆炸)": [1.2 ** i for i in range(25)],
        "平台期曲线(几乎不动)": [5.0 + 1e-6 * i for i in range(25)],
        "震荡曲线(来回跳动)": [1.0, 2.0] * 13,
        "健康下降曲线": [10.0 * 0.9 ** i for i in range(25)],
        "噪声极大曲线": None,  # 由随机噪声生成
    }
    import random
    rng = random.Random(42)
    cases["噪声极大曲线"] = [0.02 + rng.gauss(0, 0.5) for _ in range(25)]
    for name, seq in cases.items():
        d = LossDiagnoser(window=20)
        diag = None
        for v in seq:
            diag = d.update(v)
        probes[name] = dict(status=diag.status, reason=diag.reason,
                            action=diag.action)
    return probes


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows = {}
    for name, cfg in SCENARIOS.items():
        all_rows[name] = run_scenario(name, cfg)

    table = render_table(all_rows)
    print(table)

    probes = diagnosis_probes()

    md = ["# 调度对比与诊断数据", "",
          "目标函数均为同一初始点出发的二次/Rosenbrock 族函数，"
          "稳定性取末尾 50 轮真实损失统计。", "",
          "## 调度对比", "", table, "",
          "## 回滚事件明细（场景 B：一开始就发散）", ""]
    for kind, r in all_rows["B_一开始就发散"].items():
        md.append(f"### {kind}")
        if not r["rollback_events"]:
            md.append("- 无回滚")
        for e in r["rollback_events"]:
            md.append(
                f"- epoch {e['epoch']}: lr {e['pre_lr']:g} -> {e['post_lr']:g}; "
                f"触发损失 {e['bad_loss']:.4g}，历史最优 {e['best_true_loss']:.4g}"
            )
            md.append(f"  - 判定依据: {e['reason']}")
        md.append("")
    md += ["## 诊断样例（判定依据与建议动作）", ""]
    for name, p in probes.items():
        md.append(f"### {name}")
        md.append(f"- 状态: `{p['status']}`")
        md.append(f"- 判定依据: {p['reason']}")
        md.append(f"- 建议动作: {p['action']}")
        md.append("")

    with open(os.path.join(DATA_DIR, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(DATA_DIR, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": all_rows, "diagnosis_probes": probes},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {DATA_DIR}/comparison.md 与 comparison.json")


if __name__ == "__main__":
    main()
