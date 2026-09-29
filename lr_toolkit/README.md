# lr_toolkit — 学习率调度与训练诊断（纯标准库）

针对“固定步长前期慢、后期不稳，训练常卡平台期或发散”的问题，提供：

- **3 种调度器**（均支持热启动 `restart()` 与 `min_lr` 下限）
- **训练诊断器**：检测发散 / 平台期 / 震荡，输出判定依据与建议动作
- **同一目标函数上的调度对比实验**与边界用例

环境：Python 3.8+，仅标准库，无第三方依赖。

## 运行方式

```bash
cd lr_toolkit
python3 experiment.py   # 调度对比 + 平台期调度实时轨迹 + 边界用例诊断
python3 test_all.py     # 22 项自测（unittest）
```

## 文件

| 文件 | 内容 |
|---|---|
| `schedulers.py` | `StepDecay`、`CosineAnnealingWarmRestarts`(SGDR)、`ReduceOnPlateau` |
| `diagnostics.py` | `TrainingMonitor`：发散/平台期/震荡检测，`summarize_stability` 稳定性指标 |
| `experiment.py` | 同一含噪二次目标上的对比实验与边界用例 |
| `test_all.py` | 调度器、诊断器、边界用例自测 |

## 调度器接口

```python
from schedulers import StepDecay, CosineAnnealingWarmRestarts, ReduceOnPlateau

sched = CosineAnnealingWarmRestarts(base_lr=0.5, t_0=50, t_mult=2, min_lr=1e-5)
for epoch in range(200):
    loss = train_one_epoch(...)
    lr = sched.step(metric=loss)   # metric 仅 ReduceOnPlateau 必需
    if need_fresh_exploration:
        sched.restart()            # 热启动：lr 回到 base_lr，状态清零
```

- `StepDecay(base_lr, step_size, gamma, min_lr)`：每 `step_size` 轮乘 `gamma`。
- `CosineAnnealingWarmRestarts(base_lr, t_0, t_mult, min_lr)`：余弦退火，周期结束自动热启动，周期按 `t_mult` 增长。
- `ReduceOnPlateau(base_lr, factor, patience, min_delta, cooldown, min_lr)`：指标 `patience` 轮无相对改善则乘 `factor`，带冷却期。
- 所有调度器的 lr 都不会低于 `min_lr`。

## 诊断器

```python
from diagnostics import TrainingMonitor

monitor = TrainingMonitor(window=15)
for loss in losses:
    report = monitor.update(loss)
    # report.status     -> ok / diverged / plateau / oscillating
    # report.evidence   -> 判定依据（数值证据）
    # report.suggestion -> 建议动作
```

判定逻辑（优先级：发散 > 震荡 > 平台期）：

- **发散**：loss 出现 NaN/Inf；或当前 loss 超过基线的 `diverge_factor` 倍；或窗口内净上升且 ≥`rise_frac` 的步在涨。
- **震荡**：窗口内最优 loss 几乎无改善，且 loss 差分符号翻转率 ≥ `osc_ratio`（噪声主导、原地抖动）。
- **平台期**：窗口内最优 loss 相对改善 < `plateau_rtol`，且无明显震荡。

## 对比实验（同一目标函数）

含噪二次型 `f(x)=0.5·||x−x*||²`（10 维，梯度加 σ=0.1 高斯噪声，固定种子，SGD，200 轮）：

| 调度 | 收敛轮(loss<1e-3) | 最终 loss | 最优 loss | 尾部 std | 震荡率 |
|---|---|---|---|---|---|
| constant(0.5) | 永不 | 1.68e-02 | 3.68e-03 | 1.05e-02 | 0.55 |
| step(0.5, /2每40轮) | 126 | 8.89e-04 | 5.75e-04 | 1.91e-04 | 0.52 |
| cosine(T0=50, Tmult=2) | 127 | 1.41e-02* | 6.64e-04 | 9.84e-03 | 0.51 |
| plateau(×0.5, pat=8) | 129 | 4.28e-04 | 4.28e-04 | 5.24e-05 | 0.44 |

\* 余弦热启动在重启点把 lr 拉回 base_lr，最终轮恰在重启后高位——这是探索特性，其最优 loss 与阶梯衰减相当。

结论：固定步长后期被噪声主导无法收敛；三种调度都能收敛，其中按平台期衰减的最终 loss 与尾部波动最小（最稳），阶梯衰减最简单可靠，余弦热启动适合需要周期性重新探索的场景。

## 边界用例（experiment.py 输出诊断）

1. **一开始就发散**（lr=50 > 2/L）：判定 `diverged`，建议降 lr / 梯度裁剪 / 回滚检查点。
2. **长时间不下降**（lr=1e-6 冻结）：判定 `plateau`，建议 reduce-on-plateau 或热启动。
3. **噪声极大**（σ=5）：判定 `oscillating`，建议降 lr、增大 batch 或换余弦/平台期调度。

对应自测见 `test_all.py::TestDiagnostics` 与 `TestExperiment`。
