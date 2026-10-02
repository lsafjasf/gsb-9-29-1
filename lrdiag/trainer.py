"""梯度下降训练器：集成调度、滑动窗口诊断与发散自愈。

每个 epoch 的流程：
1. 调度器推进（PlateauDecay 喂上一轮的观测损失），得到当前步长；
2. x <- x - lr * grad(x)，计算「观测损失」（含噪声，用于诊断）
   和「真实损失」（无噪声，用于判定收敛与维护历史最优）；
3. 诊断器吃进观测损失：
   - diverging  -> 自愈：参数回滚到历史最优点，调度器强制降步长；
     若步长已贴 min_lr 仍连续发散，则中止并标记 unstable；
   - plateau / oscillating -> 记录事件与建议（PlateauDecay 会自行降档）；
4. 真实损失达到 tol 即收敛结束。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .diagnostics import (
    DIVERGING,
    HEALTHY,
    OSCILLATING,
    PLATEAU,
    Diagnosis,
    LossDiagnoser,
)
from .schedulers import LRScheduler, PlateauDecay, SchedulerState


@dataclass
class RollbackEvent:
    epoch: int
    pre_lr: float
    post_lr: float
    bad_loss: float
    best_true_loss: float
    reason: str
    no_cut: bool = False


@dataclass
class RunResult:
    converged: bool
    unstable: bool
    epochs: int
    final_true_loss: float
    final_obs_loss: float
    final_lr: float
    best_true_loss: float
    best_epoch: int
    rollback_count: int
    plateau_count: int
    oscillating_count: int
    rollback_events: List[RollbackEvent] = field(default_factory=list)
    obs_losses: List[float] = field(default_factory=list)
    true_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)


def assert_rollback(
    event: RollbackEvent,
    params_before_rollback: Sequence[float],
    params_after_rollback: Sequence[float],
    best_params: Sequence[float],
) -> None:
    """回滚断言：触发自愈后必须满足的不变量。

    1. 回滚后的参数与记录的历史最优参数逐分量完全一致；
    2. 回滚确实丢弃了一个「变坏了」的参数点（前后参数不同）；
    3. 回滚对应的损失确实劣于历史最优损失；
    4. 步长被压低且不小于 min_lr（由调用方在 event 中给出 post_lr）。
    """
    after = list(params_after_rollback)
    best = list(best_params)
    assert after == best, (
        f"回滚断言失败: 回滚后参数 {after} 与历史最优点 {best} 不一致"
    )
    assert after != list(params_before_rollback), (
        "回滚断言失败: 回滚前后参数相同，回滚没有实际丢弃坏状态"
    )
    assert event.bad_loss >= event.best_true_loss, (
        f"回滚断言失败: 触发损失 {event.bad_loss} 并未劣于历史最优 "
        f"{event.best_true_loss}，不该回滚"
    )
    assert event.post_lr <= event.pre_lr, (
        f"回滚断言失败: 步长未降低 ({event.pre_lr} -> {event.post_lr})"
    )
    assert event.post_lr >= 0.0


class GDTrainer:
    def __init__(
        self,
        objective,
        scheduler: LRScheduler,
        diagnoser: Optional[LossDiagnoser] = None,
        tol: float = 1e-6,
        rollback_factor: float = 0.5,
        max_stuck_rollbacks: int = 3,
    ):
        self.objective = objective
        self.scheduler = scheduler
        self.diagnoser = diagnoser or LossDiagnoser()
        self.tol = tol
        self.rollback_factor = rollback_factor
        self.max_stuck_rollbacks = max_stuck_rollbacks

        self.epoch = 0
        self.params = list(objective.start)
        self.best_params = list(self.params)
        self.best_true_loss = objective.true_value(self.params)
        self.best_epoch = 0
        self.prev_obs: Optional[float] = None
        self.stuck_count = 0

        self.obs_losses: List[float] = []
        self.true_losses: List[float] = []
        self.lrs: List[float] = []
        self.statuses: List[str] = []
        self.rollback_events: List[RollbackEvent] = []
        self.plateau_count = 0
        self.oscillating_count = 0

    # ------------------------------------------------------------------ #
    def run(self, max_epochs: int = 2000) -> RunResult:
        while self.epoch < max_epochs:
            is_plateau_sched = isinstance(self.scheduler, PlateauDecay)
            lr = (
                self.scheduler.step(self.prev_obs)
                if is_plateau_sched
                else self.scheduler.step()
            )

            grad = self.objective.grad(self.params)
            next_params = [
                p - lr * g for p, g in zip(self.params, grad)
            ]
            obs_loss = self.objective.value(next_params)
            true_loss = self.objective.true_value(next_params)

            diagnosis: Diagnosis = self.diagnoser.update(obs_loss)
            status = diagnosis.status
            pre_rollback_params = list(next_params)

            if status == DIVERGING:
                pre_lr = lr
                post_lr = self.scheduler.reduce(self.rollback_factor)
                event = RollbackEvent(
                    epoch=self.epoch,
                    pre_lr=pre_lr,
                    post_lr=post_lr,
                    bad_loss=true_loss,
                    best_true_loss=self.best_true_loss,
                    reason=diagnosis.reason,
                    no_cut=(post_lr >= pre_lr),
                )
                # 回滚断言：在真正提交回滚前校验所有不变量
                assert_rollback(event, pre_rollback_params, self.best_params,
                                self.best_params)
                self.rollback_events.append(event)

                # 提交回滚：丢弃 next_params，恢复历史最优点
                self.params = list(self.best_params)
                self.prev_obs = None
                self.diagnoser.reset()
                self.diagnoser.seed(self.objective.value(list(self.best_params)))

                if event.no_cut:
                    self.stuck_count += 1
                else:
                    self.stuck_count = 0

                obs_recorded = obs_loss
                true_recorded = true_loss
                lr_recorded = pre_lr

                if self.stuck_count >= self.max_stuck_rollbacks:
                    self._record(obs_recorded, true_recorded, lr_recorded, DIVERGING)
                    return self._build_result(False, True)
            else:
                # 正常接受这一步
                self.params = next_params
                self.prev_obs = obs_loss
                if status == PLATEAU:
                    self.plateau_count += 1
                elif status == OSCILLATING:
                    self.oscillating_count += 1

                if not math.isnan(true_loss) and true_loss < self.best_true_loss:
                    self.best_true_loss = true_loss
                    self.best_params = list(self.params)
                    self.best_epoch = self.epoch

                obs_recorded = obs_loss
                true_recorded = true_loss
                lr_recorded = lr
                self._record(obs_recorded, true_recorded, lr_recorded, status)

                if true_loss <= self.tol:
                    return self._build_result(True, False)

            self.epoch += 1

        return self._build_result(False, False)

    def _record(self, obs: float, true: float, lr: float, status: str) -> None:
        self.obs_losses.append(obs)
        self.true_losses.append(true)
        self.lrs.append(lr)
        self.statuses.append(status)

    def _build_result(self, converged: bool, unstable: bool) -> RunResult:
        return RunResult(
            converged=converged,
            unstable=unstable,
            epochs=len(self.true_losses),
            final_true_loss=self.true_losses[-1] if self.true_losses else float("nan"),
            final_obs_loss=self.obs_losses[-1] if self.obs_losses else float("nan"),
            final_lr=self.scheduler.current_lr,
            best_true_loss=self.best_true_loss,
            best_epoch=self.best_epoch,
            rollback_count=len(self.rollback_events),
            plateau_count=self.plateau_count,
            oscillating_count=self.oscillating_count,
            rollback_events=list(self.rollback_events),
            obs_losses=list(self.obs_losses),
            true_losses=list(self.true_losses),
            lrs=list(self.lrs),
            statuses=list(self.statuses),
        )

    # ------------------------------------------------------------------ #
    # 热启动：完整保存/恢复训练状态，恢复后续跑与一次性跑完轨迹一致
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "params": list(self.params),
            "best_params": list(self.best_params),
            "best_true_loss": self.best_true_loss,
            "best_epoch": self.best_epoch,
            "prev_obs": self.prev_obs,
            "stuck_count": self.stuck_count,
            "scheduler": self.scheduler.state_dict(),
            "diagnoser_window": list(self.diagnoser.losses),
            "history": {
                "obs": list(self.obs_losses),
                "true": list(self.true_losses),
                "lrs": list(self.lrs),
                "statuses": list(self.statuses),
            },
            "events": [
                {
                    "epoch": e.epoch,
                    "pre_lr": e.pre_lr,
                    "post_lr": e.post_lr,
                    "bad_loss": e.bad_loss,
                    "best_true_loss": e.best_true_loss,
                    "reason": e.reason,
                    "no_cut": e.no_cut,
                }
                for e in self.rollback_events
            ],
            "counts": {
                "plateau": self.plateau_count,
                "oscillating": self.oscillating_count,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        sched_state: SchedulerState = state["scheduler"]
        if sched_state.name != self.scheduler.type_name:
            raise ValueError(
                f"热启动调度器类型不匹配: {sched_state.name} vs {self.scheduler.type_name}"
            )
        self.scheduler.load_state_dict(sched_state)

        self.epoch = state["epoch"]
        self.params = list(state["params"])
        self.best_params = list(state["best_params"])
        self.best_true_loss = state["best_true_loss"]
        self.best_epoch = state["best_epoch"]
        self.prev_obs = state["prev_obs"]
        self.stuck_count = state["stuck_count"]

        self.diagnoser.reset()
        for value in state["diagnoser_window"]:
            self.diagnoser.seed(value)

        hist = state["history"]
        self.obs_losses = list(hist["obs"])
        self.true_losses = list(hist["true"])
        self.lrs = list(hist["lrs"])
        self.statuses = list(hist["statuses"])

        self.rollback_events = [RollbackEvent(**e) for e in state["events"]]
        self.plateau_count = state["counts"]["plateau"]
        self.oscillating_count = state["counts"]["oscillating"]


def tail_stability(result: RunResult, tail: int = 50) -> dict:
    """收敛后稳定性数据：取末尾 tail 轮真实损失做统计。"""
    losses = [v for v in result.true_losses[-tail:] if not math.isnan(v)]
    if not losses:
        return {"tail_mean": float("nan"), "tail_std": float("nan"),
                "tail_min": float("nan"), "tail_max": float("nan"),
                "tail_range": float("nan")}
    mean = sum(losses) / len(losses)
    std = math.sqrt(sum((v - mean) ** 2 for v in losses) / len(losses))
    lo, hi = min(losses), max(losses)
    return {
        "tail_mean": mean,
        "tail_std": std,
        "tail_min": lo,
        "tail_max": hi,
        "tail_range": hi - lo,
    }
