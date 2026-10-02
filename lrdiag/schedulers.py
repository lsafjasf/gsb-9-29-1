"""学习率调度器：阶梯衰减、余弦衰减、按平台期衰减。

所有调度器共享同一组约定：
- ``base_lr`` 是初始（基准）步长，``min_lr`` 是步长硬下限，任何路径都不会跌破；
- ``step(metric=None)`` 推进调度并返回当前可用步长；
- ``current_lr`` 始终等于最近一次 ``step`` 的返回值；
- ``reduce(factor)`` 供自愈模块在发散后强制降步长；
- ``state_dict()/load_state_dict()`` 用于热启动，恢复后调度与继续训练完全等价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SchedulerState:
    name: str
    base_lr: float
    min_lr: float
    last_epoch: int
    current_lr: float
    extra: dict


class LRScheduler:
    """调度器基类，统一 min_lr 钳制、epoch 计数与状态存取。"""

    type_name = "base"

    def __init__(self, base_lr: float, min_lr: float = 0.0):
        if base_lr <= 0:
            raise ValueError("base_lr 必须为正数")
        if min_lr < 0:
            raise ValueError("min_lr 不能为负数")
        if min_lr > base_lr:
            raise ValueError("min_lr 不能大于 base_lr")
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.last_epoch = -1
        self.current_lr = float(base_lr)

    # ---- 子类实现的两个钩子 ----
    def _compute_lr(self) -> float:
        raise NotImplementedError

    def _reduce_base(self, factor: float) -> None:
        """自愈降步长时对基准步长做的调整（阶梯/余弦会永久压低基准）。"""

    # ---- 公共 API ----
    def step(self, metric: Optional[float] = None) -> float:
        self.last_epoch += 1
        lr = max(self.min_lr, min(self.base_lr, self._compute_lr()))
        self.current_lr = float(lr)
        return self.current_lr

    def reduce(self, factor: float = 0.5) -> float:
        """发散自愈：按 factor 强制降低步长，返回降后实际生效的步长。"""
        if not 0.0 < factor < 1.0:
            raise ValueError("reduce 的 factor 必须落在 (0, 1)")
        self._reduce_base(factor)
        lr = max(self.min_lr, min(self.base_lr, self._compute_lr()))
        self.current_lr = float(lr)
        return self.current_lr

    def state_dict(self) -> SchedulerState:
        return SchedulerState(
            name=self.type_name,
            base_lr=self.base_lr,
            min_lr=self.min_lr,
            last_epoch=self.last_epoch,
            current_lr=self.current_lr,
            extra=self._extra_state(),
        )

    def _extra_state(self) -> dict:
        return {}

    def load_state_dict(self, state: SchedulerState) -> None:
        if state.name != self.type_name:
            raise ValueError(
                f"调度器类型不匹配: 期望 {self.type_name}, 状态来自 {state.name}"
            )
        self.base_lr = float(state.base_lr)
        self.min_lr = float(state.min_lr)
        self.last_epoch = state.last_epoch
        self.current_lr = float(state.current_lr)
        self._load_extra(state.extra)

    def _load_extra(self, extra: dict) -> None:
        pass

    def describe(self) -> str:
        raise NotImplementedError


class StepDecay(LRScheduler):
    """阶梯衰减：每 ``step_size`` 轮把步长乘一次 ``gamma``。

    更新公式（epoch 从 0 开始计数）::

        lr(e) = base_lr * gamma ** floor(e / step_size)

    特点：前期保持大步长快速推进，周期性跳水；衰减次数可预测，
    配合总轮数与 gamma 很容易安排「什么时候开始精修」。
    """

    type_name = "step"

    def __init__(
        self,
        base_lr: float,
        step_size: int = 100,
        gamma: float = 0.5,
        min_lr: float = 0.0,
    ):
        super().__init__(base_lr, min_lr)
        if step_size <= 0:
            raise ValueError("step_size 必须为正整数")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma 必须落在 (0, 1]")
        self.step_size = step_size
        self.gamma = gamma

    def _compute_lr(self) -> float:
        drops = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** drops)

    def _reduce_base(self, factor: float) -> None:
        # 永久压低基准，后续所有阶梯按新基准继续衰减
        self.base_lr = max(self.min_lr, self.base_lr * factor)

    def _extra_state(self) -> dict:
        return {"step_size": self.step_size, "gamma": self.gamma}

    def describe(self) -> str:
        return (
            f"阶梯衰减: 每 {self.step_size} 轮 lr *= {self.gamma}, "
            f"base={self.base_lr:g}, 下限={self.min_lr:g}"
        )


class CosineDecay(LRScheduler):
    """余弦衰减：在 ``total_epochs`` 内按半周期余弦平滑退火到 min_lr。

    更新公式::

        lr(e) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * e / T))

    e >= T 后保持 min_lr。特点：全程连续平滑，前期下降慢（保留探索速度）、
    中期下降最快、后期趋于平缓（稳住精修），不会出现阶梯跳水。
    """

    type_name = "cosine"

    def __init__(
        self,
        base_lr: float,
        total_epochs: int = 1000,
        min_lr: float = 1e-8,
    ):
        super().__init__(base_lr, min_lr)
        if total_epochs <= 0:
            raise ValueError("total_epochs 必须为正整数")
        self.total_epochs = total_epochs

    def _compute_lr(self) -> float:
        if self.last_epoch >= self.total_epochs:
            return self.min_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.total_epochs))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def _reduce_base(self, factor: float) -> None:
        self.base_lr = max(self.min_lr, self.base_lr * factor)

    def _extra_state(self) -> dict:
        return {"total_epochs": self.total_epochs}

    def describe(self) -> str:
        return (
            f"余弦衰减: 周期 {self.total_epochs} 轮退火到 {self.min_lr:g}, "
            f"base={self.base_lr:g}"
        )


class PlateauDecay(LRScheduler):
    """按平台期衰减（ReduceLROnPlateau 风格）：损失不见改善才降步长。

    更新规则：每次 ``step(metric)`` 喂入当前观测损失；若连续
    ``patience`` 轮都没有比历史最优改善 ``threshold``（相对阈值），
    就把步长乘一次 ``factor``，并清空等待计数。没有平台期时步长保持不变，
    因此它不按时间衰减，而是按「训练是否还有进展」衰减，
    对噪声目标可设置较大的 patience 避免被噪声误降。
    """

    type_name = "plateau"

    def __init__(
        self,
        base_lr: float,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 20,
        threshold: float = 1e-4,
        min_lr: float = 1e-8,
    ):
        super().__init__(base_lr, min_lr)
        if mode not in ("min", "max"):
            raise ValueError("mode 只能是 'min' 或 'max'")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor 必须落在 (0, 1)")
        if patience < 0:
            raise ValueError("patience 不能为负")
        if threshold < 0:
            raise ValueError("threshold 不能为负")
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        # 第一个 epoch（还没有 metric）就用 base_lr
        self._pending_lr = float(base_lr)

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best * (1.0 - self.threshold)
        return current > best * (1.0 + self.threshold)

    def step(self, metric: Optional[float] = None) -> float:
        self.last_epoch += 1
        # metric 为 None 时（首轮、或发散回滚后的下一轮）只推进 epoch，
        # 保持当前步长；正常训练每轮都应传入观测损失。
        if metric is None:
            self.current_lr = self._pending_lr
            return self.current_lr
        if self.best is None or self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience and self._pending_lr > self.min_lr:
                self._pending_lr = max(
                    self.min_lr, self._pending_lr * self.factor
                )
                self.num_bad_epochs = 0
                self.best = metric  # 降档后重新建立基线
        self.current_lr = self._pending_lr
        return self.current_lr

    def _compute_lr(self) -> float:
        return self._pending_lr

    def _reduce_base(self, factor: float) -> None:
        # 自愈降档与「平台期降档」走同一条通道，保证下限钳制一致
        self._pending_lr = max(self.min_lr, self._pending_lr * factor)

    def _extra_state(self) -> dict:
        return {
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "pending_lr": self._pending_lr,
        }

    def _load_extra(self, extra: dict) -> None:
        self.mode = extra["mode"]
        self.factor = extra["factor"]
        self.patience = extra["patience"]
        self.threshold = extra["threshold"]
        self.best = extra["best"]
        self.num_bad_epochs = extra["num_bad_epochs"]
        self._pending_lr = extra["pending_lr"]

    def describe(self) -> str:
        return (
            f"按平台期衰减: 连续 {self.patience} 轮无相对改善 "
            f"{self.threshold:g} 则 lr *= {self.factor}, "
            f"当前={self.current_lr:g}, 下限={self.min_lr:g}"
        )


def build_scheduler(kind: str, base_lr: float, min_lr: float, **kwargs) -> LRScheduler:
    """工厂函数，对比脚本和测试统一从这里构造调度器。"""
    kind = kind.lower()
    if kind == "step":
        return StepDecay(base_lr, min_lr=min_lr, **kwargs)
    if kind == "cosine":
        return CosineDecay(base_lr, min_lr=min_lr, **kwargs)
    if kind == "plateau":
        return PlateauDecay(base_lr, min_lr=min_lr, **kwargs)
    raise ValueError(f"未知调度器类型: {kind}")
