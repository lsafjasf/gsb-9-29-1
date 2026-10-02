"""训练曲线诊断：滑动窗口检测发散、平台期、震荡。

每个 epoch 喂一个观测损失，诊断器只保留最近 ``window`` 个点，
输出 :class:`Diagnosis`，其中包含：
- ``status``：diverging / plateau / oscillating / healthy；
- ``reason``：判定依据（使用的窗口、斜率、涨跌幅等具体数字）；
- ``action``：建议动作（保持 / 降步长 / 回滚 + 降步长 等）。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

DIVERGING = "diverging"
PLATEAU = "plateau"
OSCILLATING = "oscillating"
HEALTHY = "healthy"


@dataclass
class Diagnosis:
    status: str
    reason: str
    action: str
    window_mean: float
    window_std: float
    rel_slope: float

    def __str__(self) -> str:
        return f"[{self.status}] {self.action} | {self.reason}"


@dataclass
class WindowStats:
    values: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    slope: float = 0.0  # 后半窗口均值 - 前半窗口均值（绝对差）

    @property
    def rel_slope(self) -> float:
        scale = abs(self.mean) + 1e-12
        return self.slope / scale


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _mean_std(values: List[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def window_stats(window: Deque[float]) -> WindowStats:
    values = list(window)
    n = len(values)
    mean, std = _mean_std(values)
    half = max(1, n // 2)
    first = sum(values[:half]) / half
    second = sum(values[n - half:]) / half
    slope = second - first
    return WindowStats(values=values, mean=mean, std=std, slope=slope)


class LossDiagnoser:
    """滑动窗口诊断器。

    参数（均只依赖最近 ``window`` 个损失点）：
    - ``window``：窗口长度（默认 20）；
    - ``diverge_ratio``：后半窗口均值相对前半窗口均值的上涨倍数阈值，
      超过即判定发散；另外损失变成 NaN/Inf 或被判定为快速单调上涨
      （窗口后 1/4 全部高于前 1/4 且整体翻倍）时直接判发散；
    - ``plateau_slope``：窗口归一化斜率（相对变化）绝对值小于该阈值，
      且窗口波动也很小（std/mean < plateau_slope），判平台期；
    - ``oscillate_cv``：变异系数 std/(|mean|+eps) 超过该阈值，
      且方向频繁反转（相邻差分符号翻转率 >= 0.5），判震荡。

    判定优先级：发散 > 平台期 > 震荡 > 健康。发散是硬故障，
    需要触发回滚自愈，因此排在最前。

    发散证据采用「水位抬升」而非单点比较：最近 1/4 窗口的最小值
    必须高于前半窗口的最大值（整体抬升，对单个噪声尖峰免疫），
    且尾部均值相对头部均值放大 ``diverge_ratio`` 倍；证据需连续
    两个窗口成立才确认发散（NaN/Inf 除外，立即确认）。
    """

    def __init__(
        self,
        window: int = 20,
        diverge_ratio: float = 1.5,
        plateau_slope: float = 1e-3,
        oscillate_cv: float = 0.1,
        min_full_window: bool = True,
    ):
        if window < 4:
            raise ValueError("窗口长度至少为 4")
        if diverge_ratio <= 1.0:
            raise ValueError("diverge_ratio 必须大于 1")
        self.window = window
        self.diverge_ratio = diverge_ratio
        self.plateau_slope = plateau_slope
        self.oscillate_cv = oscillate_cv
        self.min_full_window = min_full_window
        self._losses: Deque[float] = deque(maxlen=window)
        self._div_streak = 0

    def reset(self) -> None:
        self._losses.clear()
        self._div_streak = 0

    def seed(self, loss: float) -> None:
        """向窗口直接注入一个损失点（不触发判定）。

        用于回滚后用历史最优点的损失重新打底，或热启动时恢复窗口。
        """
        self._losses.append(float(loss))

    @property
    def losses(self) -> List[float]:
        return list(self._losses)

    def update(self, loss: float) -> Diagnosis:
        self._losses.append(float(loss))
        values = list(self._losses)
        n = len(values)

        if not _is_finite(loss):
            return Diagnosis(
                DIVERGING,
                f"最新损失为 {loss}（NaN/Inf），优化已数值崩溃",
                "立即回滚到历史最优状态，并把步长减半后重试",
                float("nan"),
                float("nan"),
                float("nan"),
            )

        # 数据不足时不下结论，避免回滚/降档误伤开局
        if n < (self.window if self.min_full_window else 4):
            mean, std = _mean_std(values)
            return Diagnosis(
                HEALTHY,
                f"窗口仅 {n} 个点（需要 {self.window} 个），暂不判定",
                "保持当前步长继续训练",
                mean,
                std,
                0.0,
            )

        stats = window_stats(self._losses)
        mean, std = stats.mean, stats.std
        scale = abs(mean) + 1e-12
        cv = std / scale
        rel_slope = stats.rel_slope

        # ---- 1. 发散：整体水位抬升或快速单调翻倍，连续两窗确认 ----
        lifted, first_mean, tail_mean, ratio = self._level_lifted(
            values, self.diverge_ratio
        )
        fast_soaring = self._fast_soaring(values)
        if lifted or fast_soaring:
            self._div_streak += 1
        else:
            self._div_streak = 0
        if self._div_streak >= 2:
            q = max(1, n // 4)
            return Diagnosis(
                DIVERGING,
                (
                    f"连续 {self._div_streak} 个窗口满足发散证据：最近 {q} 轮"
                    f"最低值已高于前半窗最高值，尾部均值 {tail_mean:.6g} 相对"
                    f"头部均值 {first_mean:.6g} 放大 {ratio:.2f}x"
                    f"（阈值 {self.diverge_ratio:g}x），确认发散"
                ),
                "回滚到历史最优状态，并把步长减半后继续",
                mean,
                std,
                rel_slope,
            )

        # ---- 2. 平台期：几乎不下降且几乎不波动 ----
        sign_changes, _ = self._sign_changes(values)
        if abs(rel_slope) <= self.plateau_slope and cv <= self.plateau_slope:
            return Diagnosis(
                PLATEAU,
                (
                    f"窗口归一化斜率 {rel_slope:.3e} 与变异系数 {cv:.3e} 均不超过 "
                    f"{self.plateau_slope:g}，损失长时间停在 {mean:.6g} 附近"
                ),
                "适当降低步长（或切换到按平台期衰减），检查是否已到精度极限",
                mean,
                std,
                rel_slope,
            )

        # ---- 3. 震荡：波动大且方向频繁反转 ----
        flip_rate = sign_changes / max(1, n - 1)
        if cv >= self.oscillate_cv and flip_rate >= 0.5:
            return Diagnosis(
                OSCILLATING,
                (
                    f"变异系数 {cv:.3e} 超过阈值 {self.oscillate_cv:g}，"
                    f"且 {n - 1} 个相邻变化中有 {sign_changes} 次反转"
                    f"（翻转率 {flip_rate:.2f}），训练在最优点附近来回跳动"
                ),
                "降低步长并对损失做滑动平均；若数据噪声大应增大批量或使用鲁棒损失",
                mean,
                std,
                rel_slope,
            )

        return Diagnosis(
            HEALTHY,
            (
                f"窗口均值 {mean:.6g}、标准差 {std:.3g}、归一化斜率 "
                f"{rel_slope:.3e}，各项指标正常"
            ),
            "保持当前步长继续训练",
            mean,
            std,
            rel_slope,
        )

    @staticmethod
    def _level_lifted(
        values: List[float], diverge_ratio: float
    ) -> tuple[bool, float, float, float]:
        """整体水位抬升：尾部 1/4 的最低点高于前半窗最高点，且均值放大达标。"""
        n = len(values)
        half = max(1, n // 2)
        q = max(1, n // 4)
        first = values[:half]
        tail = values[n - q:]
        first_mean = sum(first) / half
        tail_mean = sum(tail) / q
        ratio = (tail_mean + 1e-12) / (abs(first_mean) + 1e-12)
        lifted = min(tail) > max(first) and ratio >= diverge_ratio
        return lifted, first_mean, tail_mean, ratio

    @staticmethod
    def _sign_changes(values: List[float]) -> tuple[int, int]:
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        changes = 0
        nonzero = 0
        prev_sign = 0
        for d in diffs:
            if d == 0:
                continue
            nonzero += 1
            sign = 1 if d > 0 else -1
            if prev_sign != 0 and sign != prev_sign:
                changes += 1
            prev_sign = sign
        return changes, nonzero

    @staticmethod
    def _fast_soaring(values: List[float]) -> bool:
        n = len(values)
        q = max(1, n // 4)
        head = values[:q]
        tail = values[n - q:]
        head_max = max(head)
        if not all(v > head_max for v in tail):
            return False
        head_mean = sum(head) / q
        tail_mean = sum(tail) / q
        return tail_mean >= 2.0 * (abs(head_mean) + 1e-12)
