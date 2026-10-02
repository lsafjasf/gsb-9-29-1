"""对比实验使用的目标函数（只用标准库，无 numpy）。

每个目标都提供：
- ``value(x)``：带噪声的观测损失（训练时实际看到的）；
- ``true_value(x)``：无噪声真实损失（用于判定收敛、记录历史最优）；
- ``grad(x)``：基于真实损失的解析梯度；
- ``start``：统一的初始点；``optimum``：理论最优值。
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence


class Quadratic:
    """各向异性二次函数 f(x) = sum a_i * x_i^2，可选加性高斯噪声。

    条件数由 axes 控制（最大/最小轴系数之比），条件数越大越容易平台期/发散。
    """

    def __init__(
        self,
        axes: Sequence[float] = (1.0, 0.01),
        noise_std: float = 0.0,
        seed: int = 0,
    ):
        self.axes = list(axes)
        self.noise_std = noise_std
        self.start = [10.0] * len(self.axes)
        self.optimum = 0.0
        self._rng = random.Random(seed)

    def true_value(self, x: Sequence[float]) -> float:
        return sum(a * xi * xi for a, xi in zip(self.axes, x))

    def value(self, x: Sequence[float]) -> float:
        loss = self.true_value(x)
        if self.noise_std > 0:
            loss += self._rng.gauss(0.0, self.noise_std)
        return loss

    def grad(self, x: Sequence[float]) -> List[float]:
        return [2.0 * a * xi for a, xi in zip(self.axes, x)]


class Rosenbrock:
    """Rosenbrock 香蕉函数：经典的弯谷函数，用于考察走出山谷的能力。

    f(x,y) = (1-x)^2 + 100*(y-x^2)^2，理论最优 (1,1)，最优值 0。
    """

    def __init__(self, noise_std: float = 0.0, seed: int = 1):
        self.noise_std = noise_std
        self.start = [-1.2, 1.0]
        self.optimum = 0.0
        self._rng = random.Random(seed)

    def true_value(self, x: Sequence[float]) -> float:
        a, b = x[0], x[1]
        return (1.0 - a) ** 2 + 100.0 * (b - a * a) ** 2

    def value(self, x: Sequence[float]) -> float:
        loss = self.true_value(x)
        if self.noise_std > 0:
            loss += self._rng.gauss(0.0, self.noise_std)
        return loss

    def grad(self, x: Sequence[float]) -> List[float]:
        a, b = x[0], x[1]
        d_a = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a)
        d_b = 200.0 * (b - a * a)
        return [d_a, d_b]
