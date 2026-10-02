"""序贯检验库：Wald 序贯概率比检验（SPRT）+ 截断 + Bonferroni 多指标校正。

只依赖 Python 3 标准库，随机源可注入（任何带 random() 方法的对象均可）。

适用模型：双臂伯努利（A/B 转化率类指标）。每组观测是一对
(x_t, x_c) ∈ {0,1}^2，分别来自处理组/对照组。

边界推导（详见 README.md）：
  设 L_n = ∏ f1/f0 为备择 H1 对原假设 H0 的似然比。
  - 拒绝 H0（上界 A = 1/α）：由 Ville 不等式（非负上鞅），
    P0(存在 n: L_n ≥ A) ≤ 1/A = α，且截断（n ≥ nmax 后强制终止）
    不会增加该概率，因此 I 类错误有严格保证。
  - 接受 H0（下界 B = β）：经典 Wald 常数（未截断时由错误成本不等式
    得 P1(误受 H0) ≤ β）；截断后下界保证不再严格成立，故本库通过
    模拟实证检验截断后的功效，nmax 默认取 3 倍理论平均样本量（ASN）
    以使"撞顶"概率可忽略。

多指标：K 个指标同时检验时，各检验使用 α' = α/K（Bonferroni），
族系错误率 FWER ≤ α（指标间任意相关均成立，用并集界证明）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

__all__ = [
    "SPRTConfig",
    "BernoulliPairSPRT",
    "bonferroni_configs",
    "FamilySPRT",
    "wald_asn",
    "bern_asn",
]

_DECIDE_ACCEPT = "accept_h0"
_DECIDE_REJECT = "reject_h0"
_DECIDE_CONTINUE = "continue"


@dataclass(frozen=True)
class SPRTConfig:
    """单个 SPRT 的配置。

    属性
      p0_t, p0_c: H0 下处理组/对照组的伯努利概率。
      p1_t, p1_c: H1 下处理组/对照组的伯努利概率（p1_t > p0_t）。
      alpha: 目标 I 类错误率（上界由 1/alpha 严格保证）。
      beta:  目标 II 类错误率（未截断时由 Wald 常数保证）。
      nmax:  最大配对样本量上限（撞顶时按 L_n 与 1 的大小关系裁决）。
    """

    p0_t: float
    p0_c: float
    p1_t: float
    p1_c: float
    alpha: float = 0.05
    beta: float = 0.20
    nmax: int = 0

    def __post_init__(self) -> None:
        for name in ("p0_t", "p0_c", "p1_t", "p1_c"):
            p = getattr(self, name)
            if not 0.0 < p < 1.0:
                raise ValueError(f"{name} 必须在 (0, 1) 内，收到 {p}")
        if self.p1_t <= self.p0_t:
            raise ValueError("备择效应必须为正：p1_t > p0_t（负向情形可交换双臂）")
        if not 0.0 < self.alpha < 1.0 or not 0.0 < self.beta < 1.0:
            raise ValueError("alpha、beta 必须在 (0, 1) 内")
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta 必须 < 1，边界 A > B 才存在")
        if self.nmax < 0:
            raise ValueError("nmax 不能为负")
        if self.nmax == 0:
            object.__setattr__(self, "nmax", max(1, math.ceil(3.0 * self.asn_h1())))

    @property
    def upper(self) -> float:
        """拒绝边界 A = 1/alpha（对数尺度为 log A）。"""
        return 1.0 / self.alpha

    @property
    def lower(self) -> float:
        """接受边界 B = beta（对数尺度为 log B）。"""
        return self.beta

    @property
    def log_upper(self) -> float:
        return -math.log(self.alpha)

    @property
    def log_lower(self) -> float:
        return math.log(self.beta)

    def llr_pair(self, x_t: int, x_c: int) -> float:
        """一对观测的对数似然比贡献 log[f1/f0]。"""
        return (
            x_t * math.log(self.p1_t / self.p0_t)
            + (1 - x_t) * math.log((1 - self.p1_t) / (1 - self.p0_t))
            + x_c * math.log(self.p1_c / self.p0_c)
            + (1 - x_c) * math.log((1 - self.p1_c) / (1 - self.p0_c))
        )

    @staticmethod
    def for_effect(p0: float, delta: float, alpha: float = 0.05,
                   beta: float = 0.20, nmax: int = 0) -> "SPRTConfig":
        """对称效应便捷构造：H0=(p0,p0)，H1=(p0+delta, p0-delta)。"""
        return SPRTConfig(p0, p0, p0 + delta, p0 - delta, alpha, beta, nmax)

    def asn_h0(self) -> float:
        return bern_asn(self.p0_t, self.p0_c, self)

    def asn_h1(self) -> float:
        return bern_asn(self.p1_t, self.p1_c, self)


def _kl_bern(p: float, q: float) -> float:
    """伯努利 KL 散度 D(p‖q)。"""
    if p == 0.0 or p == 1.0:
        return 0.0
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def wald_asn(llr_mean: float, llr_var: float, cfg: SPRTConfig) -> float:
    """Wald 近似平均样本量（配对数）。

    在“真实对数似然比每步均值为 μ、方差为 σ²”时：
      H0 下 μ<0，ASN ≈ [(1-α)log B + α log A] / μ
      H1 下 μ>0，ASN ≈ [β log B + (1-β) log A] / μ
    忽略越界过冲。返回 H1（μ>0）或 H0（μ<0）对应的近似值。
    """
    a, b = cfg.log_upper, cfg.log_lower
    if llr_mean > 0:
        return (cfg.beta * b + (1 - cfg.alpha) * a) / llr_mean
    if llr_mean < 0:
        return ((1 - cfg.beta) * b + cfg.alpha * a) / llr_mean
    return a * abs(b) / llr_var  # μ=0 时 Wald 扩散近似


def bern_asn(p_t: float, p_c: float, cfg: SPRTConfig) -> float:
    """双臂伯努利在真实概率 (p_t, p_c) 下的理论 ASN。"""
    mu = (
        _kl_bern(p_t, cfg.p0_t) - _kl_bern(p_t, cfg.p1_t)
        + _kl_bern(p_c, cfg.p0_c) - _kl_bern(p_c, cfg.p1_c)
    )
    # 四种结局 (1,1),(1,0),(0,1),(0,0) 下 LLR 的方差
    probs = (
        p_t * p_c, p_t * (1 - p_c), (1 - p_t) * p_c, (1 - p_t) * (1 - p_c)
    )
    vals = (cfg.llr_pair(1, 1), cfg.llr_pair(1, 0),
            cfg.llr_pair(0, 1), cfg.llr_pair(0, 0))
    mean = sum(pr * v for pr, v in zip(probs, vals))
    var = sum(pr * (v - mean) ** 2 for pr, v in zip(probs, vals))
    return wald_asn(mu, var, cfg)


class BernoulliPairSPRT:
    """双臂伯努利 SPRT 的在线检验状态机。

    用法：
        test = BernoulliPairSPRT(cfg, rng=rng)
        result = test.update(x_t, x_c)   # 每来一对样本调用一次
    result 为 "continue" / "reject_h0"（有显著正向效果）/ "accept_h0"（无效果）。
    """

    def __init__(self, config: SPRTConfig, rng=None):
        self.config = config
        self._rng = rng  # 检验本身不用随机数；保留以便扩展（如随机化裁决）
        self.log_lr = 0.0
        self.n = 0
        self.decision = _DECIDE_CONTINUE
        # 单步 LLR 查找表：2*x_t + x_c -> 值
        self._step = (
            config.llr_pair(0, 0), config.llr_pair(0, 1),
            config.llr_pair(1, 0), config.llr_pair(1, 1),
        )

    @property
    def stopped(self) -> bool:
        return self.decision != _DECIDE_CONTINUE

    def update(self, x_t: int, x_c: int):
        if self.stopped:
            raise RuntimeError(f"检验已终止（{self.decision}），不能继续 update")
        x_t, x_c = int(x_t), int(x_c)
        if x_t not in (0, 1) or x_c not in (0, 1):
            raise ValueError("x_t、x_c 必须为 0 或 1")
        self.n += 1
        self.log_lr += self._step[2 * x_t + x_c]
        if self.log_lr >= self.config.log_upper:
            self.decision = _DECIDE_REJECT
        elif self.log_lr <= self.config.log_lower:
            self.decision = _DECIDE_ACCEPT
        elif self.n >= self.config.nmax:
            # 撞顶一律记 accept_h0：拒绝只发生在真正越上界时，
            # Ville 给出的 P0(越上界) ≤ alpha 因此对截断版本仍然严格成立。
            self.decision = _DECIDE_ACCEPT
        return self.decision


def bonferroni_configs(configs, alpha: float = 0.05):
    """Bonferroni 校正：把 K 个检验各自的 I 类水平调到 alpha/K。

    nmax 保持不变（每个指标仍是独立的双臂流）。返回新配置列表。
    """
    k = len(configs)
    if k == 0:
        return []
    adjusted = alpha / k
    return [replace(c, alpha=adjusted) for c in configs]


class FamilySPRT:
    """多指标族：任一指标 reject_h0 即族级判“存在阳性指标”。

    各指标可独立提前停止；全部停止或所有指标已出结论时族结束。
    保守起见，撞顶的指标一律记 accept_h0（不把撞顶算作阳性）。
    """

    def __init__(self, configs, alpha: float = 0.05, rng=None,
                 correction: Optional[Callable] = None):
        if correction is None:
            correction = bonferroni_configs
        self.alpha = alpha
        self.configs = correction(list(configs), alpha)
        self.tests = [BernoulliPairSPRT(c, rng) for c in self.configs]
        self.decision = _DECIDE_CONTINUE

    @property
    def stopped(self) -> bool:
        return all(t.stopped for t in self.tests)

    def update(self, index: int, x_t: int, x_c: int):
        """给第 index 个指标送入一对观测。"""
        test = self.tests[index]
        if not test.stopped:
            test.update(x_t, x_c)
        if any(t.decision == _DECIDE_REJECT for t in self.tests):
            self.decision = _DECIDE_REJECT
        elif self.stopped:
            self.decision = _DECIDE_ACCEPT
        return self.decision

    def active_indices(self):
        return [i for i, t in enumerate(self.tests) if not t.stopped]
