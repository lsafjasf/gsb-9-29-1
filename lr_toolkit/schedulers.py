"""Learning-rate schedulers (pure stdlib).

All schedulers share a common interface:

    lr = scheduler.step(metric=None)   # advance one epoch, return new lr
    scheduler.restart()                # warm restart: jump back to base_lr
    scheduler.get_lr()                 # current lr without advancing

Every scheduler enforces a ``min_lr`` floor and supports warm restarts.
"""

import math


class BaseScheduler:
    """Common interface + min_lr floor + warm-restart bookkeeping."""

    def __init__(self, base_lr, min_lr=0.0):
        if base_lr <= 0:
            raise ValueError("base_lr must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        if min_lr > base_lr:
            raise ValueError("min_lr must be <= base_lr")
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.lr = float(base_lr)
        self.epoch = 0
        self.n_restarts = 0

    def get_lr(self):
        return self.lr

    def _clamp(self, lr):
        return max(self.min_lr, lr)

    def restart(self):
        """Warm restart: restore lr to base_lr and reset internal state."""
        self.lr = self.base_lr
        self.epoch = 0
        self.n_restarts += 1

    def step(self, metric=None):
        raise NotImplementedError


class StepDecay(BaseScheduler):
    """Step (ladder) decay: lr = base_lr * gamma ** (epoch // step_size)."""

    def __init__(self, base_lr, step_size=30, gamma=0.1, min_lr=0.0):
        super().__init__(base_lr, min_lr)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def step(self, metric=None):
        self.epoch += 1
        self.lr = self._clamp(self.base_lr * self.gamma ** (self.epoch // self.step_size))
        return self.lr


class CosineAnnealingWarmRestarts(BaseScheduler):
    """SGDR-style cosine annealing with automatic warm restarts.

    lr(t) = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * t / T_cur))

    When t reaches T_cur the schedule restarts (lr jumps back to base_lr)
    and the period grows by ``t_mult``.
    """

    def __init__(self, base_lr, t_0=10, t_mult=2, min_lr=0.0):
        super().__init__(base_lr, min_lr)
        if t_0 <= 0:
            raise ValueError("t_0 must be positive")
        if t_mult < 1:
            raise ValueError("t_mult must be >= 1")
        self.t_0 = int(t_0)
        self.t_mult = int(t_mult)
        self.t_cur = self.t_0
        self.t_i = 0  # steps since last restart

    def _cosine(self):
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * self.t_i / self.t_cur)
        )

    def step(self, metric=None):
        self.epoch += 1
        self.t_i += 1
        if self.t_i >= self.t_cur:  # automatic warm restart
            self.t_i = 0
            self.t_cur = int(self.t_cur * self.t_mult)
            self.n_restarts += 1
            self.lr = self.base_lr
        else:
            self.lr = self._clamp(self._cosine())
        return self.lr

    def restart(self):
        super().restart()
        self.t_cur = self.t_0
        self.t_i = 0


class ReduceOnPlateau(BaseScheduler):
    """Reduce lr by ``factor`` when the metric stops improving.

    A step counts as an improvement when
    ``metric < best - min_delta`` (relative to ``max(|best|, 1)``).
    After ``patience`` non-improving epochs, lr *= factor (floored at
    min_lr) and a ``cooldown`` window suppresses further reductions.
    """

    def __init__(self, base_lr, factor=0.5, patience=5, min_delta=1e-3,
                 cooldown=2, min_lr=0.0):
        super().__init__(base_lr, min_lr)
        if not 0 < factor < 1:
            raise ValueError("factor must be in (0, 1)")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.cooldown = int(cooldown)
        self.best = math.inf
        self.bad_epochs = 0
        self.cooldown_left = 0
        self.n_reductions = 0

    def step(self, metric=None):
        if metric is None:
            raise ValueError("ReduceOnPlateau.step() requires a metric")
        self.epoch += 1
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        if math.isinf(self.best):
            threshold = math.inf  # first metric always counts as improvement
        else:
            threshold = self.best - self.min_delta * max(abs(self.best), 1.0)
        if metric < threshold:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience and self.cooldown_left == 0:
                new_lr = self._clamp(self.lr * self.factor)
                if new_lr < self.lr:  # only count real reductions
                    self.n_reductions += 1
                self.lr = new_lr
                self.bad_epochs = 0
                self.cooldown_left = self.cooldown
        return self.lr

    def restart(self):
        super().restart()
        self.best = math.inf
        self.bad_epochs = 0
        self.cooldown_left = 0
