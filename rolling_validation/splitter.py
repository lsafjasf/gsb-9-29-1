"""Walk-forward splitters: expanding and sliding windows.

Overlap / purge rules
---------------------
The timeline is always sorted first (stable sort), so shuffled input is
handled and reported indices refer to the *original* input positions.

For fold k, with validation window V_k = [v_start, v_start + horizon):

* embargo (purge band): samples in [v_start - embargo, v_start) belong to
  neither training nor validation;
* expanding mode trains on [0, v_start - embargo);
* sliding mode trains on [v_start - embargo - window, v_start - embargo);
* duplicate-timestamp purge: after the positional cut, every training sample
  whose timestamp is >= the first validation timestamp is removed. Equal
  timestamps straddling the cut are therefore excluded from training, never
  silently reused;
* folds advance by `step`, and step >= horizon is enforced, so validation
  windows never overlap each other (validation samples cannot be scored
  twice). Set step == horizon for back-to-back windows, step > horizon for a
  gap between them.

If a fold has no training data left after purging it is skipped. If every
fold is skipped (e.g. all timestamps identical) split() returns [].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class SplitConfigError(ValueError):
    """Invalid splitter configuration."""


@dataclass(frozen=True)
class SplitConfig:
    mode: str = "expanding"
    min_train: int = 20
    horizon: int = 5
    step: int = 5
    embargo: int = 0
    window: Optional[int] = None


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: Tuple[int, ...]
    val_idx: Tuple[int, ...]
    train_start: float
    train_end: float
    val_start: float
    val_end: float
    n_embargo_dropped: int = 0
    n_overlap_dropped: int = 0


class RollingSplitter:
    def __init__(self, config: SplitConfig):
        self.cfg = self._validate(config)

    @staticmethod
    def _validate(cfg: SplitConfig) -> SplitConfig:
        if cfg.mode not in ("expanding", "sliding"):
            raise SplitConfigError("mode must be 'expanding' or 'sliding'")
        if cfg.min_train < 1:
            raise SplitConfigError("min_train must be >= 1")
        if cfg.horizon < 1:
            raise SplitConfigError("horizon must be >= 1")
        if cfg.step < cfg.horizon:
            raise SplitConfigError(
                "step (%d) must be >= horizon (%d), otherwise validation "
                "windows overlap and samples get scored multiple times"
                % (cfg.step, cfg.horizon)
            )
        if cfg.embargo < 0:
            raise SplitConfigError("embargo must be >= 0")
        if cfg.mode == "sliding":
            if cfg.window is None:
                raise SplitConfigError("sliding mode requires window")
            if cfg.window < 1:
                raise SplitConfigError("sliding window must be >= 1")
        return cfg

    def split(self, timestamps: Sequence[float]) -> List[Fold]:
        n = len(timestamps)
        for i, t in enumerate(timestamps):
            if isinstance(t, float) and math.isnan(t):
                raise SplitConfigError("timestamp at sample #%d is NaN" % i)

        order = sorted(range(n), key=lambda i: (timestamps[i], i))
        sorted_ts = [timestamps[i] for i in order]
        cfg = self.cfg

        folds: List[Fold] = []
        first_val = cfg.min_train + cfg.embargo
        k = 0
        while True:
            vs = first_val + k * cfg.step
            ve = vs + cfg.horizon
            if ve > n:
                break

            train_end_pos = vs - cfg.embargo
            if cfg.mode == "sliding":
                train_start_pos = max(0, train_end_pos - (cfg.window or 0))
            else:
                train_start_pos = 0

            val_start_ts = sorted_ts[vs]
            train_positions = [
                p
                for p in range(train_start_pos, train_end_pos)
                if sorted_ts[p] < val_start_ts
            ]
            n_overlap_dropped = (train_end_pos - train_start_pos) - len(
                train_positions
            )
            if train_positions:
                folds.append(
                    Fold(
                        fold_id=k,
                        train_idx=tuple(order[p] for p in train_positions),
                        val_idx=tuple(order[p] for p in range(vs, ve)),
                        train_start=sorted_ts[train_positions[0]],
                        train_end=sorted_ts[train_positions[-1]],
                        val_start=val_start_ts,
                        val_end=sorted_ts[ve - 1],
                        n_embargo_dropped=cfg.embargo,
                        n_overlap_dropped=n_overlap_dropped,
                    )
                )
            k += 1
        return folds
