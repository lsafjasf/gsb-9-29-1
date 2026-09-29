"""Training diagnostics: detect divergence, plateau and oscillation.

Feed per-epoch losses into ``TrainingMonitor.update()``; it returns a
``Report`` describing the current health of the run, the numeric evidence
behind the verdict, and a suggested action.
"""

import math
from collections import namedtuple

Report = namedtuple("Report", ["status", "evidence", "suggestion"])

OK = "ok"
DIVERGED = "diverged"
PLATEAU = "plateau"
OSCILLATING = "oscillating"


class TrainingMonitor:
    """Stateful monitor over a loss history.

    Parameters
    ----------
    window:          size of the trailing analysis window
    diverge_factor:  loss > diverge_factor * baseline  -> diverged
    rise_frac:       fraction of rising steps in window that, together with
                     a net rise, confirms divergence
    plateau_rtol:    relative best-loss improvement below which the run is
                     considered stalled
    osc_ratio:       fraction of sign flips in loss deltas that marks
                     oscillation (only when the run is not making progress)
    min_epochs:      don't judge before this many epochs
    """

    def __init__(self, window=15, diverge_factor=3.0, rise_frac=0.7,
                 plateau_rtol=1e-3, osc_ratio=0.5, min_epochs=10):
        self.window = int(window)
        self.diverge_factor = float(diverge_factor)
        self.rise_frac = float(rise_frac)
        self.plateau_rtol = float(plateau_rtol)
        self.osc_ratio = float(osc_ratio)
        self.min_epochs = int(min_epochs)
        self.losses = []
        self._baseline = None

    # ------------------------------------------------------------------ api
    def update(self, loss):
        """Record one epoch's loss and return a Report."""
        self.losses.append(float(loss))
        return self.diagnose()

    def diagnose(self):
        losses = self.losses
        n = len(losses)

        if any(math.isnan(x) or math.isinf(x) for x in losses):
            bad = next(x for x in losses if math.isnan(x) or math.isinf(x))
            return Report(
                DIVERGED,
                "loss became %s at epoch %d" % (bad, losses.index(bad)),
                "STOP: lr is far too large or gradients exploded; "
                "restart with lr x 0.1 and/or gradient clipping",
            )

        if n < self.min_epochs:
            return Report(OK, "warming up (%d/%d epochs)" % (n, self.min_epochs),
                          "keep training")

        if self._baseline is None:
            head = losses[: max(3, self.window // 3)]
            self._baseline = sorted(head)[len(head) // 2]  # median, robust

        recent = losses[-self.window:]
        verdict = self._check_diverged(recent)
        if verdict is None:
            verdict = self._check_plateau_or_oscillation(recent)
        return verdict

    # -------------------------------------------------------------- checks
    def _check_diverged(self, recent):
        cur = recent[-1]
        base = max(self._baseline, 1e-12)
        deltas = [b - a for a, b in zip(recent, recent[1:])]
        rises = sum(1 for d in deltas if d > 0)
        rise_frac = rises / max(len(deltas), 1)
        net_rise = recent[-1] - recent[0]

        if cur > self.diverge_factor * base or (
            net_rise > 0 and rise_frac >= self.rise_frac
            and cur > 1.5 * base
        ):
            evidence = (
                "loss=%.6g vs baseline=%.6g (%.3gx); %d/%d of last steps rose"
                % (cur, self._baseline, cur / base, rises, len(deltas))
            )
            return Report(
                DIVERGED, evidence,
                "reduce lr (x0.1~x0.5); if already small, check gradients "
                "and enable clipping; consider restarting from last good "
                "checkpoint",
            )
        return None

    def _check_plateau_or_oscillation(self, recent):
        scale = max(abs(self._baseline), 1e-12)
        best_recent = min(recent)
        best_before = min(self.losses[:-self.window]) if len(self.losses) > self.window else self._baseline
        rel_improve = (best_before - best_recent) / scale

        deltas = [b - a for a, b in zip(recent, recent[1:])]
        if len(deltas) >= 4:
            signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
            flips = sum(1 for a, b in zip(signs, signs[1:]) if a * b < 0)
            flip_ratio = flips / (len(signs) - 1)
            mean_abs_delta = sum(abs(d) for d in deltas) / len(deltas)
        else:
            flip_ratio, mean_abs_delta = 0.0, 0.0

        stalled = rel_improve < self.plateau_rtol

        if stalled and flip_ratio >= self.osc_ratio:
            evidence = (
                "best-loss improvement %.2e over last %d epochs; "
                "%.0f%% of steps flip direction, mean|delta|=%.3g"
                % (rel_improve, len(recent), 100 * flip_ratio, mean_abs_delta)
            )
            return Report(
                OSCILLATING, evidence,
                "lr is too high for the noise level: reduce lr (x0.3~x0.5), "
                "or average gradients / increase batch size; cosine or "
                "plateau scheduling helps",
            )
        if stalled:
            evidence = (
                "best-loss improvement %.2e over last %d epochs (rtol=%.1e)"
                % (rel_improve, len(recent), self.plateau_rtol)
            )
            return Report(
                PLATEAU, evidence,
                "decay lr (reduce-on-plateau) to fine-tune, or trigger a "
                "warm restart to escape the basin",
            )
        return Report(
            OK,
            "improving: best loss down %.2e over last %d epochs"
            % (rel_improve, len(recent)),
            "keep training",
        )


def summarize_stability(losses, tail=20):
    """Stability metrics for a finished run (used by the comparison)."""
    tail_losses = losses[-tail:] if len(losses) >= tail else losses
    mean = sum(tail_losses) / len(tail_losses)
    var = sum((x - mean) ** 2 for x in tail_losses) / len(tail_losses)
    deltas = [b - a for a, b in zip(losses, losses[1:])]
    flips = 0
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in deltas]
    for a, b in zip(signs, signs[1:]):
        if a * b < 0:
            flips += 1
    flip_ratio = flips / max(len(signs) - 1, 1) if signs else 0.0
    return {
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "tail_mean": mean,
        "tail_std": math.sqrt(var),
        "oscillation_ratio": flip_ratio,
        "max_loss": max(losses),
    }
