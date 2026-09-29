"""Compare schedulers on the SAME objective + edge-case diagnostics.

Objective: noisy quadratic  f(x) = 0.5 * ||x - x*||^2  in 10 dims,
optimized by SGD with Gaussian gradient noise. Deterministic via seeds.

Run:  python3 experiment.py
"""

import math
import random

from schedulers import StepDecay, CosineAnnealingWarmRestarts, ReduceOnPlateau
from diagnostics import TrainingMonitor, summarize_stability

DIM = 10
EPOCHS = 200
CONVERGE_TOL = 1e-3


def make_objective(seed=0):
    rng = random.Random(seed)
    x_star = [rng.uniform(-2, 2) for _ in range(DIM)]

    def loss(x):
        return 0.5 * sum((xi - si) ** 2 for xi, si in zip(x, x_star))

    def grad(x, rng, noise):
        return [(xi - si) + rng.gauss(0, noise) for xi, si in zip(x, x_star)]

    return loss, grad, x_star


def train(scheduler, noise=0.1, epochs=EPOCHS, seed=42, monitor=None):
    """Plain SGD on the noisy quadratic. Returns the loss history."""
    loss_fn, grad_fn, _ = make_objective(seed=0)
    rng = random.Random(seed)
    x = [rng.uniform(-4, 4) for _ in range(DIM)]
    losses = []
    for _ in range(epochs):
        losses.append(loss_fn(x))
        lr = scheduler.get_lr()
        g = grad_fn(x, rng, noise)
        x = [xi - lr * gi for xi, gi in zip(x, g)]
        scheduler.step(losses[-1])
        if monitor is not None:
            monitor.update(losses[-1])
    return losses


def convergence_epoch(losses, tol=CONVERGE_TOL):
    for i, l in enumerate(losses):
        if l < tol:
            return i
    return None


def compare_schedulers():
    print("=" * 88)
    print("Scheduler comparison on the SAME noisy quadratic (dim=%d, noise=0.1, seed=42)"
          % DIM)
    print("=" * 88)
    base_lr = 0.5
    scheds = {
        "constant(lr=0.5)": None,  # placeholder, handled below
        "step(0.5, /2 every 40)": StepDecay(base_lr, step_size=40, gamma=0.5, min_lr=1e-4),
        "cosine(T0=50, Tmult=2)": CosineAnnealingWarmRestarts(base_lr, t_0=50, t_mult=2, min_lr=1e-4),
        "plateau(x0.5, pat=8)": ReduceOnPlateau(base_lr, factor=0.5, patience=8, min_lr=5e-3),
    }

    class _Constant:
        def get_lr(self):
            return base_lr

        def step(self, metric=None):
            return base_lr

    scheds["constant(lr=0.5)"] = _Constant()

    header = "%-24s %10s %12s %12s %10s %10s %10s" % (
        "scheduler", "conv@epoch", "final_loss", "best_loss", "tail_std",
        "osc_ratio", "max_loss")
    print(header)
    print("-" * 88)
    results = {}
    for name, sched in scheds.items():
        losses = train(sched)
        stats = summarize_stability(losses)
        conv = convergence_epoch(losses)
        results[name] = (conv, stats, losses)
        print("%-24s %10s %12.3e %12.3e %10.3e %10.2f %10.3e" % (
            name,
            conv if conv is not None else "never",
            stats["final_loss"], stats["best_loss"], stats["tail_std"],
            stats["oscillation_ratio"], stats["max_loss"]))
    print()
    return results


def edge_cases():
    print("=" * 88)
    print("Edge cases: diagnostics on pathological runs")
    print("=" * 88)

    class _Fixed:
        def __init__(self, lr):
            self.lr = lr

        def get_lr(self):
            return self.lr

        def step(self, metric=None):
            return self.lr

    cases = [
        ("1) diverges from the start (lr=50, lr>2/L)",
         _Fixed(50.0), 0.2, 60),
        ("2) never improves (lr=1e-6, frozen)",
         _Fixed(1e-6), 0.2, 60),
        ("3) extremely noisy gradients (sigma=5, lr=0.5)",
         _Fixed(0.5), 5.0, 60),
    ]
    verdicts = []
    for title, sched, noise, epochs in cases:
        monitor = TrainingMonitor(window=15, min_epochs=12)
        losses = train(sched, noise=noise, epochs=epochs, monitor=monitor)
        report = monitor.diagnose()
        verdicts.append(report.status)
        print("\n%s" % title)
        print("  first loss=%.4g  last loss=%.4g" % (losses[0], losses[-1]))
        print("  verdict   : %s" % report.status)
        print("  evidence  : %s" % report.evidence)
        print("  suggestion: %s" % report.suggestion)
    print()
    return verdicts


def plateau_scheduler_reacts():
    """Show ReduceOnPlateau actually reacting to a stalled run."""
    print("=" * 88)
    print("ReduceOnPlateau reacting to a plateau (live lr trace)")
    print("=" * 88)
    sched = ReduceOnPlateau(0.5, factor=0.5, patience=6, min_lr=5e-3)
    losses = train(sched, noise=0.1, epochs=80)
    conv = convergence_epoch(losses)
    print("  reductions=%d  final_lr=%.2e  final_loss=%.3e  converged@%s"
          % (sched.n_reductions, sched.get_lr(), losses[-1],
             conv if conv is not None else "never"))
    print()


if __name__ == "__main__":
    compare_schedulers()
    plateau_scheduler_reacts()
    edge_cases()
