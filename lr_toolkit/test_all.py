"""Self-tests: schedulers, diagnostics, edge cases. Stdlib unittest only.

Run:  python3 test_all.py  (or: python3 -m unittest test_all -v)
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schedulers import (StepDecay, CosineAnnealingWarmRestarts,
                        ReduceOnPlateau)
from diagnostics import TrainingMonitor, summarize_stability, OK, DIVERGED, \
    PLATEAU, OSCILLATING
from experiment import train, convergence_epoch, edge_cases


class TestStepDecay(unittest.TestCase):
    def test_decay_values(self):
        s = StepDecay(1.0, step_size=10, gamma=0.1)
        for _ in range(10):
            lr = s.step()
        self.assertAlmostEqual(lr, 0.1)
        for _ in range(10):
            lr = s.step()
        self.assertAlmostEqual(lr, 0.01, places=7)

    def test_min_lr_floor(self):
        s = StepDecay(1.0, step_size=1, gamma=0.01, min_lr=0.05)
        for _ in range(10):
            lr = s.step()
        self.assertAlmostEqual(lr, 0.05)

    def test_warm_restart(self):
        s = StepDecay(1.0, step_size=5, gamma=0.5)
        for _ in range(20):
            s.step()
        self.assertLess(s.get_lr(), 1.0)
        s.restart()
        self.assertEqual(s.get_lr(), 1.0)
        self.assertEqual(s.epoch, 0)
        self.assertEqual(s.n_restarts, 1)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            StepDecay(1.0, step_size=0)
        with self.assertRaises(ValueError):
            StepDecay(1.0, gamma=1.5)
        with self.assertRaises(ValueError):
            StepDecay(-1.0)
        with self.assertRaises(ValueError):
            StepDecay(0.1, min_lr=0.2)


class TestCosine(unittest.TestCase):
    def test_half_period_value(self):
        s = CosineAnnealingWarmRestarts(1.0, t_0=10, t_mult=1, min_lr=0.0)
        for _ in range(5):
            lr = s.step()
        self.assertAlmostEqual(lr, 0.5, places=6)

    def test_min_lr_floor(self):
        s = CosineAnnealingWarmRestarts(1.0, t_0=10, t_mult=1, min_lr=0.1)
        for _ in range(9):
            lr = s.step()
        self.assertGreaterEqual(lr, 0.1 - 1e-12)

    def test_automatic_warm_restart(self):
        s = CosineAnnealingWarmRestarts(1.0, t_0=10, t_mult=2)
        for _ in range(10):
            lr = s.step()
        self.assertEqual(lr, 1.0)          # restarted at t_i == t_cur
        self.assertEqual(s.n_restarts, 1)
        self.assertEqual(s.t_cur, 20)      # period grew by t_mult

    def test_manual_restart(self):
        s = CosineAnnealingWarmRestarts(1.0, t_0=10, t_mult=2)
        for _ in range(7):
            s.step()
        s.restart()
        self.assertEqual(s.get_lr(), 1.0)
        self.assertEqual(s.t_i, 0)
        self.assertEqual(s.t_cur, 10)


class TestReduceOnPlateau(unittest.TestCase):
    def test_requires_metric(self):
        s = ReduceOnPlateau(1.0)
        with self.assertRaises(ValueError):
            s.step()

    def test_reduces_after_patience(self):
        s = ReduceOnPlateau(1.0, factor=0.5, patience=3, cooldown=0)
        s.step(1.0)
        for _ in range(3):
            lr = s.step(1.0)  # no improvement
        self.assertAlmostEqual(lr, 0.5)
        self.assertEqual(s.n_reductions, 1)

    def test_no_reduce_while_improving(self):
        s = ReduceOnPlateau(1.0, factor=0.5, patience=3, cooldown=0)
        loss = 10.0
        for _ in range(20):
            loss -= 0.1  # steady improvement above min_delta
            lr = s.step(loss)
        self.assertAlmostEqual(lr, 1.0)
        self.assertEqual(s.n_reductions, 0)

    def test_min_lr_floor_and_cooldown(self):
        s = ReduceOnPlateau(0.1, factor=0.1, patience=2, cooldown=1,
                            min_lr=0.005)
        for _ in range(30):
            lr = s.step(1.0)
        self.assertAlmostEqual(lr, 0.005)

    def test_warm_restart(self):
        s = ReduceOnPlateau(1.0, factor=0.5, patience=2, cooldown=0)
        for _ in range(5):
            s.step(1.0)
        self.assertLess(s.get_lr(), 1.0)
        s.restart()
        self.assertEqual(s.get_lr(), 1.0)
        self.assertEqual(s.best, math.inf)


class TestDiagnostics(unittest.TestCase):
    def feed(self, monitor, losses):
        report = None
        for l in losses:
            report = monitor.update(l)
        return report

    def test_nan_divergence(self):
        m = TrainingMonitor(min_epochs=3)
        m.update(1.0)
        report = m.update(float("nan"))
        self.assertEqual(report.status, DIVERGED)
        self.assertIn("STOP", report.suggestion)

    def test_divergence_from_start(self):
        m = TrainingMonitor(window=10, min_epochs=8)
        losses = [1.0 * (1.5 ** i) for i in range(15)]  # exponential blow-up
        report = self.feed(m, losses)
        self.assertEqual(report.status, DIVERGED)
        self.assertIn("reduce lr", report.suggestion)

    def test_plateau(self):
        m = TrainingMonitor(window=10, min_epochs=8)
        losses = [5.0] * 30  # never improves
        report = self.feed(m, losses)
        self.assertEqual(report.status, PLATEAU)
        self.assertIn("restart", report.suggestion.lower())

    def test_oscillation(self):
        m = TrainingMonitor(window=12, min_epochs=8)
        # zig-zag around 2.0: no net progress, constant sign flips
        losses = [2.0 + (0.1 if i % 2 else -0.1) for i in range(30)]
        report = self.feed(m, losses)
        self.assertEqual(report.status, OSCILLATING)
        self.assertIn("too high", report.suggestion)

    def test_healthy_progress(self):
        m = TrainingMonitor(window=10, min_epochs=8)
        losses = [10.0 * (0.9 ** i) for i in range(30)]
        report = self.feed(m, losses)
        self.assertEqual(report.status, OK)

    def test_warmup_period(self):
        m = TrainingMonitor(min_epochs=10)
        report = m.update(1.0)
        self.assertEqual(report.status, OK)
        self.assertIn("warming up", report.evidence)

    def test_stability_summary(self):
        stats = summarize_stability([1.0, 0.5, 0.25, 0.25, 0.25], tail=3)
        self.assertAlmostEqual(stats["best_loss"], 0.25)
        self.assertAlmostEqual(stats["final_loss"], 0.25)
        self.assertIn("oscillation_ratio", stats)


class TestExperiment(unittest.TestCase):
    def test_train_converges_with_step_decay(self):
        s = StepDecay(0.5, step_size=40, gamma=0.5, min_lr=1e-4)
        losses = train(s, epochs=200)
        self.assertIsNotNone(convergence_epoch(losses))
        self.assertLess(losses[-1], losses[0])

    def test_edge_case_verdicts(self):
        verdicts = edge_cases()
        self.assertEqual(verdicts[0], DIVERGED)
        self.assertEqual(verdicts[1], PLATEAU)
        self.assertEqual(verdicts[2], OSCILLATING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
