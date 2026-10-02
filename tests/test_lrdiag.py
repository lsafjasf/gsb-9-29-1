"""lrdiag 单元测试：调度器、诊断器、自愈回滚、热启动、边界用例。

运行方式：python3 -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrdiag import (
    CosineDecay,
    GDTrainer,
    LossDiagnoser,
    PlateauDecay,
    Quadratic,
    StepDecay,
    assert_rollback,
)
from lrdiag.diagnostics import DIVERGING, HEALTHY, OSCILLATING, PLATEAU


class TestStepDecay(unittest.TestCase):
    def test_lr_sequence_and_floor(self):
        sched = StepDecay(base_lr=1.0, step_size=2, gamma=0.5, min_lr=0.1)
        lrs = [sched.step() for _ in range(9)]
        # 1,1 | 0.5,0.5 | 0.25,0.25 | 0.125,0.125 | 0.1(被下限钳制)
        self.assertEqual(lrs, [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125, 0.1])

    def test_reduce_lowers_base(self):
        sched = StepDecay(base_lr=1.0, step_size=10, gamma=0.5, min_lr=0.01)
        sched.step()
        post = sched.reduce(0.5)
        self.assertAlmostEqual(post, 0.5)
        self.assertAlmostEqual(sched.base_lr, 0.5)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            StepDecay(base_lr=0.0)
        with self.assertRaises(ValueError):
            StepDecay(base_lr=1.0, min_lr=2.0)


class TestCosineDecay(unittest.TestCase):
    def test_endpoints_and_monotonic(self):
        sched = CosineDecay(base_lr=1.0, total_epochs=100, min_lr=0.01)
        lrs = [sched.step() for _ in range(101)]
        self.assertAlmostEqual(lrs[0], 1.0)
        self.assertAlmostEqual(lrs[100], 0.01, places=6)
        for a, b in zip(lrs, lrs[1:]):
            self.assertGreaterEqual(a, b - 1e-12)
        # 超过周期后保持 min_lr
        self.assertAlmostEqual(sched.step(), 0.01)


class TestPlateauDecay(unittest.TestCase):
    def test_holds_then_drops(self):
        sched = PlateauDecay(base_lr=0.5, factor=0.5, patience=3,
                             threshold=1e-4, min_lr=0.05)
        self.assertEqual(sched.step(), 0.5)  # 首轮无 metric
        loss = 10.0
        for _ in range(5):
            loss *= 0.5
            sched.step(loss)  # 持续改善，不降
        self.assertEqual(sched.current_lr, 0.5)
        for _ in range(4):  # patience=3，第 4 个坏 epoch 触发降档
            sched.step(loss)
        self.assertAlmostEqual(sched.current_lr, 0.25)

    def test_min_lr_floor(self):
        sched = PlateauDecay(base_lr=0.4, factor=0.5, patience=0,
                             threshold=0.0, min_lr=0.1)
        sched.step()
        for _ in range(10):
            sched.step(1.0)  # 一直不改善
        self.assertAlmostEqual(sched.current_lr, 0.1)

    def test_none_metric_keeps_lr(self):
        # 首轮以及回滚后的续跑轮允许无 metric：仅推进 epoch、保持步长
        sched = PlateauDecay(base_lr=0.1, factor=0.5, patience=1)
        self.assertEqual(sched.step(), 0.1)
        self.assertEqual(sched.step(None), 0.1)
        self.assertEqual(sched.step(1.0), 0.1)


class TestSchedulerWarmStart(unittest.TestCase):
    def test_warm_start_continuity(self):
        for make in (
            lambda: StepDecay(0.8, step_size=5, gamma=0.5, min_lr=0.01),
            lambda: CosineDecay(0.8, total_epochs=40, min_lr=0.01),
            lambda: PlateauDecay(0.8, factor=0.5, patience=2, min_lr=0.01),
        ):
            sched = make()
            for i in range(12):
                sched.step(1.0 / (i + 1) if sched.type_name == "plateau" else None)
            state = sched.state_dict()

            restored = make()
            restored.load_state_dict(state)
            for i in range(8):
                metric = 1.0 / (13 + i) if sched.type_name == "plateau" else None
                a = sched.step(metric)
                b = restored.step(metric)
                self.assertEqual(a, b, f"{sched.type_name} 热启动后轨迹不一致")

    def test_type_mismatch_rejected(self):
        sched = CosineDecay(0.5, total_epochs=10)
        state = StepDecay(0.5).state_dict()
        with self.assertRaises(ValueError):
            sched.load_state_dict(state)


class TestDiagnoser(unittest.TestCase):
    def feed(self, values, **kwargs):
        d = LossDiagnoser(**kwargs)
        diag = None
        for v in values:
            diag = d.update(v)
        return diag

    def test_diverging(self):
        diag = self.feed([1.2 ** i for i in range(25)])
        self.assertEqual(diag.status, DIVERGING)
        self.assertIn("发散", diag.reason)
        self.assertIn("回滚", diag.action)

    def test_nan_is_diverging(self):
        diag = self.feed([1.0] * 19 + [float("nan")])
        self.assertEqual(diag.status, DIVERGING)
        self.assertIn("NaN", diag.reason)

    def test_plateau(self):
        diag = self.feed([5.0 + 1e-6 * i for i in range(25)])
        self.assertEqual(diag.status, PLATEAU)
        self.assertIn("平台", diag.action)

    def test_oscillating(self):
        diag = self.feed([1.0, 2.0] * 13)
        self.assertEqual(diag.status, OSCILLATING)
        self.assertIn("降低步长", diag.action)

    def test_healthy_descent(self):
        diag = self.feed([10.0 * 0.9 ** i for i in range(25)])
        self.assertEqual(diag.status, HEALTHY)

    def test_insufficient_window(self):
        d = LossDiagnoser(window=20)
        diag = d.update(100.0)
        self.assertEqual(diag.status, HEALTHY)
        self.assertIn("暂不判定", diag.reason)


class TestSelfHealing(unittest.TestCase):
    """一开始就发散：必须回滚到历史最优并降步长，最终收敛。"""

    def test_diverges_from_start_then_heals(self):
        objective = Quadratic(axes=(1.0, 0.1), seed=7)
        sched = StepDecay(base_lr=3.0, step_size=1000, gamma=0.5, min_lr=1e-6)
        trainer = GDTrainer(objective, sched, tol=1e-6)
        result = trainer.run(max_epochs=3000)

        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.rollback_count, 1)
        for event in result.rollback_events:
            # 回滚断言：损失确实变坏、步长确实被压低
            self.assertGreaterEqual(event.bad_loss, event.best_true_loss)
            self.assertLess(event.post_lr, event.pre_lr)
            self.assertGreaterEqual(event.post_lr, sched.min_lr)
        # 自愈后调度器基准步长已被永久压低
        self.assertLess(sched.base_lr, 3.0)
        self.assertLessEqual(result.best_true_loss, 1e-6)

    def test_rollback_assertion_function(self):
        from lrdiag.trainer import RollbackEvent
        event = RollbackEvent(epoch=5, pre_lr=1.0, post_lr=0.5,
                              bad_loss=10.0, best_true_loss=2.0, reason="t")
        # 正常情形不抛异常
        assert_rollback(event, [9.0, 9.0], [1.0, 2.0], [1.0, 2.0])
        # 回滚后参数与最优点不一致 -> 断言失败
        with self.assertRaises(AssertionError):
            assert_rollback(event, [9.0, 9.0], [1.0, 3.0], [1.0, 2.0])
        # 步长未降低 -> 断言失败
        bad = RollbackEvent(epoch=5, pre_lr=1.0, post_lr=2.0,
                            bad_loss=10.0, best_true_loss=2.0, reason="t")
        with self.assertRaises(AssertionError):
            assert_rollback(bad, [9.0], [1.0], [1.0])

    def test_min_lr_stuck_marks_unstable(self):
        """步长已到下限仍发散：连续回滚无效后标记 unstable 并中止。"""
        objective = Quadratic(axes=(1.0, 0.01), seed=3)
        sched = StepDecay(base_lr=5.0, step_size=100, gamma=1.0, min_lr=4.0)
        trainer = GDTrainer(objective, sched, max_stuck_rollbacks=3)
        result = trainer.run(max_epochs=500)
        self.assertTrue(result.unstable)
        self.assertFalse(result.converged)
        self.assertGreaterEqual(result.final_lr, sched.min_lr)


class TestPlateauAndNoiseScenarios(unittest.TestCase):
    def test_long_no_descent_triggers_plateau_cut(self):
        """长时间不下降：诊断判平台期，PlateauDecay 自动降档。"""
        objective = Quadratic(axes=(1.0, 1.0), seed=1)
        sched = PlateauDecay(base_lr=1e-6, factor=0.5, patience=10,
                             threshold=1e-4, min_lr=1e-8)
        trainer = GDTrainer(objective, sched, tol=1e-12)
        result = trainer.run(max_epochs=80)
        self.assertGreater(result.plateau_count, 0)
        self.assertLess(sched.current_lr, 1e-6)
        self.assertGreaterEqual(sched.current_lr, sched.min_lr)

    def test_heavy_noise_no_crash(self):
        """噪声极大：出现震荡/平台期判定，但不误判发散、不崩溃。"""
        objective = Quadratic(axes=(1.0, 1.0), noise_std=0.5, seed=11)
        sched = CosineDecay(base_lr=0.05, total_epochs=300, min_lr=1e-4)
        trainer = GDTrainer(objective, sched, tol=1e-9)
        result = trainer.run(max_epochs=300)
        self.assertEqual(result.rollback_count, 0)
        self.assertGreater(result.plateau_count + result.oscillating_count, 0)
        self.assertLess(result.best_true_loss, objective.true_value(objective.start))


class TestTrainerWarmStart(unittest.TestCase):
    def test_resume_matches_single_run(self):
        """训练一半存档 -> 新实例热启动续跑，轨迹与一次性跑完逐点一致。"""
        def make_trainer():
            objective = Quadratic(axes=(1.0, 0.01), seed=5)
            sched = CosineDecay(base_lr=0.4, total_epochs=600, min_lr=1e-6)
            return GDTrainer(objective, sched, tol=1e-9)

        full = make_trainer()
        full_result = full.run(max_epochs=400)

        first = make_trainer()
        first.run(max_epochs=150)
        state = first.state_dict()

        resumed = make_trainer()
        resumed.load_state_dict(state)
        resumed_result = resumed.run(max_epochs=400)

        self.assertEqual(full_result.true_losses, resumed_result.true_losses)
        self.assertEqual(full_result.lrs, resumed_result.lrs)
        self.assertEqual(full_result.epochs, resumed_result.epochs)


if __name__ == "__main__":
    unittest.main()
