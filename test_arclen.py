"""arclen 库自测：精度、均匀性、往返一致性、边界用例。

运行：python3 test_arclen.py   （或 python3 -m unittest -v）
"""

import math
import random
import unittest

from arclen import ArcLengthMap, cubic_bezier


def spacing_report(amap, n):
    """等弧长 n 等分后，相邻段实际弧长相对均值的偏差统计。"""
    ts = amap.uniform_parameters(n)
    seg = [amap.length(ts[i], ts[i + 1]) for i in range(n)]
    mean = amap.total_length / n
    max_dev = max(abs(x - mean) for x in seg)
    return mean, max_dev, max_dev / mean if mean else 0.0


def naive_spacing_report(curve, amap, n):
    """对照：按参数等分（旧做法）的相邻段弧长偏差。"""
    t0, t1 = amap._t0, amap._t1
    ts = [t0 + (t1 - t0) * k / n for k in range(n + 1)]
    seg = [amap.length(ts[i], ts[i + 1]) for i in range(n)]
    mean = sum(seg) / n
    return max(abs(x - mean) for x in seg) / mean if mean else 0.0


def circle(r, t0=0.0, t1=2.0 * math.pi):
    curve = lambda t: (r * math.cos(t), r * math.sin(t))
    deriv = lambda t: (-r * math.sin(t), r * math.cos(t))
    return curve, deriv, t0, t1


def figure_eight():
    # 自交曲线（8 字），t in [0, 2pi]
    curve = lambda t: (math.sin(t), math.sin(t) * math.cos(t))
    deriv = lambda t: (math.cos(t), math.cos(2.0 * t))
    return curve, deriv, 0.0, 2.0 * math.pi


class TestAccuracy(unittest.TestCase):
    def test_line_exact(self):
        curve = lambda t: (3.0 * t, 4.0 * t)
        deriv = lambda t: (3.0, 4.0)
        m = ArcLengthMap(curve, 0.0, 1.0, deriv, tol=1e-12)
        self.assertAlmostEqual(m.total_length, 5.0, places=12)
        self.assertAlmostEqual(m.length(0.2, 0.7), 2.5, places=10)

    def test_circle_length(self):
        curve, deriv, t0, t1 = circle(2.0)
        m = ArcLengthMap(curve, t0, t1, deriv, tol=1e-10)
        exact = 4.0 * math.pi
        err = abs(m.total_length - exact)
        print(f"\n[精度] 圆周长 exact={exact:.15f} got={m.total_length:.15f} "
              f"err={err:.3e} (上界 eps={m.abs_tol:.3e})")
        self.assertLessEqual(err, 10.0 * m.abs_tol)

    def test_no_derivative_finite_difference(self):
        curve = lambda t: (math.cos(t), math.sin(t))
        m = ArcLengthMap(curve, 0.0, math.pi, tol=1e-8)
        self.assertAlmostEqual(m.total_length, math.pi, places=5)


class TestUniformity(unittest.TestCase):
    def test_circle_uniform(self):
        curve, deriv, t0, t1 = circle(1.0)
        m = ArcLengthMap(curve, t0, t1, deriv, tol=1e-10)
        n = 36
        mean, max_dev, rel = spacing_report(m, n)
        print(f"\n[均匀性] 圆 n={n} 均值={mean:.12f} "
              f"相邻间距最大偏差={max_dev:.3e} (相对 {rel:.3e})")
        self.assertLessEqual(rel, 1e-6)

    def test_bezier_uniform_vs_naive(self):
        # 控制点挤在一端的三次贝塞尔：参数等分会严重不均
        curve, deriv = cubic_bezier((0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (1.0, 1.0))
        m = ArcLengthMap(curve, 0.0, 1.0, deriv, tol=1e-10)
        n = 20
        mean, max_dev, rel = spacing_report(m, n)
        naive_rel = naive_spacing_report(curve, m, n)
        print(f"\n[均匀性] 贝塞尔(控制点密集) n={n} 均值={mean:.12f}")
        print(f"         弧长等分: 相邻间距最大偏差={max_dev:.3e} (相对 {rel:.3e})")
        print(f"         参数等分(对照): 相对偏差={naive_rel:.3e}")
        self.assertLessEqual(rel, 1e-6)
        self.assertGreater(naive_rel, 0.1)  # 确认对照组确实不均


class TestRoundTrip(unittest.TestCase):
    def check_roundtrip(self, m, label):
        rng = random.Random(42)
        L = m.total_length
        worst_s = 0.0
        for _ in range(200):
            s = rng.uniform(0.0, L)
            t = m.parameter_at(s)
            err = abs(m.length(m._t0, t) - s)
            worst_s = max(worst_s, err)
            self.assertLessEqual(err, 10.0 * m.abs_tol)
        # 端点
        self.assertEqual(m.parameter_at(0.0), m._t0)
        self.assertEqual(m.parameter_at(L), m._t1)
        # t -> s -> t 往返
        worst_t = 0.0
        span = m._t1 - m._t0
        for _ in range(200):
            t = rng.uniform(m._t0, m._t1)
            t2 = m.parameter_at(m.length(m._t0, t))
            worst_t = max(worst_t, abs(t2 - t))
        print(f"\n[往返] {label}: s->t->s 最大误差={worst_s:.3e} "
              f"(上界 eps={m.abs_tol:.3e}), t->s->t 最大参数误差={worst_t:.3e} (跨度 {span:.3g})")

    def test_circle_roundtrip(self):
        curve, deriv, t0, t1 = circle(1.0)
        self.check_roundtrip(ArcLengthMap(curve, t0, t1, deriv), "圆")

    def test_bezier_roundtrip(self):
        curve, deriv = cubic_bezier((0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (1.0, 1.0))
        self.check_roundtrip(ArcLengthMap(curve, 0.0, 1.0, deriv), "贝塞尔")

    def test_figure_eight_roundtrip(self):
        curve, deriv, t0, t1 = figure_eight()
        self.check_roundtrip(ArcLengthMap(curve, t0, t1, deriv), "自交8字")


class TestEdgeCases(unittest.TestCase):
    def test_zero_length_curve(self):
        curve = lambda t: (1.0, 2.0)
        deriv = lambda t: (0.0, 0.0)
        m = ArcLengthMap(curve, 0.0, 1.0, deriv)
        self.assertEqual(m.total_length, 0.0)
        self.assertEqual(m.parameter_at(0.0), 0.0)
        with self.assertRaises(ValueError):
            m.parameter_at(1e-6)
        pts = m.uniform_points(8)  # 不崩溃，全部落在同一点
        self.assertTrue(all(p == (1.0, 2.0) for p in pts))
        print("\n[边界] 零长度曲线: L=0, s=0 可达, s>0 抛 ValueError, 取点不崩溃")

    def test_coincident_control_points(self):
        curve, deriv = cubic_bezier((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
        m = ArcLengthMap(curve, 0.0, 1.0, deriv)
        self.assertEqual(m.total_length, 0.0)
        self.assertEqual(m.parameter_at(0.0), 0.0)
        with self.assertRaises(ValueError):
            m.parameter_at(0.5)
        print("[边界] 控制点全重合: L=0, 行为同零长度曲线")

    def test_self_intersecting(self):
        curve, deriv, t0, t1 = figure_eight()
        m = ArcLengthMap(curve, t0, t1, deriv, tol=1e-10)
        # 8 字总长 = 2 个对称瓣，数值参考值（高精度对照）：约 6.0972237...
        self.assertGreater(m.total_length, 6.0)
        self.assertLess(m.total_length, 6.2)
        # 交点处（t=0 与 t=pi 同为原点）两侧弧长各自累计，不回退
        s_quarter = m.length(0.0, math.pi / 2)
        s_half = m.length(0.0, math.pi)
        self.assertAlmostEqual(s_half, 2.0 * s_quarter, places=8)
        n = 40
        mean, max_dev, rel = spacing_report(m, n)
        print(f"\n[边界] 自交8字: L={m.total_length:.10f}, 等分 n={n} "
              f"相邻间距最大相对偏差={rel:.3e}")
        self.assertLessEqual(rel, 1e-6)

    def test_very_long_curve(self):
        scale = 1e7
        curve = lambda t: (scale * math.cos(t), scale * math.sin(t))
        deriv = lambda t: (-scale * math.sin(t), scale * math.cos(t))
        m = ArcLengthMap(curve, 0.0, 2.0 * math.pi, deriv, tol=1e-10)
        exact = 2.0 * math.pi * scale
        rel_err = abs(m.total_length - exact) / exact
        # 往返在巨大坐标下仍一致
        s = 0.37 * m.total_length
        t = m.parameter_at(s)
        rt_err = abs(m.length(0.0, t) - s) / m.total_length
        print(f"\n[边界] 超长曲线(R=1e7): 周长相对误差={rel_err:.3e}, "
              f"往返相对误差={rt_err:.3e}")
        self.assertLessEqual(rel_err, 1e-8)
        self.assertLessEqual(rt_err, 1e-8)

    def test_out_of_range_raises(self):
        curve, deriv, t0, t1 = circle(1.0)
        m = ArcLengthMap(curve, t0, t1, deriv)
        with self.assertRaises(ValueError):
            m.parameter_at(-1.0)
        with self.assertRaises(ValueError):
            m.parameter_at(m.total_length + 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
