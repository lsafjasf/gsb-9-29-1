"""弧长参数化自测：精度、均匀性、往返一致性、边界用例。

运行：python3 test_arc_length.py
"""

import math
import random

from arc_length import ArcLengthParameterization, BezierCurve, PolylineCurve

REL_TOL = 1e-10


def report(name, al: ArcLengthParameterization, n_samples=100, n_roundtrip=200,
           check_roundtrip=True):
    """对一条曲线输出精度/均匀性/往返数据并做断言。"""
    L = al.total_length
    tol = al.tolerance

    # ---- 等弧长均匀性：相邻采样点的实际弧长间距与 L/n 的最大偏差 ----
    if L > 0:
        samples = al.sample_even(n_samples)
        step = L / n_samples
        max_dev = 0.0
        for k in range(n_samples):
            t0, t1 = samples[k][1], samples[k + 1][1]
            ds = al.length_at(t1) - al.length_at(t0)
            max_dev = max(max_dev, abs(ds - step))
        rel_dev = max_dev / step
    else:
        max_dev = rel_dev = 0.0

    # ---- 往返一致性：t -> s -> t' 与 s -> t -> s' ----
    rng = random.Random(20260930)
    max_rt_t = 0.0
    max_rt_s = 0.0
    if check_roundtrip and L > 0:
        for _ in range(n_roundtrip):
            t = rng.random()
            t2 = al.param_at_length(al.length_at(t))
            max_rt_t = max(max_rt_t, abs(t2 - t))
        for _ in range(n_roundtrip):
            s = rng.random() * L
            s2 = al.length_at(al.param_at_length(s))
            max_rt_s = max(max_rt_s, abs(s2 - s))

    print(f"[{name}]")
    print(f"  总弧长 L            = {L:.12g}   (误差上界 tol = {tol:.3g})")
    print(f"  等弧长 {n_samples} 段: 相邻间距最大偏差 = {max_dev:.3g}"
          f"  (相对 {rel_dev:.3g})")
    if check_roundtrip and L > 0:
        print(f"  往返 t->s->t 最大误差 = {max_rt_t:.3g}")
        print(f"  往返 s->t->s 最大误差 = {max_rt_s:.3g}  (tol = {tol:.3g})")
    print()

    # ---- 断言 ----
    if L > 0:
        assert rel_dev < 1e-6, f"{name}: 均匀性不足 rel_dev={rel_dev}"
        if check_roundtrip:
            assert max_rt_t < 1e-6, f"{name}: t 往返误差过大 {max_rt_t}"
            assert max_rt_s < 10 * tol + 1e-9 * max(L, 1.0), \
                f"{name}: s 往返误差过大 {max_rt_s}"
    return L


def main():
    print("=" * 72)
    print("1. 解析解对照（精度验证）")
    print("=" * 72)

    # 直线：精确长度已知
    line = BezierCurve([(0, 0), (3, 4)])
    al_line = ArcLengthParameterization(line, rel_tol=REL_TOL)
    L = report("直线 (0,0)-(3,4)", al_line)
    assert abs(L - 5.0) < 1e-9, L

    # 抛物线 y = x^2, x in [-1,1]（二次 Bezier 精确表示）
    # 解析弧长 = sqrt(5) + asinh(2)/2
    parabola = BezierCurve([(-1, 1), (0, -1), (1, 1)])
    al_par = ArcLengthParameterization(parabola, rel_tol=REL_TOL)
    exact = math.sqrt(5.0) + math.asinh(2.0) / 2.0
    L = report("抛物线 y=x^2 (二次 Bezier)", al_par)
    err = abs(L - exact)
    print(f"  解析弧长 = {exact:.15f}, 数值弧长 = {L:.15f}, 误差 = {err:.3g}")
    assert err < 1e-9, err
    print()

    # 三次 Bezier 近似单位半圆（与 pi 比较，含逼近误差，只验证收敛性）
    k = 4.0 / 3.0 * math.tan(math.pi / 8.0)  # 半圆拆两段
    semi = BezierCurve([(1, 0), (1, k), (k, 1), (0, 1)])
    al_q = ArcLengthParameterization(semi, rel_tol=REL_TOL)
    Lq = al_q.total_length
    print(f"[四分之一圆弧近似] L = {Lq:.12f}, pi/2 = {math.pi/2:.12f}, "
          f"差 = {abs(Lq - math.pi/2):.3g} (含 Bezier 逼近误差)")
    assert abs(Lq - math.pi / 2) < 1e-3
    print()

    # 容差收敛性：tol 缩小 100 倍，抛物线弧长误差应同步下降
    e1 = abs(ArcLengthParameterization(parabola, rel_tol=1e-6).total_length - exact)
    e2 = abs(ArcLengthParameterization(parabola, rel_tol=1e-8).total_length - exact)
    e3 = abs(ArcLengthParameterization(parabola, rel_tol=1e-10).total_length - exact)
    print(f"[容差收敛] rel_tol=1e-6: {e1:.3g}  1e-8: {e2:.3g}  1e-10: {e3:.3g}")
    assert e3 <= e2 <= e1 * 10 and e3 < 1e-9
    print()

    print("=" * 72)
    print("2. 一般曲线：均匀性 + 往返一致性")
    print("=" * 72)

    cubic = BezierCurve([(0, 0), (1, 3), (4, -2), (6, 2)])
    report("普通三次 Bezier", ArcLengthParameterization(cubic, rel_tol=REL_TOL))

    quartic = BezierCurve([(0, 0), (2, 5), (5, -3), (7, 4), (9, 0)])
    report("四次 Bezier", ArcLengthParameterization(quartic, rel_tol=REL_TOL))

    print("=" * 72)
    print("3. 边界用例")
    print("=" * 72)

    # 3a. 零长度曲线：所有控制点重合
    zero = BezierCurve([(2, 2), (2, 2), (2, 2), (2, 2)])
    al_zero = ArcLengthParameterization(zero, rel_tol=REL_TOL)
    print(f"[零长度曲线] L = {al_zero.total_length}")
    assert al_zero.total_length == 0.0
    assert al_zero.param_at_length(0.0) == 0.0
    pts = al_zero.sample_even(10)
    assert all(p[2] == (2.0, 2.0) for p in pts)
    try:
        al_zero.param_at_length(1.0)
        raise AssertionError("应当对超程弧长抛异常")
    except ValueError:
        pass
    print("  sample_even 全部返回 (2,2)；超程弧长正确抛出 ValueError\n")

    # 3b. 部分控制点重合（前三个重合）：退化为线段，t=0 处速度为 0
    deg = BezierCurve([(0, 0), (0, 0), (0, 0), (5, 0)])
    al_deg = ArcLengthParameterization(deg, rel_tol=REL_TOL)
    L = report("部分控制点重合（退化为线段）", al_deg)
    assert abs(L - 5.0) < 1e-9, L

    # 3c. 自交曲线（环形三次 Bezier，起点=终点，中途自交）
    loop = BezierCurve([(0, 0), (3, 5), (-3, 5), (0, 0)])
    al_loop = ArcLengthParameterization(loop, rel_tol=REL_TOL)
    report("自交曲线（环形）", al_loop)
    # 自交点两侧采样仍应均匀，且曲线确实回到原点
    assert al_loop.curve.eval(0.0) == al_loop.curve.eval(1.0) == (0.0, 0.0)

    # 3d. 超长曲线：1000 段折线 zigzag，总长 ~ 1e6
    n_seg = 1000
    pts = [(float(i), (1e3 if i % 2 else 0.0)) for i in range(n_seg + 1)]
    poly = PolylineCurve(pts)
    al_poly = ArcLengthParameterization(poly, rel_tol=REL_TOL)
    exact_poly = n_seg * math.hypot(1.0, 1e3)
    L = report(f"超长折线 ({n_seg} 段, L≈{exact_poly:.3g})", al_poly,
               n_samples=200)
    assert abs(L - exact_poly) <= al_poly.tolerance * 2, (L, exact_poly)
    # 端点反查
    assert al_poly.param_at_length(0.0) == 0.0
    assert al_poly.param_at_length(L) == 1.0

    # 3e. 超长 Bezier（坐标尺度 1e6，验证相对容差生效）
    big = BezierCurve([(0, 0), (1e6, 1e6), (2e6, -1e6), (3e6, 0)])
    al_big = ArcLengthParameterization(big, rel_tol=REL_TOL)
    report("超长 Bezier（尺度 1e6）", al_big)

    print("=" * 72)
    print("全部断言通过 ✔")
    print("=" * 72)


if __name__ == "__main__":
    main()
