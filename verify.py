"""解析解对拍与收敛阶验证, 打印误差数据表。

运行: python3 verify.py            # 打印到终端
      python3 verify.py > verify_output.txt

对拍问题（均有解析解）:
  A. 两端 0 度定温, u0 = sin(pi x),  u = sin(pi x) e^{-pi^2 t}
  B. 两端绝热,       u0 = 0.5+cos(pi x), u = 0.5 + cos(pi x) e^{-pi^2 t}

理论阶:
  显式 FTCS:  O(dt) + O(dx^2)
  隐式 BE :   O(dt) + O(dx^2)
  => 固定 dx 加密 dt: 一阶;  固定 r 同步加密: 二阶
"""

import math

from heat1d import DIRICHLET, NEUMANN, energy, solve

ALPHA = 1.0
L = 1.0
T_FINAL = 0.05


def exact_a(x, t):
    return math.sin(math.pi * x) * math.exp(-math.pi * math.pi * t)


def exact_b(x, t):
    return 0.5 + math.cos(math.pi * x) * math.exp(-math.pi * math.pi * t)


def run(scheme, n, dt, bc, u0_fn):
    dx = L / n
    steps = round(T_FINAL / dt)
    dt = T_FINAL / steps  # 对齐末时刻
    u0 = [u0_fn(i * dx) for i in range(n + 1)]
    u = solve(u0, ALPHA, dx, dt, steps,
              bc_left=bc, bc_right=bc, scheme=scheme)
    return u, dx, steps * dt


def linf_error(u, exact, t):
    n = len(u) - 1
    return max(abs(u[i] - exact(i * L / n, t)) for i in range(n + 1))


def order(e1, e2):
    return math.log(e1 / e2) / math.log(2.0)


def table_time_refinement_explicit(bc, u0_fn, label):
    """固定 dx, 加密 dt, 与同网格超细 dt 参考解比较（消去空间误差）-> 期望一阶。"""
    n, t_final = 200, 0.01
    dx = L / n
    dt_ref = 1e-5 / 16.0
    steps_ref = round(t_final / dt_ref)
    u0 = [u0_fn(i * dx) for i in range(n + 1)]
    u_ref = solve(u0, ALPHA, dx, t_final / steps_ref, steps_ref,
                  bc_left=bc, bc_right=bc, scheme="explicit")
    print("  [%s] 固定 dx = 1/%d, 加密 dt, 参考解 dt=%.2e（期望阶 ~1）"
          % (label, n, t_final / steps_ref))
    print("  %-12s %-14s %s" % ("dt", "Linf 误差", "阶"))
    prev = None
    for k in range(4):
        dt = 1e-5 / 2 ** k
        steps = round(t_final / dt)
        u = solve(u0, ALPHA, dx, t_final / steps, steps,
                  bc_left=bc, bc_right=bc, scheme="explicit")
        e = max(abs(a - b) for a, b in zip(u, u_ref))
        print("  %-12.3e %-14.6e %s" % (t_final / steps, e,
              "-" if prev is None else "%.2f" % order(prev, e)))
        prev = e
    print()


def table_time_refinement_implicit(bc, exact, u0_fn, label):
    """固定 dx (N=1024, 空间误差可忽略), 加密 dt, 对拍解析解 -> 期望一阶。"""
    n = 1024
    print("  [%s] 固定 dx = 1/%d, 加密 dt（期望阶 ~1）" % (label, n))
    print("  %-12s %-14s %s" % ("dt", "Linf 误差", "阶"))
    prev = None
    for k in range(4):
        dt = 4e-4 / 2 ** k
        u, _, t = run("implicit", n, dt, bc, u0_fn)
        e = linf_error(u, exact, t)
        print("  %-12.3e %-14.6e %s" % (dt, e, "-" if prev is None else "%.2f" % order(prev, e)))
        prev = e
    print()


def table_joint_refinement(scheme, r, bc, exact, u0_fn, label):
    """固定 r = alpha*dt/dx^2, dx/dt 同步加密 -> 期望二阶。"""
    print("  [%s] 固定 r = %g, 同步加密 dx,dt（期望阶 ~2）" % (label, r))
    print("  %-8s %-12s %-14s %s" % ("N", "dx", "Linf 误差", "阶"))
    prev = None
    for n in (20, 40, 80, 160):
        dx = L / n
        dt = r * dx * dx / ALPHA
        u, _, t = run(scheme, n, dt, bc, u0_fn)
        e = linf_error(u, exact, t)
        print("  %-8d %-12.4e %-14.6e %s" % (n, dx, e, "-" if prev is None else "%.2f" % order(prev, e)))
        prev = e
    print()


def table_conservation():
    """两端绝热 + 阶跃初值: 能量守恒数据。"""
    print("  [绝热边界能量守恒] 阶跃初值 u0 = 1 (x<0.5), 0 (x>=0.5), t 到 0.5")
    n = 100
    dx = L / n
    u0 = [1.0 if i * dx < 0.5 else 0.0 for i in range(n + 1)]
    e0 = energy(u0, dx)
    print("  初始能量 E(0) = %.15f" % e0)
    print("  %-10s %-8s %-20s %s" % ("格式", "r", "E(t)", "|E(t)-E(0)|"))
    for scheme, r in (("explicit", 0.4), ("implicit", 10.0)):
        dt = r * dx * dx
        steps = round(0.5 / dt)
        u = solve(u0, ALPHA, dx, dt, steps,
                  bc_left=(NEUMANN, 0.0), bc_right=(NEUMANN, 0.0),
                  scheme=scheme)
        e1 = energy(u, dx)
        print("  %-10s %-8g %-20.15f %.3e" % (scheme, r, e1, abs(e1 - e0)))
    print()


def main():
    bc_d = (DIRICHLET, 0.0)
    bc_n = (NEUMANN, 0.0)
    u0_a = lambda x: math.sin(math.pi * x)
    u0_b = lambda x: 0.5 + math.cos(math.pi * x)

    print("=" * 56)
    print("问题 A: 两端定温 0, u0 = sin(pi x), t = %g" % T_FINAL)
    print("=" * 56)
    table_time_refinement_explicit(bc_d, u0_a, "显式")
    table_time_refinement_implicit(bc_d, exact_a, u0_a, "隐式")
    table_joint_refinement("explicit", 0.4, bc_d, exact_a, u0_a, "显式")
    table_joint_refinement("implicit", 2.0, bc_d, exact_a, u0_a, "隐式")

    print("=" * 56)
    print("问题 B: 两端绝热, u0 = 0.5 + cos(pi x), t = %g" % T_FINAL)
    print("=" * 56)
    table_joint_refinement("explicit", 0.4, bc_n, exact_b, u0_b, "显式")
    table_joint_refinement("implicit", 2.0, bc_n, exact_b, u0_b, "隐式")

    print("=" * 56)
    print("能量守恒验证")
    print("=" * 56)
    table_conservation()


if __name__ == "__main__":
    main()
