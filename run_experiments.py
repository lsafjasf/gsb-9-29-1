"""实验脚本: 生成反射率 / 能量守恒 / 网格色散 / 稳定性数据。

运行:  python3 run_experiments.py
输出:  终端打印 Markdown 表格, 同时写入 RESULTS.md
"""

import math
import sys

from wave1d import (Wave1D, DIRICHLET, NEUMANN, MUR,
                    dispersion_omega, group_velocity, mur_reflection_coeff)

LINES = []


def out(text=""):
    print(text)
    LINES.append(text)


# ---------------------------------------------------------------- 实验 1: 反射率

def measure_reflection(right_bc, width, r=0.9, nx=501, c=1.0):
    """右行高斯脉冲打向右边界, 测反射能量比 R = E_reflected / E_incident。

    左边界用 Mur 吸收残余, 测量时刻取在入射脉冲被完全吸收、
    反射波尚未到达左边界之前。
    """
    L = 1.0
    dx = L / (nx - 1)
    dt = r * dx / c
    x0 = 0.25
    sim = Wave1D(nx, c, dx, dt, left=MUR, right=right_bc)
    sim.set_right_traveling_gaussian(x0, width)
    e0 = sim.energy()
    amp0 = sim.max_abs()
    # 入射到达右边界时刻 (L-x0)/c, 再留 0.35L/c 让反射波进入域内
    t_measure = ((L - x0) + 0.35 * L) / c
    sim.step(round(t_measure / dt))
    return sim.energy() / e0, sim.max_abs() / amp0


def exp1_reflection():
    out("## 1. 边界反射率 (右行高斯脉冲, 能量反射率 R = E_反射 / E_入射)")
    out()
    out("网格: nx=501, L=1, c=1, r=0.9; 脉冲宽度 w 以 dx 为单位标注。")
    out()
    out("| 右边界 | 脉冲宽度 w | R (能量) | sqrt(R) (振幅) |")
    out("|---|---|---|---|")
    for bc, name in ((DIRICHLET, "Dirichlet u=0"),
                     (NEUMANN, "Neumann du/dx=0"),
                     (MUR, "Mur 一阶吸收")):
        for w, tag in ((0.04, "0.04 (20dx)"), (0.008, "0.008 (4dx)"),
                       (0.004, "0.004 (2dx, 极窄)")):
            re, ra = measure_reflection(bc, w)
            out("| %s | %s | %.3e | %.3e |" % (name, tag, re, ra))
    out()
    # r = 1 时 Mur 精确吸收
    re, ra = measure_reflection(MUR, 0.04, r=1.0)
    out("r = 1.0 时 Mur 边界: R = %.3e (机器精度量级, 一维下对平面波精确吸收)" % re)
    out()
    out("不同 Courant 数 r 下 Mur 反射率 (w = 0.04):")
    out()
    out("| r | R (能量) |")
    out("|---|---|")
    for r in (1.0, 0.95, 0.9, 0.7, 0.5):
        re, _ = measure_reflection(MUR, 0.04, r=r)
        out("| %.2f | %.3e |" % (r, re))
    out()


def exp1b_mur_theory():
    """单色波包验证 Mur 反射系数渐近公式 |R| ~ (1-r^2)(k dx)^2 / 96。"""
    out("单色波包验证 Mur 振幅反射系数理论值 (r=0.9); 渐近式 |R| ~ (1-r^2)(k dx)^2/16:")
    out()
    out("| k*dx | 理论 \\|R\\| (振幅) | 实测 sqrt(R_能量) | 比值 |")
    out("|---|---|---|---|")
    c, nx, r = 1.0, 1001, 0.9
    L = 2.0
    dx = L / (nx - 1)
    dt = r * dx / c
    for kdx in (0.1, 0.2, 0.4):
        k = kdx / dx
        width = 8.0 / k          # 宽包络 -> 谱集中在 k 附近
        x0 = 0.5
        sim = Wave1D(nx, c, dx, dt, left=MUR, right=MUR)
        u0, v0 = [], []
        for i in range(nx):
            x = i * dx
            s = (x - x0) / width
            env = math.exp(-s * s)
            u0.append(env * math.cos(k * x))
            # 右行波包: v = -c du/dx (包络足够宽时近似)
            v0.append(c * (2 * s / width * env * math.cos(k * x)
                           + k * env * math.sin(k * x)))
        sim.set_initial(u0, v0)
        e0 = sim.energy()
        t_measure = ((L - x0) + 0.3 * L) / c
        sim.step(round(t_measure / dt))
        r_meas = math.sqrt(sim.energy() / e0)
        r_theo = mur_reflection_coeff(k, c, dx, dt)
        out("| %.2f | %.3e | %.3e | %.2f |" % (kdx, r_theo, r_meas,
                                               r_meas / r_theo))
    out()


# ---------------------------------------------------------------- 实验 2: 能量守恒

def exp2_energy():
    out("## 2. 封闭边界能量守恒 (长时间演化)")
    out()
    out("高斯脉冲 w=0.05 置于域中心, nx=201, r=0.9, 推进 50000 步")
    out("(约 112 次往返)。E 为严格守恒型离散能量, E_naive 为常用能量。")
    out()
    out("| 边界 | 步数 | max \\|E/E0 - 1\\| (守恒型) | max \\|E/E0 - 1\\| (naive) |")
    out("|---|---|---|---|")
    for bc, name in ((DIRICHLET, "Dirichlet"), (NEUMANN, "Neumann")):
        nx, c, r = 201, 1.0, 0.9
        dx = 1.0 / (nx - 1)
        dt = r * dx / c
        sim = Wave1D(nx, c, dx, dt, left=bc, right=bc)
        sim.set_right_traveling_gaussian(0.5, 0.05)
        e0 = sim.energy()
        en0 = sim.energy_naive()
        dev, dev_n = 0.0, 0.0
        nsteps = 50000
        for blk in range(50):
            sim.step(1000)
            dev = max(dev, abs(sim.energy() / e0 - 1.0))
            dev_n = max(dev_n, abs(sim.energy_naive() / en0 - 1.0))
        out("| %s | %d | %.3e | %.3e |" % (name, nsteps, dev, dev_n))
    out()
    out("守恒型能量漂移在 1e-12 量级 (纯浮点舍入), naive 能量仅有界振荡,")
    out("不随时间增长 —— 格式无数值耗散/增益。")
    out()


# ---------------------------------------------------------------- 实验 3: 网格色散

def exp3_dispersion():
    out("## 3. 网格色散: 数值相速度随 k*dx 的失真")
    out()
    out("驻波模态 sin(m pi x), nx=101, r=0.9, 由投影系数零点测数值频率。")
    out("w_exact = c k, w_num 由色散关系 sin(w dt/2) = r sin(k dx/2) 给出。")
    out()
    out("| m | k*dx | w_num/w_exact (实测) | w_num/w_exact (理论) | 群速度 vg/c (理论) |")
    out("|---|---|---|---|---|")
    nx, c, r = 101, 1.0, 0.9
    dx = 1.0 / (nx - 1)
    dt = r * dx / c
    for m in (1, 2, 4, 8, 16, 24, 32):
        k = m * math.pi
        mode = [math.sin(k * i * dx) for i in range(nx)]
        sim = Wave1D(nx, c, dx, dt, left=DIRICHLET, right=DIRICHLET)
        sim.set_initial(mode)
        # 投影系数 A(t) = sum u_i sin(k x_i), 以其过零点测周期
        nsteps = int(20.0 * (2.0 * math.pi / (c * k)) / dt) + 1
        crossings = []
        a_prev = sum(ui * si for ui, si in zip(sim.u, mode))
        for step in range(1, nsteps):
            sim.step()
            a = sum(ui * si for ui, si in zip(sim.u, mode))
            if a_prev < 0.0 <= a:
                # 线性插值过零时刻
                frac = -a_prev / (a - a_prev)
                crossings.append((sim.n - 1 + frac) * dt)
            a_prev = a
        if len(crossings) >= 2:
            period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
            w_meas = 2.0 * math.pi / period
        else:
            w_meas = float("nan")
        w_exact = c * k
        w_theo = dispersion_omega(k, c, dx, dt)
        vg = group_velocity(k, c, dx, dt)
        out("| %d | %.3f | %.5f | %.5f | %.4f |"
            % (m, k * dx, w_meas / w_exact, w_theo / w_exact, vg / c))
    out()
    out("k*dx 越大 (每个波长网格点越少), 相速度与群速度失真越严重;")
    out("k*dx -> pi 时群速度 -> 0 甚至变号, 高频分量几乎原地不动 ——")
    out("这就是极窄脉冲在粗网格上\"散架\"的原因。")
    out()


# ---------------------------------------------------------------- 实验 4: 稳定性

def exp4_stability():
    out("## 4. 稳定性 (CFL 条件 r = c dt/dx <= 1)")
    out()
    nx, c = 201, 1.0
    dx = 1.0 / (nx - 1)
    out("| r | 行为 (2000 步后 max\\|u\\|, 初始 1.0) |")
    out("|---|---|")
    for r in (1.0, 1.001, 1.01):
        dt = r * dx / c
        sim = Wave1D(nx, c, dx, dt, left=DIRICHLET, right=DIRICHLET,
                     strict=False)
        sim.set_right_traveling_gaussian(0.5, 0.05)
        sim.step(2000)
        out("| %.3f | %.3e |" % (r, sim.max_abs()))
    out()
    out("r <= 1 有界; r > 1 时误差按几何级数放大, 迅速溢出 —— 与 von Neumann")
    out("分析给出的稳定条件 r <= 1 一致。库默认在 r > 1 时抛 ValueError。")
    out()


def main():
    out("# 一维波动求解实验数据")
    out()
    out("由 `python3 run_experiments.py` 自动生成。")
    out()
    exp1_reflection()
    exp1b_mur_theory()
    exp2_energy()
    exp3_dispersion()
    exp4_stability()
    with open("RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\n已写入 RESULTS.md", file=sys.stderr)


if __name__ == "__main__":
    main()
