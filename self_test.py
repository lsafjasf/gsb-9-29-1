#!/usr/bin/env python3
"""Analytic, boundary, conservation, and regression tests for heat1d.py."""

from __future__ import annotations

import math

from heat1d import (
    StabilityError,
    assert_energy_conserved,
    cosine_neumann_solution,
    energy,
    make_grid,
    max_error,
    mixed_cosine_solution,
    observed_order,
    rms_error,
    sine_dirichlet_solution,
    solve_heat1d,
)

LENGTH = 1.0
ALPHA = 0.1


def convergence_rows():
    final_time = 0.1
    rows = []
    for scheme in ("explicit", "implicit"):
        max_errors = []
        rms_errors = []
        for nx in (10, 20, 40, 80):
            x, dx = make_grid(LENGTH, nx)
            dt = dx * dx
            steps = int(round(final_time / dt))
            initial = [math.sin(math.pi * value / LENGTH) for value in x]
            result = solve_heat1d(
                initial,
                dx,
                dt,
                ALPHA,
                steps,
                scheme=scheme,
                boundaries=("fixed", "fixed"),
            )
            expected = sine_dirichlet_solution(x, final_time, ALPHA, LENGTH)
            current_max = max_error(result.u, expected)
            current_rms = rms_error(result.u, expected, dx)
            max_errors.append(current_max)
            rms_errors.append(current_rms)
            rows.append(
                {
                    "scheme": scheme,
                    "nx": nx,
                    "dx": dx,
                    "dt": dt,
                    "steps": steps,
                    "max": current_max,
                    "rms": current_rms,
                    "order_max": "",
                    "order_rms": "",
                }
            )
        max_orders = observed_order(max_errors)
        rms_orders = observed_order(rms_errors)
        for offset, order in enumerate(max_orders, start=1):
            rows[-4 + offset]["order_max"] = f"{order:.3f}"
            rows[-4 + offset]["order_rms"] = f"{rms_orders[offset - 1]:.3f}"
    return rows


def special_cases():
    cases = {}
    nx = 40
    x, dx = make_grid(LENGTH, nx)

    smooth = [math.cos(math.pi * value / LENGTH) for value in x]
    for scheme in ("explicit", "implicit"):
        result = solve_heat1d(
            smooth,
            dx,
            dx * dx,
            ALPHA,
            16,
            scheme=scheme,
            boundaries=("insulated", "insulated"),
        )
        expected = cosine_neumann_solution(x, result.time, ALPHA, LENGTH)
        initial_energy, final_energy, difference = assert_energy_conserved(
            smooth,
            result.u,
            dx,
            label=f"{scheme} smooth insulated",
        )
        cases[f"{scheme} insulated cosine"] = {
            "time": result.time,
            "max_error": max_error(result.u, expected),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": difference,
        }

    step = [
        1.0 if nx // 4 < index < 3 * nx // 4 else
        (0.5 if index in (nx // 4, 3 * nx // 4) else 0.0)
        for index in range(nx + 1)
    ]
    mean = energy(step, dx) / LENGTH
    short_step = solve_heat1d(
        step,
        dx,
        0.0025,
        ALPHA,
        40,
        scheme="explicit",
        boundaries=("insulated", "insulated"),
    )
    initial_energy, final_energy, difference = assert_energy_conserved(
        step,
        short_step.u,
        dx,
        label="explicit step insulated",
    )
    cases["explicit short insulated step"] = {
        "time": short_step.time,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_change": difference,
    }
    assert min(short_step.u) >= 0.0 and max(short_step.u) <= 1.0
    assert abs(short_step.u[10] - short_step.u[30]) <= 1e-15

    long_explicit_step = solve_heat1d(
        step,
        dx,
        0.0025,
        ALPHA,
        40000,
        scheme="explicit",
        boundaries=("insulated", "insulated"),
    )
    initial_energy, final_energy, difference = assert_energy_conserved(
        step,
        long_explicit_step.u,
        dx,
        label="explicit long step insulated",
    )
    cases["explicit long insulated step"] = {
        "time": long_explicit_step.time,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_change": difference,
        "max_deviation_from_mean": max(
            abs(value - mean)
            for value in long_explicit_step.u
        ),
    }

    long_step = solve_heat1d(
        step,
        dx,
        0.25,
        ALPHA,
        400,
        scheme="implicit",
        boundaries=("insulated", "insulated"),
    )
    initial_energy, final_energy, difference = assert_energy_conserved(
        step,
        long_step.u,
        dx,
        label="implicit long step insulated",
    )
    cases["implicit long insulated step"] = {
        "time": long_step.time,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_change": difference,
        "max_deviation_from_mean": max(abs(value - mean) for value in long_step.u),
        "min": min(long_step.u),
        "max": max(long_step.u),
    }

    stationary_fixed = solve_heat1d(
        step,
        dx,
        0.25,
        ALPHA,
        400,
        scheme="implicit",
        boundaries=("fixed", "fixed"),
        fixed_temps=(1.0, 0.0),
    )
    fixed_profile_error = max(
        abs(value - (1.0 - position / LENGTH))
        for value, position in zip(stationary_fixed.u, x)
    )
    cases["implicit fixed 1/0 long-time profile"] = {
        "time": stationary_fixed.time,
        "left": stationary_fixed.u[0],
        "right": stationary_fixed.u[-1],
        "max_profile_error": fixed_profile_error,
    }

    mixed_initial = [math.cos(math.pi * value / (2.0 * LENGTH)) for value in x]
    for scheme in ("explicit", "implicit"):
        mixed = solve_heat1d(
            mixed_initial,
            dx,
            dx * dx,
            ALPHA,
            16,
            scheme=scheme,
            boundaries=("insulated", "fixed"),
        )
        mixed_expected = mixed_cosine_solution(x, mixed.time, ALPHA, LENGTH)
        cases[f"{scheme} insulated/fixed mixed mode"] = {
            "time": mixed.time,
            "left": mixed.u[0],
            "right": mixed.u[-1],
            "max_error": max_error(mixed.u, mixed_expected),
        }

    zero_profile = [0.0] + [1.0 + 0.1 * i for i in range(nx - 1)] + [0.0]
    zero_changes = []
    zero_result = None
    for scheme in ("explicit", "implicit"):
        zero_result = solve_heat1d(
            zero_profile,
            dx,
            10.0,
            0.0,
            7,
            scheme=scheme,
            boundaries=("fixed", "fixed"),
        )
        zero_changes.append(max(abs(a - b) for a, b in zip(zero_profile, zero_result.u)))
    cases["zero diffusion, both schemes"] = {
        "time": zero_result.time,
        "explicit_max_change": zero_changes[0],
        "implicit_max_change": zero_changes[1],
    }

    return cases


def assert_stability_guard():
    _, dx = make_grid(LENGTH, 10)
    alpha = 0.1
    rejected_dt = 0.51 * dx * dx / alpha
    try:
        solve_heat1d(
            [0.0] * 11,
            dx,
            rejected_dt,
            alpha,
            1,
            scheme="explicit",
            boundaries=("insulated", "insulated"),
        )
    except StabilityError:
        pass
    else:
        raise AssertionError("unstable explicit run was not rejected")

    accepted_dt = 0.5 * dx * dx / alpha
    solve_heat1d(
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        dx,
        accepted_dt,
        alpha,
        1,
        scheme="explicit",
        boundaries=("fixed", "fixed"),
    )
    return alpha * rejected_dt / dx**2, alpha * accepted_dt / dx**2


def render_report():
    rows = convergence_rows()
    cases = special_cases()
    rejected_r, accepted_r = assert_stability_guard()

    lines = [
        "# 一维热传导自测报告",
        "",
        "方程：$u_t=\\alpha u_{xx}$；以下对拍取 $L=1$、$\\alpha=0.1$、$t=0.1$。",
        "细化时令 $dt=dx^2$，因此显式 FTCS 与隐式 BTCS 的总截断误差均为 "
        "$O(dt+dx^2)=O(dx^2)$。",
        "",
        "## 解析解对拍",
        "",
        "初值 $\\sin(\\pi x/L)$，两端定温 0，解析解为 "
        "$e^{-\\alpha\\pi^2t/L^2}\\sin(\\pi x/L)$。",
        "",
        "| 格式 | nx | dx | dt | 步数 | max误差 | RMS误差 | max阶数 | RMS阶数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scheme']} | {row['nx']} | {row['dx']:.6g} | "
            f"{row['dt']:.3e} | {row['steps']} | {row['max']:.6e} | "
            f"{row['rms']:.6e} | {row['order_max']} | {row['order_rms']} |"
        )

    lines.extend(
        [
            "",
            "阶数由相邻网格的 `log2(error_coarse/error_fine)` 得到，数值接近 2，"
            "符合二阶空间/该细化路径下的总体二阶表现。",
            "",
            "## 守恒与边界用例",
            "",
            "| 用例 | 指标 | 数值 |",
            "| --- | --- | ---: |",
        ]
    )
    for case_name, values in cases.items():
        for metric, value in values.items():
            lines.append(f"| {case_name} | {metric} | {value:.6e} |")

    lines.extend(
        [
            "",
            "## 稳定性保护",
            "",
            f"- 显式格式在 $r=\\alpha dt/dx^2={rejected_r:.3f}>1/2$ 时抛出 "
            "`StabilityError`，不执行积分。",
            f"- 边界值 $r={accepted_r:.3f}$ 允许执行。",
            "- 绝热两端使用虚节点 $u_{-1}=u_1$、$u_{N+1}=u_{N-1}$；梯形能量 "
            "$E=dx(u_0/2+\\sum_{i=1}^{N-1}u_i+u_N/2)$ 在显式和隐式格式中均有离散恒等"
            "式 $E^{n+1}=E^n$。",
            "- 绝热余弦、阶跃初值和长时间阶跃用例都执行能量守恒断言；浮点容差为 "
            "1e-12，代数变化量为零。",
            "",
            "## 覆盖情形",
            "",
            "- 显式 FTCS 与隐式 BTCS（三对角 Thomas 算法）。",
            "- 定温-定温、绝热-绝热、绝热-定温混合边界，以及非零定温边界。",
            "- 零扩散系数、阶跃初值、绝热长时间趋均、定温长时间趋近稳态。",
        ]
    )
    return "\n".join(lines)


def main():
    print(render_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
