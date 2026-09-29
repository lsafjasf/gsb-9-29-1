# 一维热传导自测报告

方程：$u_t=\alpha u_{xx}$；以下对拍取 $L=1$、$\alpha=0.1$、$t=0.1$。
细化时令 $dt=dx^2$，因此显式 FTCS 与隐式 BTCS 的总截断误差均为 $O(dt+dx^2)=O(dx^2)$。

## 解析解对拍

初值 $\sin(\pi x/L)$，两端定温 0，解析解为 $e^{-\alpha\pi^2t/L^2}\sin(\pi x/L)$。

| 格式 | nx | dx | dt | 步数 | max误差 | RMS误差 | max阶数 | RMS阶数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| explicit | 10 | 0.1 | 1.000e-02 | 10 | 2.961660e-04 | 2.094210e-04 |  |  |
| explicit | 20 | 0.05 | 2.500e-03 | 40 | 7.366931e-05 | 5.209207e-05 | 2.007 | 2.007 |
| explicit | 40 | 0.025 | 6.250e-04 | 160 | 1.839408e-05 | 1.300658e-05 | 2.002 | 2.002 |
| explicit | 80 | 0.0125 | 1.563e-04 | 640 | 4.597067e-06 | 3.250617e-06 | 2.000 | 2.000 |
| implicit | 10 | 0.1 | 1.000e-02 | 10 | 1.165041e-03 | 8.238086e-04 |  |  |
| implicit | 20 | 0.05 | 2.500e-03 | 40 | 2.934448e-04 | 2.074968e-04 | 1.989 | 1.989 |
| implicit | 40 | 0.025 | 6.250e-04 | 160 | 7.349920e-05 | 5.197178e-05 | 1.997 | 1.997 |
| implicit | 80 | 0.0125 | 1.563e-04 | 640 | 1.838345e-05 | 1.299906e-05 | 1.999 | 1.999 |

阶数由相邻网格的 `log2(error_coarse/error_fine)` 得到，数值接近 2，符合二阶空间/该细化路径下的总体二阶表现。

## 守恒与边界用例

| 用例 | 指标 | 数值 |
| --- | --- | ---: |
| explicit insulated cosine | time | 1.000000e-02 |
| explicit insulated cosine | max_error | 2.010254e-06 |
| explicit insulated cosine | initial_energy | -2.775558e-18 |
| explicit insulated cosine | final_energy | -2.081668e-17 |
| explicit insulated cosine | energy_change | -1.804112e-17 |
| implicit insulated cosine | time | 1.000000e-02 |
| implicit insulated cosine | max_error | 8.032368e-06 |
| implicit insulated cosine | initial_energy | -2.775558e-18 |
| implicit insulated cosine | final_energy | 1.387779e-17 |
| implicit insulated cosine | energy_change | 1.665335e-17 |
| explicit short insulated step | time | 1.000000e-01 |
| explicit short insulated step | initial_energy | 5.000000e-01 |
| explicit short insulated step | final_energy | 5.000000e-01 |
| explicit short insulated step | energy_change | -5.551115e-17 |
| explicit long insulated step | time | 1.000000e+02 |
| explicit long insulated step | initial_energy | 5.000000e-01 |
| explicit long insulated step | final_energy | 5.000000e-01 |
| explicit long insulated step | energy_change | 1.443290e-15 |
| explicit long insulated step | max_deviation_from_mean | 5.329071e-15 |
| implicit long insulated step | time | 1.000000e+02 |
| implicit long insulated step | initial_energy | 5.000000e-01 |
| implicit long insulated step | final_energy | 5.000000e-01 |
| implicit long insulated step | energy_change | 1.653122e-13 |
| implicit long insulated step | max_deviation_from_mean | 1.667555e-13 |
| implicit long insulated step | min | 5.000000e-01 |
| implicit long insulated step | max | 5.000000e-01 |
| implicit fixed 1/0 long-time profile | time | 1.000000e+02 |
| implicit fixed 1/0 long-time profile | left | 1.000000e+00 |
| implicit fixed 1/0 long-time profile | right | 0.000000e+00 |
| implicit fixed 1/0 long-time profile | max_profile_error | 7.771561e-16 |
| explicit insulated/fixed mixed mode | time | 1.000000e-02 |
| explicit insulated/fixed mixed mode | left | 9.975358e-01 |
| explicit insulated/fixed mixed mode | right | 0.000000e+00 |
| explicit insulated/fixed mixed mode | max_error | 1.265352e-07 |
| implicit insulated/fixed mixed mode | time | 1.000000e-02 |
| implicit insulated/fixed mixed mode | left | 9.975361e-01 |
| implicit insulated/fixed mixed mode | right | 0.000000e+00 |
| implicit insulated/fixed mixed mode | max_error | 5.060043e-07 |
| zero diffusion, both schemes | time | 7.000000e+01 |
| zero diffusion, both schemes | explicit_max_change | 0.000000e+00 |
| zero diffusion, both schemes | implicit_max_change | 0.000000e+00 |

## 稳定性保护

- 显式格式在 $r=\alpha dt/dx^2=0.510>1/2$ 时抛出 `StabilityError`，不执行积分。
- 边界值 $r=0.500$ 允许执行。
- 绝热两端使用虚节点 $u_{-1}=u_1$、$u_{N+1}=u_{N-1}$；梯形能量 $E=dx(u_0/2+\sum_{i=1}^{N-1}u_i+u_N/2)$ 在显式和隐式格式中均有离散恒等式 $E^{n+1}=E^n$。
- 绝热余弦、阶跃初值和长时间阶跃用例都执行能量守恒断言；浮点容差为 1e-12，代数变化量为零。

## 覆盖情形

- 显式 FTCS 与隐式 BTCS（三对角 Thomas 算法）。
- 定温-定温、绝热-绝热、绝热-定温混合边界，以及非零定温边界。
- 零扩散系数、阶跃初值、绝热长时间趋均、定温长时间趋近稳态。
