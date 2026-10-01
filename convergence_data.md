# 收敛对比数据（BFGS vs 梯度下降）

同一套强 Wolfe/Armijo 线搜索，c1=1e-4, c2=0.9；停机 \|g\|_inf <= gtol。

| 问题 | 方法 | 收敛轮数 | f 求值 | g 求值 | \|f-f*\| | \|g\|inf | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| 二次函数(n=10, cond=1e4) | BFGS | 33 | 47 | 35 | 4.044e-22 | 6.302e-10 | converged_grad |
| 二次函数(n=10, cond=1e4) | GD | 300 | 623 | 325 | 4.362e-01 | 2.351e+00 | max_iter |
| Rosenbrock(n=2) | BFGS | 34 | 64 | 47 | 1.049e-20 | 5.882e-10 | converged_grad |
| Rosenbrock(n=2) | GD | 1000 | 2080 | 1114 | 4.816e-03 | 9.692e-02 | max_iter |
| Rosenbrock(n=10) | BFGS | 30 | 41 | 31 | 3.987e+00 | 5.175e-07 | converged_grad |
| Rosenbrock(n=10) | GD | 3000 | 6093 | 3179 | 3.987e+00 | 8.399e-04 | max_iter |
| Rastrigin 非凸多峰(n=2) | BFGS | 4 | 138 | 6 | 0.000e+00 | 1.750e-08 | converged_grad |
| Rastrigin 非凸多峰(n=2) | GD | 25 | 178 | 27 | 0.000e+00 | 5.037e-07 | converged_grad |
| 四次函数近零梯度(n=3) | BFGS | 6 | 27 | 27 | 7.467e-19 | 4.457e-14 | converged_grad |
| 四次函数近零梯度(n=3) | GD | 500 | 10501 | 10501 | 3.000e-16 | 4.000e-12 | max_iter |
| 起点即最优 | BFGS | 0 | 1 | 1 | 0.000e+00 | 0.000e+00 | converged_grad |
| 起点即最优 | GD | 0 | 1 | 1 | 0.000e+00 | 0.000e+00 | converged_grad |
