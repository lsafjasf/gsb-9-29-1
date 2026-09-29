"""demo.py — 端到端演示：训练、收敛曲线、IRLS 对拍、正则效应、发散检测。

运行: python3 demo.py
产出: convergence_curve.csv   每次迭代的 loss / step / grad_norm
      reference_check.csv     本库 vs 独立 IRLS 参考实现的系数对拍
"""

import math

from logistic_model import DivergenceError, LogisticScorer
from selftest import irls_reference, make_data


def main():
    # 量纲差异很大的特征：收入(万)、年龄、申请次数、负债率
    X, y = make_data(n=300, d=4, seed=42,
                     scales=[20000.0, 15.0, 3.0, 0.05])
    l2 = 0.3
    model = LogisticScorer(l2=l2, tol=1e-12).fit(X, y)

    print("== 收敛曲线（首尾各 5 行）==")
    h = model.history_
    print(" iter        loss        step     grad_norm")
    for row in h[:5] + h[-5:]:
        print("%5d  %.10f  %9.3g  %11.4g"
              % (row["iter"], row["loss"], row["step"], row["grad_norm"]))
    print("converged: %s, iterations: %d" % (model.converged_, len(h)))
    with open("convergence_curve.csv", "w") as f:
        f.write(model.convergence_csv())
    print("-> convergence_curve.csv")

    print("\n== 与独立 IRLS(Newton) 参考实现对拍 ==")
    Xs = model.scaler_.transform(X)
    ref_w, ref_b = irls_reference(Xs, y, l2)
    rows = ["param,ours,irls_reference,abs_diff"]
    max_diff = 0.0
    for j in range(len(ref_w)):
        diff = abs(model.coef_[j] - ref_w[j])
        max_diff = max(max_diff, diff)
        rows.append("w%d,%.10f,%.10f,%.3g" % (j, model.coef_[j], ref_w[j], diff))
        print("  w%d  ours=% .10f  irls=% .10f  |diff|=%.3g"
              % (j, model.coef_[j], ref_w[j], diff))
    diff = abs(model.intercept_ - ref_b)
    max_diff = max(max_diff, diff)
    rows.append("b,%.10f,%.10f,%.3g" % (model.intercept_, ref_b, diff))
    print("  b   ours=% .10f  irls=% .10f  |diff|=%.3g"
          % (model.intercept_, ref_b, diff))
    with open("reference_check.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print("max |diff| = %.3g  -> reference_check.csv" % max_diff)
    assert max_diff < 1e-4, "reference check failed"

    print("\n== 正则对系数的影响（l2 越大，||w|| 越小）==")
    for reg in (0.01, 0.1, 1.0, 10.0):
        m = LogisticScorer(l2=reg, max_iter=5000).fit(X, y)
        norm = math.sqrt(sum(c * c for c in m.coef_))
        acc = sum(int(p == t) for p, t in zip(m.predict(X), y)) / len(y)
        print("  l2=%6.2f   ||w||=%8.4f   train_acc=%.4f" % (reg, norm, acc))

    print("\n== 发散检测：完全可分 + 无正则 ==")
    Xs2, ys2 = make_data(n=80, d=2, seed=5, separable=True)
    try:
        LogisticScorer(l2=0.0).fit(Xs2, ys2)
        print("  ERROR: should have raised")
    except DivergenceError as e:
        print("  DivergenceError: %s" % e)


if __name__ == "__main__":
    main()
