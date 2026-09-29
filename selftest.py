"""selftest.py — logistic_model 的断言自测与对拍。仅标准库，直接 python3 运行。

覆盖：
  1. 梯度 vs 中心有限差分对拍
  2. 训练结果 vs 独立 IRLS（Newton）实现对拍
  3. 标准化训练/预测一致性断言
  4. 边界用例：完全可分（有/无正则）、单类别、特征全常数、样本量为零
  5. 收敛曲线单调下降 & 正则对系数的收缩效应
"""

import math
import random

from logistic_model import (
    DivergenceError,
    LogisticScorer,
    StandardScaler,
    _sigmoid,
)

PASS = []


def check(name, cond):
    assert cond, "FAILED: %s" % name
    PASS.append(name)
    print("  ok - %s" % name)


def make_data(n=200, d=4, seed=7, separable=False, scales=None):
    rng = random.Random(seed)
    true_w = [1.5, -2.0, 0.8, 0.3][:d]
    scales = scales or [1.0] * d
    X, y = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 1.0) * scales[j] for j in range(d)]
        z = sum(w * v for w, v in zip(true_w, row)) / max(scales) + 0.5
        if separable:
            label = 1 if z > 0 else 0  # 无噪声 => 完全可分
        else:
            label = 1 if rng.random() < _sigmoid(z) else 0
        X.append(row)
        y.append(label)
    return X, y


# ---------- 1. 梯度 vs 有限差分 ----------

def test_gradient_finite_difference():
    print("[1] gradient vs finite difference")
    rng = random.Random(1)
    n, d = 30, 3
    X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    y = [rng.randint(0, 1) for _ in range(n)]
    m = LogisticScorer(l2=0.7)
    w = [rng.gauss(0, 1) for _ in range(d)]
    b = rng.gauss(0, 1)
    loss, gw, gb = m._loss_grad(X, y, w, b)
    eps = 1e-6
    for j in range(d):
        wp, wm = list(w), list(w)
        wp[j] += eps
        wm[j] -= eps
        lp = m._loss_grad(X, y, wp, b)[0]
        lm = m._loss_grad(X, y, wm, b)[0]
        num = (lp - lm) / (2 * eps)
        check("grad_w[%d] close to finite diff" % j, abs(num - gw[j]) < 1e-5)
    lp = m._loss_grad(X, y, w, b + eps)[0]
    lm = m._loss_grad(X, y, w, b - eps)[0]
    check("grad_b close to finite diff", abs((lp - lm) / (2 * eps) - gb) < 1e-5)


# ---------- 2. 与独立 IRLS（Newton）实现对拍 ----------

def _solve_linear(A, b):
    """高斯消元解 Ax=b（部分主元），仅用于对拍参考。"""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))) / M[r][r]
    return x


def irls_reference(Xs, y, l2, iters=50):
    """独立参考实现：IRLS/Newton 解带 L2 的逻辑回归（截距不正则）。"""
    n, d = len(Xs), len(Xs[0])
    w = [0.0] * (d + 1)  # 最后一维是截距
    for _ in range(iters):
        g = [0.0] * (d + 1)
        H = [[0.0] * (d + 1) for _ in range(d + 1)]
        for xi, yi in zip(Xs, y):
            x1 = list(xi) + [1.0]
            z = sum(wj * xj for wj, xj in zip(w, x1))
            p = _sigmoid(z)
            diff = (p - yi) / n
            s = max(p * (1.0 - p), 1e-12) / n
            for a in range(d + 1):
                g[a] += diff * x1[a]
                for c in range(d + 1):
                    H[a][c] += s * x1[a] * x1[c]
        for a in range(d):  # 截距不正则
            g[a] += l2 * w[a]
            H[a][a] += l2
        step = _solve_linear(H, g)
        w = [wj - sj for wj, sj in zip(w, step)]
    return w[:d], w[d]


def test_against_irls():
    print("[2] fit vs independent IRLS reference")
    X, y = make_data(n=300, d=4, seed=11, scales=[1.0, 50.0, 0.01, 5.0])
    l2 = 0.3
    m = LogisticScorer(l2=l2, tol=1e-12).fit(X, y)
    Xs = m.scaler_.transform(X)
    ref_w, ref_b = irls_reference(Xs, y, l2)
    for j, (a, b_) in enumerate(zip(m.coef_, ref_w)):
        check("coef_[%d] matches IRLS (%.8f vs %.8f)" % (j, a, b_),
              abs(a - b_) < 1e-4)
    check("intercept_ matches IRLS (%.8f vs %.8f)" % (m.intercept_, ref_b),
          abs(m.intercept_ - ref_b) < 1e-4)
    check("model converged", m.converged_)


# ---------- 3. 标准化训练/预测一致性 ----------

def test_scaler_consistency():
    print("[3] train/predict standardization consistency")
    X, y = make_data(n=150, d=3, seed=3, scales=[1000.0, 0.001, 10.0])
    m = LogisticScorer(l2=0.5).fit(X, y)
    # 预测路径内部变换 == 训练时 scaler 的变换
    Xnew = [[123.0, 0.005, 8.0], [999.0, 0.0007, 12.0]]
    manual = [[(v - m.scaler_.mean_[j]) / m.scaler_.scale_[j]
               for j, v in enumerate(row)] for row in Xnew]
    z_manual = [m.intercept_ + sum(wj * xj for wj, xj in zip(m.coef_, row))
                for row in manual]
    z_model = m.decision_function(Xnew)
    for a, b_ in zip(z_manual, z_model):
        check("decision_function uses training scaler", abs(a - b_) < 1e-12)
    # 篡改 scaler 后指纹断言必须触发
    m.scaler_.mean_[0] += 1.0
    try:
        m.decision_function(Xnew)
        raised = False
    except AssertionError:
        raised = True
    check("fingerprint assertion fires on scaler tampering", raised)


# ---------- 4. 边界用例 ----------

def test_edge_cases():
    print("[4] edge cases")
    # 4a. 完全可分 + 无正则 => 必须报 DivergenceError 而不是返回垃圾参数
    Xs_, ys_ = make_data(n=80, d=2, seed=5, separable=True)
    try:
        LogisticScorer(l2=0.0, max_iter=2000).fit(Xs_, ys_)
        raised = False
    except DivergenceError:
        raised = True
    check("separable + l2=0 raises DivergenceError", raised)

    # 4b. 完全可分 + 弱 L2 => 收敛且训练集 100% 正确
    m = LogisticScorer(l2=0.01).fit(Xs_, ys_)
    acc = sum(int(p == t) for p, t in zip(m.predict(Xs_), ys_)) / len(ys_)
    check("separable + l2>0 converges", m.converged_)
    check("separable + l2>0 trains to 100% accuracy", acc == 1.0)

    # 4c. 单类别数据 => ValueError
    for labels in ([0] * 10, [1] * 10):
        try:
            LogisticScorer().fit([[float(i)] for i in range(10)], labels)
            raised = False
        except ValueError:
            raised = True
        check("single-class (all %d) raises ValueError" % labels[0], raised)

    # 4d. 特征全常数 => 可训练，常数特征系数恒为 0（标准化后列全 0）
    Xc = [[3.14, 2.71] for _ in range(40)]
    yc = [i % 2 for i in range(40)]
    mc = LogisticScorer(l2=0.5).fit(Xc, yc)
    check("constant features: fit converges", mc.converged_)
    check("constant features: coef == 0",
          all(abs(c) < 1e-12 for c in mc.coef_))
    check("constant features: scaler scale == 1 (no div-by-zero)",
          all(s == 1.0 for s in mc.scaler_.scale_))

    # 4e. 样本量为零 => ValueError
    try:
        LogisticScorer().fit([], [])
        raised = False
    except ValueError:
        raised = True
    check("zero samples raises ValueError", raised)

    # 4f. 其他输入校验
    try:
        LogisticScorer().fit([[1.0], [2.0]], [0])
        raised = False
    except ValueError:
        raised = True
    check("X/y length mismatch raises ValueError", raised)


# ---------- 5. 收敛曲线与正则效应 ----------

def test_convergence_and_regularization():
    print("[5] convergence curve & regularization effect")
    X, y = make_data(n=200, d=4, seed=9, scales=[1.0, 100.0, 0.01, 20.0])
    m = LogisticScorer(l2=0.5).fit(X, y)
    losses = [h["loss"] for h in m.history_]
    check("loss decreases monotonically along curve",
          all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])))
    check("grad_norm shrinks by 100x over training",
          m.history_[-1]["grad_norm"] < 0.01 * m.history_[0]["grad_norm"])
    check("line-search steps stay positive",
          all(h["step"] > 0 for h in m.history_))

    # 正则收缩效应：l2 越大，系数范数越小
    norms = []
    for l2 in (0.01, 0.5, 5.0):
        mm = LogisticScorer(l2=l2, max_iter=5000).fit(X, y)
        norms.append(math.sqrt(sum(c * c for c in mm.coef_)))
    check("larger l2 => smaller ||w|| (%.4f > %.4f > %.4f)" % tuple(norms),
          norms[0] > norms[1] > norms[2])


if __name__ == "__main__":
    test_gradient_finite_difference()
    test_against_irls()
    test_scaler_consistency()
    test_edge_cases()
    test_convergence_and_regularization()
    print("\nALL %d CHECKS PASSED" % len(PASS))
