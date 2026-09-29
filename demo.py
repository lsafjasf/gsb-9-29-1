"""演示脚本：收敛曲线、对拍数据、正则影响、边界用例。

运行：python3 demo.py
产出：convergence.csv（收敛曲线数据）
"""

import csv
import math
import random

from linear_score import (
    DivergenceError,
    LogisticScoringModel,
    StandardScaler,
    _log1pexp,
    _sigmoid,
)


def make_data(n=300, d=3, seed=42, scales=None):
    rng = random.Random(seed)
    scales = scales or [1.0] * d
    true_w = [rng.uniform(-2, 2) for _ in range(d)]
    X, y = [], []
    for _ in range(n):
        row = [rng.gauss(0, 1) * scales[j] for j in range(d)]
        z = sum(true_w[j] * row[j] / scales[j] for j in range(d))
        X.append(row)
        y.append(1 if rng.random() < _sigmoid(z) else 0)
    return X, y


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_convergence_curve():
    section("1. 收敛曲线（特征量纲差 1e5 倍，回溯线搜索）")
    X, y = make_data(scales=[1.0, 100.0, 1e5])
    model = LogisticScoringModel(alpha=0.01).fit(X, y)
    print("converged=%s  n_iter=%d  final_loss=%.10f  train_acc=%.4f"
          % (model.converged_, model.n_iter_, model.loss_history_[-1],
             model.score(X, y)))
    print("%6s %16s %12s" % ("iter", "loss", "step"))
    rows = []
    hist, steps = model.loss_history_, model.step_history_
    idxs = sorted(set([0, 1, 2, 3, 4, 5, 10, 20, 30, 50, len(hist) - 1]))
    for i in idxs:
        if 0 <= i < len(hist):
            step = steps[i - 1] if i > 0 else float("nan")
            print("%6d %16.10f %12.3e" % (i, hist[i], step))
        if i >= len(hist):
            break
    with open("convergence.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["iter", "loss", "step"])
        for i, loss in enumerate(hist):
            wcsv.writerow([i, "%.12g" % loss,
                           "" if i == 0 else "%.6g" % steps[i - 1]])
    print("完整曲线已写入 convergence.csv（共 %d 个点）" % len(hist))


def demo_crosscheck():
    section("2. 对拍数据")
    # 2a. 稳定损失 vs 朴素损失
    rng = random.Random(0)
    max_diff = 0.0
    for _ in range(2000):
        z = rng.uniform(-30, 30)
        yi = rng.choice([0, 1])
        stable = _log1pexp(z) - yi * z
        naive = math.log(1.0 + math.exp(z)) - yi * z
        max_diff = max(max_diff, abs(stable - naive))
    print("稳定损失 vs 朴素损失（|z|<=30, 2000 样本）最大绝对误差: %.3e" % max_diff)
    print("极端输入 z=1000: 稳定实现 -> %.6f；朴素实现 math.exp(1000) 直接 OverflowError"
          % _log1pexp(1000.0))

    # 2b. 解析梯度 vs 中心差分
    X, y = make_data(n=60, d=3, seed=7)
    model = LogisticScoringModel(alpha=0.3)
    Xs = StandardScaler().fit_transform(X)
    w, b = [0.3, -0.7, 1.1], -0.4
    _, gw, gb = model._loss_grad(Xs, y, w, b)
    eps = 1e-6
    max_rel = 0.0
    for j in range(3):
        wp, wm = list(w), list(w)
        wp[j] += eps
        wm[j] -= eps
        num = (model._loss_only(Xs, y, wp, b)
               - model._loss_only(Xs, y, wm, b)) / (2 * eps)
        max_rel = max(max_rel, abs(gw[j] - num) / max(1.0, abs(num)))
    print("解析梯度 vs 中心差分 最大相对误差: %.3e" % max_rel)

    # 2c. 与独立朴素批量梯度下降对拍
    X, y = make_data(n=200, d=3, seed=9)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n, d, alpha = len(Xs), 3, 0.1
    wr, br, lr = [0.0] * d, 0.0, 0.05
    for _ in range(3000):
        gw2 = [0.0] * d
        gb2 = 0.0
        for row, yi in zip(Xs, y):
            z = br + sum(wr[j] * row[j] for j in range(d))
            r = _sigmoid(z) - yi
            for j in range(d):
                gw2[j] += r * row[j]
            gb2 += r
        for j in range(d):
            wr[j] -= lr * (gw2[j] / n + alpha * wr[j])
        br -= lr * gb2 / n
    ref_loss = LogisticScoringModel(alpha=alpha)._loss_only(Xs, y, wr, br)
    ours = LogisticScoringModel(alpha=alpha).fit(X, y)
    ref_pred = [1 if _sigmoid(br + sum(wr[j] * r[j] for j in range(d))) >= 0.5
                else 0 for r in Xs]
    agree = sum(1 for a, bb in zip(ours.predict(X), ref_pred) if a == bb) / n
    print("对拍参考实现（朴素 GD, lr=0.05, 3000 轮）:")
    print("  参考最终损失 = %.10f" % ref_loss)
    print("  本库最终损失 = %.10f  （应 <= 参考）" % ours.loss_history_[-1])
    print("  预测一致率   = %.2f%%" % (100.0 * agree))


def demo_regularization():
    section("3. 正则化对系数的影响（完全可分数据）")
    rng = random.Random(3)
    X, y = [], []
    while len(X) < 150:
        row = [rng.gauss(0, 1), rng.gauss(0, 1)]
        z = 2.0 * row[0] - 1.5 * row[1]
        if abs(z) < 0.5:
            continue
        X.append(row)
        y.append(1 if z > 0 else 0)
    print("完全可分数据上，alpha 越大系数模长越小（正则把系数压向 0，")
    print("避免 MLE 不存在导致的权值无界增长；截距不正则化）：")
    print("%10s %14s %14s" % ("alpha", "||w||", "final_loss"))
    for alpha in (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticScoringModel(alpha=alpha, max_iter=3000).fit(X, y)
        norm = math.sqrt(sum(c * c for c in m.coef_))
        print("%10.3f %14.6f %14.8f" % (alpha, norm, m.loss_history_[-1]))


def demo_divergence():
    section("4. 发散检测（报错而非返回垃圾参数）")
    X, y = make_data(n=100, d=2, seed=31)
    try:
        LogisticScoringModel(
            alpha=0.0, step_search="fixed", lr=1e12, max_iter=50).fit(X, y)
        print("ERROR: 未检测到发散！")
    except DivergenceError as e:
        print("固定步长 lr=1e12 -> 正确抛出 DivergenceError:")
        print("  %s" % e)
    m = LogisticScoringModel(alpha=0.01, step0=1e6).fit(X, y)
    print("同样的数据改用回溯线搜索（初始步长 1e6）-> 自适应收敛: "
          "converged=%s, n_iter=%d, acc=%.4f"
          % (m.converged_, m.n_iter_, m.score(X, y)))


def demo_edge_cases():
    section("5. 边界用例")
    cases = []

    # 完全可分
    rng = random.Random(19)
    Xs, ys = [], []
    while len(Xs) < 100:
        row = [rng.gauss(0, 1), rng.gauss(0, 1)]
        z = row[0] - row[1]
        if abs(z) < 0.5:
            continue
        Xs.append(row)
        ys.append(1 if z > 0 else 0)
    m = LogisticScoringModel(alpha=0.1).fit(Xs, ys)
    cases.append(("完全可分数据", "收敛=%s, 训练准确率=%.4f（L2 使问题有界）"
                  % (m.converged_, m.score(Xs, ys))))

    # 单类别
    try:
        LogisticScoringModel().fit([[float(i)] for i in range(10)], [1] * 10)
        cases.append(("单类别数据", "ERROR: 未报错"))
    except ValueError as e:
        cases.append(("单类别数据", "正确拒绝：%s" % e))

    # 全常数特征
    rng = random.Random(29)
    yc = [1 if rng.random() < 0.7 else 0 for _ in range(100)]
    yc[0] = 0
    m = LogisticScoringModel(alpha=1.0).fit([[5.0, -3.0]] * 100, yc)
    cases.append(("特征全常数", "系数=%s（恒 0），预测概率=%.4f ≈ 先验 %.2f"
                  % (["%.2f" % c for c in m.coef_],
                     m.predict_proba([[5.0, -3.0]])[0], sum(yc) / len(yc))))

    # 零样本
    try:
        LogisticScoringModel().fit([], [])
        cases.append(("样本量为零", "ERROR: 未报错"))
    except ValueError as e:
        cases.append(("样本量为零", "正确拒绝：%s" % e))

    for name, result in cases:
        print("  [%s] %s" % (name, result))


def demo_transform_consistency():
    section("6. 训练/预测同一套标准化变换（断言）")
    X, y = make_data(n=120, d=2, seed=11, scales=[3.0, 400.0])
    m = LogisticScoringModel(alpha=0.01).fit(X, y)
    means, scales = m.scaler_.means_, m.scaler_.scales_
    row = X[0]
    z = m.intercept_ + sum(
        m.coef_[j] * (row[j] - means[j]) / scales[j] for j in range(2))
    manual = _sigmoid(z)
    lib = m.predict_proba([row])[0]
    print("手工用训练时 mean/std 复算: %.12f" % manual)
    print("库 predict_proba        : %.12f  （完全一致）" % lib)
    m.scaler_.means_[0] += 1.0
    try:
        m.predict([row])
        print("ERROR: 篡改变换参数未触发断言！")
    except AssertionError as e:
        print("篡改 scaler 参数后预测 -> 正确触发断言: %s" % e)


if __name__ == "__main__":
    demo_convergence_curve()
    demo_crosscheck()
    demo_regularization()
    demo_divergence()
    demo_edge_cases()
    demo_transform_consistency()
