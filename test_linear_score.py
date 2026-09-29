"""linear_score 的断言测试与边界用例（仅标准库，unittest）。

运行：python3 -m unittest test_linear_score -v
"""

import math
import random
import unittest

from linear_score import (
    DivergenceError,
    NotFittedError,
    StandardScaler,
    LogisticScoringModel,
    _log1pexp,
    _sigmoid,
)


def make_data(n=200, d=3, seed=42, separable=False, scales=None, return_w=False):
    """生成二分类数据；scales 可指定每个特征的量纲放大倍数。"""
    rng = random.Random(seed)
    scales = scales or [1.0] * d
    true_w = [rng.uniform(-2, 2) for _ in range(d)]
    X, y = [], []
    while len(X) < n:
        row = [rng.gauss(0, 1) * scales[j] for j in range(d)]
        z = sum(true_w[j] * row[j] / scales[j] for j in range(d))
        if separable:
            label = 1 if z > 0 else 0
            margin = abs(z)
            if margin < 0.5:  # 拉开间隔，保证完全可分
                continue
        else:
            p = _sigmoid(z)
            label = 1 if rng.random() < p else 0
        X.append(row)
        y.append(label)
    if return_w:
        return X, y, true_w
    return X, y


class TestNumericalStability(unittest.TestCase):
    def test_log1pexp_extreme(self):
        # 朴素实现 math.log(1+math.exp(1000)) 会 OverflowError
        self.assertAlmostEqual(_log1pexp(1000.0), 1000.0, places=10)
        self.assertAlmostEqual(_log1pexp(-1000.0), 0.0, places=10)
        self.assertAlmostEqual(_log1pexp(0.0), math.log(2.0), places=12)

    def test_sigmoid_extreme(self):
        self.assertEqual(_sigmoid(1000.0), 1.0)
        self.assertEqual(_sigmoid(-1000.0), 0.0)
        self.assertAlmostEqual(_sigmoid(0.0), 0.5, places=12)

    def test_stable_loss_matches_naive_on_moderate_inputs(self):
        # 对拍：稳定实现 vs 朴素实现（在朴素实现不溢出的范围内）
        rng = random.Random(0)
        for _ in range(500):
            z = rng.uniform(-30, 30)
            yi = rng.choice([0, 1])
            stable = _log1pexp(z) - yi * z
            naive = math.log(1.0 + math.exp(z)) - yi * z
            self.assertAlmostEqual(stable, naive, places=10)

    def test_gradient_finite_difference(self):
        # 对拍：解析梯度 vs 中心差分
        X, y = make_data(n=60, d=4, seed=7)
        model = LogisticScoringModel(alpha=0.3)
        Xs = StandardScaler().fit_transform(X)
        w = [0.3, -0.7, 1.1, 0.2]
        b = -0.4
        _, gw, gb = model._loss_grad(Xs, y, w, b)
        eps = 1e-6
        for j in range(len(w)):
            wp, wm = list(w), list(w)
            wp[j] += eps
            wm[j] -= eps
            lp = model._loss_only(Xs, y, wp, b)
            lm = model._loss_only(Xs, y, wm, b)
            num = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(gw[j], num, delta=1e-5)
        lp = model._loss_only(Xs, y, w, b + eps)
        lm = model._loss_only(Xs, y, w, b - eps)
        self.assertAlmostEqual(gb, (lp - lm) / (2 * eps), delta=1e-5)


class TestTraining(unittest.TestCase):
    def test_converges_and_convergence_curve_monotone(self):
        X, y, true_w = make_data(n=300, d=3, seed=1,
                                 scales=[1.0, 100.0, 1e5], return_w=True)
        model = LogisticScoringModel(alpha=0.01).fit(X, y)
        self.assertTrue(model.converged_)
        self.assertGreater(model.n_iter_, 1)
        hist = model.loss_history_
        # 线搜索保证收敛曲线单调不增
        for a, b in zip(hist, hist[1:]):
            self.assertLessEqual(b, a + 1e-12)
        # 准确率应接近用真实系数打分的“先知”基准
        scales = [1.0, 100.0, 1e5]
        oracle = sum(
            1 for row, yi in zip(X, y)
            if (1 if sum(true_w[j] * row[j] / scales[j] for j in range(3)) > 0
                else 0) == yi) / len(y)
        self.assertGreaterEqual(model.score(X, y), oracle - 0.05)

    def test_standardization_handles_huge_scale_gap(self):
        # 量纲差异 1e5 倍也能稳定训练
        X, y = make_data(n=200, d=2, seed=3, scales=[1.0, 1e5])
        model = LogisticScoringModel(alpha=0.01).fit(X, y)
        self.assertTrue(model.converged_)
        for v in model.coef_ + [model.intercept_]:
            self.assertTrue(math.isfinite(v))

    def test_regularization_shrinks_coefficients(self):
        # 正则强度越大，系数模长越小（单调压缩，不会变号/爆炸）
        X, y = make_data(n=200, d=3, seed=5)
        norms = []
        for alpha in (0.0, 0.1, 1.0, 10.0, 100.0):
            m = LogisticScoringModel(alpha=alpha).fit(X, y)
            norms.append(math.sqrt(sum(c * c for c in m.coef_)))
        for small, big in zip(norms, norms[1:]):
            self.assertLess(big, small + 1e-9)

    def test_reference_implementation_crosscheck(self):
        # 对拍：与独立实现的朴素批量梯度下降比较最终损失与预测一致性
        X, y = make_data(n=200, d=3, seed=9)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        n, d = len(Xs), len(Xs[0])
        alpha = 0.1
        w = [0.0] * d
        b = 0.0
        lr = 0.05
        for _ in range(3000):
            gw = [0.0] * d
            gb = 0.0
            for row, yi in zip(Xs, y):
                z = b + sum(w[j] * row[j] for j in range(d))
                r = _sigmoid(z) - yi
                for j in range(d):
                    gw[j] += r * row[j]
                gb += r
            for j in range(d):
                w[j] -= lr * (gw[j] / n + alpha * w[j])
            b -= lr * gb / n
        ref_loss = LogisticScoringModel(alpha=alpha)._loss_only(Xs, y, w, b)

        model = LogisticScoringModel(alpha=alpha).fit(X, y)
        self.assertLessEqual(model.loss_history_[-1], ref_loss + 1e-6)
        ref_pred = [1 if _sigmoid(b + sum(w[j] * r[j] for j in range(d))) >= 0.5
                    else 0 for r in Xs]
        agree = sum(1 for a, bb in zip(model.predict(X), ref_pred) if a == bb)
        self.assertGreaterEqual(agree / n, 0.95)


class TestTransformConsistency(unittest.TestCase):
    def test_predict_uses_training_transform(self):
        X, y = make_data(n=150, d=2, seed=11, scales=[3.0, 400.0])
        model = LogisticScoringModel(alpha=0.01).fit(X, y)
        # 手工用训练时保存的 mean/std 复算，必须与 predict_proba 完全一致
        means, scales = model.scaler_.means_, model.scaler_.scales_
        for row, p in zip(X[:10], model.predict_proba(X[:10])):
            z = model.intercept_
            for j in range(2):
                z += model.coef_[j] * (row[j] - means[j]) / scales[j]
            self.assertAlmostEqual(p, _sigmoid(z), places=12)

    def test_transform_fingerprint_assertion(self):
        X, y = make_data(n=100, d=2, seed=13)
        model = LogisticScoringModel(alpha=0.01).fit(X, y)
        model.predict(X[:1])  # 正常时不触发断言
        model.scaler_.means_[0] += 1.0  # 篡改变换参数
        with self.assertRaises(AssertionError):
            model.predict(X[:1])

    def test_predict_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            LogisticScoringModel().predict([[1.0, 2.0]])

    def test_predict_dimension_mismatch(self):
        X, y = make_data(n=80, d=3, seed=17)
        model = LogisticScoringModel().fit(X, y)
        with self.assertRaises(ValueError):
            model.predict([[1.0, 2.0]])


class TestEdgeCases(unittest.TestCase):
    def test_perfectly_separable_with_regularization(self):
        # 完全可分：MLE 不存在，L2 正则使优化问题有界、可收敛
        X, y = make_data(n=150, d=2, seed=19, separable=True)
        model = LogisticScoringModel(alpha=0.1).fit(X, y)
        self.assertTrue(model.converged_)
        self.assertEqual(model.score(X, y), 1.0)
        for v in model.coef_ + [model.intercept_]:
            self.assertTrue(math.isfinite(v))

    def test_perfectly_separable_unregularized_no_garbage(self):
        # 无正则时可分数据损失单调趋 0、权值增长，但绝不能产出 NaN/Inf
        X, y = make_data(n=100, d=2, seed=23, separable=True)
        model = LogisticScoringModel(alpha=0.0, max_iter=2000).fit(X, y)
        for v in model.coef_ + [model.intercept_]:
            self.assertTrue(math.isfinite(v))
        self.assertEqual(model.score(X, y), 1.0)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_single_class_raises(self):
        X = [[float(i), 2.0] for i in range(20)]
        with self.assertRaises(ValueError):
            LogisticScoringModel().fit(X, [1] * 20)
        with self.assertRaises(ValueError):
            LogisticScoringModel().fit(X, [0] * 20)

    def test_all_constant_features(self):
        rng = random.Random(29)
        n = 100
        X = [[5.0, -3.0] for _ in range(n)]  # 全部常数
        y = [1 if rng.random() < 0.7 else 0 for _ in range(n)]
        if 0 not in y:
            y[0] = 0
        model = LogisticScoringModel(alpha=1.0).fit(X, y)
        self.assertTrue(model.converged_)
        # 常数特征标准化后恒为 0，系数必须恒为 0，只剩截距学先验
        for c in model.coef_:
            self.assertAlmostEqual(c, 0.0, places=12)
        p = model.predict_proba([[5.0, -3.0]])[0]
        self.assertAlmostEqual(p, sum(y) / n, delta=0.05)

    def test_zero_samples_raises(self):
        with self.assertRaises(ValueError):
            LogisticScoringModel().fit([], [])
        with self.assertRaises(ValueError):
            StandardScaler().fit([])

    def test_invalid_labels_raise(self):
        with self.assertRaises(ValueError):
            LogisticScoringModel().fit([[1.0], [2.0]], [1, 2])

    def test_nan_features_raise(self):
        with self.assertRaises(ValueError):
            LogisticScoringModel().fit([[float("nan")], [1.0]], [0, 1])


class TestDivergenceDetection(unittest.TestCase):
    def test_huge_fixed_step_raises_divergence(self):
        # 固定巨大步长：必须报 DivergenceError 而不是返回垃圾参数
        X, y = make_data(n=100, d=2, seed=31)
        model = LogisticScoringModel(
            alpha=0.0, step_search="fixed", lr=1e12, max_iter=50)
        with self.assertRaises(DivergenceError):
            model.fit(X, y)

    def test_divergence_error_not_silent_params(self):
        # 发散时报错，且模型不会把爆炸参数标记为可用
        X, y = make_data(n=100, d=2, seed=37)
        model = LogisticScoringModel(
            alpha=0.0, step_search="fixed", lr=1e15, max_iter=50)
        try:
            model.fit(X, y)
        except DivergenceError:
            pass
        self.assertIsNone(model.coef_)  # 未正常结束，参数不可用
        with self.assertRaises(NotFittedError):
            model.predict(X[:1])

    def test_backtracking_recovers_from_large_initial_step(self):
        # 自适应步长：初始步长给得离谱也能通过回溯收敛
        X, y, true_w = make_data(n=150, d=2, seed=41, return_w=True)
        model = LogisticScoringModel(alpha=0.01, step0=1e6).fit(X, y)
        self.assertTrue(model.converged_)
        oracle = sum(
            1 for row, yi in zip(X, y)
            if (1 if sum(true_w[j] * row[j] for j in range(2)) > 0
                else 0) == yi) / len(y)
        self.assertGreaterEqual(model.score(X, y), oracle - 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
