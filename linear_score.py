"""线性打分模型（逻辑回归），仅依赖 Python 标准库。

设计要点：
1. 数值稳定：log-loss 用 logaddexp 形式、sigmoid 分支计算，|z| 再大也不溢出。
2. 正则化：L2（岭惩罚），损失 = 平均 logloss + 0.5 * alpha * ||w||^2（截距不罚）。
   alpha 越大系数被压缩得越接近 0，可解完全可分数据下 MLE 不存在（权值发散）的问题。
3. 步长：默认回溯线搜索（Armijo 充分下降条件），也支持固定步长用于演示发散检测。
4. 发散检测：损失出现 NaN/Inf、或相对初始损失放大超过 divergence_factor 倍时
   抛出 DivergenceError，绝不返回垃圾参数。
5. 标准化：训练时拟合 StandardScaler 并随模型保存，预测强制走同一套变换，
   并用指纹断言防止变换参数被篡改。
"""

import math

__all__ = [
    "DivergenceError",
    "NotFittedError",
    "StandardScaler",
    "LogisticScoringModel",
]


class DivergenceError(RuntimeError):
    """训练发散（损失 NaN/Inf 或爆炸）时抛出。"""


class NotFittedError(RuntimeError):
    """模型尚未训练就调用预测时抛出。"""


# ---------------------------------------------------------------------------
# 数值稳定的基本函数
# ---------------------------------------------------------------------------

def _log1pexp(z):
    """稳定计算 log(1 + exp(z))。"""
    if z > 0.0:
        # exp(-z) 在 z 很大时安全地下溢为 0
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _sigmoid(z):
    """稳定计算 1 / (1 + exp(-z))。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# 特征标准化
# ---------------------------------------------------------------------------

class StandardScaler:
    """均值/标准差标准化。常数特征的标准差置 1（变换后恒为 0，避免除零）。"""

    def __init__(self):
        self.means_ = None
        self.scales_ = None

    @property
    def fitted_(self):
        return self.means_ is not None

    def fit(self, X):
        n = len(X)
        if n == 0:
            raise ValueError("无法用空样本拟合 StandardScaler")
        d = len(X[0])
        means = [0.0] * d
        for row in X:
            if len(row) != d:
                raise ValueError("特征行长度不一致")
            for j, v in enumerate(row):
                means[j] += v
        means = [m / n for m in means]
        var = [0.0] * d
        for row in X:
            for j, v in enumerate(row):
                diff = v - means[j]
                var[j] += diff * diff
        scales = []
        for j in range(d):
            std = math.sqrt(var[j] / n)
            scales.append(std if std > 0.0 else 1.0)  # 常数特征：scale=1
        self.means_ = means
        self.scales_ = scales
        return self

    def transform(self, X):
        if not self.fitted_:
            raise NotFittedError("StandardScaler 尚未拟合")
        d = len(self.means_)
        out = []
        for row in X:
            if len(row) != d:
                raise ValueError(
                    "特征维度不匹配：期望 %d，实际 %d" % (d, len(row)))
            out.append([(v - self.means_[j]) / self.scales_[j]
                        for j, v in enumerate(row)])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fingerprint(self):
        """变换参数指纹，用于训练/预测一致性断言。"""
        if not self.fitted_:
            raise NotFittedError("StandardScaler 尚未拟合")
        return (tuple(round(m, 12) for m in self.means_),
                tuple(round(s, 12) for s in self.scales_))


# ---------------------------------------------------------------------------
# 线性打分模型（逻辑回归）
# ---------------------------------------------------------------------------

class LogisticScoringModel:
    """带 L2 正则、回溯线搜索与发散检测的逻辑回归打分模型。

    参数
    ----
    alpha : L2 正则强度。损失 = 平均 logloss + 0.5*alpha*||w||^2（截距不罚）。
            alpha 越大，系数模长被压得越狠；alpha=0 在完全可分数据上权值无界。
    max_iter : 最大迭代次数。
    tol : 相对损失下降小于该值判定收敛。
    step0 : 初始步长。
    step_search : 'backtracking'（Armijo 回溯线搜索）或 'fixed'（固定步长，
                  主要用于演示/触发发散检测）。
    lr : step_search='fixed' 时的固定学习率。
    c1 : Armijo 充分下降系数。
    shrink : 线搜索步长收缩因子。
    divergence_factor : 损失相对初始值放大超过该倍数判定发散。
    verbose : 训练时打印收敛曲线。
    """

    def __init__(self, alpha=1.0, max_iter=500, tol=1e-9, step0=1.0,
                 step_search="backtracking", lr=None, c1=1e-4, shrink=0.5,
                 divergence_factor=1e6, verbose=False):
        if alpha < 0.0:
            raise ValueError("alpha 必须 >= 0")
        if step_search not in ("backtracking", "fixed"):
            raise ValueError("step_search 只能是 'backtracking' 或 'fixed'")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.step0 = step0
        self.step_search = step_search
        self.lr = lr
        self.c1 = c1
        self.shrink = shrink
        self.divergence_factor = divergence_factor
        self.verbose = verbose

        self.scaler_ = None
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []
        self.step_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        self._transform_fingerprint = None

    # -- 内部：损失与梯度（数值稳定） -------------------------------------

    def _loss_grad(self, Xs, y, w, b):
        n = len(Xs)
        d = len(w)
        loss = 0.0
        gw = [0.0] * d
        gb = 0.0
        for row, yi in zip(Xs, y):
            z = b
            for j in range(d):
                z += w[j] * row[j]
            # 稳定逐项损失：log(1+exp(z)) - y*z
            loss += _log1pexp(z) - yi * z
            r = _sigmoid(z) - yi
            for j in range(d):
                gw[j] += r * row[j]
            gb += r
        loss /= n
        reg = 0.5 * self.alpha * sum(wj * wj for wj in w)
        loss += reg
        inv_n = 1.0 / n
        for j in range(d):
            gw[j] = gw[j] * inv_n + self.alpha * w[j]  # 截距不正则化
        gb *= inv_n
        return loss, gw, gb

    def _loss_only(self, Xs, y, w, b):
        n = len(Xs)
        loss = 0.0
        for row, yi in zip(Xs, y):
            z = b
            for j, wj in enumerate(w):
                z += wj * row[j]
            loss += _log1pexp(z) - yi * z
        loss /= n
        return loss + 0.5 * self.alpha * sum(wj * wj for wj in w)

    @staticmethod
    def _check_finite(value, what):
        if math.isnan(value) or math.isinf(value):
            raise DivergenceError(
                "训练发散：%s 出现非有限值 %r，请减小步长或检查数据" % (what, value))

    # -- 训练 -------------------------------------------------------------

    def fit(self, X, y):
        X = [list(row) for row in X]
        y = list(y)
        n = len(X)
        if n == 0:
            raise ValueError("训练样本为空（n_samples=0）")
        if len(y) != n:
            raise ValueError("X 与 y 的样本数不一致")
        label_set = set(y)
        if not label_set <= {0, 1}:
            raise ValueError("标签必须是 0/1，实际取值：%r" % sorted(label_set))
        if len(label_set) < 2:
            raise ValueError("只有一个类别 %r，无法训练二分类模型" % sorted(label_set))
        d = len(X[0])
        if d == 0:
            raise ValueError("特征维度为 0")
        for row in X:
            if len(row) != d:
                raise ValueError("特征行长度不一致")
            for v in row:
                if math.isnan(v) or math.isinf(v):
                    raise ValueError("特征含 NaN/Inf")

        # 标准化：scaler 随模型保存，预测路径强制复用同一套变换
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self._transform_fingerprint = self.scaler_.fingerprint()

        w = [0.0] * d
        b = 0.0
        loss, gw, gb = self._loss_grad(Xs, y, w, b)
        self._check_finite(loss, "初始损失")
        initial_loss = max(loss, 1e-300)
        step = self.lr if self.step_search == "fixed" else self.step0

        self.loss_history_ = [loss]
        self.step_history_ = []
        self.converged_ = False

        for it in range(1, self.max_iter + 1):
            grad_norm2 = sum(g * g for g in gw) + gb * gb
            self._check_finite(grad_norm2, "梯度")

            if self.step_search == "fixed":
                new_w = [w[j] - step * gw[j] for j in range(d)]
                new_b = b - step * gb
                new_loss, new_gw, new_gb = self._loss_grad(Xs, y, new_w, new_b)
                used_step = step
            else:
                # Armijo 回溯线搜索：沿负梯度方向寻找充分下降步长
                t = step
                new_w = new_b = new_loss = new_gw = new_gb = None
                while True:
                    cand_w = [w[j] - t * gw[j] for j in range(d)]
                    cand_b = b - t * gb
                    cand_loss = self._loss_only(Xs, y, cand_w, cand_b)
                    if (not math.isnan(cand_loss) and not math.isinf(cand_loss)
                            and cand_loss <= loss - self.c1 * t * grad_norm2):
                        new_w, new_b = cand_w, cand_b
                        new_loss, new_gw, new_gb = self._loss_grad(
                            Xs, y, new_w, new_b)
                        break
                    t *= self.shrink
                    if t < 1e-16:
                        if grad_norm2 < self.tol:
                            new_w, new_b = w, b
                            new_loss, new_gw, new_gb = loss, gw, gb
                            break
                        raise DivergenceError(
                            "线搜索失败：步长收缩到 1e-16 仍无充分下降，"
                            "当前损失 %r" % (loss,))
                used_step = t
                # 成功一步后略微放大下一步的试探步长，减少回溯次数
                step = min(t / self.shrink, 1e4)

            # ---- 发散检测：宁可报错，绝不返回垃圾参数 ----
            self._check_finite(new_loss, "损失（第 %d 次迭代）" % it)
            if new_loss > self.divergence_factor * initial_loss:
                raise DivergenceError(
                    "训练发散：损失从 %.6g 爆炸到 %.6g（第 %d 次迭代），"
                    "请减小步长或加强正则" % (initial_loss, new_loss, it))

            self.loss_history_.append(new_loss)
            self.step_history_.append(used_step)
            if self.verbose:
                print("iter %4d  loss=%.10f  step=%.3e" % (it, new_loss, used_step))

            if abs(loss - new_loss) <= self.tol * max(1.0, abs(loss)):
                self.converged_ = True
                w, b = new_w, new_b
                self.n_iter_ = it
                break
            w, b, loss, gw, gb = new_w, new_b, new_loss, new_gw, new_gb
            self.n_iter_ = it

        self.coef_ = list(w)
        self.intercept_ = b
        return self

    # -- 预测（强制走训练时同一套标准化变换） ------------------------------

    def _assert_transform_consistent(self):
        if self.scaler_ is None or self.coef_ is None:
            raise NotFittedError("模型尚未训练")
        # 断言：预测用的变换与训练时保存的指纹完全一致
        assert self.scaler_.fitted_, "预测路径必须使用已拟合的 StandardScaler"
        assert self.scaler_.fingerprint() == self._transform_fingerprint, (
            "标准化参数与训练时不一致：预测路径必须使用训练时的同一套变换")

    def decision_function(self, X):
        self._assert_transform_consistent()
        Xs = self.scaler_.transform([list(row) for row in X])
        out = []
        for row in Xs:
            z = self.intercept_
            for j, wj in enumerate(self.coef_):
                z += wj * row[j]
            out.append(z)
        return out

    def predict_proba(self, X):
        return [_sigmoid(z) for z in self.decision_function(X)]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def score(self, X, y):
        y = list(y)
        pred = self.predict(X)
        if not y:
            raise ValueError("评估样本为空")
        return sum(1 for a, b in zip(pred, y) if a == b) / len(y)
