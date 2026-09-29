"""logistic_model.py — 风控线性打分模型（逻辑回归），仅依赖 Python 标准库。

设计要点
========
1. 数值稳定
   - 损失用 log(1+exp(-y*z)) 的稳定形式: max(0,-y*z) + log1p(exp(-|y*z|))，
     避免 exp 上溢 / log(0)。
   - sigmoid 分正负两支计算，避免 exp(-z) 溢出。

2. 正则化（L2，系数 l2）
   - 目标函数: mean(logloss) + 0.5 * l2 * ||w||^2，只惩罚权重 w，不惩罚截距 b。
   - 对系数的影响: l2 把 w 向 0 收缩（ridge 效应），l2 越大 ||w|| 越小、
     模型越平滑、方差越低；在完全可分数据上，无正则时 ||w|| 会趋向无穷
     （发散），L2 使最优解存在且唯一，是本库防止"垃圾参数"的关键手段。

3. 步长策略
   - 自适应回溯线搜索（Armijo 充分下降条件）：每步从上一步步长放大 1.2 倍
     起步，不满足 Armijo 就减半，兼顾收敛速度与稳定性，无需手调学习率。

4. 发散检测（抛 DivergenceError，绝不返回垃圾参数）
   - 损失出现 NaN / Inf；
   - 线搜索步长被压到 MIN_STEP 以下仍无法满足 Armijo；
   - 参数范数超过 max_param_norm（典型场景：可分数据 + 无正则）。
   - 完全可分证书：l2=0 时若当前参数已把所有样本正确分类（最小间隔 > 0），
     则将 w 按比例放大可令损失趋向 0，MLE 不存在，立即报错。

5. 特征标准化
   - StandardScaler 在 fit 时记录 mean_/scale_（零方差特征 scale 取 1，避免除零），
     模型持有同一份 scaler，predict 路径强制复用，并用指纹断言保证
     训练与预测使用的是同一套变换参数。
"""

import math

__all__ = [
    "DivergenceError",
    "StandardScaler",
    "LogisticScorer",
]

MIN_STEP = 1e-12          # 线搜索最小步长，低于此值判定失败
ARMIJO_C = 1e-4           # Armijo 充分下降系数
SHRINK = 0.5              # 回溯收缩因子
GROW = 1.2                # 步长自适应放大因子


class DivergenceError(RuntimeError):
    """训练发散时抛出。携带原因说明，调用方应视为训练失败。"""


def _sigmoid(z):
    """数值稳定的 sigmoid。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _log_loss_term(y, z):
    """数值稳定的逐样本 logloss: softplus(z) - y*z，y ∈ {0,1}。

    softplus(z) = log(1+exp(z)) = max(0,z) + log1p(exp(-|z|))，避免溢出。
    """
    return max(0.0, z) + math.log1p(math.exp(-abs(z))) - y * z


class StandardScaler:
    """特征标准化：z = (x - mean) / scale。零方差特征 scale=1（变换后为全 0 列）。"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        n = len(X)
        if n == 0:
            raise ValueError("cannot fit StandardScaler on 0 samples")
        d = len(X[0])
        means = [0.0] * d
        for row in X:
            if len(row) != d:
                raise ValueError("ragged input: rows have different lengths")
            for j, v in enumerate(row):
                means[j] += v
        means = [m / n for m in means]
        vars_ = [0.0] * d
        for row in X:
            for j, v in enumerate(row):
                diff = v - means[j]
                vars_[j] += diff * diff
        self.mean_ = means
        # 零方差（常数）特征 scale 置 1，避免除零；变换后该列恒为 0。
        # 用相对容差判定：常数列的浮点方差是 ~eps^2 量级而非精确 0。
        self.scale_ = []
        for m, v in zip(means, vars_):
            var = v / n
            if var > 1e-16 * max(1.0, m * m):
                self.scale_.append(math.sqrt(var))
            else:
                self.scale_.append(1.0)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("StandardScaler is not fitted")
        d = len(self.mean_)
        out = []
        for row in X:
            if len(row) != d:
                raise ValueError(
                    "feature dimension mismatch: scaler fitted with %d, got %d"
                    % (d, len(row))
                )
            out.append([
                (v - self.mean_[j]) / self.scale_[j] for j, v in enumerate(row)
            ])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fingerprint(self):
        """变换参数的指纹，用于断言训练/预测使用同一套变换。"""
        if self.mean_ is None:
            raise ValueError("StandardScaler is not fitted")
        return (tuple(round(m, 12) for m in self.mean_),
                tuple(round(s, 12) for s in self.scale_))


class LogisticScorer:
    """带 L2 正则的逻辑回归打分器。

    参数
    ----
    l2 : float
        L2 正则强度。>0 时把系数向 0 收缩，保证完全可分数据上也有有限最优解；
        =0 时在可分数据上会触发 DivergenceError（参数范数爆炸检测）。
    tol : float
        相邻迭代损失相对变化小于 tol 判定收敛。
    max_iter : int
        最大迭代次数，超过仍未收敛抛 DivergenceError。
    max_param_norm : float
        参数（含截距）无穷范数上限，超过判定发散。
    """

    def __init__(self, l2=1.0, tol=1e-9, grad_tol=1e-6, max_iter=1000,
                 max_param_norm=1e6):
        if l2 < 0.0:
            raise ValueError("l2 must be >= 0")
        self.l2 = float(l2)
        self.tol = float(tol)
        self.grad_tol = float(grad_tol)
        self.max_iter = int(max_iter)
        self.max_param_norm = float(max_param_norm)
        # fit 后可用
        self.scaler_ = None
        self.coef_ = None       # 标准化空间中的权重
        self.intercept_ = None
        self.history_ = []      # 收敛曲线: dict(iter, loss, step, grad_norm)
        self.converged_ = False
        self._fit_fingerprint = None

    # ---------- 内部：损失与梯度 ----------

    @staticmethod
    def _min_margin(Xs, y, w, b):
        """所有样本的最小分类间隔 (2y-1)*z；>0 表示被完全正确分开。"""
        m = float("inf")
        for xi, yi in zip(Xs, y):
            z = b
            for j in range(len(w)):
                z += w[j] * xi[j]
            margin = (2.0 * yi - 1.0) * z
            if margin < m:
                m = margin
        return m

    def _loss_grad(self, Xs, y, w, b):
        """返回 (正则化平均损失, grad_w, grad_b)。全部使用稳定公式。"""
        n = len(Xs)
        d = len(w)
        grad_w = [0.0] * d
        grad_b = 0.0
        loss = 0.0
        for xi, yi in zip(Xs, y):
            z = b
            for j in range(d):
                z += w[j] * xi[j]
            loss += _log_loss_term(yi, z)
            # 梯度项 p - y，p 用稳定 sigmoid
            p = _sigmoid(z)
            diff = p - yi
            for j in range(d):
                grad_w[j] += diff * xi[j]
            grad_b += diff
        loss /= n
        inv_n = 1.0 / n
        for j in range(d):
            grad_w[j] = grad_w[j] * inv_n + self.l2 * w[j]
        grad_b *= inv_n
        loss += 0.5 * self.l2 * sum(wj * wj for wj in w)
        return loss, grad_w, grad_b

    # ---------- 训练 ----------

    def fit(self, X, y):
        # ---- 输入校验（边界情形） ----
        if len(X) == 0:
            raise ValueError("cannot fit on 0 samples")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch: %d vs %d"
                             % (len(X), len(y)))
        y = [float(v) for v in y]
        classes = set(y)
        if not classes <= {0.0, 1.0}:
            raise ValueError("labels must be 0/1, got %s" % sorted(classes))
        if len(classes) < 2:
            raise ValueError(
                "single-class data (all y=%d): logistic model is undefined"
                % int(y[0]))

        # ---- 标准化（fit 时确定变换，predict 强制复用同一份） ----
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self._fit_fingerprint = self.scaler_.fingerprint()

        n, d = len(Xs), len(Xs[0])
        w = [0.0] * d
        b = 0.0
        self.history_ = []
        self.converged_ = False

        loss, gw, gb = self._loss_grad(Xs, y, w, b)
        step = 1.0
        prev_loss = loss

        for it in range(self.max_iter):
            grad_norm = math.sqrt(sum(g * g for g in gw) + gb * gb)
            self.history_.append({
                "iter": it, "loss": loss, "step": step,
                "grad_norm": grad_norm,
            })
            if not math.isfinite(loss) or not math.isfinite(grad_norm):
                raise DivergenceError(
                    "non-finite loss/gradient at iter %d (loss=%r)" % (it, loss))

            # ---- 收敛判定：损失相对变化足够小 且 梯度范数足够小 ----
            # （可分数据上 loss->0 时 Δloss 也会变小，但梯度仍在推动参数外扩，
            #   双条件避免把"趋向无穷的过程"误判为收敛）
            if (it > 0
                    and abs(prev_loss - loss) <= self.tol * max(1.0, abs(prev_loss))
                    and grad_norm <= self.grad_tol):
                self.converged_ = True
                break
            prev_loss = loss

            # ---- 自适应回溯线搜索（Armijo） ----
            g_sq = sum(g * g for g in gw) + gb * gb
            trial_step = step * GROW
            ok = False
            while trial_step >= MIN_STEP:
                w_new = [w[j] - trial_step * gw[j] for j in range(d)]
                b_new = b - trial_step * gb
                loss_new, gw_new, gb_new = self._loss_grad(Xs, y, w_new, b_new)
                if (math.isfinite(loss_new)
                        and loss_new <= loss - ARMIJO_C * trial_step * g_sq):
                    ok = True
                    break
                trial_step *= SHRINK
            if not ok:
                raise DivergenceError(
                    "line search failed at iter %d: no step >= %g satisfies "
                    "Armijo decrease (loss=%.6g, grad_norm=%.6g)"
                    % (it, MIN_STEP, loss, grad_norm))

            w, b, gw, gb = w_new, b_new, gw_new, gb_new
            step = trial_step
            loss = loss_new

            # ---- 发散检测：参数范数爆炸（可分 + 无正则的典型表现） ----
            param_norm = max([abs(v) for v in w] + [abs(b)])
            if param_norm > self.max_param_norm:
                raise DivergenceError(
                    "parameter norm %g exceeded max_param_norm=%g at iter %d; "
                    "data may be linearly separable — set l2 > 0"
                    % (param_norm, self.max_param_norm, it))

            # ---- 发散检测：完全可分证书（l2=0 时 MLE 不存在） ----
            if self.l2 == 0.0 and self._min_margin(Xs, y, w, b) > 0.0:
                raise DivergenceError(
                    "data is perfectly separated at iter %d and l2=0: the "
                    "unregularized MLE does not exist (||w|| -> inf). "
                    "Set l2 > 0 to get finite, well-defined coefficients." % it)
        else:
            raise DivergenceError(
                "not converged within max_iter=%d (last loss=%.6g, "
                "grad_norm=%.6g)" % (self.max_iter, loss, grad_norm))

        self.coef_ = w
        self.intercept_ = b
        return self

    # ---------- 预测（强制复用训练时的变换） ----------

    def _check_ready(self, X):
        if self.scaler_ is None or self.coef_ is None:
            raise ValueError("model is not fitted")
        # 断言：预测路径使用的变换与训练时拟合的变换是同一套参数
        assert self.scaler_.fingerprint() == self._fit_fingerprint, \
            "scaler changed after fit: train/predict transform mismatch"
        d = len(self.coef_)
        for row in X:
            if len(row) != d:
                raise ValueError(
                    "feature dimension mismatch: model fitted with %d, got %d"
                    % (d, len(row)))

    def decision_function(self, X):
        """对原始（未标准化）特征打分，内部自动套用训练时的标准化。"""
        self._check_ready(X)
        Xs = self.scaler_.transform(X)
        out = []
        for xi in Xs:
            z = self.intercept_
            for j, wj in enumerate(self.coef_):
                z += wj * xi[j]
            out.append(z)
        return out

    def predict_proba(self, X):
        return [_sigmoid(z) for z in self.decision_function(X)]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    # ---------- 收敛曲线导出 ----------

    def convergence_csv(self):
        """返回收敛曲线 CSV 文本：iter,loss,step,grad_norm。"""
        lines = ["iter,loss,step,grad_norm"]
        for h in self.history_:
            lines.append("%d,%.12g,%.6g,%.6g"
                         % (h["iter"], h["loss"], h["step"], h["grad_norm"]))
        return "\n".join(lines) + "\n"
