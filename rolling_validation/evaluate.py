"""评估执行器：把切分器、预测函数与指标串起来，输出每折指标与折间区间。"""

import math
from dataclasses import dataclass, field

from .metrics import METRICS, summarize, warn, _is_missing
from .splitter import RollingValidator


@dataclass
class FoldResult:
    fold_id: int
    n_train: int
    n_val: int
    n_missing_val: int           # 验证集中被剔除的缺失样本数
    val_start_ts: object
    val_end_ts: object
    metrics: dict                # 指标名 -> float 或 None（无法计算）
    warnings: list = field(default_factory=list)


@dataclass
class EvaluationResult:
    mode: str
    folds: list                  # FoldResult
    summary: dict                # 指标名 -> summarize() 结果
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {
            "mode": self.mode,
            "folds": [
                {"fold_id": f.fold_id, "n_train": f.n_train,
                 "n_val": f.n_val, "n_missing_val": f.n_missing_val,
                 "val_start_ts": f.val_start_ts, "val_end_ts": f.val_end_ts,
                 "metrics": f.metrics, "warnings": f.warnings}
                for f in self.folds
            ],
            "summary": self.summary,
            "warnings": self.warnings,
        }

    def pretty(self):
        lines = ["mode=%s  folds=%d" % (self.mode, len(self.folds))]
        names = list(self.summary)
        header = "fold  n_train  n_val  miss  " + "  ".join(
            "%8s" % n for n in names)
        lines.append(header)
        for f in self.folds:
            cells = []
            for n in names:
                v = f.metrics.get(n)
                cells.append("%8.4f" % v if v is not None else "    None")
            lines.append("%4d  %7d  %5d  %4d  %s"
                         % (f.fold_id, f.n_train, f.n_val,
                            f.n_missing_val, "  ".join(cells)))
        for n in names:
            s = self.summary[n]
            if s["mean"] is None:
                lines.append("%s: 无法计算（无有效折）" % n)
            elif s["ci"] is None:
                lines.append("%s: mean=%.4f (单折，无波动与区间)"
                             % (n, s["mean"]))
            else:
                lo, hi = s["ci"]
                lines.append("%s: mean=%.4f  std=%.4f  CI%.0f%%=[%.4f, %.4f]  (n=%d)"
                             % (n, s["mean"], s["std"], s["confidence"] * 100,
                                lo, hi, s["n"]))
        for w in self.warnings:
            lines.append("[WARN] " + w)
        return "\n".join(lines)


def evaluate(timestamps, y, forecaster, validator=None, metrics=("mae", "rmse"),
             confidence=0.95):
    """滚动评估。

    forecaster: callable(train_indices, val_indices) -> 预测序列
                （长度须等于 len(val_indices)，缺失用 None/NaN 占位）
    validator:  RollingValidator；缺省为扩张窗口 min_train=20, horizon=5
    """
    validator = validator or RollingValidator()
    for name in metrics:
        if name not in METRICS:
            raise ValueError("未知指标 %r，可选：%s" % (name, sorted(METRICS)))

    warnings_out = []
    folds = validator.split(timestamps)
    fold_results = []
    for fold in folds:
        fw = []
        preds = list(forecaster(list(fold.train_indices),
                                list(fold.val_indices)))
        if len(preds) != len(fold.val_indices):
            raise ValueError(
                "forecaster 返回 %d 个预测，但验证集有 %d 个样本"
                % (len(preds), len(fold.val_indices)))
        y_val = [y[j] for j in fold.val_indices]
        n_missing = sum(1 for v in y_val if _is_missing(v))
        if n_missing == len(y_val):
            warn("折 %d 验证值全部缺失，该折指标记为 None" % fold.fold_id, fw)
        m = {}
        for name in metrics:
            value = METRICS[name](y_val, preds)
            if value is None and n_missing < len(y_val):
                warn("折 %d 指标 %s 无法计算（真值缺失或全零），记为 None"
                     % (fold.fold_id, name), fw)
            m[name] = value
        fold_results.append(FoldResult(
            fold_id=fold.fold_id,
            n_train=len(fold.train_indices),
            n_val=len(fold.val_indices),
            n_missing_val=n_missing,
            val_start_ts=fold.val_start_ts,
            val_end_ts=fold.val_end_ts,
            metrics=m,
            warnings=fw,
        ))
        warnings_out.extend(fw)

    summary = {}
    for name in metrics:
        values = [f.metrics[name] for f in fold_results]
        valid = [v for v in values if v is not None]
        if not valid:
            warn("指标 %s 在所有折上均无法计算" % name, warnings_out)
        elif len(valid) < len(values):
            warn("指标 %s 仅有 %d/%d 折有效，汇总基于有效折"
                 % (name, len(valid), len(values)), warnings_out)
        if len(valid) == 1:
            warn("指标 %s 只有单折有效：无法估计折间波动与置信区间，"
                 "结果仅供参考" % name, warnings_out)
        elif 1 < len(valid) < 5:
            warn("指标 %s 有效折数 (%d) 很少，置信区间很不稳定，"
                 "建议增加数据或减小 horizon" % (name, len(valid)),
                 warnings_out)
        summary[name] = summarize(values, confidence=confidence)

    return EvaluationResult(mode=validator.mode, folds=fold_results,
                            summary=summary, warnings=warnings_out)
