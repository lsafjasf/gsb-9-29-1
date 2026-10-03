"""rolling_validation：仅标准库的时序滚动验证库。

- 扩张/滑动窗口切分，支持 gap / embargo / purge（重叠样本剔除）
- 泄漏检测（报错含样本编号与时间戳）
- 嵌套验证（外层评估、内层选参，验证段零重叠）
- 每折指标、折间波动与置信区间，含小样本/缺失/单折降级告警
"""

from .splitter import (
    Fold, RollingValidator, LeakageError, InsufficientDataError, check_leakage,
)
from .metrics import METRICS, mae, rmse, mape, smape, summarize
from .evaluate import evaluate, EvaluationResult, FoldResult
from .nested import (
    NestedFold, nested_splits, assert_nested_integrity, run_nested,
)

__all__ = [
    "Fold", "RollingValidator", "LeakageError", "InsufficientDataError",
    "check_leakage", "METRICS", "mae", "rmse", "mape", "smape", "summarize",
    "evaluate", "EvaluationResult", "FoldResult",
    "NestedFold", "nested_splits", "assert_nested_integrity", "run_nested",
]
