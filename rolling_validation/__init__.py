"""Rolling (walk-forward) validation on time series, stdlib only.

Public API:
    SplitConfig, RollingSplitter, Fold
    LeakageError, assert_no_leakage, check_folds
    evaluate, summarize, MetricSummary, FoldMetric
    nested_split, assert_nested_disjoint, NestedFold
"""
from .splitter import (
    Fold,
    RollingSplitter,
    SplitConfig,
    SplitConfigError,
)
from .leakage import LeakageError, assert_no_leakage, check_folds
from .metrics import (
    FoldMetric,
    MetricSummary,
    accuracy,
    evaluate,
    mae,
    mse,
    rmse,
    summarize,
)
from .nested import NestedFold, assert_nested_disjoint, nested_split

__all__ = [
    "Fold",
    "RollingSplitter",
    "SplitConfig",
    "SplitConfigError",
    "LeakageError",
    "assert_no_leakage",
    "check_folds",
    "FoldMetric",
    "MetricSummary",
    "accuracy",
    "evaluate",
    "mae",
    "mse",
    "rmse",
    "summarize",
    "NestedFold",
    "assert_nested_disjoint",
    "nested_split",
]
