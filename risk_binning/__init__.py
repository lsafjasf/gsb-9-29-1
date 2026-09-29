"""Risk-control feature binning: quantile & optimal binners, stdlib only."""

from .binners import (
    BaseBinner,
    OptimalBinner,
    QuantileBinner,
    MISSING_LABEL,
    OUTLIER_HIGH_LABEL,
    OUTLIER_LOW_LABEL,
    UNSEEN_LABEL,
)
from .core import (
    BinStat,
    detect_direction,
    find_violations,
    is_missing,
    monotonicity_report,
    quantile,
    suggest_merges,
    woe_iv,
)

__all__ = [
    "BaseBinner",
    "OptimalBinner",
    "QuantileBinner",
    "BinStat",
    "MISSING_LABEL",
    "OUTLIER_HIGH_LABEL",
    "OUTLIER_LOW_LABEL",
    "UNSEEN_LABEL",
    "detect_direction",
    "find_violations",
    "is_missing",
    "monotonicity_report",
    "quantile",
    "suggest_merges",
    "woe_iv",
]

__version__ = "0.1.0"
