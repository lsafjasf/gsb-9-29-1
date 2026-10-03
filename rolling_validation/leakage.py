"""Leakage detection between training and validation samples.

Rule: every training sample must be *strictly earlier* than the validation
start. A training sample whose timestamp equals the validation start (e.g.
duplicate timestamps straddling the cut) is treated as leakage, because the
two samples cannot be ordered causally.

Violations report the ORIGINAL sample index (position in the input list,
which may be shuffled) together with the offending timestamp.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

Violation = Tuple[int, float, float]


class LeakageError(ValueError):
    """Raised when training samples are not strictly before validation."""

    def __init__(self, violations: List[Violation]):
        self.violations = violations
        lines = [
            "leakage detected: %d training sample(s) at or after the "
            "validation start:" % len(violations)
        ]
        for idx, ts, vstart in violations[:20]:
            lines.append(
                "  sample #%d: t=%r >= validation start t=%r"
                % (idx, ts, vstart)
            )
        if len(violations) > 20:
            lines.append("  ... and %d more" % (len(violations) - 20))
        super().__init__("\n".join(lines))


def find_leaks(
    timestamps: Sequence[float],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> List[Violation]:
    if not val_idx:
        raise ValueError("validation index set is empty")
    vstart = min(timestamps[j] for j in val_idx)
    return [
        (i, timestamps[i], vstart)
        for i in train_idx
        if timestamps[i] >= vstart
    ]


def assert_no_leakage(
    timestamps: Sequence[float],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> bool:
    violations = find_leaks(timestamps, train_idx, val_idx)
    if violations:
        raise LeakageError(violations)
    return True


def check_folds(timestamps: Sequence[float], folds) -> bool:
    for fold in folds:
        assert_no_leakage(timestamps, fold.train_idx, fold.val_idx)
    return True
