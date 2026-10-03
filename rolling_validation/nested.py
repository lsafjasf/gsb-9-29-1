"""Nested walk-forward validation.

For every outer fold, the inner (hyper-parameter selection) splitter only
sees samples strictly earlier than the outer validation start. Inner
validation windows therefore can never coincide with, or extend into, the
outer validation window: model selection and final evaluation never share
validation data.

NestedFold.outer : Fold evaluated against the held-out outer window.
NestedFold.inner : tuple of Fold usable for inner model selection; indices
                   are remapped to the original input positions. May be
                   empty if the pre-outer history is too short.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .splitter import Fold, RollingSplitter, SplitConfig


@dataclass(frozen=True)
class NestedFold:
    outer: Fold
    inner: Tuple[Fold, ...]


def nested_split(
    timestamps: Sequence[float],
    outer_config: SplitConfig,
    inner_config: SplitConfig,
) -> List[NestedFold]:
    outer_folds = RollingSplitter(outer_config).split(timestamps)
    result: List[NestedFold] = []
    for outer in outer_folds:
        inner_positions = [
            i for i, t in enumerate(timestamps) if t < outer.val_start
        ]
        inner_ts = [timestamps[i] for i in inner_positions]
        inner_folds = RollingSplitter(inner_config).split(inner_ts)
        remapped = tuple(
            Fold(
                fold_id=f.fold_id,
                train_idx=tuple(inner_positions[p] for p in f.train_idx),
                val_idx=tuple(inner_positions[p] for p in f.val_idx),
                train_start=f.train_start,
                train_end=f.train_end,
                val_start=f.val_start,
                val_end=f.val_end,
                n_embargo_dropped=f.n_embargo_dropped,
                n_overlap_dropped=f.n_overlap_dropped,
            )
            for f in inner_folds
        )
        result.append(NestedFold(outer=outer, inner=remapped))
    return result


def assert_nested_disjoint(nested_folds: Sequence[NestedFold]) -> bool:
    """Assert inner and outer validation data never overlap."""
    for nf in nested_folds:
        outer_val = set(nf.outer.val_idx)
        seen_inner_val = set()
        for f in nf.inner:
            if not f.val_idx:
                raise AssertionError(
                    "outer fold %d: inner fold %d has empty validation"
                    % (nf.outer.fold_id, f.fold_id)
                )
            if f.val_end >= nf.outer.val_start:
                raise AssertionError(
                    "outer fold %d: inner fold %d validation ends at t=%r, "
                    "reaching the outer validation start t=%r"
                    % (nf.outer.fold_id, f.fold_id, f.val_end,
                       nf.outer.val_start)
                )
            shared = outer_val & set(f.val_idx)
            if shared:
                raise AssertionError(
                    "outer fold %d: inner fold %d shares %d validation "
                    "sample(s) with the outer validation set"
                    % (nf.outer.fold_id, f.fold_id, len(shared))
                )
            overlap = seen_inner_val & set(f.val_idx)
            if overlap:
                raise AssertionError(
                    "outer fold %d: inner fold %d validation overlaps an "
                    "earlier inner validation window"
                    % (nf.outer.fold_id, f.fold_id)
                )
            seen_inner_val |= set(f.val_idx)
    return True
