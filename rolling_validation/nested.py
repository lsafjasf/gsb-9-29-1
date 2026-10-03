"""嵌套滚动验证：外层折用于泛化评估，内层折只在外层训练集内做选参。

外层评估与内层选参绝不共用同一段验证数据，由 assert_nested_integrity 强制。
支持按目标(target)逐目标进行：同一套时间轴折划分，每个目标独立选参与评估。
"""

from dataclasses import dataclass

from .splitter import RollingValidator, Fold, check_leakage


@dataclass(frozen=True)
class NestedFold:
    """一折外层 + 其全部内层折。所有 indices 均为原始输入下标。"""
    outer: Fold
    inner: tuple        # tuple[Fold, ...]，其 train/val 全部来自 outer.train_indices


def nested_splits(timestamps, outer_validator, inner_validator=None):
    """生成嵌套折。

    内层折是在外层训练集（其子时间序列）上用 inner_validator 切出来的，
    外层验证集对内层完全不可见。
    """
    if inner_validator is None:
        inner_validator = outer_validator
    result = []
    for outer in outer_validator.split(timestamps):
        outer_train = list(outer.train_indices)
        sub_ts = [timestamps[i] for i in outer_train]
        inner_folds = []
        for sub_fold in inner_validator.split(sub_ts):
            # 子序列下标映射回全局原始下标
            train_idx = tuple(outer_train[j] for j in sub_fold.train_indices)
            val_idx = tuple(outer_train[j] for j in sub_fold.val_indices)
            purged = tuple(outer_train[j] for j in sub_fold.purged_indices)
            embargoed = tuple(outer_train[j]
                              for j in sub_fold.embargoed_indices)
            fold = Fold(
                fold_id=sub_fold.fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                train_end_ts=sub_fold.train_end_ts,
                val_start_ts=sub_fold.val_start_ts,
                val_end_ts=sub_fold.val_end_ts,
                purged_indices=purged,
                embargoed_indices=embargoed,
            )
            check_leakage(fold.train_indices, fold.val_indices, timestamps,
                          allow_ties=inner_validator.allow_ties)
            inner_folds.append(fold)
        result.append(NestedFold(outer=outer, inner=tuple(inner_folds)))
    return result


def assert_nested_integrity(nf, timestamps):
    """硬性断言：内外层验证数据零重叠、时间上完全隔离、内层训练无泄漏。

    任何一条不满足即 AssertionError。供测试与生产代码直接调用。
    """
    outer_val = set(nf.outer.val_indices)
    outer_train = set(nf.outer.train_indices)

    for k, inner in enumerate(nf.inner):
        inner_val = set(inner.val_indices)
        inner_train = set(inner.train_indices)

        # 1. 内外层验证段零重叠（最核心的一条）
        overlap = inner_val & outer_val
        assert not overlap, (
            "内层折 %d 与外层折共用 %d 个验证样本: %s"
            % (k, len(overlap), sorted(overlap)))

        # 2. 内层的一切都必须来自外层训练集
        assert not (inner_val - outer_train), (
            "内层折 %d 的验证样本越出外层训练集: %s"
            % (k, sorted(inner_val - outer_train)))
        assert not (inner_train - outer_train), (
            "内层折 %d 的训练样本越出外层训练集" % k)

        # 3. 时间隔离：内层最晚的验证时间戳必须早于外层验证起点
        latest_inner_val = max(timestamps[i] for i in inner.val_indices)
        assert latest_inner_val < nf.outer.val_start_ts, (
            "内层折 %d 验证结束时间 %r 不早于外层验证起点 %r"
            % (k, latest_inner_val, nf.outer.val_start_ts))

        # 4. 内层折自身泄漏检查（严格：含并列时间戳）
        check_leakage(inner.train_indices, inner.val_indices, timestamps)

    # 5. 外层折自身泄漏检查
    check_leakage(nf.outer.train_indices, nf.outer.val_indices, timestamps)
    return True


def run_nested(timestamps, targets, score, outer_validator, inner_validator,
               param_grid):
    """按目标做嵌套验证的通用执行骨架。

    targets: {目标名: 该目标的 y 序列}
    score:   callable(target_name, params, train_idx, val_idx, y) -> 标量分数，
             越小越好（如均方误差）
    param_grid: 待选参数列表（任意可哈希/可比较对象）
    返回: [{outer_fold_id, target, best_param, outer_score, inner_scores}]
    """
    nf_list = nested_splits(timestamps, outer_validator, inner_validator)
    records = []
    for nf in nf_list:
        assert_nested_integrity(nf, timestamps)
        for target, y in targets.items():
            inner_scores = {}
            for params in param_grid:
                vals = [score(target, params, list(f.train_indices),
                              list(f.val_indices), y) for f in nf.inner]
                vals = [v for v in vals if v is not None]
                inner_scores[params] = sum(vals) / len(vals) if vals else None
            valid = {p: s for p, s in inner_scores.items() if s is not None}
            best_param = min(valid, key=valid.get) if valid else None
            outer_score = (
                score(target, best_param, list(nf.outer.train_indices),
                      list(nf.outer.val_indices), y)
                if best_param is not None else None
            )
            records.append({
                "outer_fold_id": nf.outer.fold_id,
                "target": target,
                "best_param": best_param,
                "outer_score": outer_score,
                "inner_scores": inner_scores,
            })
    return records
