"""滚动验证切分器：扩张窗口 / 滑动窗口，含净化(purge)与隔离带(embargo)。

所有切分在"按时间戳排序后的位置空间"中进行，再映射回原始输入下标，
因此乱序输入与重复时间戳都能得到确定性的折划分。
"""

from dataclasses import dataclass, field


class LeakageError(ValueError):
    """训练样本时间戳晚于（或并列于）验证起点时抛出。"""


class InsufficientDataError(ValueError):
    """序列过短，连一折都构造不出来时抛出。"""


@dataclass(frozen=True)
class Fold:
    """一折划分。indices 均为原始输入下标（不是排序后位置）。"""
    fold_id: int
    train_indices: tuple
    val_indices: tuple
    train_end_ts: object   # 训练集最大时间戳
    val_start_ts: object   # 验证集最小时间戳
    val_end_ts: object     # 验证集最大时间戳
    purged_indices: tuple = ()    # 因标签窗口与验证区间重叠被剔除的训练样本
    embargoed_indices: tuple = () # 因落在上一折验证区之后隔离带内被剔除的样本


def _sorted_order(timestamps):
    """稳定排序：按 (时间戳, 原始下标)，重复时间戳保持输入先后。"""
    return sorted(range(len(timestamps)), key=lambda i: (timestamps[i], i))


def check_leakage(train_indices, val_indices, timestamps, allow_ties=False):
    """泄漏检测：任何训练样本时间戳晚于验证起点即报错。

    默认 (allow_ties=False) 下，与验证起点时间戳相同的训练样本也算泄漏
    （同一时刻的样本无法区分先后，存在信息串扰风险）。

    报错信息包含具体样本编号（原始输入下标）与其时间戳。
    """
    if not val_indices:
        raise ValueError("验证集为空，无法执行泄漏检测")
    val_start = min(timestamps[j] for j in val_indices)
    offenders = []
    for i in train_indices:
        ts = timestamps[i]
        if ts > val_start or (ts == val_start and not allow_ties):
            offenders.append((i, ts))
    if offenders:
        details = "; ".join(
            "样本 #%d (t=%r)" % (i, ts) for i, ts in offenders[:10]
        )
        if len(offenders) > 10:
            details += "; ... 共 %d 个" % len(offenders)
        raise LeakageError(
            "检测到时间泄漏：%d 个训练样本的时间戳不早于验证起点 %r -> %s"
            % (len(offenders), val_start, details)
        )
    return True


class RollingValidator:
    """滚动（walk-forward）验证切分器。

    参数（除 mode/window/step 外均以"样本数"计）：
      mode:      "expanding" 扩张窗口 | "sliding" 滑动窗口
      min_train: 最小训练样本量（第一折验证起点之前必须有的样本数）
      horizon:   预测长度，即每折验证集大小
      gap:       训练结束与验证起点之间的间隔样本数（隔离带前段）
      embargo:   每折验证区间之后、禁止进入后续训练集的样本数（隔离带后段）
      window:    sliding 模式的训练窗口长度；缺省等于 min_train
      step:      相邻两折验证起点间距；缺省等于 horizon（不重叠）
      allow_ties: 是否允许训练/验证边界两侧出现相同时间戳
    """

    def __init__(self, mode="expanding", min_train=20, horizon=5,
                 gap=0, embargo=0, window=None, step=None, allow_ties=False):
        if mode not in ("expanding", "sliding"):
            raise ValueError("mode 必须是 'expanding' 或 'sliding'")
        if min_train < 1 or horizon < 1:
            raise ValueError("min_train 与 horizon 必须 >= 1")
        if gap < 0 or embargo < 0:
            raise ValueError("gap 与 embargo 必须 >= 0")
        self.mode = mode
        self.min_train = min_train
        self.horizon = horizon
        self.gap = gap
        self.embargo = embargo
        self.window = window if window is not None else min_train
        self.step = step if step is not None else horizon
        self.allow_ties = allow_ties
        if self.mode == "sliding" and self.window < 1:
            raise ValueError("sliding 模式 window 必须 >= 1")

    def split(self, timestamps):
        """生成 Fold 列表。timestamps 可乱序、可含重复值。"""
        n = len(timestamps)
        if n == 0:
            raise InsufficientDataError("输入为空，无法切分")
        order = _sorted_order(timestamps)
        sorted_ts = [timestamps[i] for i in order]

        folds = []
        embargo_zones = []  # 排序位置空间中的 [start, end) 禁区
        # 隔离量 = max(gap, horizon-1)：gap 负责前段间隔，horizon-1 来自
        # 净化——训练样本 i 的标签窗口 [i, i+horizon) 不得探入验证起点 v。
        separation = max(self.gap, self.horizon - 1)
        base = self.window if self.mode == "sliding" else self.min_train
        v = base + separation  # 第一个验证起点（已保证净化后训练量充足）
        fold_id = 0
        while v + self.horizon <= n:
            val_pos = list(range(v, v + self.horizon))
            cutoff = v - separation  # 净化+间隔后训练右端（不含）

            if self.mode == "expanding":
                cand = list(range(0, cutoff))
            else:
                # 滑动窗口锚定在净化截止处，保证净化后窗口长度恰为 window
                cand = list(range(max(0, cutoff - self.window), cutoff))

            # 净化(purge)信息带：这些样本的标签窗口与验证区间重叠
            # （i + horizon > v），即使没有 gap 也必须排除，这里记录供审计。
            overlap_start = v - self.horizon + 1
            overlap_end = v - self.gap
            purged = [i for i in range(overlap_start, overlap_end)
                      if i + self.horizon > v and i >= 0]
            train_pos = cand

            # 隔离带(embargo)：落在既往验证区间之后 embargo 样本内的点，
            # 不得进入当前及以后的训练集。
            embargoed = [
                i for i in train_pos
                if any(a <= i < b for a, b in embargo_zones)
            ]
            train_pos = [
                i for i in train_pos
                if not any(a <= i < b for a, b in embargo_zones)
            ]
            embargo_zones.append(
                (v + self.horizon, min(n, v + self.horizon + self.embargo))
            )

            if len(train_pos) >= self.min_train:
                train_idx = tuple(order[i] for i in train_pos)
                val_idx = tuple(order[i] for i in val_pos)
                # 每折构造后都做泄漏自检（防御性：切分逻辑之外的最后一道闸）
                check_leakage(train_idx, val_idx, timestamps,
                              allow_ties=self.allow_ties)
                folds.append(Fold(
                    fold_id=fold_id,
                    train_indices=train_idx,
                    val_indices=val_idx,
                    train_end_ts=sorted_ts[train_pos[-1]],
                    val_start_ts=sorted_ts[val_pos[0]],
                    val_end_ts=sorted_ts[val_pos[-1]],
                    purged_indices=tuple(order[i] for i in purged),
                    embargoed_indices=tuple(order[i] for i in embargoed),
                ))
                fold_id += 1
            v += self.step

        if not folds:
            need = base + separation + self.horizon
            raise InsufficientDataError(
                "序列过短：n=%d，构造一折至少需要 %d 个样本"
                "（%s 基准 %d + 隔离量 max(gap=%d, horizon-1=%d)=%d "
                "+ horizon=%d；隔离带 embargo 可能进一步减少可用样本）"
                % (n, need, self.mode, base, self.gap, self.horizon - 1,
                   separation, self.horizon)
            )
        return folds

    def n_splits(self, timestamps):
        return len(self.split(timestamps))
