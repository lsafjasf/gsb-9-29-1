"""
morphology.py — 二值 / 灰度形态学运算（纯标准库）。

支持：腐蚀 erode、膨胀 dilate、开运算 opening、闭运算 closing。
图像表示：list[list[number]]，二值图取 0/1，灰度图取任意可比较数值（如 0..255）。
结构元素（SE）：list[list[bool]]，真值表示属于结构元素，锚点取几何中心 (h//2, w//2)。

边界填充规则（所有运算一致）：
    越界像素取对应运算的“中性元”——腐蚀视为 +inf、膨胀视为 -inf，
    等价于“窗口裁剪”：超出图像边界的采样点直接被忽略。
    推论：若某像素的窗口完全落在图像外（仅当 SE 不含锚点且 SE 远大于图像时
    可能出现），腐蚀结果为 +inf、膨胀结果为 -inf；只要锚点属于 SE 就不会发生。

开/闭运算定义（伴随对，保证任意 SE 下幂等）：
    opening = dilate(erode(img, B), reflect(B))
    closing = erode(dilate(img, B), reflect(B))
    其中 reflect 为绕锚点 180° 反射。对对称 SE 反射是恒等，无差别。

大结构元素加速（decompose）：
    1. 若 SE 是实心矩形（Minkowski 和 H ⊕ V），拆成水平 + 垂直两次 1D 滤波，
       每像素 O(1)，与窗口大小无关（van Herk / Gil-Werman 算法）。
    2. 任意形状 SE 拆成若干水平线段（按 dx 区间分组去重），
       腐蚀 = 各线段 1D 腐蚀的逐点 min，膨胀 = 逐点 max。
       复杂度 O(不同行段数) 每像素，优于暴力的 O(|SE|)。
"""

from math import inf

__all__ = [
    "erode", "dilate", "opening", "closing",
    "erode_brute", "dilate_brute", "opening_brute", "closing_brute",
    "decompose",
]


# ---------------------------------------------------------------- 基础校验

def _check_image(img):
    if not img or not img[0]:
        raise ValueError("image must be a non-empty 2D list")
    w = len(img[0])
    if any(len(row) != w for row in img):
        raise ValueError("image rows must all have the same length")
    return len(img), w


def _se_to_offsets(se):
    """SE 网格 -> 相对锚点的偏移列表。"""
    if not se or not se[0]:
        raise ValueError("structuring element must be a non-empty 2D list")
    h, w = len(se), len(se[0])
    if any(len(row) != w for row in se):
        raise ValueError("structuring element rows must all have the same length")
    ay, ax = h // 2, w // 2
    offsets = [(y - ay, x - ax) for y in range(h) for x in range(w) if se[y][x]]
    if not offsets:
        raise ValueError("structuring element must contain at least one True cell")
    return offsets


# ------------------------------------------------------- van Herk 1D 滤波

def _van_herk_1d(data, k, origin, op, neutral):
    """对 1D 序列做窗口大小 k 的滑动 min/max，每元素 O(1)。

    origin 为锚点在窗口内的下标（0..k-1），即窗口覆盖 data[i-origin .. i-origin+k-1]。
    越界位置取 neutral（等价于窗口裁剪）。
    """
    n = len(data)
    if n == 0:
        return []
    left = origin
    right = k - 1 - origin
    g = [neutral] * n  # 前向：块内从块首到 i 的累积 op
    for i in range(n):
        g[i] = data[i] if i % k == 0 else op(g[i - 1], data[i])
    h = [neutral] * n  # 后向：块内从 i 到块尾的累积 op
    for i in range(n - 1, -1, -1):
        h[i] = data[i] if (i == n - 1 or i % k == k - 1) else op(h[i + 1], data[i])
    out = [neutral] * n
    for i in range(n):
        lo = i - left
        hi = i + right
        if hi < 0 or lo > n - 1:
            # 窗口完全越界（SE 线段不含锚点列时可能出现）
            out[i] = neutral
        elif lo < 0:
            # 左端裁剪：窗口 [0, hi] 完全落在第 0 块内，g 精确覆盖
            out[i] = g[min(hi, n - 1)]
        elif hi > n - 1:
            if lo // k == (n - 1) // k:
                # 右端裁剪且窗口在末块内：h 精确覆盖 [lo, n-1]
                out[i] = h[lo]
            else:
                # 右端裁剪且跨块：末块起点在窗口内，g[n-1] + h[lo] 精确拼接
                out[i] = op(g[n - 1], h[lo])
        else:
            out[i] = op(g[hi], h[lo])
    return out


def _filter_rows(img, k, origin, op, neutral):
    return [_van_herk_1d(row, k, origin, op, neutral) for row in img]


def _filter_cols(img, k, origin, op, neutral):
    cols = [_van_herk_1d(list(col), k, origin, op, neutral) for col in zip(*img)]
    return [list(row) for row in zip(*cols)]


# ------------------------------------------------------- 结构元素分解

def decompose(se):
    """把 SE 分解为可分离/分段形式，返回描述 dict（供检查与测试）。"""
    offsets = _se_to_offsets(se)
    rows = {}
    for dy, dx in offsets:
        rows.setdefault(dy, []).append(dx)
    dys = sorted(rows)
    runs = []
    for dy in dys:
        xs = sorted(rows[dy])
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((dy, start, prev))
                start = prev = x
        runs.append((dy, start, prev))
    # 实心矩形检测：行连续、每行恰好一段且区间一致
    rect = (
        dys == list(range(dys[0], dys[0] + len(dys)))
        and len(runs) == len(dys)
        and len({(x0, x1) for _, x0, x1 in runs}) == 1
    )
    if rect:
        x0, x1 = runs[0][1], runs[0][2]
        return {
            "kind": "rectangle",
            "h": (x1 - x0 + 1, -x0),          # (窗口长度, origin)
            "v": (dys[-1] - dys[0] + 1, -dys[0]),
        }
    return {"kind": "runs", "runs": runs}


# ------------------------------------------------------- 快速滤波核心

def _apply(img, offsets, op, neutral):
    _check_image(img)
    rows = {}
    for dy, dx in offsets:
        rows.setdefault(dy, []).append(dx)
    dys = sorted(rows)
    runs = []
    for dy in dys:
        xs = sorted(rows[dy])
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((dy, start, prev))
                start = prev = x
        runs.append((dy, start, prev))

    rect = (
        dys == list(range(dys[0], dys[0] + len(dys)))
        and len(runs) == len(dys)
        and len({(x0, x1) for _, x0, x1 in runs}) == 1
    )
    if rect:  # 矩形：H ⊕ V 两次 1D 滤波
        x0, x1 = runs[0][1], runs[0][2]
        tmp = _filter_rows(img, x1 - x0 + 1, -x0, op, neutral)
        return _filter_cols(tmp, dys[-1] - dys[0] + 1, -dys[0], op, neutral)

    # 一般形状：按 (x0, x1) 分组，水平滤波结果复用，再按行偏移合并
    height, width = len(img), len(img[0])
    cache = {}
    out = [[neutral] * width for _ in range(height)]
    for dy, x0, x1 in runs:
        key = (x0, x1)
        if key not in cache:
            cache[key] = _filter_rows(img, x1 - x0 + 1, -x0, op, neutral)
        filtered = cache[key]
        for y in range(height):
            sy = y + dy
            if 0 <= sy < height:
                out[y] = list(map(op, out[y], filtered[sy]))
    return out


def _apply_brute(img, offsets, op, neutral):
    height, width = _check_image(img)
    out = [[neutral] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = neutral
            for dy, dx in offsets:
                yy, xx = y + dy, x + dx
                if 0 <= yy < height and 0 <= xx < width:
                    acc = op(acc, img[yy][xx])
            out[y][x] = acc
    return out


def _reflect(offsets):
    return [(-dy, -dx) for dy, dx in offsets]


# ------------------------------------------------------- 公开 API（快速）

def erode(img, se):
    """灰度/二值腐蚀：逐点取 SE 窗口内的最小值。"""
    return _apply(img, _se_to_offsets(se), min, inf)


def dilate(img, se):
    """灰度/二值膨胀：逐点取 SE 窗口内的最大值。"""
    return _apply(img, _se_to_offsets(se), max, -inf)


def opening(img, se):
    """开运算：先腐蚀后膨胀（第二步用反射 SE，构成伴随对，保证幂等）。"""
    offsets = _se_to_offsets(se)
    return _apply(_apply(img, offsets, min, inf), _reflect(offsets), max, -inf)


def closing(img, se):
    """闭运算：先膨胀后腐蚀（第二步用反射 SE，构成伴随对，保证幂等）。"""
    offsets = _se_to_offsets(se)
    return _apply(_apply(img, offsets, max, -inf), _reflect(offsets), min, inf)


# ------------------------------------------------------- 公开 API（暴力对拍）

def erode_brute(img, se):
    return _apply_brute(img, _se_to_offsets(se), min, inf)


def dilate_brute(img, se):
    return _apply_brute(img, _se_to_offsets(se), max, -inf)


def opening_brute(img, se):
    offsets = _se_to_offsets(se)
    return _apply_brute(_apply_brute(img, offsets, min, inf), _reflect(offsets), max, -inf)


def closing_brute(img, se):
    offsets = _se_to_offsets(se)
    return _apply_brute(_apply_brute(img, offsets, max, -inf), _reflect(offsets), min, inf)
