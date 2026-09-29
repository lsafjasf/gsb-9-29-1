"""精确对拍实现：用 Fraction 精确算术判定三角形与体素盒是否相交。

算法与 voxelgrid.tri_box_overlap（分离轴）完全独立：
把三角形当作平面多边形，依次用盒子的 6 个半空间做
Sutherland-Hodgman 裁剪；裁剪结果非空 <=> 三角形与盒相交。
全部运算使用 fractions.Fraction，无浮点舍入误差，作为对拍基准。
"""

from fractions import Fraction


def _clip_halfspace(poly, axis, value, keep_leq):
    """用半空间（x[axis] <= value 或 >= value）裁剪多边形。"""
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[i - 1]
        d_cur = cur[axis] - value
        d_prv = prv[axis] - value
        cur_in = d_cur <= 0 if keep_leq else d_cur >= 0
        prv_in = d_prv <= 0 if keep_leq else d_prv >= 0
        if cur_in != prv_in:
            t = (value - prv[axis]) / (cur[axis] - prv[axis])
            out.append(tuple(prv[c] + t * (cur[c] - prv[c]) for c in range(3)))
        if cur_in:
            out.append(cur)
    return out


def tri_box_intersect_exact(tri, bmin, bmax):
    """精确判定：三角形（float 顶点）与盒 [bmin, bmax] 是否相交。"""
    poly = [tuple(Fraction(coord) for coord in p) for p in tri]
    for axis in range(3):
        poly = _clip_halfspace(poly, axis, Fraction(bmin[axis]), keep_leq=False)
        poly = _clip_halfspace(poly, axis, Fraction(bmax[axis]), keep_leq=True)
        if not poly:
            return False
    return True


def voxelize_exact(triangles, h, origin=(0.0, 0.0, 0.0)):
    """精确体素化：返回占据体素索引集合，作为对拍基准。"""
    import math

    cells = set()
    # 体素盒定义与 voxelgrid.SparseVoxelGrid 完全一致：
    # 体素 idx 的盒 = [center - h/2, center + h/2]，
    # 其中 center = origin + (idx + 0.5) * h 按 float 求值后再转 Fraction，
    # 保证对拍双方判定的是同一个数学对象。
    half_f = Fraction(h * 0.5)
    for tri in triangles:
        lo = [min(tri[v][c] for v in range(3)) for c in range(3)]
        hi = [max(tri[v][c] for v in range(3)) for c in range(3)]
        rng = [
            range(
                math.floor((lo[c] - origin[c]) / h),
                math.floor((hi[c] - origin[c]) / h) + 1,
            )
            for c in range(3)
        ]
        for i in rng[0]:
            for j in rng[1]:
                for k in rng[2]:
                    idx = (i, j, k)
                    center = tuple(
                        origin[c] + (idx[c] + 0.5) * h for c in range(3)
                    )
                    bmin = tuple(Fraction(center[c]) - half_f for c in range(3))
                    bmax = tuple(Fraction(center[c]) + half_f for c in range(3))
                    if tri_box_intersect_exact(tri, bmin, bmax):
                        cells.add(idx)
    return cells
