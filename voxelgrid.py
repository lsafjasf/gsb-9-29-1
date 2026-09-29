"""稀疏体素网格：体素化与范围查询（仅标准库）。

存储结构：dict[(i, j, k)] -> value，只保存被占据的体素，
内存占用与场景边长无关，只随占据体素数增长。

相交判定：分离轴（SAT）+ 浮点过滤器 + Fraction 精确回退，
保证结果与精确算术一致（含薄面/穿越/退化三角形）。
"""

import math
import sys
from fractions import Fraction

_EPS = 2.0 ** -53          # 双精度机器 epsilon
_FILTER_K = 64.0           # 过滤器安全系数（覆盖点积/叉积的累积舍入）


def _edge_axes(edges):
    """9 条 盒边 x 三角形边 候选分离轴（与坐标轴叉积，仅置换/取负，无舍入）。"""
    axes = []
    for ex, ey, ez in edges:
        axes.append((0.0, -ez, ey))   # edge x x-hat
        axes.append((ez, 0.0, -ex))   # edge x y-hat
        axes.append((-ey, ex, 0.0))   # edge x z-hat
    return axes


def _normal(e0, e1):
    return (
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    )


def _min_clearance(v, half, axes):
    """13 条分离轴上的最小间隙（泛型：float 或 Fraction）。

    间隙 >= 0 表示该轴上投影区间重叠（接触也算相交）；
    所有轴间隙 >= 0 <=> 三角形与盒相交。
    """
    hx, hy, hz = half
    worst = None
    for ax, ay, az in axes:
        p0 = ax * v[0][0] + ay * v[0][1] + az * v[0][2]
        p1 = ax * v[1][0] + ay * v[1][1] + az * v[1][2]
        p2 = ax * v[2][0] + ay * v[2][1] + az * v[2][2]
        rad = hx * abs(ax) + hy * abs(ay) + hz * abs(az)
        c = min(rad - min(p0, p1, p2), max(p0, p1, p2) + rad)
        if worst is None or c < worst:
            worst = c
    for c in range(3):  # 3 条盒面法向
        col = (v[0][c], v[1][c], v[2][c])
        cc = min(half[c] - min(col), max(col) + half[c])
        if cc < worst:
            worst = cc
    return worst


def _float_bound(v, half, axes):
    """float 路径的舍入误差上界：K * eps * 各轴项绝对值之和的最大值。"""
    hx, hy, hz = half
    mag = 0.0
    for ax, ay, az in axes:
        s = hx * abs(ax) + hy * abs(ay) + hz * abs(az)
        for i in range(3):
            s += abs(ax * v[i][0]) + abs(ay * v[i][1]) + abs(az * v[i][2])
        if s > mag:
            mag = s
    for c in range(3):
        s = half[c] + max(abs(v[0][c]), abs(v[1][c]), abs(v[2][c]))
        if s > mag:
            mag = s
    return _FILTER_K * _EPS * mag


def _tri_box_overlap_exact(center, half, tri):
    """Fraction 精确 SAT：仅当 float 过滤器无法确定时调用。"""
    vf = [[Fraction(tri[i][c]) - Fraction(center[c]) for c in range(3)]
          for i in range(3)]
    ef = [[vf[(i + 1) % 3][c] - vf[i][c] for c in range(3)] for i in range(3)]
    axes = _edge_axes(ef) + [_normal(ef[0], ef[1])]
    hf = tuple(Fraction(x) for x in half)
    return _min_clearance(vf, hf, axes) >= 0


def tri_box_overlap(center, half, tri):
    """三角形与 AABB 是否相交（分离轴判据，SAT）。

    center: 体素中心 (cx, cy, cz)
    half:   体素半尺寸 (hx, hy, hz)
    tri:    三角形三顶点 ((x,y,z), (x,y,z), (x,y,z))

    判据：两凸体相交 当且仅当 在全部 13 条候选分离轴上投影区间均重叠：
      - 3 条盒面法向（坐标轴）
      - 1 条三角形法向
      - 9 条 盒边 x 三角形边 的叉积轴
    因此即使三角形没有任何顶点落在盒内（薄面/穿越），
    只要三角形面片与盒体有交，就会被正确判定为占据。
    退化三角形（零面积、退化为线段/点）同样适用。

    数值鲁棒性：先用 float 计算并估计舍入误差界；间隙超出误差界
    直接定论，否则用 Fraction 精确算术重判，保证与精确几何一致。
    """
    v = [[tri[i][c] - center[c] for c in range(3)] for i in range(3)]
    e = [[v[(i + 1) % 3][c] - v[i][c] for c in range(3)] for i in range(3)]
    axes = _edge_axes(e) + [_normal(e[0], e[1])]
    cmin = _min_clearance(v, half, axes)
    bound = _float_bound(v, half, axes)
    if cmin > bound:
        return True
    if cmin < -bound:
        return False
    return _tri_box_overlap_exact(center, half, tri)


class SparseVoxelGrid:
    """稀疏体素网格。

    voxel_size: 体素边长 h（必须 > 0）
    origin:     体素索引 (0,0,0) 对应的世界坐标角点
    """

    def __init__(self, voxel_size, origin=(0.0, 0.0, 0.0)):
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        self.h = float(voxel_size)
        self.origin = tuple(float(o) for o in origin)
        self._cells = {}  # (i, j, k) -> value

    # ---- 基本访问 ----

    def index_of(self, point):
        """世界坐标 -> 体素索引（floor 划分，负坐标同样正确）。"""
        return tuple(
            math.floor((point[c] - self.origin[c]) / self.h) for c in range(3)
        )

    def add(self, idx, value=1):
        self._cells[tuple(idx)] = value

    def __contains__(self, idx):
        return tuple(idx) in self._cells

    def __len__(self):
        return len(self._cells)

    def cells(self):
        return self._cells.keys()

    # ---- 体素化 ----

    def voxelize(self, triangles):
        """将三角形集合体素化，标记所有与三角形相交的体素。"""
        half = (self.h * 0.5,) * 3
        ox, oy, oz = self.origin
        h = self.h
        for tri in triangles:
            lo = [min(tri[v][c] for v in range(3)) for c in range(3)]
            hi = [max(tri[v][c] for v in range(3)) for c in range(3)]
            i0 = math.floor((lo[0] - ox) / h)
            i1 = math.floor((hi[0] - ox) / h)
            j0 = math.floor((lo[1] - oy) / h)
            j1 = math.floor((hi[1] - oy) / h)
            k0 = math.floor((lo[2] - oz) / h)
            k1 = math.floor((hi[2] - oz) / h)
            for i in range(i0, i1 + 1):
                cx = ox + (i + 0.5) * h
                for j in range(j0, j1 + 1):
                    cy = oy + (j + 0.5) * h
                    for k in range(k0, k1 + 1):
                        if tri_box_overlap((cx, cy, oz + (k + 0.5) * h), half, tri):
                            self._cells[(i, j, k)] = 1

    # ---- 范围查询与统计 ----

    def iter_range(self, lo, hi):
        """产出闭区间 [lo, hi]（体素索引，逐轴）内所有被占据体素的 (idx, value)。"""
        for idx, val in self._cells.items():
            if all(lo[c] <= idx[c] <= hi[c] for c in range(3)):
                yield idx, val

    def count_range(self, lo, hi):
        """统计闭区间 [lo, hi] 内被占据体素数。O(占据体素总数)。"""
        return sum(1 for _ in self.iter_range(lo, hi))

    def bounds(self):
        """占据体素索引的 (lo, hi)；空场景返回 None。"""
        if not self._cells:
            return None
        lo = [min(idx[c] for idx in self._cells) for c in range(3)]
        hi = [max(idx[c] for idx in self._cells) for c in range(3)]
        return tuple(lo), tuple(hi)

    def estimated_bytes(self):
        """稀疏存储的实际内存估计（dict + 键值对象）。"""
        total = sys.getsizeof(self._cells)
        for key, val in self._cells.items():
            total += sys.getsizeof(key) + sys.getsizeof(val)
        return total
