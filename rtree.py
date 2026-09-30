"""R-tree 空间索引（Python 3，仅标准库）。

功能：
- 批量加载：STR（Sort-Tile-Recursive）打包，O(n log n)，得到紧凑的静态树
- 动态插入 / 删除：Guttman 经典算法（最小面积增量选择子树 + 二次分裂 + 压缩重插）
- 查询：query_intersect（相交查询）、query_within（范围查询，对象完全落在窗口内）
- 退化检测：check_degradation 在兄弟节点包围盒重叠严重时发出 RuntimeWarning

矩形约定：(minx, miny, maxx, maxy)，边界相接视为相交（闭区间）。
"""

from __future__ import annotations

import math
import sys
import warnings

__all__ = [
    "RTree",
    "rect_area",
    "rect_union",
    "rects_intersect",
    "rect_contains",
    "rect_intersection_area",
]


# ---------------------------------------------------------------- 矩形基础运算

def rect_area(r):
    w = r[2] - r[0]
    h = r[3] - r[1]
    return w * h if w > 0.0 and h > 0.0 else 0.0


def rect_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def rects_intersect(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def rect_intersection_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0.0 and dy > 0.0 else 0.0


def rect_contains(outer, inner):
    """inner 是否完全落在 outer 内（含边界）。"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1]
            and outer[2] >= inner[2] and outer[3] >= inner[3])


def _normalize(rect):
    minx, miny, maxx, maxy = rect
    if minx > maxx:
        minx, maxx = maxx, minx
    if miny > maxy:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------- 节点

class _Node:
    __slots__ = ("is_leaf", "entries")

    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        # 叶子: [(rect, item), ...]  内部: [(child_bbox, _Node), ...]
        self.entries = []


def _node_bbox(node):
    it = iter(node.entries)
    bbox = next(it)[0]
    for e in it:
        bbox = rect_union(bbox, e[0])
    return bbox


def _str_groups(entries, max_children):
    """STR 打包：按 x 中心排序切片，片内按 y 中心排序，再按 max_children 分页。"""
    n = len(entries)
    pages = math.ceil(n / max_children)
    slices = math.ceil(math.sqrt(pages))
    entries.sort(key=lambda e: (e[0][0] + e[0][2]) * 0.5)
    groups = []
    step = slices * max_children
    for s in range(0, n, step):
        band = entries[s:s + step]
        band.sort(key=lambda e: (e[0][1] + e[0][3]) * 0.5)
        for i in range(0, len(band), max_children):
            groups.append(band[i:i + max_children])
    return groups


# ---------------------------------------------------------------- RTree

class RTree:
    """二维 R-tree。item 为任意用户对象（建议可哈希、可比较）。"""

    def __init__(self, max_children=16):
        if max_children < 4:
            raise ValueError("max_children 至少为 4")
        self.max_children = max_children
        self.min_children = max(2, max_children // 2)
        self._root = None
        self._size = 0
        self._last_query_visited = 0

    # ------------------------------------------------------------ 构造

    @classmethod
    def bulk_load(cls, items, max_children=16):
        """items: 可迭代的 (rect, item)。返回新树（STR 打包）。"""
        tree = cls(max_children=max_children)
        tree.load(items)
        return tree

    def load(self, items):
        """STR 批量加载，替换现有内容。"""
        entries = [(_normalize(r), it) for r, it in items]
        self._size = len(entries)
        self._root = None
        self._last_query_visited = 0
        if not entries:
            return
        nodes = []
        for group in _str_groups(entries, self.max_children):
            leaf = _Node(True)
            leaf.entries = group
            nodes.append(leaf)
        while len(nodes) > 1:
            up = [(_node_bbox(nd), nd) for nd in nodes]
            parents = []
            for group in _str_groups(up, self.max_children):
                p = _Node(False)
                p.entries = group
                parents.append(p)
            nodes = parents
        self._root = nodes[0]

    # ------------------------------------------------------------ 插入

    def insert(self, rect, item):
        self._add(_normalize(rect), item)
        self._size += 1

    def _add(self, rect, item):
        """树内插入（不维护 _size），供删除后的重插复用。"""
        if self._root is None:
            self._root = _Node(True)
            self._root.entries.append((rect, item))
            return
        split = self._insert(self._root, rect, item)
        if split is not None:
            new_root = _Node(False)
            new_root.entries = [(_node_bbox(self._root), self._root),
                                (_node_bbox(split), split)]
            self._root = new_root

    def _insert(self, node, rect, item):
        if node.is_leaf:
            node.entries.append((rect, item))
        else:
            idx = self._choose_subtree(node, rect)
            child = node.entries[idx][1]
            split = self._insert(child, rect, item)
            node.entries[idx] = (_node_bbox(child), child)
            if split is not None:
                node.entries.append((_node_bbox(split), split))
        if len(node.entries) > self.max_children:
            return self._split_node(node)
        return None

    def _choose_subtree(self, node, rect):
        """选面积增量最小的孩子，平局取面积更小者。"""
        best_idx = 0
        best_key = None
        for idx, (r, _child) in enumerate(node.entries):
            enl = rect_area(rect_union(r, rect)) - rect_area(r)
            key = (enl, rect_area(r))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        return best_idx

    # ------------------------------------------------------- 分裂（二次）

    def _split_node(self, node):
        """Guttman 二次分裂：PickSeeds 找浪费最大的一对种子，再按增量差分配。"""
        entries = node.entries
        i, j = self._pick_seeds(entries)
        group_a = [entries[i]]
        group_b = [entries[j]]
        rest = [entries[k] for k in range(len(entries)) if k != i and k != j]
        bbox_a = group_a[0][0]
        bbox_b = group_b[0][0]
        while rest:
            # 若某组加上剩余全部才够 min_children，则全部分给它
            if len(group_a) + len(rest) <= self.min_children:
                group_a.extend(rest)
                break
            if len(group_b) + len(rest) <= self.min_children:
                group_b.extend(rest)
                break
            area_a = rect_area(bbox_a)
            area_b = rect_area(bbox_b)
            best_k = -1
            best_diff = -1.0
            best_d = (0.0, 0.0)
            for k, e in enumerate(rest):
                r = e[0]
                d1 = rect_area(rect_union(bbox_a, r)) - area_a
                d2 = rect_area(rect_union(bbox_b, r)) - area_b
                diff = d1 - d2
                if diff < 0:
                    diff = -diff
                if diff > best_diff:
                    best_diff = diff
                    best_k = k
                    best_d = (d1, d2)
            e = rest.pop(best_k)
            d1, d2 = best_d
            if d1 < d2:
                group_a.append(e)
                bbox_a = rect_union(bbox_a, e[0])
            elif d2 < d1:
                group_b.append(e)
                bbox_b = rect_union(bbox_b, e[0])
            else:
                # 平局：面积小者优先，再平局 entries 少者优先
                if (rect_area(bbox_a), len(group_a)) <= (rect_area(bbox_b), len(group_b)):
                    group_a.append(e)
                    bbox_a = rect_union(bbox_a, e[0])
                else:
                    group_b.append(e)
                    bbox_b = rect_union(bbox_b, e[0])
        node.entries = group_a
        sibling = _Node(node.is_leaf)
        sibling.entries = group_b
        return sibling

    @staticmethod
    def _pick_seeds(entries):
        """二次 PickSeeds：合并后浪费面积最大的一对作为两组种子。"""
        worst = -1.0
        pair = (0, 1)
        for i in range(len(entries)):
            ri = entries[i][0]
            ai = rect_area(ri)
            for j in range(i + 1, len(entries)):
                rj = entries[j][0]
                waste = rect_area(rect_union(ri, rj)) - ai - rect_area(rj)
                if waste > worst:
                    worst = waste
                    pair = (i, j)
        return pair

    # ------------------------------------------------------------ 删除

    def delete(self, rect, item):
        """删除与 (rect, item) 精确匹配的一条记录；不存在则抛 KeyError。"""
        rect = _normalize(rect)
        found = self._find_leaf(rect, item)
        if found is None:
            raise KeyError("未找到匹配的 (rect, item)")
        path, entry_idx = found
        leaf = path[-1][0]
        del leaf.entries[entry_idx]
        self._size -= 1
        reinsert = []
        # 自叶向上压缩：下溢节点摘下并收集其全部数据项稍后重插
        for level in range(len(path) - 1, 0, -1):
            node, idx_in_parent = path[level]
            parent = path[level - 1][0]
            if len(node.entries) < self.min_children:
                del parent.entries[idx_in_parent]
                self._collect_items(node, reinsert)
            else:
                parent.entries[idx_in_parent] = (_node_bbox(node), node)
        # 根收缩
        while self._root is not None and not self._root.is_leaf \
                and len(self._root.entries) == 1:
            self._root = self._root.entries[0][1]
        if self._root is not None and not self._root.entries:
            self._root = None
        for r, it in reinsert:
            self._add(r, it)

    def _find_leaf(self, rect, item):
        if self._root is None:
            return None
        path = []

        def rec(node, idx_in_parent):
            path.append((node, idx_in_parent))
            if node.is_leaf:
                for k, (r, it) in enumerate(node.entries):
                    if r == rect and it == item:
                        return k
                path.pop()
                return None
            for k, (r, child) in enumerate(node.entries):
                if rect_contains(r, rect):
                    found = rec(child, k)
                    if found is not None:
                        return found
            path.pop()
            return None

        k = rec(self._root, None)
        if k is None:
            return None
        return path, k

    def _collect_items(self, node, out):
        stack = [node]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                out.extend(nd.entries)
            else:
                for _r, child in nd.entries:
                    stack.append(child)

    # ------------------------------------------------------------ 查询

    def query_intersect(self, rect):
        """返回所有与窗口相交（含边界相接）的 item。"""
        q = _normalize(rect)
        out = []
        visited = 0
        if self._root is not None:
            stack = [self._root]
            while stack:
                node = stack.pop()
                visited += 1
                if node.is_leaf:
                    for r, it in node.entries:
                        if rects_intersect(r, q):
                            out.append(it)
                else:
                    for r, child in node.entries:
                        if rects_intersect(r, q):
                            stack.append(child)
        self._last_query_visited = visited
        return out

    def query_within(self, rect):
        """范围查询：返回完全落在窗口内的 item。"""
        q = _normalize(rect)
        out = []
        visited = 0
        if self._root is not None:
            stack = [self._root]
            while stack:
                node = stack.pop()
                visited += 1
                if node.is_leaf:
                    for r, it in node.entries:
                        if rect_contains(q, r):
                            out.append(it)
                else:
                    for r, child in node.entries:
                        if rects_intersect(r, q):
                            stack.append(child)
        self._last_query_visited = visited
        return out

    def items(self):
        """遍历全部 (rect, item)。"""
        out = []
        if self._root is None:
            return out
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.extend(node.entries)
            else:
                for _r, child in node.entries:
                    stack.append(child)
        return out

    # ------------------------------------------------------------ 统计

    def __len__(self):
        return self._size

    @property
    def last_query_visited(self):
        """上一次查询访问的节点数（衡量查询放大）。"""
        return self._last_query_visited

    def stats(self, sample_limit=2048):
        """返回结构统计：节点数、高度、填充率、兄弟重叠度等。"""
        result = {
            "size": self._size,
            "height": 0,
            "nodes": 0,
            "leaf_nodes": 0,
            "internal_nodes": 0,
            "avg_fill": 0.0,
            "avg_overlap": 0.0,
            "last_query_visited": self._last_query_visited,
        }
        if self._root is None:
            return result
        fill_sum = 0
        overlap_sum = 0.0
        overlap_cnt = 0
        stack = [(self._root, 1)]
        while stack:
            node, lvl = stack.pop()
            result["nodes"] += 1
            if lvl > result["height"]:
                result["height"] = lvl
            if node.is_leaf:
                result["leaf_nodes"] += 1
                continue
            result["internal_nodes"] += 1
            fill_sum += len(node.entries)
            if overlap_cnt < sample_limit and len(node.entries) > 1:
                rs = [r for r, _c in node.entries]
                pair_iou = 0.0
                pairs = 0
                for a in range(len(rs)):
                    for b in range(a + 1, len(rs)):
                        inter = rect_intersection_area(rs[a], rs[b])
                        if inter <= 0.0:
                            pairs += 1
                            continue
                        union = rect_area(rect_union(rs[a], rs[b]))
                        pair_iou += inter / union if union > 0.0 else 1.0
                        pairs += 1
                if pairs:
                    overlap_sum += pair_iou / pairs
                    overlap_cnt += 1
            for _r, child in node.entries:
                stack.append((child, lvl + 1))
        if result["internal_nodes"]:
            result["avg_fill"] = fill_sum / result["internal_nodes"]
        if overlap_cnt:
            result["avg_overlap"] = overlap_sum / overlap_cnt
        return result

    def check_degradation(self, overlap_threshold=0.5):
        """重叠退化告警。

        avg_overlap = 内部节点兄弟包围盒的平均成对 IoU（0=互不交叠，
        1=完全重合），越接近 1 查询越会退化成近乎全树扫描。
        超过阈值时发出 RuntimeWarning 并返回 True。
        """
        s = self.stats()
        ov = s["avg_overlap"]
        if ov > overlap_threshold:
            warnings.warn(
                "R-tree 退化告警：兄弟节点平均重叠度 {:.1%} 超过阈值 {:.1%}，"
                "查询可能退化为大面积扫描；建议重建索引、增大 max_children "
                "或改用对重叠不敏感的结构（如网格/线段树）。".format(
                    ov, overlap_threshold),
                RuntimeWarning,
                stacklevel=2,
            )
            return True
        return False

    def memory_bytes(self):
        """估算索引结构自身占用的字节数（含 rect 元组与 item 引用目标，去重）。"""
        seen = set()
        total = 0

        def add(o):
            nonlocal total
            oid = id(o)
            if oid in seen:
                return
            seen.add(oid)
            total += sys.getsizeof(o)

        if self._root is None:
            return 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            add(node)
            add(node.entries)
            for e in node.entries:
                add(e)
                add(e[0])
                for v in e[0]:
                    add(v)
                if node.is_leaf:
                    add(e[1])
                else:
                    stack.append(e[1])
        return total
