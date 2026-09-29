"""内存对比 demo：稀疏存储 vs 稠密数组（运行：python3 demo_memory.py）"""

import random

from voxelgrid import SparseVoxelGrid


def fmt_bytes(n):
    if n is None:
        return "N/A"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if abs(n) < 1024.0 or unit == "EiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def dense_cells(lo, hi):
    n = 1
    for c in range(3):
        n *= hi[c] - lo[c] + 1
    return n


def report(name, g, dense_lo, dense_hi):
    n_dense = dense_cells(dense_lo, dense_hi)
    sparse = g.estimated_bytes()
    print(f"[{name}]")
    print(f"  体素尺寸 h        : {g.h}")
    print(f"  占据体素数        : {len(g)}")
    print(f"  稀疏存储内存      : {fmt_bytes(sparse)}"
          f"（约 {sparse / max(len(g), 1):.0f} B/体素）")
    print(f"  稠密网格规模      : {n_dense:.3g} 体素")
    print(f"  稠密内存(1B/体素) : {fmt_bytes(n_dense)}")
    print(f"  稠密内存(1bit/体素): {fmt_bytes(n_dense / 8)}")
    print()


def main():
    # 场景 1：空场景
    g = SparseVoxelGrid(1.0)
    g.voxelize([])
    report("空场景", g, (0, 0, 0), (99, 99, 99))

    # 场景 2：单体素（一个小三角形）
    g = SparseVoxelGrid(1.0)
    g.voxelize([((0.1, 0.1, 0.1), (0.2, 0.1, 0.1), (0.1, 0.2, 0.1))])
    report("单体素", g, (0, 0, 0), (0, 0, 0))

    # 场景 3：体素尺寸远小于物体（h=0.01，物体尺寸 ~1）
    g = SparseVoxelGrid(0.01)
    g.voxelize([((0.0, 0.0, 0.3), (1.0, 0.0, 0.3), (0.0, 1.0, 0.7))])
    report("体素尺寸 << 物体（h=0.01，单个三角形）", g, (0, 0, 0), (100, 100, 100))

    # 场景 4：超大场景（坐标 ~1e6，100 个随机三角形，h=1）
    rng = random.Random(42)
    tris = []
    for _ in range(100):
        p = tuple(rng.uniform(-1e6, 1e6) for _ in range(3))
        tris.append(tuple(
            tuple(p[c] + rng.uniform(0, 5) for c in range(3)) for _ in range(3)
        ))
    g = SparseVoxelGrid(1.0)
    g.voxelize(tris)
    report("超大场景（±1e6，100 个三角形）", g,
           (-1_000_000,) * 3, (1_000_006,) * 3)

    # 场景 5：同一场景、不同体素尺寸 -> 展示 稀疏 O(1/h^2) vs 稠密 O(1/h^3)
    print("体素尺寸 scaling（同一物体：半径 1 的球面近似，1280 三角形）")
    tris = []
    import math
    n_lat, n_lon = 20, 32
    for a in range(n_lat):
        for b in range(n_lon):
            def pt(aa, bb):
                th = math.pi * aa / n_lat
                ph = 2 * math.pi * bb / n_lon
                return (math.sin(th) * math.cos(ph),
                        math.sin(th) * math.sin(ph), math.cos(th))
            tris.append((pt(a, b), pt(a + 1, b), pt(a, b + 1)))
            tris.append((pt(a + 1, b), pt(a + 1, b + 1), pt(a, b + 1)))
    print(f"  {'h':>8} {'占据体素':>10} {'稀疏内存':>12} {'稠密(1bit)':>14}")
    for h in (0.2, 0.1, 0.05, 0.025):
        g = SparseVoxelGrid(h, origin=(-1.0, -1.0, -1.0))
        g.voxelize(tris)
        n_dense = dense_cells((0, 0, 0), tuple(int(2 / h) for _ in range(3)))
        print(f"  {h:>8} {len(g):>10} {fmt_bytes(g.estimated_bytes()):>12}"
              f" {fmt_bytes(n_dense / 8):>14}")


if __name__ == "__main__":
    main()
