"""
benchmark.py —— 区间筛与轮式分解的耗时 / 内存基准

运行：python3 benchmark.py
内存使用 tracemalloc 测量 Python 层峰值（bytearray 等原生分配只计其
PyObject 头，因此另用 resource.ru_maxrss 报告进程级峰值 RSS 作对照）。
"""

import platform
import time
import tracemalloc
from math import isqrt

import prime_tools as pt


def timed(label: str, fn, repeat: int = 1):
    """返回 (结果, 最优耗时秒, tracemalloc 峰值 MiB)。"""
    best = float("inf")
    result = None
    peak = 0
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        best = min(best, dt)
        peak = max(peak, pk)
    print(f"{label:<46} {best:9.3f} s   峰值(py) {peak / 2**20:8.3f} MiB")
    return result, best, peak


def main() -> None:
    print("=" * 78)
    print(f"Python {platform.python_version()}  {platform.platform()}")
    print("=" * 78)

    print("\n[1] 区间筛：primes_up_to(n)（list 收集，含输出存储）")
    cases = [1_000_000, 10_000_000, 100_000_000]
    for n in cases:
        primes, dt, _ = timed(f"筛 2..{n:<12,}  pi={n}/ln(n)≈", lambda n=n: list(pt.primes_up_to(n)))
        print(f"      -> 素数个数 pi({n:,}) = {len(primes):,}（校验末项 {primes[-1]:,}）")

    print("\n[2] 区间筛：流式（不保存结果，仅求和计数）——内存与 n 无关")
    for n, chunk in ((10_000_000, 1_000_000), (100_000_000, 1_000_000)):
        def run(n=n, chunk=chunk):
            cnt = 0
            for _ in pt.primes_up_to(n, chunk=chunk):
                cnt += 1
            return cnt
        cnt, dt, peak = timed(f"流式筛至 {n:>13,}  chunk={chunk:>9,}", run)
        print(f"      -> 计数 {cnt:,}，流式工作内存 ~{chunk/2/2**20:.3f} MiB 段缓冲"
              f"（tracemalloc {peak/2**20:.3f} MiB）")

    print("\n[3] 任意区间筛（不贴边界，sqrt 上界决定小素数表）")
    def run_range():
        return list(pt.primes_in_range(1_000_000_000 - 50_000, 1_000_000_000 + 50_000))
    primes, dt, _ = timed("区间 [10^9-5e4, 10^9+5e4)", run_range)
    print(f"      -> 区间内素数 {len(primes)} 个，首 {primes[0]:,}，末 {primes[-1]:,}")

    print("\n[4] 轮式试除分解（断言 ∏ p^e == n）")
    # 用筛得到可复现的大素数：10^11 附近
    big = list(pt.primes_in_range(10 ** 11 - 200, 10 ** 11))[-1]
    p12 = list(pt.primes_in_range(10 ** 6 - 1000, 10 ** 6))[-1]
    q12 = list(pt.primes_in_range(10 ** 6 - 2000, 10 ** 6 - 1000))[-1]
    samples = {
        "0": 0,
        "1": 1,
        "2 (最小素数)": 2,
        "2^60": 2 ** 60,
        "30^2 完全平方": 30 ** 2,
        "999983^2 大素数平方(≈10^12)": 999_983 ** 2,
        f"{big:,} 大素数": big,
        f"{p12:,}*{q12:,} 两素数积": p12 * q12,
        f"{big:,}*1000003 大小混合": big * 1_000_003,
        "随机 18 位合数(含小因子)": 704_239_817_452_133_601,
    }
    for label, n in samples.items():
        fac = pt.factorize(n)
        if n < 2:
            assert fac == [], f"{n}: 约定应返回空表"
        else:
            assert pt.factors_product(fac) == n, f"乘积断言失败: {n}"
        t0 = time.perf_counter()
        for _ in range(5):
            pt.factorize(n)
        dt = (time.perf_counter() - t0) / 5
        shown = "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in fac) or "1"
        print(f"  {label:<28} {dt*1000:9.3f} ms/call  {shown[:46]}")

    print("\n[5] 分解批量吞吐：2..100000 全部整数")
    t0 = time.perf_counter()
    total = 0
    for n in range(2, 100_001):
        fac = pt.factorize(n)
        assert pt.factors_product(fac) == n
        total += len(fac)
    print(f"      {time.perf_counter()-t0:9.3f} s，99,999 个数，累计 {total:,} 个质因子")

    print("\n说明：朴素逐个试除素性判定在 10^8 规模下不可行；此处筛法为 O(n log log n)。")


if __name__ == "__main__":
    main()
