"""性能实测：批量求逆 vs 逐个求逆、缓存命中、大整数扩展欧几里得。

运行：python3 bench.py
"""

import random
import sys
import time

sys.set_int_max_str_digits(200000)  # 允许打印超大整数位数信息

from numtheory_pkg import InverseCache, egcd_count, gcd, modinv

rng = random.Random(7)


def bench_inverse(n=20000, bits=512):
    m = rng.getrandbits(bits) | 1
    values = []
    while len(values) < n:
        v = rng.getrandbits(bits - 1) + 1
        if gcd(v, m) == 1:
            values.append(v)

    t0 = time.perf_counter()
    r1 = [modinv(v, m) for v in values]
    t1 = time.perf_counter()

    cache = InverseCache(m)
    t2 = time.perf_counter()
    r2 = cache.batch(values)
    t3 = time.perf_counter()

    assert r1 == r2
    for v, inv in zip(values[:100], r1[:100]):
        assert (v * inv) % m == 1

    # 缓存命中：同一批值再查一轮
    t4 = time.perf_counter()
    r3 = [cache.inverse(v) for v in values]
    t5 = time.perf_counter()
    assert r3 == r1

    print("== 批量求逆（n=%d，模数 %d 位）==" % (n, bits))
    print("  逐个 modinv      : %8.3f ms  (%6.2f us/次)" % ((t1 - t0) * 1e3, (t1 - t0) / n * 1e6))
    print("  batch 前缀积技巧 : %8.3f ms  (%6.2f us/次，加速 %.1fx)"
          % ((t3 - t2) * 1e3, (t3 - t2) / n * 1e6, (t1 - t0) / (t3 - t2)))
    print("  缓存命中再查     : %8.3f ms  (%6.2f us/次，加速 %.1fx)"
          % ((t5 - t4) * 1e3, (t5 - t4) / n * 1e6, (t1 - t0) / (t5 - t4)))
    print()


def bench_egcd(bits_list=(64, 256, 1024, 2048, 4096, 8192), trials=200):
    print("== 扩展欧几里得：迭代次数与耗时（每组 %d 次随机对）==" % trials)
    print("  %6s | %8s %8s %8s | %10s | %s" % ("位数", "平均迭代", "最大迭代", "理论均值*", "平均耗时", "单次最大耗时"))
    for bits in bits_list:
        iters, times = [], []
        for _ in range(trials):
            a = rng.getrandbits(bits) | 1
            b = rng.getrandbits(bits)
            t0 = time.perf_counter()
            g, x, y, n = egcd_count(a, b)
            times.append(time.perf_counter() - t0)
            iters.append(n)
            assert a * x + b * y == g and g == gcd(a, b)
        import math
        theo = 12 * math.log(2) / math.pi ** 2 * (bits * math.log(2))
        print("  %6d | %8.1f %8d %8.1f | %8.1f us | %9.1f us"
              % (bits, sum(iters) / len(iters), max(iters), theo,
                 sum(times) / len(times) * 1e6, max(times) * 1e6))
    print("  * 理论均值 = (12 ln2 / pi^2) * ln N，随机输入下带余除法步数的期望值")
    print()


def bench_crt_scaling():
    print("== 同余组合并：k 个 64 位模数（非互素）==")
    for k in (10, 100, 1000, 5000):
        x_true = rng.getrandbits(63)
        cons = []
        for _ in range(k):
            m = rng.getrandbits(64) | 1
            cons.append((x_true % m, m))
        from numtheory_pkg import crt
        t0 = time.perf_counter()
        sol = crt(cons)
        t1 = time.perf_counter()
        ok = all((sol.x0 - r) % m == 0 for r, m in cons)
        assert ok
        print("  k=%5d : %8.2f ms（lcm 约 %d 位）"
              % (k, (t1 - t0) * 1e3, sol.modulus.bit_length()))
    print()


if __name__ == "__main__":
    bench_inverse()
    bench_inverse(n=20000, bits=2048)
    bench_egcd()
    bench_crt_scaling()
