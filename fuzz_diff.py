"""对拍脚本：库实现 vs 独立参照实现，随机 + 边界输入，全部解回代校验。

参照实现：
- 模逆：Python 内置 pow(a, -1, m)（CPython 官方实现）；
- 线性同余 / 同余组：O(m) / O(lcm) 暴力枚举（逻辑独立、显然正确）。

运行：python3 fuzz_diff.py [轮数倍数]
"""

import random
import sys

from numtheory_pkg import (
    modinv,
    solve_linear_congruence,
    linear_congruence_solutions,
    crt,
    gcd,
    lcm,
    NoInverseError,
    NoSolutionError,
    CongruenceConflictError,
)

TRIAL_SCALE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rng = random.Random(20261003)

stats = {"modinv": 0, "linear": 0, "crt": 0, "crt_conflict": 0, "huge": 0}


# ---------- 参照实现 ----------

def ref_modinv(a, m):
    """参照：内置 pow；m == 0 双方都应报错。"""
    return pow(a, -1, abs(m)) if abs(m) != 1 else 0


def ref_linear_bruteforce(a, b, m):
    """暴力枚举 [0, |m|) 内全部解；m == 0 按精确等式处理。"""
    if m == 0:
        if a == 0:
            return "all" if b == 0 else None
        return [b // a] if b % a == 0 else None
    mm = abs(m)
    return [x for x in range(mm) if (a * x - b) % mm == 0]


def ref_crt_bruteforce(congruences):
    """暴力枚举 [0, lcm) 内全部解；仅用于小模数（不含 m=0）。"""
    big_lcm = 1
    for _, m in congruences:
        big_lcm = lcm(big_lcm, abs(m))
    sols = []
    for x in range(big_lcm):
        if all((x - r) % abs(m) == 0 for r, m in congruences):
            sols.append(x)
    return sols, big_lcm


def ref_crt_consistent(congruences):
    """两两一致判定（参照）：全部相容 <=> 有解。"""
    n = len(congruences)
    for i in range(n):
        for j in range(i + 1, n):
            r1, m1 = congruences[i]
            r2, m2 = congruences[j]
            g = gcd(abs(m1), abs(m2))
            if (r2 - r1) % g != 0:
                return False
    return True


# ---------- 对拍：模逆 ----------

def check_modinv(a, m):
    if m == 0:
        try:
            modinv(a, m)
        except NoInverseError:
            stats["modinv"] += 1
            return
        raise AssertionError("modinv(%d, 0) 应当报错" % a)
    g = gcd(a, abs(m))
    if g != 1:
        try:
            modinv(a, m)
        except NoInverseError:
            stats["modinv"] += 1
            return
        raise AssertionError("modinv(%d, %d) 应当报错（gcd=%d）" % (a, m, g))
    got = modinv(a, m)
    want = ref_modinv(a, m)
    assert got == want, "modinv(%d, %d): got %d, want %d" % (a, m, got, want)
    assert 0 <= got < abs(m)
    if abs(m) != 1:
        assert (a * got) % abs(m) == 1  # 回代校验
    stats["modinv"] += 1


# ---------- 对拍：线性同余 ----------

def check_linear(a, b, m):
    want = ref_linear_bruteforce(a, b, m)
    if want is None or want == []:
        try:
            solve_linear_congruence(a, b, m)
        except NoSolutionError:
            stats["linear"] += 1
            return
        raise AssertionError("(%d)x ≡ %d (mod %d) 应当无解" % (a, b, m))
    got = linear_congruence_solutions(a, b, m)
    if want == "all":
        assert got == [0] and solve_linear_congruence(a, b, m).modulus == 1
        stats["linear"] += 1
        return
    assert sorted(got) == want, (
        "(%d)x ≡ %d (mod %d): got %s, want %s" % (a, b, m, sorted(got), want))
    for x in got:  # 回代校验
        if m != 0:
            assert (a * x - b) % abs(m) == 0
        else:
            assert a * x == b
    stats["linear"] += 1


# ---------- 对拍：同余组 ----------

def check_crt(congruences):
    consistent = ref_crt_consistent(congruences)
    if not consistent:
        try:
            crt(congruences)
        except CongruenceConflictError as e:
            # 异常指出的冲突对必须真的冲突
            r1, m1 = congruences[e.index1]
            r2, m2 = congruences[e.index2]
            g = gcd(abs(m1), abs(m2))
            assert (r2 - r1) % g != 0, "报告的冲突对实际相容！"
            stats["crt_conflict"] += 1
            return
        raise AssertionError("同余组 %s 应当冲突" % (congruences,))
    want, big_lcm = ref_crt_bruteforce(congruences)
    sol = crt(congruences)
    assert sol.modulus == big_lcm, (
        "%s: 通解模 got %d, want lcm %d" % (congruences, sol.modulus, big_lcm))
    assert want == [sol.x0], (
        "%s: got x0=%d, want %s" % (congruences, sol.x0, want))
    for r, m in congruences:  # 回代校验
        assert (sol.x0 - r) % abs(m) == 0
    stats["crt"] += 1


# ---------- 大整数专项（无法暴力，用 pow 交叉验证 + 回代） ----------

def check_huge(bits):
    m = rng.getrandbits(bits) | 1
    a = rng.getrandbits(bits)
    while gcd(a, m) != 1:
        a = rng.getrandbits(bits)
    got = modinv(a, m)
    assert got == pow(a, -1, m)
    assert (a * got) % m == 1

    b = rng.getrandbits(bits)
    sol = solve_linear_congruence(a, b, m)  # gcd=1，必有唯一解
    assert sol.modulus == m
    assert (a * sol.x0 - b) % m == 0

    # 非互素大模数同余组：人为构造保证有解
    g = rng.getrandbits(bits // 4) | 1
    m1, m2 = g * (rng.getrandbits(bits // 2) | 1), g * (rng.getrandbits(bits // 2) | 1)
    x_true = rng.getrandbits(bits)
    sol2 = crt([(x_true, m1), (x_true, -m2)])
    assert (sol2.x0 - x_true) % m1 == 0
    assert (sol2.x0 - x_true) % m2 == 0
    stats["huge"] += 1


def main():
    # 1) 边界用例（含模 0、模 ±1、负数）
    edge_cases = [0, 1, -1, 2, -2]
    for a in edge_cases + [3, -3, 100, -100]:
        for m in edge_cases + [7, -7, 12, -12]:
            check_modinv(a, m)
    for a in edge_cases + [6, -6]:
        for b in edge_cases + [12, -12]:
            for m in edge_cases + [18, -18]:
                check_linear(a, b, m)
    for cons in [
        [(0, 1)], [(5, -1), (3, 7)], [(0, 2), (0, 4), (0, 8)],
        [(1, 2), (2, 4), (3, 8)], [(2, 6), (8, 12), (14, 18)],
        [(-3, -4), (5, -6)], [(7, 12), (7, 18), (7, 30)],
    ]:
        check_crt(cons)

    # 2) 随机小模数对拍
    for _ in range(3000 * TRIAL_SCALE):
        check_modinv(rng.randint(-500, 500), rng.randint(-60, 60))
    for _ in range(2000 * TRIAL_SCALE):
        check_linear(rng.randint(-100, 100), rng.randint(-100, 100),
                     rng.randint(-80, 80))
    for _ in range(1500 * TRIAL_SCALE):
        n = rng.randint(1, 5)
        cons = [(rng.randint(-40, 40), rng.choice([-1, 1]) * rng.randint(1, 40))
                for _ in range(n)]
        check_crt(cons)

    # 3) 超大整数（256 ~ 4096 位）
    for bits in (256, 512, 1024, 2048, 4096):
        for _ in range(20 * TRIAL_SCALE):
            check_huge(bits)

    print("对拍全部通过：")
    for k, v in stats.items():
        print("  %-12s %d 例" % (k, v))


if __name__ == "__main__":
    main()
