"""数论运算库（仅依赖 Python 标准库，支持任意大整数）。

约定说明
--------
1. 所有模数按其绝对值处理：`x ≡ r (mod m)` 与 `x ≡ r (mod -m)` 等价，
   返回的代表解一律归一化到最小非负区间 ``[0, |m|)``。
2. 模数 ``m = 0`` 表示“精确相等”约束：``x ≡ r (mod 0)`` 即 ``x = r``。
   模逆在模 0 下无定义，直接抛 ``NoInverseError``。
3. 模数 ``|m| = 1`` 时所有整数彼此同余；模 1 的逆元按约定为 ``0``
   （与 Python ``pow(a, -1, 1)`` 一致）。
4. 线性同余 ``a*x ≡ b (mod m)`` 令 ``g = gcd(a, m)``：
   - ``g ∤ b``：无解；
   - ``g | b``：恰有 ``g`` 个模 ``|m|`` 不同余解，通解为
     ``x ≡ x0 (mod |m|/g)``，``x0`` 为最小非负解。
5. 同余组可解性按“两两一致”判定：``x ≡ r1 (mod m1)`` 与
   ``x ≡ r2 (mod m2)`` 相容当且仅当 ``gcd(m1,m2) | (r2-r1)``。
   冲突时抛出的异常会指明冲突的两个条件下标、模数、余数与 gcd。
6. 同余组有解时解唯一（模 lcm 意义下），最小非负代表为 ``x0``。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd as _gcd


class NoInverseError(ArithmeticError):
    """模逆不存在。判据：``gcd(a, m) != 1``（或模数为 0）。"""


class NoSolutionError(ArithmeticError):
    """线性同余无解。"""


class CongruenceConflictError(NoSolutionError):
    """同余组中某两个条件互相矛盾。

    属性记录冲突的两个原始条件（下标从 0 开始）、模数、余数及 gcd。
    """

    def __init__(self, index1, residue1, modulus1,
                 index2, residue2, modulus2, common_divisor):
        self.index1 = index1
        self.residue1 = residue1
        self.modulus1 = modulus1
        self.index2 = index2
        self.residue2 = residue2
        self.modulus2 = modulus2
        self.common_divisor = common_divisor
        super().__init__(
            "第 %d 条条件 x ≡ %d (mod %d) 与第 %d 条条件 x ≡ %d (mod %d) "
            "冲突：gcd(%d, %d) = %d 不能整除余数差 %d"
            % (index1, residue1, modulus1, index2, residue2, modulus2,
               abs(modulus1), abs(modulus2), common_divisor,
               residue2 - residue1)
        )


@dataclass(frozen=True)
class CongruenceSolution:
    """同余方程/同余组的解。

    ``x ≡ x0 (mod modulus)`` 为通解；``modulus == 0`` 表示唯一精确解
    ``x = x0``（由模 0 的精确相等约束产生）。
    """

    x0: int
    modulus: int

    def contains(self, value: int) -> bool:
        if self.modulus == 0:
            return value == self.x0
        return (value - self.x0) % self.modulus == 0


def gcd(a: int, b: int) -> int:
    """最大公约数，结果非负；``gcd(0, 0) == 0``。"""
    return abs(_gcd(a, b))


def lcm(a: int, b: int) -> int:
    """最小公倍数（非负）；任一参数为 0 时结果为 0。"""
    if a == 0 or b == 0:
        return 0
    return abs(a // _gcd(a, b) * b)


def egcd(a: int, b: int):
    """扩展欧几里得（迭代版，支持负输入）。

    返回 ``(g, x, y)``，满足 ``a*x + b*y == g``，且 ``g == gcd(a,b) >= 0``。
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def egcd_count(a: int, b: int):
    """与 :func:`egcd` 相同，额外返回带余除法（迭代）次数。"""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    iters = 0
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
        iters += 1
    if old_r < 0:
        return -old_r, -old_s, -old_t, iters
    return old_r, old_s, old_t, iters


def modinv(a: int, m: int) -> int:
    """求 ``a`` 在模 ``m`` 下的乘法逆元，返回 ``[0, |m|)`` 内的最小非负解。

    逆元不存在（``m == 0`` 或 ``gcd(a, m) != 1``）时抛 :class:`NoInverseError`，
    绝不返回错误结果。
    """
    if m == 0:
        raise NoInverseError("模数为 0 时模逆没有定义")
    modulus = abs(m)
    if modulus == 1:
        return 0
    g, x, _ = egcd(a, modulus)
    if g != 1:
        raise NoInverseError(
            "模逆不存在：gcd(%d, %d) = %d ≠ 1（a 与模数必须互素）" % (a, modulus, g)
        )
    return x % modulus


def solve_linear_congruence(a: int, b: int, m: int) -> CongruenceSolution:
    """解线性同余方程 ``a*x ≡ b (mod m)``，含负输入。

    返回 :class:`CongruenceSolution`（通解的模为 ``|m|/gcd(a,m)``）。
    无解时抛 :class:`NoSolutionError` 并给出判据。
    ``m == 0`` 按精确方程 ``a*x = b`` 处理。
    """
    a, b = int(a), int(b)
    if m == 0:
        if a == 0:
            if b == 0:
                return CongruenceSolution(0, 1)  # 任意整数都是解
            raise NoSolutionError("方程 0*x = %d 无解" % b)
        if b % a != 0:
            raise NoSolutionError("方程 %d*x = %d 无整数解" % (a, b))
        return CongruenceSolution(b // a, 0)  # 唯一精确解

    modulus = abs(m)
    g, x, _ = egcd(a, modulus)
    if b % g != 0:
        raise NoSolutionError(
            "线性同余 %d*x ≡ %d (mod %d) 无解：gcd(%d, %d) = %d 不整除 %d"
            % (a, b, modulus, a, modulus, g, b)
        )
    step = modulus // g
    x0 = (x * (b // g)) % step
    return CongruenceSolution(x0, step)


def linear_congruence_solutions(a: int, b: int, m: int):
    """返回 ``a*x ≡ b (mod m)`` 在 ``[0, |m|)`` 内的全部代表解（升序）。

    有解时解的个数恰为 ``g = gcd(a, m)``；``m == 0`` 的精确解返回单元素列表。
    """
    sol = solve_linear_congruence(a, b, m)
    if m == 0:
        return [sol.x0]
    modulus = abs(m)
    g = gcd(a, modulus)
    return [(sol.x0 + k * sol.modulus) % modulus for k in range(g)]


def _pairwise_conflict(i, r_i, m_i, j, r_j, m_j):
    g = gcd(m_i, m_j)
    if (r_j - r_i) % g != 0:
        return CongruenceConflictError(i, r_i, m_i, j, r_j, m_j, g)
    return None


def crt(congruences) -> CongruenceSolution:
    """线性同余组求解（广义 CRT，模数不必两两互素）。

    参数 ``congruences`` 为 ``(r, m)`` 序列，表示 ``x ≡ r (mod m)``；
    允许负模数、``m = 0``（精确相等）与 ``|m| = 1``（恒真）。

    有解时解唯一（模所有模数的 lcm），返回最小非负代表与通解模。
    条件矛盾时抛 :class:`CongruenceConflictError`，指明冲突条件对。
    """
    norm = []
    for i, item in enumerate(congruences):
        r, m = int(item[0]), int(item[1])
        if m == 0:
            norm.append((i, r, 0))
        else:
            modulus = abs(m)
            norm.append((i, r % modulus, modulus))

    acc_r, acc_m = 0, 1        # x ≡ 0 (mod 1)：恒真的初始约束
    acc_indices = []           # 已并入 acc 的原始条件下标
    exact = None               # (index, value)：已确定的精确值约束

    for idx, r, m in norm:
        if m == 1:
            continue

        if m == 0:
            if exact is not None:
                if exact[1] != r:
                    err = CongruenceConflictError(
                        exact[0], exact[1], 0, idx, r, 0, 0
                    )
                    raise err
                continue
            if acc_m != 1 and r % acc_m != acc_r:
                for k in acc_indices:
                    _, rk, mk = norm[k]
                    err = _pairwise_conflict(k, rk, mk, idx, r, 0)
                    if err is not None:
                        raise err
                raise CongruenceConflictError(
                    acc_indices[0] if acc_indices else idx,
                    acc_r, acc_m, idx, r, 0, gcd(acc_m, 0),
                )
            exact = (idx, r)
            continue

        if exact is not None:
            if exact[1] % m != r:
                raise CongruenceConflictError(
                    exact[0], exact[1], 0, idx, r, m, gcd(0, m)
                )
            continue

        g = gcd(acc_m, m)
        if (r - acc_r) % g != 0:
            # acc 与新条件不一致；由两两一致定理找出具体冲突的原始条件对
            for k in acc_indices:
                _, rk, mk = norm[k]
                err = _pairwise_conflict(k, rk, mk, idx, r, m)
                if err is not None:
                    raise err
            raise AssertionError("内部错误：未能定位冲突条件对")

        # 合并：acc_r + acc_m*t ≡ r (mod m)
        m1g, m2g = acc_m // g, m // g
        t0 = modinv(m1g, m2g) * ((r - acc_r) // g) % m2g
        new_m = acc_m * m2g  # == lcm(acc_m, m)
        new_r = (acc_r + acc_m * t0) % new_m
        acc_r, acc_m = new_r, new_m
        acc_indices.append(idx)

    if exact is not None:
        return CongruenceSolution(exact[1], 0)
    return CongruenceSolution(acc_r, acc_m)


def crt_representatives(congruences, copies: int = 1):
    """同余组求解并返回代表解列表。

    同余组在 lcm 内恰有一个代表解，返回 ``[x0]``；
    ``copies > 1`` 时返回 ``[0, lcm)`` 内 1 个 + 后续若干个周期解，
    便于观察“通解 + k*lcm”的形式。
    """
    sol = crt(congruences)
    if sol.modulus == 0:
        return [sol.x0]
    return [sol.x0 + k * sol.modulus for k in range(copies)]


class InverseCache:
    """同一模数下的逆元缓存与批量求解器。

    - 逐个调用 :meth:`inverse` 时，已算过的余数直接命中缓存；
    - :meth:`batch` 用“前缀积 + 单次扩展欧几里得 + 回代”技巧，
      只做一次扩展欧几里得即可得到一批元素各自的逆元。
    """

    def __init__(self, modulus: int):
        if modulus == 0:
            raise ValueError("模数不能为 0")
        self.modulus = abs(modulus)
        self._cache = {}

    def inverse(self, a: int) -> int:
        modulus = self.modulus
        a_mod = a % modulus
        cached = self._cache.get(a_mod)
        if cached is not None:
            return cached
        inv = modinv(a_mod, modulus)
        self._cache[a_mod] = inv
        return inv

    def cache_info(self):
        return {"modulus": self.modulus, "cached": len(self._cache)}

    def batch(self, values):
        """批量求逆，结果与逐个 :meth:`inverse` 完全一致。

        任一元素与模数不互素时抛 :class:`NoInverseError` 并指出位置。
        """
        modulus = self.modulus
        if modulus == 1:
            result = [0] * len(values)
            self._cache.setdefault(0, 0)
            return result

        mods = [v % modulus for v in values]
        n = len(mods)
        if n == 0:
            return []

        prefix = [0] * n
        prod = 1
        for i, v in enumerate(mods):
            prod = (prod * v) % modulus
            prefix[i] = prod

        g, s, _ = egcd(prod, modulus)
        if g != 1:
            for i, v in enumerate(mods):
                d = gcd(v, modulus)
                if d != 1:
                    raise NoInverseError(
                        "批量求逆失败：第 %d 个输入 %d 与模数 %d 不互素"
                        "（gcd = %d ≠ 1），其逆元不存在"
                        % (i, values[i], modulus, d)
                    )
            raise NoInverseError("批量求逆失败：存在不可逆元素")

        result = [0] * n
        inv_suffix = s % modulus
        for i in range(n - 1, 0, -1):
            result[i] = (inv_suffix * prefix[i - 1]) % modulus
            inv_suffix = (inv_suffix * mods[i]) % modulus
        result[0] = inv_suffix

        for v, inv in zip(mods, result):
            self._cache.setdefault(v, inv)
        return result
