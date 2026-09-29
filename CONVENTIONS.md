# 数论运算约定

本文档规定 `number_theory.py` 中扩展欧几里得、模逆和线性同余求解的行为。

## 输入

- 所有输入参数必须是 Python `int`；浮点整数、字符串、复数和布尔值都会抛出 `NonIntegerInputError`。
- Python 整数为任意精度整数，支持超大整数，不存在固定位数溢出。
- `a`、`b` 可以为负数；负数按 Python 的向下取整整除和非负余数规则参与欧几里得算法。
- 负模数统一解释为其绝对值：`m ≡ abs(m)`，因为 `x ≡ y (mod m)` 与 `x ≡ y (mod -m)` 表示同一同余关系。
- 模数 `m = 0` 无定义，抛出 `ZeroModulusError`。
- 如调用方希望禁止负模数，可传入 `allow_negative_modulus=False`，此时抛出 `NegativeModulusError`。

## 扩展欧几里得

`extended_gcd(a, b)` 返回三元组 `(g, x, y)`：

```text
g = gcd(a, b)
g = a*x + b*y
g >= 0
```

贝祖系数不唯一。当输入符号不同或比较路径不同时，其它正确实现可能返回另一组合法系数；对拍因此同时比较参照实现并验证贝祖恒等式。

零输入的系数约定为：

- `extended_gcd(a, 0) = (abs(a), sign(a), 0)`，其中 `a = 0` 时为 `(0, 1, 0)`。
- `extended_gcd(0, b) = (abs(b), 0, sign(b))`，其中 `b = 0` 时为 `(0, 1, 0)`。

## 模逆

`mod_inverse(a, m)` 返回最小非负逆元：

```text
0 <= result < abs(m)
a*result ≡ 1 (mod abs(m))
```

存在判据是：

```text
gcd(a, abs(m)) = 1
```

若最大公约数不是 1，则抛出 `NoInverseError`，错误信息明确给出实际 gcd，绝不返回错误结果。

特殊模数：

- `m = 0`：抛出 `ZeroModulusError`。
- `abs(m) = 1`：只有一个剩余类 `0`，且任意两整数模 1 都同余，因此返回 `0`，满足 `a*0 ≡ 1 (mod 1)`。

## 线性同余

`solve_linear_congruence(a, b, m)` 求解：

```text
a*x ≡ b (mod m)
```

令：

```text
d = gcd(a, abs(m))
```

有解判据是：

```text
d | b
```

若不满足，抛出 `NoSolutionError`，错误信息给出 `gcd(a, m)` 和 `b`。

有解时先约分：

```text
(a/d)*x ≡ b/d (mod abs(m)/d)
```

由于 `gcd(a/d, abs(m)/d) = 1`，存在唯一的最小非负约分特解 `x0`。完整整数解为：

```text
x = x0 + (abs(m)/d)*k,    k ∈ Z
```

返回对象字段：

- `x0`：最小非负约分特解，满足 `0 <= x0 < step`。
- `step`：通解步长，即 `abs(m)/d`。
- `divisor`：`d = gcd(a, abs(m))`，也是原模数下不同剩余类的数量。
- `all_integers`：当 `step = 1` 时为真，表示所有整数都是解。
- `residue_count`：原模数下剩余类数量，即 `d`。
- `representative(i)`：第 `i` 个最小非负代表元 `x0 + step*i`，要求 `0 <= i < d`。
- `general_solution()`：可读的通解字符串。

边界情形：

- `m = 0`：抛出 `ZeroModulusError`。
- `abs(m) = 1`：有解时 `x0 = 0`、`step = 1`，所有整数均为解。
- `a = 0, b = 0`：所有整数都是解。
- `a = 0, b != 0`：当且仅当 `abs(m)` 整除 `b` 时有解，否则无解。
- `m < 0`：按 `abs(m)` 处理，输出与正模数完全相同。
