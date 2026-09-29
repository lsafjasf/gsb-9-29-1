"""一维热传导方程求解库（仅标准库）。

方程: u_t = alpha * u_xx,  x in [0, L],  t > 0

网格: N 个单元, N+1 个节点, x_i = i*dx, i = 0..N, dx = L/N。

边界条件（左右可独立指定）:
    ("dirichlet", T)  定温边界: u = T
    ("neumann", 0.0)  绝热边界: du/dx = 0（零热流，用虚节点实现）

格式:
    explicit  显式 FTCS, 稳定性条件 r = alpha*dt/dx^2 <= 1/2, 越界拒绝执行
    implicit  隐式 backward Euler, 无条件稳定, 三对角 Thomas 算法求解

能量守恒: 两端绝热时, 梯形权重离散能量
    E = dx * (u_0/2 + u_1 + ... + u_{N-1} + u_N/2)
在两种格式下均严格守恒（到浮点精度）。
"""

DIRICHLET = "dirichlet"
NEUMANN = "neumann"

_STAB_TOL = 1e-12


def stability_limit(alpha, dx):
    """显式格式允许的最大 dt: dx^2 / (2*alpha)。alpha<=0 时无限制。"""
    if alpha <= 0.0:
        return float("inf")
    return dx * dx / (2.0 * alpha)


def check_stable(alpha, dx, dt):
    """显式稳定性检查, 越界抛 ValueError。"""
    limit = stability_limit(alpha, dx)
    if dt > limit * (1.0 + _STAB_TOL):
        r = alpha * dt / (dx * dx)
        raise ValueError(
            "显式格式不稳定: r = alpha*dt/dx^2 = %.6g > 1/2 "
            "(alpha=%g, dx=%g, dt=%g, 允许的最大 dt=%g)" % (r, alpha, dx, dt, limit)
        )


def energy(u, dx):
    """梯形权重离散总热量（两端绝热时应守恒的量）。"""
    return dx * (0.5 * u[0] + sum(u[1:-1]) + 0.5 * u[-1])


def _apply_dirichlet_row(diag, rhs, value):
    diag.append(1.0)
    rhs.append(value)


def _step_explicit(u, r, bc_left, bc_right):
    n = len(u)
    v = [0.0] * n

    if bc_left[0] == DIRICHLET:
        v[0] = bc_left[1]
    else:  # 绝热: 虚节点 u_{-1} = u_1
        v[0] = u[0] + 2.0 * r * (u[1] - u[0])

    for i in range(1, n - 1):
        v[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])

    if bc_right[0] == DIRICHLET:
        v[n - 1] = bc_right[1]
    else:  # 绝热: 虚节点 u_{n} = u_{n-2}
        v[n - 1] = u[n - 1] + 2.0 * r * (u[n - 2] - u[n - 1])
    return v


def _thomas(lower, diag, upper, rhs):
    """三对角求解: lower[i]*x[i-1] + diag[i]*x[i] + upper[i]*x[i+1] = rhs[i]。"""
    n = len(diag)
    c = [0.0] * n
    d = [0.0] * n
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m
    x = [0.0] * n
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _step_implicit(u, r, bc_left, bc_right):
    n = len(u)
    lower = [0.0] * n
    diag = []
    upper = [0.0] * n
    rhs = []

    if bc_left[0] == DIRICHLET:
        _apply_dirichlet_row(diag, rhs, bc_left[1])
    else:  # 绝热: (1+2r)u_0 - 2r u_1 = u_0_old
        diag.append(1.0 + 2.0 * r)
        upper[0] = -2.0 * r
        rhs.append(u[0])

    for i in range(1, n - 1):
        lower[i] = -r
        diag.append(1.0 + 2.0 * r)
        upper[i] = -r
        rhs.append(u[i])

    if bc_right[0] == DIRICHLET:
        _apply_dirichlet_row(diag, rhs, bc_right[1])
    else:  # 绝热: -2r u_{n-2} + (1+2r)u_{n-1} = u_{n-1}_old
        lower[n - 1] = -2.0 * r
        diag.append(1.0 + 2.0 * r)
        rhs.append(u[n - 1])

    return _thomas(lower, diag, upper, rhs)


def solve(u0, alpha, dx, dt, steps,
          bc_left=(DIRICHLET, 0.0), bc_right=(DIRICHLET, 0.0),
          scheme="explicit"):
    """推进 steps 步, 返回末态温度分布（list）。

    u0: 初始温度分布（长度 N+1 的序列）
    alpha: 扩散系数（>=0）
    dx, dt: 空间/时间步长
    bc_left / bc_right: (类型, 值) 元组
    scheme: "explicit" 或 "implicit"
    """
    u = [float(x) for x in u0]
    if len(u) < 2:
        raise ValueError("至少需要 2 个网格节点")
    if alpha < 0.0:
        raise ValueError("扩散系数不能为负")
    if dx <= 0.0 or dt <= 0.0:
        raise ValueError("步长必须为正")
    for bc in (bc_left, bc_right):
        if bc[0] not in (DIRICHLET, NEUMANN):
            raise ValueError("未知边界类型: %r" % (bc[0],))

    if alpha == 0.0:
        return u  # 零扩散: 温度场不演化

    r = alpha * dt / (dx * dx)

    if scheme == "explicit":
        check_stable(alpha, dx, dt)
        step = _step_explicit
    elif scheme == "implicit":
        step = _step_implicit
    else:
        raise ValueError("未知格式: %r" % (scheme,))

    for _ in range(steps):
        u = step(u, r, bc_left, bc_right)
    return u
