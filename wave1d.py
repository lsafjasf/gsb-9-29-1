"""1D 波动方程求解器:  u_tt = c^2 u_xx

格式: 时间/空间均为二阶中心差分 (leapfrog):

    u_i^{n+1} = 2 u_i^n - u_i^{n-1} + r^2 (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)

其中 r = c*dt/dx 为 Courant 数。稳定条件 (CFL): r <= 1。

边界条件:
  - "dirichlet": 封闭端, u = 0, 全反射且反相
  - "neumann":   自由端, du/dx = 0 (鬼点 u_{-1}=u_1), 全反射不反相
  - "mur":       一阶吸收边界 (Engquist-Majda / Mur), 出射波近似无反射;
                 r = 1 时对平面波精确无反射

仅使用 Python 标准库。
"""

import math
import cmath

DIRICHLET = "dirichlet"
NEUMANN = "neumann"
MUR = "mur"
BOUNDARIES = (DIRICHLET, NEUMANN, MUR)


class Wave1D:
    """一维波动方程中心差分求解器。

    参数:
        nx     网格点数 (含两个边界点)
        c      波速 (>0)
        dx     空间步长 (>0)
        dt     时间步长 (>0), 需满足 c*dt/dx <= 1
        left, right  左右边界类型, 取 DIRICHLET / NEUMANN / MUR
        strict 为 True 时违反 CFL 条件直接抛 ValueError
    """

    def __init__(self, nx, c, dx, dt, left=DIRICHLET, right=DIRICHLET, strict=True):
        if nx < 4:
            raise ValueError("nx 至少为 4")
        if c <= 0.0 or dx <= 0.0 or dt <= 0.0:
            raise ValueError("c, dx, dt 必须为正")
        if left not in BOUNDARIES or right not in BOUNDARIES:
            raise ValueError("未知边界类型: %r / %r" % (left, right))
        self.nx = nx
        self.c = float(c)
        self.dx = float(dx)
        self.dt = float(dt)
        self.r = self.c * self.dt / self.dx
        if strict and self.r > 1.0 + 1e-12:
            raise ValueError(
                "违反 CFL 稳定条件: r = c*dt/dx = %.6g > 1" % self.r)
        self.left = left
        self.right = right
        self.u = [0.0] * nx        # 当前层 u^n
        self.u_prev = [0.0] * nx   # 上一层 u^{n-1}
        self.n = 0                 # 已推进的步数

    # ---------------------------------------------------------------- 初始化

    def set_initial(self, u0, v0=None):
        """设置初始位移 u0 与初速度 v0 (可选)。

        起步层用二阶泰勒展开:
            u^{-1} = u0 - dt*v0 + 0.5*dt^2*c^2*D2(u0)
        保证整体二阶精度。
        """
        if len(u0) != self.nx:
            raise ValueError("u0 长度必须等于 nx")
        if v0 is None:
            v0 = [0.0] * self.nx
        if len(v0) != self.nx:
            raise ValueError("v0 长度必须等于 nx")
        self.u = list(u0)
        r2 = self.r * self.r
        prev = [u0[i] - self.dt * v0[i] for i in range(self.nx)]
        for i in range(1, self.nx - 1):
            prev[i] += 0.5 * r2 * (u0[i + 1] - 2.0 * u0[i] + u0[i - 1])
        # 边界点的起步修正 (Neumann 鬼点)
        if self.left == NEUMANN:
            prev[0] += r2 * (u0[1] - u0[0])
        if self.right == NEUMANN:
            prev[self.nx - 1] += r2 * (u0[self.nx - 2] - u0[self.nx - 1])
        self.u_prev = prev
        self.n = 0

    def set_right_traveling_gaussian(self, x0, width, amp=1.0):
        """设置一个向右传播的高斯脉冲 u(x,0) = amp*exp(-((x-x0)/w)^2)。"""
        u0, v0 = traveling_gaussian(self.nx, self.dx, self.c, x0, width, amp)
        self.set_initial(u0, v0)

    # ---------------------------------------------------------------- 推进

    def step(self, nsteps=1):
        """推进 nsteps 个时间步。"""
        for _ in range(nsteps):
            self._step_once()

    def _step_once(self):
        u, up = self.u, self.u_prev
        nx = self.nx
        r2 = self.r * self.r
        un = [0.0] * nx
        un[1:nx - 1] = [
            2.0 * uc - up_ + r2 * (ur - 2.0 * uc + ul)
            for ul, uc, ur, up_ in zip(u[:-2], u[1:-1], u[2:], up[1:-1])
        ]
        # 左边界
        if self.left == DIRICHLET:
            un[0] = 0.0
        elif self.left == NEUMANN:
            un[0] = 2.0 * u[0] - up[0] + 2.0 * r2 * (u[1] - u[0])
        else:  # MUR: u_0^{n+1} = u_1^n + q (u_1^{n+1} - u_0^n)
            q = (self.r - 1.0) / (self.r + 1.0)
            un[0] = u[1] + q * (un[1] - u[0])
        # 右边界
        if self.right == DIRICHLET:
            un[nx - 1] = 0.0
        elif self.right == NEUMANN:
            un[nx - 1] = (2.0 * u[nx - 1] - up[nx - 1]
                          + 2.0 * r2 * (u[nx - 2] - u[nx - 1]))
        else:  # MUR
            q = (self.r - 1.0) / (self.r + 1.0)
            un[nx - 1] = u[nx - 2] + q * (un[nx - 2] - u[nx - 1])
        self.u_prev, self.u = u, un
        self.n += 1

    # ---------------------------------------------------------------- 诊断

    def energy(self):
        """离散守恒能量 E^{n-1/2}:

            E = sum_i dx/2 ((u_i^n - u_i^{n-1})/dt)^2
              + sum_i c^2 dx/2 (du_{i+1/2}^n)(du_{i+1/2}^{n-1})

        动能采用梯形权重 (边界点权重 1/2)。对 Dirichlet / Neumann 封闭
        边界, 该量被格式严格守恒 (仅受浮点舍入影响)。
        """
        dx, dt, c2 = self.dx, self.dt, self.c * self.c
        u, up = self.u, self.u_prev
        w = [1.0] * self.nx
        w[0] = w[-1] = 0.5
        ke = 0.5 * dx * sum(wi * ((a - b) / dt) ** 2
                            for wi, a, b in zip(w, u, up))
        pe = 0.5 * c2 / dx * sum(
            (u[i + 1] - u[i]) * (up[i + 1] - up[i]) for i in range(self.nx - 1))
        return ke + pe

    def energy_naive(self):
        """常用(非严格守恒)能量: 动能 + 当前层势能。有 O(dt^2) 有界振荡。"""
        dx, dt, c2 = self.dx, self.dt, self.c * self.c
        u, up = self.u, self.u_prev
        w = [1.0] * self.nx
        w[0] = w[-1] = 0.5
        ke = 0.5 * dx * sum(wi * ((a - b) / dt) ** 2
                            for wi, a, b in zip(w, u, up))
        pe = 0.5 * c2 / dx * sum((u[i + 1] - u[i]) ** 2
                                 for i in range(self.nx - 1))
        return ke + pe

    def max_abs(self):
        return max(abs(v) for v in self.u)

    def l2_norm(self):
        return math.sqrt(sum(v * v for v in self.u) * self.dx)


# ---------------------------------------------------------------- 辅助函数

def traveling_gaussian(nx, dx, c, x0, width, amp=1.0, direction=+1):
    """构造单向传播高斯脉冲的 (u0, v0)。direction=+1 向右, -1 向左。"""
    u0, v0 = [], []
    for i in range(nx):
        x = i * dx
        s = (x - x0) / width
        g = amp * math.exp(-s * s)
        dg = -2.0 * s / width * g
        u0.append(g)
        v0.append(-direction * c * dg)
    return u0, v0


def cfl_max_dt(c, dx):
    """CFL 稳定条件允许的最大时间步长。"""
    return dx / c


def dispersion_omega(k, c, dx, dt):
    """数值色散关系: sin(w dt/2) = r sin(k dx/2) 解出的数值角频率。"""
    r = c * dt / dx
    s = r * math.sin(0.5 * k * dx)
    if s > 1.0:  # 超出可传播频段
        return float("nan")
    return 2.0 / dt * math.asin(s)


def phase_velocity(k, c, dx, dt):
    """数值相速度 w/k 与精确值 c 之比即网格色散误差。"""
    return dispersion_omega(k, c, dx, dt) / k


def group_velocity(k, c, dx, dt):
    """数值群速度 dw/dk = c cos(k dx/2) / sqrt(1 - r^2 sin^2(k dx/2))。"""
    r = c * dt / dx
    s2 = (r * math.sin(0.5 * k * dx)) ** 2
    if s2 >= 1.0:
        return float("nan")
    return c * math.cos(0.5 * k * dx) / math.sqrt(1.0 - s2)


def mur_reflection_coeff(k, c, dx, dt):
    """一阶 Mur 边界对平面波 e^{i(kx-wt)} 的振幅反射系数 (精确离散理论值)。

    将平面波叠加代入离散 Mur 边界条件解出 (w 取数值色散频率):
        R = [q(E/p - 1) - (E - 1/p)] / [(E - p) - q(E p - 1)]
        E = e^{-i w dt}, p = e^{i k dx}, q = (r-1)/(r+1)
    渐近式 |R| ~ (1 - r^2) (k dx)^2 / 16 (k dx -> 0), r = 1 时精确为零。
    """
    r = c * dt / dx
    q = (r - 1.0) / (r + 1.0)
    w = dispersion_omega(k, c, dx, dt)
    e = cmath.exp(-1j * w * dt)
    p = cmath.exp(1j * k * dx)
    num = q * (e / p - 1.0) - (e - 1.0 / p)
    den = (e - p) - q * (e * p - 1.0)
    return abs(num / den)
