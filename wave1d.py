"""One-dimensional wave equation solver (standard library only).

Solves u_tt = c^2 u_xx on [0, L] with a uniform grid, using
second-order central differences in space and leapfrog in time:

    u_i^{n+1} = 2 u_i^n - u_i^{n-1}
                + r^2 (u_{i+1}^n - 2 u_i^n + u_{i-1}^n),

where r = c*dt/dx is the Courant number.

Boundary conditions
-------------------
DIRICHLET : u = 0 at the boundary (closed, reflects with sign flip).
NEUMANN   : du/dx = 0 at the boundary (closed, reflects without flip),
            enforced with a ghost point u_{-1} = u_1, i.e. the scheme
            is applied on the boundary with a 2 r^2 (u_1 - u_0) term.
            With trapezoidal endpoint weights in the kinetic energy
            this conserves the discrete energy exactly, like Dirichlet.
MUR       : first-order absorbing boundary condition (Mur, 1981).
            At the left boundary the one-way wave equation u_t = c u_x
            is discretised as

                u_0^{n+1} = u_1^n + (r-1)/(r+1) * (u_1^{n+1} - u_0^n),

            and symmetrically at the right boundary.  For r = 1 this is
            an exact discrete absorber; for r < 1 a small spurious
            reflection remains.

Stability (CFL condition)
-------------------------
Von Neumann analysis of the scheme gives the amplification equation

    g^2 - 2 (1 - 2 r^2 sin^2(k dx / 2)) g + 1 = 0.

|g| <= 1 for all wavenumbers k iff r <= 1, i.e.  c*dt/dx <= 1.

Numerical dispersion
--------------------
Substituting u = exp(i(k x - omega t)) into the scheme gives the
discrete dispersion relation

    sin^2(omega dt / 2) = r^2 sin^2(k dx / 2).

The continuum relation omega = c k is recovered only as k dx -> 0.
Numerical phase and group velocities:

    c_phase = omega / k,
    c_group = d omega / dk = c cos(k dx/2) / cos(omega dt/2).

Both drop below c for short wavelengths (k dx -> pi), so wave packets
with few grid points per wavelength travel too slowly and disperse:
this is grid (numerical) dispersion, not physics.
"""

import math

DIRICHLET = "dirichlet"
NEUMANN = "neumann"
MUR = "mur"

_BOUNDARIES = (DIRICHLET, NEUMANN, MUR)


class WaveSolver1D:
    """Leapfrog central-difference solver for u_tt = c^2 u_xx."""

    def __init__(self, nx, dx, c, dt, left=DIRICHLET, right=DIRICHLET):
        if nx < 4:
            raise ValueError("need at least 4 grid points")
        if dx <= 0 or c <= 0 or dt <= 0:
            raise ValueError("dx, c, dt must be positive")
        if left not in _BOUNDARIES or right not in _BOUNDARIES:
            raise ValueError("unknown boundary condition")
        self.nx = nx
        self.dx = dx
        self.c = c
        self.dt = dt
        self.left = left
        self.right = right
        self.u_prev = [0.0] * nx
        self.u_cur = [0.0] * nx
        self.u_next = [0.0] * nx
        self.t = 0.0
        self.nstep = 0

    @property
    def courant(self):
        """Courant number r = c*dt/dx; stability requires r <= 1."""
        return self.c * self.dt / self.dx

    @property
    def length(self):
        return (self.nx - 1) * self.dx

    def x(self, i):
        return i * self.dx

    def set_initial(self, u0, v0):
        """Set displacement u0 and velocity v0 (lists of length nx).

        The first leapfrog level is built with a Taylor step,
        u^1 = u^0 + dt v^0 + (r^2/2) D2(u^0), then the boundary
        condition is applied so all levels are consistent.
        """
        if len(u0) != self.nx or len(v0) != self.nx:
            raise ValueError("initial data must have length nx")
        r2 = self.courant ** 2
        u1 = list(u0)
        for i in range(1, self.nx - 1):
            u1[i] = (u0[i] + self.dt * v0[i]
                     + 0.5 * r2 * (u0[i + 1] - 2.0 * u0[i] + u0[i - 1]))
        n = self.nx - 1
        q = (self.courant - 1.0) / (self.courant + 1.0)
        for side, i0, i1 in ((self.left, 0, 1), (self.right, n, n - 1)):
            if side == DIRICHLET:
                u0[i0] = 0.0
                u1[i0] = 0.0
            elif side == NEUMANN:
                u1[i0] = u0[i0] + self.dt * v0[i0] + r2 * (u0[i1] - u0[i0])
            else:  # MUR
                u1[i0] = u0[i1] + q * (u1[i1] - u0[i0])
        self.u_prev = list(u0)
        self.u_cur = u1
        self.t = self.dt
        self.nstep = 1

    def _apply_boundary(self, u_new, u_old, u_cur):
        """Apply boundary conditions to u_new.

        u_new: level n+1 (interior already updated), u_old: level n-1 is
        unused, u_cur: level n.  Mur needs u_cur and the new interior.
        """
        q = (self.courant - 1.0) / (self.courant + 1.0)
        r2 = self.courant ** 2
        n = self.nx - 1
        if self.left == DIRICHLET:
            u_new[0] = 0.0
        elif self.left == NEUMANN:
            u_new[0] = (2.0 * u_cur[0] - u_old[0]
                        + 2.0 * r2 * (u_cur[1] - u_cur[0]))
        else:  # MUR
            u_new[0] = u_cur[1] + q * (u_new[1] - u_cur[0])
        if self.right == DIRICHLET:
            u_new[n] = 0.0
        elif self.right == NEUMANN:
            u_new[n] = (2.0 * u_cur[n] - u_old[n]
                        + 2.0 * r2 * (u_cur[n - 1] - u_cur[n]))
        else:  # MUR
            u_new[n] = u_cur[n - 1] + q * (u_new[n - 1] - u_cur[n])

    def step(self, nsteps=1):
        """Advance the solution by nsteps time steps."""
        r2 = self.courant ** 2
        for _ in range(nsteps):
            u, up, un = self.u_cur, self.u_prev, self.u_next
            for i in range(1, self.nx - 1):
                un[i] = (2.0 * u[i] - up[i]
                         + r2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]))
            self._apply_boundary(un, up, u)
            self.u_prev, self.u_cur, self.u_next = u, un, up
            self.t += self.dt
            self.nstep += 1

    def max_abs(self):
        return max(abs(v) for v in self.u_cur)

    def energy(self):
        """Discrete energy at the current time level (simple form).

        E = sum_i [ (1/2)((u_i^n - u_i^{n-1})/dt)^2
                    + (1/2) c^2 ((u_{i+1}^n - u_i^n)/dx)^2 ] dx

        Bounded under the CFL condition but oscillates slightly;
        use conserved_energy() for the exactly conserved variant.
        """
        c, dt, dx = self.c, self.dt, self.dx
        u, up = self.u_cur, self.u_prev
        e = 0.0
        for i in range(self.nx):
            v = (u[i] - up[i]) / dt
            w = 0.5 if i == 0 or i == self.nx - 1 else 1.0
            e += w * 0.5 * v * v * dx
        for i in range(self.nx - 1):
            g = (u[i + 1] - u[i]) / dx
            e += 0.5 * c * c * g * g * dx
        return e

    def conserved_energy(self):
        """Leapfrog energy that is exactly conserved for closed boundaries.

        E^{n-1/2} = sum_i (1/2)((u_i^n-u_i^{n-1})/dt)^2 dx
                  + sum_i (1/2) c^2 (du_i^n du_i^{n-1}) / dx,

        with du_i^n = u_{i+1}^n - u_i^n.  For Dirichlet/Neumann ends this
        is conserved to roundoff by the scheme.
        """
        c, dt, dx = self.c, self.dt, self.dx
        u, up = self.u_cur, self.u_prev
        e = 0.0
        for i in range(self.nx):
            v = (u[i] - up[i]) / dt
            w = 0.5 if i == 0 or i == self.nx - 1 else 1.0
            e += w * 0.5 * v * v * dx
        for i in range(self.nx - 1):
            e += (0.5 * c * c / dx
                  * (u[i + 1] - u[i]) * (up[i + 1] - up[i]))
        return e


# ---------------------------------------------------------------------------
# Initial-condition helpers
# ---------------------------------------------------------------------------

def gaussian_pulse(nx, dx, x0, sigma, c=1.0, moving=True):
    """Gaussian pulse; if moving, velocity chosen for +x propagation."""
    u0, v0 = [], []
    for i in range(nx):
        x = i * dx
        s = (x - x0) / sigma
        amp = math.exp(-0.5 * s * s)
        u0.append(amp)
        # u_t = -c u_x for a right-moving wave; u_x = -(x-x0)/sigma^2 u
        v0.append(c * (x - x0) / (sigma * sigma) * amp if moving else 0.0)
    return u0, v0


def wave_packet(nx, dx, x0, sigma, k, c=1.0):
    """Gaussian-modulated cos(k x) packet moving in +x direction."""
    u0, v0 = [], []
    for i in range(nx):
        x = i * dx
        s = (x - x0) / sigma
        env = math.exp(-0.5 * s * s)
        phase = k * (x - x0)
        u0.append(env * math.cos(phase))
        # u_t = -c u_x
        ux = env * (-(x - x0) / (sigma * sigma) * math.cos(phase)
                    - k * math.sin(phase))
        v0.append(-c * ux)
    return u0, v0


# ---------------------------------------------------------------------------
# Numerical dispersion (exact for the discrete scheme)
# ---------------------------------------------------------------------------

def numerical_omega(k, c, dt, dx):
    """omega from sin^2(omega dt/2) = r^2 sin^2(k dx/2)."""
    r = c * dt / dx
    s = r * math.sin(0.5 * k * dx)
    if s > 1.0:
        raise ValueError("unstable: r*sin(k dx/2) > 1")
    return 2.0 * math.asin(s) / dt


def numerical_phase_velocity(k, c, dt, dx):
    """c_phase = omega/k of the discrete scheme (equals c only as kdx->0)."""
    return numerical_omega(k, c, dt, dx) / k


def numerical_group_velocity(k, c, dt, dx):
    """c_group = d omega/dk = c cos(k dx/2) / cos(omega dt/2)."""
    r = c * dt / dx
    om = numerical_omega(k, c, dt, dx)
    return c * math.cos(0.5 * k * dx) / math.cos(0.5 * om * dt)
