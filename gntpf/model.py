"""Kinematic model of the Generalised N-Trailer (GNT) vehicle.

State ordering (matches the MATLAB implementation ``refGenMinOffTrackingNtrailers.m``)::

    q = [ beta_1 ... beta_N ,            # N joint angles
          theta_0 ... theta_N ,          # N+1 headings (tractor first)
          x_0, y_0 ,                     # tractor position
          x_1, y_1, ..., x_N, y_N ]      # trailer positions

    nq = 4N + 3

Input ordering::

    u = [ omega_0 , v_0 ]                # tractor angular rate and forward speed

Geometry.  Trailer ``i`` (1-indexed) has drawbar/hitch offset ``Lh[i-1]`` measured
on segment ``i-1`` and length ``L[i-1]``.  A non-zero ``Lh`` makes the hitch
off-axle, which is what distinguishes a Generalised N-Trailer from a Standard one:

    Lh_i == 0  for all i        -> Standard N-Trailer (SNT)
    Lh_i != 0  for all i        -> non-Standard N-Trailer (nSNT)
    mixed signs / zeros         -> Generalised N-Trailer (GNT)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

__all__ = ["GNT", "wrap_to_pi", "atan2c"]


def wrap_to_pi(a):
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def atan2c(dy, dx, previous):
    """Continuous ``atan2``: the branch of ``atan2(dy, dx)`` nearest ``previous``.

    Port of the MATLAB helper ``atan2c``.  Used when sampling the tangent angle of
    a path so that the heading reference does not jump by 2*pi between samples.
    """
    a = np.arctan2(dy, dx)
    return a + 2 * np.pi * np.round((previous - a) / (2 * np.pi))


@dataclass
class GNT:
    """Drift-free kinematics of a GNT vehicle with ``N`` passive trailers.

    Parameters
    ----------
    N : int
        Number of trailers.
    Lh : array_like, shape (N,)
        Hitch offsets.  ``Lh[i]`` belongs to trailer ``i+1``.
    L : array_like, shape (N,)
        Trailer lengths.  ``L[i]`` belongs to trailer ``i+1``.
    Ts : float
        Sampling period used by the RK4 discretisation.
    """

    N: int
    Lh: np.ndarray
    L: np.ndarray
    Ts: float = 0.05

    # populated in __post_init__
    nq: int = field(init=False)
    nu: int = field(init=False)
    ny: int = field(init=False)
    f: ca.Function = field(init=False, repr=False)
    F: ca.Function = field(init=False, repr=False)
    h: ca.Function = field(init=False, repr=False)
    jac_fx: ca.Function = field(init=False, repr=False)
    jac_fu: ca.Function = field(init=False, repr=False)
    jac_hx: ca.Function = field(init=False, repr=False)

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.Lh = np.asarray(self.Lh, dtype=float).reshape(-1)
        self.L = np.asarray(self.L, dtype=float).reshape(-1)
        if self.Lh.size != self.N or self.L.size != self.N:
            raise ValueError("Lh and L must each have N entries")
        if np.any(np.abs(self.L) < 1e-9):
            raise ValueError("trailer lengths L must be non-zero")

        self.nq = 4 * self.N + 3
        self.nu = 2
        self.ny = self.N + 3  # beta_1..beta_N, theta_0, x_0, y_0

        q = ca.MX.sym("q", self.nq)
        u = ca.MX.sym("u", self.nu)

        f_rhs = self._f_rhs(q, u)
        self.f = ca.Function("f", [q, u], [f_rhs], ["q", "u"], ["dq"])

        # fixed-step RK4, as in the MATLAB code
        k1 = self.f(q, u)
        k2 = self.f(q + self.Ts / 2 * k1, u)
        k3 = self.f(q + self.Ts / 2 * k2, u)
        k4 = self.f(q + self.Ts * k3, u)
        q_next = q + self.Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.F = ca.Function("F", [q, u], [q_next], ["q", "u"], ["q_next"])

        h_rhs = ca.vertcat(q[: self.N], q[self.N], q[2 * self.N + 1], q[2 * self.N + 2])
        self.h = ca.Function("h", [q], [h_rhs], ["q"], ["y"])

        self.jac_fx = ca.Function("A", [q, u], [ca.jacobian(f_rhs, q)])
        self.jac_fu = ca.Function("B", [q, u], [ca.jacobian(f_rhs, u)])
        self.jac_hx = ca.Function("C", [q], [ca.jacobian(h_rhs, q)])

    # ------------------------------------------------------------- index maps
    @property
    def i_beta(self) -> slice:
        return slice(0, self.N)

    @property
    def i_theta(self) -> slice:
        return slice(self.N, 2 * self.N + 1)

    @property
    def i_xy(self) -> slice:
        """All segment positions, ``[x_0, y_0, x_1, y_1, ...]``."""
        return slice(2 * self.N + 1, 4 * self.N + 3)

    def i_pos(self, seg: int) -> slice:
        """Position slice of segment ``seg`` (0 = tractor)."""
        if not 0 <= seg <= self.N:
            raise IndexError(f"segment {seg} out of range for N={self.N}")
        b = 2 * self.N + 1 + 2 * seg
        return slice(b, b + 2)

    def positions(self, q) -> np.ndarray:
        """Return an ``(N+1, 2)`` array of segment positions."""
        q = np.asarray(q).reshape(-1)
        return q[self.i_xy].reshape(self.N + 1, 2)

    # -------------------------------------------------------------- kinematics
    def _J(self, beta_k, k: int):
        """Velocity transformation from segment ``k`` to segment ``k+1``."""
        Lh, L = self.Lh[k], self.L[k]
        return ca.vertcat(
            ca.horzcat(-Lh * ca.cos(beta_k) / L, ca.sin(beta_k) / L),
            ca.horzcat(Lh * ca.sin(beta_k), ca.cos(beta_k)),
        )

    def _chain(self, q):
        """``chain[m] = J_m @ ... @ J_1`` with ``chain[0] = I``."""
        chain = [ca.MX.eye(2)]
        for k in range(self.N):
            chain.append(ca.mtimes(self._J(q[k], k), chain[k]))
        return chain

    def _f_rhs(self, q, u):
        N = self.N
        chain = self._chain(q)
        e1 = ca.DM([[1.0, 0.0]])

        rows = []
        # joint-angle rates: d(beta_k) = omega_{k-1} - omega_k
        for k in range(N):
            Jk = self._J(q[k], k)
            rows.append(ca.mtimes(e1, ca.mtimes(ca.MX.eye(2) - Jk, ca.mtimes(chain[k], u))))
        # heading rates
        for m in range(N + 1):
            rows.append(ca.mtimes(e1, ca.mtimes(chain[m], u)))
        # positions
        for m in range(N + 1):
            th = q[N + m]
            sel = ca.horzcat(ca.DM(0.0), ca.cos(th))
            rows.append(ca.mtimes(sel, ca.mtimes(chain[m], u)))
            sel = ca.horzcat(ca.DM(0.0), ca.sin(th))
            rows.append(ca.mtimes(sel, ca.mtimes(chain[m], u)))
        return ca.vertcat(*rows)

    # -------------------------------------------------------------- geometry
    def chain_offsets(self, thetas):
        """Cumulative offsets from the tractor to each trailer.

        ``thetas`` is a length ``N+1`` symbolic or numeric vector of headings.
        Returns a list of ``N`` 2-vectors, where entry ``k`` is the vector that
        must be *subtracted* from the tractor position to obtain the position of
        trailer ``k+1``.
        """
        offs, acc_x, acc_y = [], 0, 0
        for k in range(self.N):
            acc_x = acc_x + self.Lh[k] * ca.cos(thetas[k]) + self.L[k] * ca.cos(thetas[k + 1])
            acc_y = acc_y + self.Lh[k] * ca.sin(thetas[k]) + self.L[k] * ca.sin(thetas[k + 1])
            offs.append(ca.vertcat(acc_x, acc_y))
        return offs

    def state_from_pose(self, x0, y0, thetas) -> np.ndarray:
        """Build a kinematically consistent state from a tractor pose and headings.

        ``thetas`` has ``N+1`` entries.  Joint angles and trailer positions follow
        from the rigid-body geometry, so the returned state always satisfies the
        algebraic constraints of the chain.
        """
        thetas = np.asarray(thetas, dtype=float).reshape(-1)
        if thetas.size != self.N + 1:
            raise ValueError("thetas must have N+1 entries")
        betas = thetas[:-1] - thetas[1:]
        q = np.zeros(self.nq)
        q[self.i_beta] = betas
        q[self.i_theta] = thetas
        q[2 * self.N + 1] = x0
        q[2 * self.N + 2] = y0
        ax = ay = 0.0
        for k in range(self.N):
            ax += self.Lh[k] * np.cos(thetas[k]) + self.L[k] * np.cos(thetas[k + 1])
            ay += self.Lh[k] * np.sin(thetas[k]) + self.L[k] * np.sin(thetas[k + 1])
            q[self.i_pos(k + 1)] = [x0 - ax, y0 - ay]
        return q

    @property
    def total_length(self) -> float:
        """Sum of all hitch offsets and trailer lengths -- the vehicle's reach."""
        return float(np.sum(np.abs(self.Lh)) + np.sum(np.abs(self.L)))

    @property
    def hitching(self) -> str:
        """Classify the vehicle as ``SNT``, ``nSNT`` or ``GNT``."""
        z = np.isclose(self.Lh, 0.0)
        if z.all():
            return "SNT"
        if (~z).all():
            return "nSNT"
        return "GNT"

    def step(self, q, u) -> np.ndarray:
        """One RK4 step of the true plant."""
        return np.asarray(self.F(q, u)).reshape(-1)

    def output(self, q) -> np.ndarray:
        return np.asarray(self.h(q)).reshape(-1)
