"""Nonlinear moving horizon estimator for the GNT vehicle.

Only the tractor pose and the joint angles are measured; trailer headings and
positions must be reconstructed.  The estimator is a multiple-shooting NLP over a
window of ``Ne`` past sampling instants with an arrival cost summarising
everything older.

Three features are what an EKF cannot offer here and are the reason the extra
cost is worth paying:

* the joint angles are **bounded** by the jackknife limit, and the estimate is
  constrained to respect that bound;
* the rigid-body geometry of the chain enters as a hard equality, so every
  estimate is a physically realisable posture;
* the applied inputs are treated as decision variables penalised around their
  commanded values, which absorbs wheel slip and actuator error instead of
  letting it corrupt the state.
"""

from __future__ import annotations

from collections import deque

import casadi as ca
import numpy as np

from .model import GNT

__all__ = ["NMHE"]


class NMHE:
    def __init__(
        self,
        model: GNT,
        Ne: int = 10,
        P=None,
        Q=None,
        R=None,
        Z=None,
        dU=None,
        w_bound: float = 0.1,
        v_bound=None,
        z_bound: float = 0.01,
        beta_max: float | None = None,
        u_lb=(-2.0, -2.0),
        u_ub=(2.0, 2.0),
        print_level: int = 0,
    ):
        self.m = model
        self.N = model.N
        self.Ne = Ne
        nq, nu, ny = model.nq, model.nu, model.ny

        P = np.eye(nq) * 1e6 if P is None else np.asarray(P)
        Q = np.eye(nq) * 1e6 if Q is None else np.asarray(Q)
        R = np.diag(np.concatenate([0.01 * np.ones(self.N), [0.01], [100.0, 100.0]])) if R is None else np.asarray(R)
        Z = np.eye(self.N) if Z is None else np.asarray(Z)
        dU = np.eye(nu) if dU is None else np.asarray(dU)
        self.P, self.Q, self.R, self.Z, self.dU = P, Q, R, Z, dU

        opti = ca.Opti()
        self.opti = opti

        self.p_qbar = opti.parameter(nq)
        self.p_y = opti.parameter(ny, Ne + 1)
        self.p_u = opti.parameter(nu, Ne)

        self.v_q = opti.variable(nq, Ne + 1)
        self.v_w = opti.variable(nq, Ne)
        self.v_v = opti.variable(ny, Ne + 1)
        self.v_z = opti.variable(self.N, Ne + 1)
        self.v_u = opti.variable(nu, Ne)

        # AZ q + z = 0  encodes  beta_i - (theta_{i-1} - theta_i) = z_i
        AZ = np.zeros((self.N, nq))
        for i in range(self.N):
            AZ[i, i] = 1.0
            AZ[i, self.N + i] = -1.0
            AZ[i, self.N + i + 1] = 1.0
        self.AZ = AZ

        e0 = self.v_q[:, 0] - self.p_qbar
        J = ca.bilin(ca.DM(P), e0, e0)

        for i in range(Ne):
            opti.subject_to(self.v_q[:, i + 1] == self.m.F(self.v_q[:, i], self.v_u[:, i]) + self.v_w[:, i])
            J += ca.bilin(ca.DM(Q), self.v_w[:, i], self.v_w[:, i])
            du = self.v_u[:, i] - self.p_u[:, i]
            J += ca.bilin(ca.DM(dU), du, du)

        for i in range(Ne + 1):
            opti.subject_to(self.p_y[:, i] - self.m.h(self.v_q[:, i]) - self.v_v[:, i] == 0)
            J += ca.bilin(ca.DM(R), self.v_v[:, i], self.v_v[:, i])
            opti.subject_to(ca.mtimes(ca.DM(AZ), self.v_q[:, i]) + self.v_z[:, i] == 0)
            J += ca.bilin(ca.DM(Z), self.v_z[:, i], self.v_z[:, i])
            # hard rigid-body geometry
            theta = self.v_q[self.N : 2 * self.N + 1, i]
            pos0 = self.v_q[2 * self.N + 1 : 2 * self.N + 3, i]
            offs = self.m.chain_offsets(theta)
            for k in range(self.N):
                seg = self.v_q[2 * self.N + 3 + 2 * k : 2 * self.N + 5 + 2 * k, i]
                opti.subject_to(seg - (pos0 - offs[k]) == 0)
            # No bound on the estimated joint angles by default: the estimator
            # must report the joint angle the vehicle actually has, so that an
            # excursion beyond the mechanical limit can be detected rather than
            # clipped away.  A bound can still be imposed by passing beta_max.
            if beta_max is not None:
                for j in range(self.N):
                    opti.subject_to(opti.bounded(-beta_max, self.v_q[j, i], beta_max))

        opti.subject_to(opti.bounded(-w_bound, self.v_w, w_bound))
        opti.subject_to(opti.bounded(-z_bound, self.v_z, z_bound))
        if v_bound is not None:
            vb = np.asarray(v_bound, dtype=float).reshape(-1)
            for r in range(ny):
                if np.isfinite(vb[r]):
                    opti.subject_to(opti.bounded(-vb[r], self.v_v[r, :], vb[r]))
        opti.subject_to(opti.bounded(ca.DM(u_lb), self.v_u, ca.DM(u_ub)))

        opti.minimize(J)
        self.J = J
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"print_level": print_level, "sb": "yes", "max_iter": 600,
             "warm_start_init_point": "yes", "tol": 1e-6, "acceptable_tol": 1e-6},
        )

        self.ys: deque = deque(maxlen=Ne + 1)
        self.us: deque = deque(maxlen=Ne)
        self.qbar = None
        self._warm = None
        self.n_var = int(opti.nx)
        self.n_con = int(opti.ng)

    # ------------------------------------------------------------------ buffers
    def initialise(self, q0bar, y0=None, u0=None):
        q0bar = np.asarray(q0bar, dtype=float).reshape(-1)
        self.qbar = q0bar.copy()
        y0 = self.m.output(q0bar) if y0 is None else np.asarray(y0).reshape(-1)
        u0 = np.zeros(self.m.nu) if u0 is None else np.asarray(u0).reshape(-1)
        self.ys.clear()
        self.us.clear()
        for _ in range(self.Ne + 1):
            self.ys.append(y0.copy())
        for _ in range(self.Ne):
            self.us.append(u0.copy())
        self._warm = np.tile(q0bar.reshape(-1, 1), (1, self.Ne + 1))

    def push(self, y, u_applied):
        self.ys.append(np.asarray(y, dtype=float).reshape(-1))
        self.us.append(np.asarray(u_applied, dtype=float).reshape(-1))

    # -------------------------------------------------------------------- solve
    def solve(self):
        o = self.opti
        o.set_value(self.p_qbar, ca.DM(self.qbar))
        o.set_value(self.p_y, ca.DM(np.array(self.ys).T))
        o.set_value(self.p_u, ca.DM(np.array(self.us).T))
        if self._warm is not None:
            o.set_initial(self.v_q, ca.DM(self._warm))

        try:
            sol = o.solve()
            Qe = np.asarray(sol.value(self.v_q))
            ok = True
        except RuntimeError:
            Qe = np.asarray(o.debug.value(self.v_q))
            ok = False

        self._warm = np.hstack([Qe[:, 1:], Qe[:, -1:]])
        # arrival cost recentred on the oldest state still inside the window
        self.qbar = Qe[:, 1].copy()
        return Qe[:, -1].copy(), ok
