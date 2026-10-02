"""Stage 2 -- tracking NMPC.

A conventional nonlinear model predictive controller that steers the estimated
state to the posture produced by :mod:`gntpf.refgen`.  Multiple shooting with an
RK4 integrator; box constraints on the joint angles, the inputs and the input
rates.  Because the reference generated in stage 1 is reachable by construction,
no terminal set or terminal control law is needed -- the terminal ingredient is
replaced by a reference the vehicle can actually attain.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .model import GNT

__all__ = ["TrackingNMPC"]


class TrackingNMPC:
    def __init__(
        self,
        model: GNT,
        Nc: int = 20,
        Q=None,
        QN=None,
        R=None,
        S_du=None,
        u_lb=(-2.0, -2.0),
        u_ub=(2.0, 2.0),
        du_lb=None,
        du_ub=None,
        beta_max: float = np.deg2rad(80.0),
        print_level: int = 0,
    ):
        self.m = model
        self.N = model.N
        self.Nc = Nc
        nq, nu = model.nq, model.nu

        if Q is None:
            Q = self.default_Q(model)
        if QN is None:
            QN = 10.0 * np.asarray(Q)
        if R is None:
            R = np.diag([0.01, 0.05])
        if S_du is None:
            # Penalising the input *change* as well as its magnitude is what keeps
            # the commanded angular rate from chattering against its rate limit in
            # response to estimation noise.  Without it the constraint is active on
            # a large fraction of steps and the input is not something one would
            # send to a real steering loop.
            S_du = np.diag([0.5, 0.1])
        if du_lb is None:
            du_lb = model.Ts * np.array([-6.0, -3.0])
        if du_ub is None:
            du_ub = model.Ts * np.array([6.0, 3.0])

        self.Q, self.QN, self.R = np.asarray(Q), np.asarray(QN), np.asarray(R)
        self.S_du = np.asarray(S_du)

        opti = ca.Opti()
        self.opti = opti
        self.p_q0 = opti.parameter(nq)
        self.p_qref = opti.parameter(nq)
        self.p_ulast = opti.parameter(nu)
        self.p_Q = opti.parameter(nq, nq)
        self.p_QN = opti.parameter(nq, nq)
        self.p_R = opti.parameter(nu, nu)
        self.p_S = opti.parameter(nu, nu)

        self.v_Q = opti.variable(nq, Nc + 1)
        self.v_U = opti.variable(nu, Nc)

        opti.subject_to(self.v_Q[:, 0] == self.p_q0)
        # joint-angle limits on the predicted trajectory (jackknife avoidance)
        for j in range(self.N):
            opti.subject_to(opti.bounded(-beta_max, self.v_Q[j, 1:], beta_max))
        opti.subject_to(opti.bounded(ca.DM(u_lb), self.v_U, ca.DM(u_ub)))
        opti.subject_to(opti.bounded(ca.DM(du_lb), self.v_U[:, 0] - self.p_ulast, ca.DM(du_ub)))
        if Nc > 1:
            opti.subject_to(opti.bounded(ca.DM(du_lb),
                                         self.v_U[:, 1:] - self.v_U[:, :-1],
                                         ca.DM(du_ub)))

        J = 0
        for i in range(Nc):
            opti.subject_to(self.v_Q[:, i + 1] == self.m.F(self.v_Q[:, i], self.v_U[:, i]))
            e = self.v_Q[:, i] - self.p_qref
            J += ca.bilin(self.p_Q, e, e) + ca.bilin(self.p_R, self.v_U[:, i], self.v_U[:, i])
            du = self.v_U[:, i] - (self.p_ulast if i == 0 else self.v_U[:, i - 1])
            J += ca.bilin(self.p_S, du, du)
        eN = self.v_Q[:, Nc] - self.p_qref
        J += ca.bilin(self.p_QN, eN, eN)

        opti.minimize(J)
        self.J = J
        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"print_level": print_level, "sb": "yes", "max_iter": 400,
             "warm_start_init_point": "yes", "acceptable_tol": 1e-6},
        )
        opti.set_value(self.p_Q, ca.DM(self.Q))
        opti.set_value(self.p_QN, ca.DM(self.QN))
        opti.set_value(self.p_R, ca.DM(self.R))
        opti.set_value(self.p_S, ca.DM(self.S_du))
        opti.set_value(self.p_ulast, ca.DM(np.zeros(nu)))

        self._Qwarm = None
        self._Uwarm = None
        self.n_var = int(opti.nx)
        self.n_con = int(opti.ng)

    # ------------------------------------------------------------------ tuning
    @staticmethod
    def default_Q(model: GNT, last_trailer_weight: float = 5.0) -> np.ndarray:
        """Diagonal state weighting: joint angles, headings, then positions with
        the last trailer weighted up."""
        N = model.N
        d = np.concatenate([
            np.ones(N),                     # beta
            0.5 * np.ones(N + 1),           # theta
            np.ones(2 * N),                 # x,y of tractor .. trailer N-1
            last_trailer_weight * np.ones(2),   # x,y of the last trailer
        ])
        return np.diag(d)

    @staticmethod
    def Q_last_only(model: GNT, w: float = 8.0) -> np.ndarray:
        """Weighting that cares only about the last trailer's position -- the
        classical objective, used as a baseline."""
        N = model.N
        d = np.zeros(4 * N + 3)
        d[-2:] = w
        return np.diag(d)

    def set_Q(self, Q, QN=None):
        self.Q = np.asarray(Q)
        self.opti.set_value(self.p_Q, ca.DM(self.Q))
        self.QN = np.asarray(QN) if QN is not None else 10.0 * self.Q
        self.opti.set_value(self.p_QN, ca.DM(self.QN))

    # ------------------------------------------------------------------- solve
    def solve(self, q0, q_ref, u_last):
        o = self.opti
        o.set_value(self.p_q0, ca.DM(np.asarray(q0).reshape(-1)))
        o.set_value(self.p_qref, ca.DM(np.asarray(q_ref).reshape(-1)))
        o.set_value(self.p_ulast, ca.DM(np.asarray(u_last).reshape(-1)))

        if self._Qwarm is not None:
            o.set_initial(self.v_Q, ca.DM(self._Qwarm))
            o.set_initial(self.v_U, ca.DM(self._Uwarm))
        else:
            o.set_initial(self.v_Q, ca.DM(np.tile(np.asarray(q0).reshape(-1, 1), (1, self.Nc + 1))))

        try:
            sol = o.solve()
            Qtraj = np.asarray(sol.value(self.v_Q))
            Utraj = np.asarray(sol.value(self.v_U)).reshape(self.m.nu, self.Nc)
            ok = True
        except RuntimeError:
            Qtraj = np.asarray(o.debug.value(self.v_Q))
            Utraj = np.asarray(o.debug.value(self.v_U)).reshape(self.m.nu, self.Nc)
            ok = False

        # shift for the next warm start
        self._Qwarm = np.hstack([Qtraj[:, 1:], Qtraj[:, -1:]])
        self._Uwarm = np.hstack([Utraj[:, 1:], Utraj[:, -1:]])
        return Utraj[:, 0], Qtraj, Utraj, ok
