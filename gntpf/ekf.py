"""Extended Kalman filter baseline.

Included so that the choice of a moving-horizon estimator can be argued from
evidence rather than asserted.  The EKF is given the same model, the same
measurements and a covariance tuning matched to the true noise -- i.e. every
advantage -- so that the comparison isolates what the MHE actually adds:
constraint handling and resistance to bounded outliers.
"""

from __future__ import annotations

import numpy as np

from .model import GNT

__all__ = ["EKF"]


class EKF:
    def __init__(self, model: GNT, P0=None, Q=None, R=None, project_geometry: bool = False):
        self.m = model
        nq, ny = model.nq, model.ny
        self.P = np.eye(nq) * 1e-2 if P0 is None else np.asarray(P0, float).copy()
        self.Q = np.eye(nq) * 1e-4 if Q is None else np.asarray(Q, float)
        self.R = np.eye(ny) * 1e-4 if R is None else np.asarray(R, float)
        self.q = np.zeros(nq)
        self.project_geometry = project_geometry
        self.n_var = None

    def initialise(self, q0, **_):
        self.q = np.asarray(q0, float).reshape(-1).copy()

    def push(self, y, u_applied):
        self._y = np.asarray(y, float).reshape(-1)
        self._u = np.asarray(u_applied, float).reshape(-1)

    def solve(self):
        m = self.m
        # ---- predict (RK4 of the same model, linearised about the estimate)
        A_c = np.asarray(m.jac_fx(self.q, self._u))
        F = np.eye(m.nq) + m.Ts * A_c          # first-order transition matrix
        q_pred = np.asarray(m.F(self.q, self._u)).reshape(-1)
        P_pred = F @ self.P @ F.T + self.Q

        # ---- update
        H = np.asarray(m.jac_hx(q_pred))
        S = H @ P_pred @ H.T + self.R
        try:
            K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
        except np.linalg.LinAlgError:
            K = P_pred @ H.T @ np.linalg.pinv(S)
        innov = self._y - np.asarray(m.h(q_pred)).reshape(-1)
        q_new = q_pred + K @ innov
        I_KH = np.eye(m.nq) - K @ H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        if self.project_geometry:
            # optional: re-impose the rigid-body chain after the linear update
            th = q_new[m.i_theta]
            q_new = m.state_from_pose(q_new[2 * m.N + 1], q_new[2 * m.N + 2], th)

        self.q = q_new
        return self.q.copy(), True
