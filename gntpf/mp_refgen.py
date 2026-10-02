"""Reference reconstruction after Michałek and Pazderski (EJC 58:60-73, 2021).

The closest prior work to the reference generator of this paper: given a
prescribed output-reference for the *last trailer*, reconstruct joint references
for the whole chain that are admissible and free of a jackknife.  It is the
natural comparison, because it addresses the same problem on the same vehicle
class -- both nSNT and GNT -- rather than a different problem on a narrower one.

Structure of the method, following the paper's Section 4:

* joints are partitioned into on-axle (``Lh = 0``) and off-axle (``Lh != 0``);
* an on-axle joint is determined algebraically by the segment behind it,
  ``tan(beta_s) = L_s * omega_s / v_s``                                 (Eq. 29);
* the off-axle joints obey an ODE in the path parameter whose solution is *not*
  unique -- there are at least ``2^N`` of them and only one avoids a jackknife;
* for constant curvature the admissible one is available in closed form
  (Eqs. 32-37), which is what :func:`closed_form_betas` implements;
* for varying curvature it is found by expressing each off-axle joint reference
  in a Fourier basis and fitting the coefficients so that the ODE residual
  vanishes in least squares (Eqs. 40-47), with a homotopy in the curvature to
  keep the iteration on the admissible branch.

Two properties of the method matter for the comparison and are not incidental:
it prescribes a reference for **one** segment only, so the remaining segments go
wherever the reconstruction puts them; and the fit is **offline** over a whole
periodic reference, so it is not a receding-horizon computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .model import GNT, atan2c, wrap_to_pi
from .paths import Path

__all__ = ["MPReferenceGenerator", "closed_form_betas", "joint_rhs"]


# --------------------------------------------------------------------------- #
#  Chain velocity recursion
# --------------------------------------------------------------------------- #
def _Jinv(Lh, L, b):
    """``J^{-1}`` for an off-axle joint; singular when ``Lh = 0``."""
    return np.array([[-L * np.cos(b) / Lh, np.sin(b) / Lh],
                     [L * np.sin(b), np.cos(b)]])


def joint_rhs(model: GNT, beta, u_N, dbeta_on=None):
    """``d(beta)/dt`` and the segment velocities, working *backwards* from ``u_N``.

    ``u_N = [omega_N, v_N]`` is the prescribed guiding velocity of the last
    trailer.  Returns ``(dbeta, U)`` where ``U[k]`` is the velocity of segment
    ``k``.

    An off-axle joint is inverted directly.  An on-axle joint cannot be: its
    ``J`` has rank one, so the velocity of the segment *ahead* of it is not
    determined by the segment behind.  What is determined instead is the joint
    angle itself, and the missing angular rate follows from
    ``d(beta_s)/dt = omega_{s-1} - omega_s`` once ``d(beta_s)/dt`` is known --
    which is why ``dbeta_on`` has to be supplied for on-axle joints.
    """
    N = model.N
    U = [None] * (N + 1)
    U[N] = np.asarray(u_N, dtype=float).reshape(2)
    dbeta = np.zeros(N)
    for i in range(N, 0, -1):                       # joint i couples i-1 and i
        Lh, L, b = model.Lh[i - 1], model.L[i - 1], beta[i - 1]
        if abs(Lh) > 1e-12:
            U[i - 1] = _Jinv(Lh, L, b) @ U[i]
            dbeta[i - 1] = U[i - 1][0] - U[i][0]
        else:
            d = 0.0 if dbeta_on is None else float(dbeta_on[i - 1])
            w_prev = U[i][0] + d
            c = np.cos(b)
            v_prev = U[i][1] / c if abs(c) > 1e-9 else U[i][1] * 1e9
            U[i - 1] = np.array([w_prev, v_prev])
            dbeta[i - 1] = d
    return dbeta, U


def closed_form_betas(model: GNT, kappa: float, xi: float = 1.0) -> np.ndarray:
    """Admissible joint angles at constant curvature -- Eqs. (32)-(37).

    ``R_{i-1} = xi * sqrt(R_i^2 + L_i^2 - Lh_i^2)`` with ``xi = sgn(R_N)`` is the
    branch that satisfies the anti-jackknife condition; flipping ``xi`` on
    individual joints enumerates the other ``2^N - 1`` solutions.
    """
    N = model.N
    if abs(kappa) < 1e-12:
        return np.zeros(N)
    R = np.empty(N + 1)
    R[N] = 1.0 / kappa                              # signed radius of the last trailer
    beta = np.zeros(N)
    for i in range(N, 0, -1):
        L, Lh = model.L[i - 1], model.Lh[i - 1]
        R[i - 1] = np.sign(R[N]) * xi * np.sqrt(max(R[i] ** 2 + L ** 2 - Lh ** 2, 0.0))
        z1 = L * R[i - 1] + Lh * R[i]
        z2 = R[i] * R[i - 1] - L * Lh
        beta[i - 1] = np.arctan2(z1, z2)
    return wrap_to_pi(beta)


# --------------------------------------------------------------------------- #
#  Fourier-collocation reconstruction
# --------------------------------------------------------------------------- #
@dataclass
class MPReferenceGenerator:
    """Offline reconstruction of an admissible joint-reference.

    Parameters
    ----------
    model : GNT
    n_harm : int
        Number of Fourier harmonics per off-axle joint.
    n_coll : int
        Number of collocation points in the path parameter.
    homotopy : int
        Number of continuation stages.  The curvature profile is scaled from
        zero to its true value; at zero the admissible solution is ``beta = 0``,
        and each stage is started from the previous stage's solution, which is
        what keeps the fit on the anti-jackknife branch rather than on one of the
        other ``2^N - 1`` solutions.
    """

    model: GNT
    n_harm: int = 12
    n_coll: int = 400
    homotopy: int = 6
    verbose: bool = False

    i_off: np.ndarray = field(init=False)
    i_on: np.ndarray = field(init=False)

    def __post_init__(self):
        lh = np.abs(np.asarray(self.model.Lh, dtype=float))
        self.i_off = np.where(lh > 1e-12)[0]
        self.i_on = np.where(lh <= 1e-12)[0]
        self.name = "michalek-pazderski"
        self.target_segment = self.model.N

    # ------------------------------------------------------------------ basis
    def _basis(self, p, P):
        """Fourier basis and its derivative at the collocation points."""
        w = 2.0 * np.pi / P
        cols, dcols = [np.ones_like(p)], [np.zeros_like(p)]
        for h in range(1, self.n_harm + 1):
            cols += [np.cos(h * w * p), np.sin(h * w * p)]
            dcols += [-h * w * np.sin(h * w * p), h * w * np.cos(h * w * p)]
        return np.stack(cols, axis=1), np.stack(dcols, axis=1)

    # ------------------------------------------------------- on-axle relation
    def _beta_on(self, kappa, dkappa):
        """On-axle joints from Eq. (29), and their derivative in ``p``.

        Unit-speed parameterisation, so ``omega_s = kappa`` and ``v_s = 1`` for
        the segment behind the joint, giving ``beta_s = atan(L_s * kappa)``.
        """
        b = {}
        for s in self.i_on:
            L = self.model.L[s]
            b[s] = (np.arctan(L * kappa), L * dkappa / (1.0 + (L * kappa) ** 2))
        return b

    # --------------------------------------------------------------- residual
    def _residual(self, w, Phi, dPhi, kappa, dkappa):
        n_off = self.i_off.size
        W = w.reshape(n_off, -1)
        B = Phi @ W.T                     # (n_coll, n_off) off-axle angles
        dB = dPhi @ W.T                   # their p-derivatives
        bon = self._beta_on(kappa, dkappa)
        res = np.empty((kappa.size, n_off))
        beta = np.zeros(self.model.N)
        dbeta_on = np.zeros(self.model.N)
        for k in range(kappa.size):
            for j, idx in enumerate(self.i_off):
                beta[idx] = B[k, j]
            for s in self.i_on:
                beta[s], dbeta_on[s] = bon[s][0][k], bon[s][1][k]
            # unit-speed guiding velocity of the last trailer
            d, _ = joint_rhs(self.model, beta, np.array([kappa[k], 1.0]), dbeta_on)
            res[k] = dB[k] - d[self.i_off]
        return res.reshape(-1)

    # ------------------------------------------------------------------ solve
    def fit(self, kappa, p, P=None):
        """Fit the joint reference to a curvature profile ``kappa(p)``.

        Returns an ``(n_coll, N)`` array of joint references.
        """
        p = np.asarray(p, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        P = float(p[-1] - p[0]) if P is None else float(P)
        Phi, dPhi = self._basis(p - p[0], P)
        dkappa = np.gradient(kappa, p, edge_order=2)

        n_off = self.i_off.size
        if n_off == 0:                                  # pure SNT: all algebraic
            return self._assemble(np.zeros((p.size, 0)), kappa, dkappa)

        w = np.zeros(n_off * Phi.shape[1])
        for stage in range(1, self.homotopy + 1):
            a = stage / self.homotopy
            sol = least_squares(self._residual, w, method="lm", max_nfev=4000,
                                args=(Phi, dPhi, a * kappa, a * dkappa))
            w = sol.x
            if self.verbose:
                print(f"    homotopy {stage}/{self.homotopy}  "
                      f"residual {np.sqrt(np.mean(sol.fun**2)):.3e}", flush=True)
        self.resid_rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        B = Phi @ w.reshape(n_off, -1).T
        return self._assemble(B, kappa, dkappa)

    def _assemble(self, B, kappa, dkappa):
        beta = np.zeros((kappa.size, self.model.N))
        for j, idx in enumerate(self.i_off):
            beta[:, idx] = B[:, j]
        bon = self._beta_on(kappa, dkappa)
        for s in self.i_on:
            beta[:, s] = bon[s][0]
        return beta

    # ------------------------------------------------------------- path front
    def reference_states(self, path: Path, n: int = 400) -> tuple:
        """Full state references along ``path``, with the last trailer on it.

        Returns ``(s, q_ref)`` with ``q_ref`` of shape ``(n, nq)``.
        """
        s = np.linspace(0.0, path.length, n)
        kappa = self._signed_curvature(path, s)
        beta = self.fit(kappa, s)

        th_last = np.empty(n)
        prev = 0.0
        for k in range(n):
            t = path.tangent(s[k])
            prev = atan2c(t[1], t[0], prev)
            th_last[k] = prev

        m = self.model
        Q = np.empty((n, m.nq))
        for k in range(n):
            thetas = np.empty(m.N + 1)
            thetas[m.N] = th_last[k]
            for i in range(m.N, 0, -1):
                thetas[i - 1] = thetas[i] + beta[k, i - 1]
            q = m.state_from_pose(0.0, 0.0, thetas)
            shift = path.at(s[k]) - m.positions(q)[m.N]
            Q[k] = m.state_from_pose(shift[0], shift[1], thetas)
        return s, Q

    @staticmethod
    def _signed_curvature(path: Path, s, smooth: float = 0.25) -> np.ndarray:
        """Signed curvature of the path, differenced over a physical length.

        Differencing the heading between adjacent samples of a dense path
        amplifies the sampling noise, and at a junction where the curvature is
        discontinuous it produces a spike many times the true value.  Using a
        fixed arc-length stencil of ``smooth`` metres instead reports the
        curvature the vehicle actually experiences over its own scale.
        """
        s = np.atleast_1d(np.asarray(s, dtype=float))
        h = max(smooth, 1e-3)

        def heading(sv, prev):
            t = path.tangent(sv)
            return atan2c(t[1], t[0], prev)

        prev = 0.0
        kap = np.empty(s.size)
        for k, sv in enumerate(s):
            a = heading(max(sv - h / 2, 0.0), prev)
            b = heading(min(sv + h / 2, path.length), a)
            prev = a
            span = min(sv + h / 2, path.length) - max(sv - h / 2, 0.0)
            kap[k] = wrap_to_pi(b - a) / max(span, 1e-9)
        return kap
