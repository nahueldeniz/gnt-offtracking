"""Stage 1 -- off-tracking-minimising reference generation.

At each sampling instant a small nonlinear programme chooses one complete vehicle
configuration ``q_ref`` -- headings, joint angles and the position of every
segment -- that

1. satisfies the rigid-body geometry of the chain **exactly**,
2. is reachable in one step from the previous reference under admissible inputs,
   which is what makes the generated reference kinematically feasible even when
   the nominal path is not, and
3. places every segment as close to the nominal path as 1 and 2 allow.

Only requirement 3 is relaxed, through the slacks ``w_path``.  Requirement 1 is a
hard equality, so the reference is always a physically realisable posture of the
vehicle; requirement 2 is relaxed only by ``w_prop``, whose weight controls how
hard the reference is allowed to jump when the path turns faster than the vehicle
can follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from .model import GNT

__all__ = ["RefGenWeights", "ReferenceGenerator"]


@dataclass
class RefGenWeights:
    """Weights of the reference-generation cost.

    ``w_path`` may be a scalar, a length ``N+1`` sequence (one weight per
    segment), or an ``(N+1, 2)`` array.  Weighting the last trailer far above the
    others is what drives its off-tracking down.
    """

    w_path: object = 5.0
    w_theta: object = 5.0
    w_prop: float = 50.0
    w_s1: float = 1.0
    w_guide: float = 0.0        # weight on the virtual guidance point
    guide_alpha: object = None  # its location: N+1 coefficients summing to one

    def guide_vector(self, N: int) -> np.ndarray:
        """Barycentric coefficients defining the guidance point.

        ``None`` puts the point on the last trailer, which is the usual default
        in the literature.  Any non-negative vector summing to one is admissible;
        the point it defines lies in the convex hull of the segment positions and
        need not coincide with any segment.
        """
        if self.guide_alpha is None:
            a = np.zeros(N + 1)
            a[-1] = 1.0
            return a
        a = np.asarray(self.guide_alpha, dtype=float).reshape(-1)
        if a.size != N + 1:
            raise ValueError("guide_alpha must have N+1 entries")
        if np.any(a < -1e-12):
            raise ValueError("guide_alpha must be non-negative")
        tot = a.sum()
        if tot <= 0:
            raise ValueError("guide_alpha must sum to a positive number")
        return a / tot

    def path_matrix(self, N: int) -> np.ndarray:
        w = np.asarray(self.w_path, dtype=float)
        if w.ndim == 0:
            return np.tile(w, (N + 1, 2))
        if w.ndim == 1:
            if w.size != N + 1:
                raise ValueError("w_path sequence must have N+1 entries")
            return np.repeat(w[:, None], 2, axis=1)
        if w.shape != (N + 1, 2):
            raise ValueError("w_path array must be (N+1, 2)")
        return w

    def theta_vector(self, N: int) -> np.ndarray:
        w = np.asarray(self.w_theta, dtype=float)
        if w.ndim == 0:
            return np.full(N + 1, float(w))
        if w.size != N + 1:
            raise ValueError("w_theta must have N+1 entries")
        return w.reshape(-1)


class ReferenceGenerator:
    """The stage-1 NLP, built once and re-solved with new parameters each step."""

    def __init__(
        self,
        model: GNT,
        poly_order: int = 7,
        u_lb=(-2.0, -2.0),
        u_ub=(2.0, 2.0),
        beta_max: float = np.deg2rad(80.0),
        weights: RefGenWeights | None = None,
        track_heading: bool = True,
        print_level: int = 0,
    ):
        self.m = model
        self.N = model.N
        self.order = poly_order
        self.beta_max = beta_max
        self.weights = weights or RefGenWeights()
        self.track_heading = track_heading

        N, nq, nu = self.N, model.nq, model.nu
        opti = ca.Opti()
        self.opti = opti

        # ---------------------------------------------------------- parameters
        self.p_ref_old = opti.parameter(nq)
        self.p_theta_path = opti.parameter(N + 1)
        self.p_cx = opti.parameter(poly_order + 1)
        self.p_cy = opti.parameter(poly_order + 1)
        self.p_s_target = opti.parameter(1)
        self.p_Wpath = opti.parameter(N + 1, 2)
        self.p_Wtheta = opti.parameter(N + 1)
        self.p_Wprop = opti.parameter(nq)
        self.p_Ws1 = opti.parameter(1)
        self.p_Wguide = opti.parameter(1)
        self.p_alpha = opti.parameter(N + 1)

        # ----------------------------------------------------------- variables
        self.v_q = opti.variable(nq)
        self.v_u = opti.variable(nu)
        self.v_s = opti.variable(N + 1)
        self.w_path = opti.variable(N + 1, 2)
        self.w_prop = opti.variable(nq)
        self.w_theta = opti.variable(N + 1)
        self.w_s1 = opti.variable(1)
        # The guidance-point term is optional and is built only when it is
        # weighted, so that a generator without it carries neither the two
        # extra variables nor the two extra constraints.  The problem sizes
        # reported in the paper are those of the generator as used everywhere
        # except the guidance sweep of Section IX-H.
        self.has_guide = float(self.weights.w_guide) > 0.0
        self.w_guide = opti.variable(2) if self.has_guide else None

        q, u, s = self.v_q, self.v_u, self.v_s
        beta = q[0:N]
        theta = q[N : 2 * N + 1]
        pos0 = q[2 * N + 1 : 2 * N + 3]

        J = 0

        # -------------------------------------------- progression along the path
        # The tractor's reference is pulled to a fixed station inside the moving
        # window; the window itself advances at the virtual speed sigma, so the
        # reference progresses even when the vehicle lags behind.
        opti.subject_to(s[0] - self.p_s_target == self.w_s1)
        opti.subject_to(s[N] >= 1e-3)
        for i in range(N):
            opti.subject_to(s[i + 1] <= s[i])
        J += self.p_Ws1 * self.w_s1**2

        # ------------------------------------------------ chain consistency
        for j in range(N):
            opti.subject_to(beta[j] - theta[j] + theta[j + 1] == 0)
            opti.subject_to(opti.bounded(-beta_max, beta[j], beta_max))

        # ------------------------------------------------ heading references
        if track_heading:
            for j in range(N + 1):
                opti.subject_to(theta[j] - self.p_theta_path[j] == self.w_theta[j])
                J += self.p_Wtheta[j] * self.w_theta[j] ** 2
        else:
            # Still keep the chain straight-ish by asking for zero joint angles,
            # which is the variant used when p_theta cannot be evaluated.
            for j in range(N):
                opti.subject_to(beta[j] == self.w_theta[j])
                J += self.p_Wtheta[j] * self.w_theta[j] ** 2
            opti.subject_to(self.w_theta[N] == 0)

        # --------------------------------- one-step reachability (feasibility)
        q_prop = self.m.F(self.p_ref_old, u)
        opti.subject_to(q - q_prop == self.w_prop)
        opti.subject_to(opti.bounded(ca.DM(u_lb), u, ca.DM(u_ub)))
        J += ca.dot(self.p_Wprop * self.w_prop, self.w_prop)

        # --------------------------------- exact geometry + distance to path
        opti.subject_to(pos0 - ca.vertcat(self._poly(s[0], self.p_cx), self._poly(s[0], self.p_cy))
                        == self.w_path[0, :].T)
        J += self.p_Wpath[0, 0] * self.w_path[0, 0] ** 2 + self.p_Wpath[0, 1] * self.w_path[0, 1] ** 2

        offsets = self.m.chain_offsets(theta)
        for k in range(N):
            pos_k = pos0 - offsets[k]
            seg = q[2 * N + 3 + 2 * k : 2 * N + 5 + 2 * k]
            opti.subject_to(seg - pos_k == 0)  # hard: rigid-body geometry
            opti.subject_to(seg - ca.vertcat(self._poly(s[k + 1], self.p_cx),
                                             self._poly(s[k + 1], self.p_cy))
                            == self.w_path[k + 1, :].T)
            J += (self.p_Wpath[k + 1, 0] * self.w_path[k + 1, 0] ** 2
                  + self.p_Wpath[k + 1, 1] * self.w_path[k + 1, 1] ** 2)

        # ------------------------------------------- virtual guidance point
        # A single point defined as a barycentric combination of the segment
        # positions, held to the path.  This is the objective of the cascaded
        # guidance-point literature, and it is genuinely weaker than referencing
        # every segment: one point cannot distinguish the postures that place it
        # identically, so whatever the combination leaves undetermined is free to
        # drift.  Including it costs one extra slack and lets the comparison be
        # made on the same generator rather than against a caricature.
        self.v_guide = None
        if self.has_guide:
            p_guide = ca.MX.zeros(2)
            s_guide = 0
            for k in range(N + 1):
                seg_k = pos0 if k == 0 else q[2 * N + 1 + 2 * k : 2 * N + 3 + 2 * k]
                p_guide = p_guide + self.p_alpha[k] * seg_k
                s_guide = s_guide + self.p_alpha[k] * s[k]
            # The point is held to the path at *its own* arc length, formed from
            # the same combination as the point itself.  Pinning it to some
            # externally chosen station instead would make the comparison depend
            # on a second tuning choice that the guidance-point formulation does
            # not have.
            opti.subject_to(p_guide - ca.vertcat(self._poly(s_guide, self.p_cx),
                                                 self._poly(s_guide, self.p_cy))
                            == self.w_guide)
            J += self.p_Wguide * ca.dot(self.w_guide, self.w_guide)
            self.v_guide = p_guide

        opti.minimize(J)
        self.J = J

        opti.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {"print_level": print_level, "sb": "yes", "max_iter": 500,
             "warm_start_init_point": "yes", "acceptable_tol": 1e-6},
        )

        self.set_weights(self.weights)
        self._last = None
        self.n_var = int(opti.nx)
        self.n_con = int(opti.ng)

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _poly(s, c):
        """Horner evaluation of a descending-power coefficient vector."""
        v = c[0]
        for k in range(1, c.shape[0]):
            v = v * s + c[k]
        return v

    # ----------------------------------------------------------------- setters
    def set_weights(self, w: RefGenWeights):
        self.weights = w
        self.opti.set_value(self.p_Wpath, ca.DM(w.path_matrix(self.N)))
        self.opti.set_value(self.p_Wtheta, ca.DM(w.theta_vector(self.N)))
        self.opti.set_value(self.p_Wprop, ca.DM(np.full(self.m.nq, float(w.w_prop))))
        self.opti.set_value(self.p_Ws1, w.w_s1)
        if float(w.w_guide) > 0.0 and not self.has_guide:
            raise ValueError(
                "this generator was built without the guidance-point term; pass "
                "weights with w_guide > 0 to the constructor to include it")
        self.opti.set_value(self.p_Wguide, float(w.w_guide))
        self.opti.set_value(self.p_alpha, ca.DM(w.guide_vector(self.N)))

    def set_reference_seed(self, q_ref: np.ndarray):
        self.opti.set_value(self.p_ref_old, ca.DM(np.asarray(q_ref).reshape(-1)))
        self._last = np.asarray(q_ref, dtype=float).reshape(-1)

    # ------------------------------------------------------------------- solve
    def solve(self, fit, s_target: float, theta_path: np.ndarray, s_guess=None):
        """Generate one reference posture.

        Parameters
        ----------
        fit : LocalFit
            Current polynomial window.
        s_target : float
            Local arc length the tractor's reference is pulled towards.
        theta_path : array, shape (N+1,)
            Path tangent angles sampled at the previous ``lambda_i``, continuous
            in the sense of :func:`~gntpf.model.atan2c`.
        """
        o = self.opti
        o.set_value(self.p_cx, ca.DM(fit.cx))
        o.set_value(self.p_cy, ca.DM(fit.cy))
        o.set_value(self.p_s_target, s_target)
        o.set_value(self.p_theta_path, ca.DM(np.asarray(theta_path).reshape(-1)))

        if self._last is not None:
            o.set_initial(self.v_q, ca.DM(self._last))
        if s_guess is not None:
            o.set_initial(self.v_s, ca.DM(np.asarray(s_guess).reshape(-1)))

        try:
            sol = o.solve()
            q = np.asarray(sol.value(self.v_q)).reshape(-1)
            s = np.asarray(sol.value(self.v_s)).reshape(-1)
            cost = float(sol.value(self.J))
            # e_r of the paper: how far the new reference is from a configuration
            # reachable in one sampling period from the previous one.
            self.last_e_r = np.asarray(sol.value(self.w_prop)).reshape(-1)
            ok = True
        except RuntimeError:
            # Keep the last feasible reference rather than aborting the run; the
            # closed loop degrades gracefully and the event is reported.
            q = o.debug.value(self.v_q)
            q = np.asarray(q).reshape(-1)
            s = np.asarray(o.debug.value(self.v_s)).reshape(-1)
            self.last_e_r = np.asarray(o.debug.value(self.w_prop)).reshape(-1)
            cost = float("nan")
            ok = False

        self._last = q
        o.set_value(self.p_ref_old, ca.DM(q))
        return q, s, cost, ok
