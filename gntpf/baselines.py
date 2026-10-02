"""Comparison controllers.

Two reference points for the proposed two-stage framework, both driven by the
*nominal path directly* rather than by a generated reference.  That is the
substantive difference under test: the baselines demand a path the vehicle may
not be able to realise, and whatever they cannot achieve appears as tracking
error; the proposed method decides in advance what is achievable and asks only
for that.

``PurePursuit``
    Geometric look-ahead steering applied to the tractor.  The trailers go
    wherever the kinematics take them.  This is the naive floor: it makes no
    reference to the chain at all, it cannot fail for want of an inverse, and it
    is what an implementer reaches for first.

``MichalekCascade``
    A port of the cascaded controller of Michałek and co-workers: an outer
    path-tracking law for the *last* trailer's posture, a disturbance observer
    that lumps the unmodelled terms, and an inner loop that recovers the tractor
    input by inverting the chain's velocity transmission.  This is the
    established method for off-tracking reduction on off-axle hitched vehicles.

    The inner loop is where the structural limitation lives.  Inverting the
    transmission requires every ``J_i`` to be invertible, and

        det J_i = -Lh_i / L_i ,

    so the cascade is defined only when every hitch is off-axle -- an nSNT.  It
    has no value at all on a Generalised N-Trailer with an on-axle joint, and
    its gain grows like ``1/Lh_i`` as a joint approaches one.  That is not a
    tuning difficulty; it is the reason a different formulation is needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .model import GNT, atan2c, wrap_to_pi
from .nmhe import NMHE
from .paths import Path
from .sim import SensorModel, SimConfig, SimResult, initial_state_on_path

__all__ = ["PurePursuit", "MichalekCascade", "simulate_baseline", "chain_condition",
           "scale_to_bounds", "steady_state_betas", "settled_state_on_path",
           "simulate_with_reference"]

U_LB = np.array([-2.0, -2.0])
U_UB = np.array([2.0, 2.0])


def scale_to_bounds(u, u_lb, u_ub):
    """Shrink ``u`` uniformly until it fits inside the box.

    This is Michałek's scaling procedure, and it is not interchangeable with
    clipping.  The pair ``(omega, v)`` encodes a commanded curvature through its
    ratio; clipping one component and not the other changes that ratio, so a
    saturated controller is asked to follow a different path from the one it
    computed.  Scaling both by a common factor preserves the curvature and only
    slows the vehicle down, which is the behaviour the cascade is designed
    around.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    if not np.all(np.isfinite(u)):
        return u
    lim = np.where(u >= 0.0, np.asarray(u_ub, float), np.abs(np.asarray(u_lb, float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lim > 0.0, np.abs(u) / lim, np.inf)
    s = float(max(1.0, np.max(ratio)))
    return u / s


def chain_condition(model: GNT) -> float:
    """Worst-case ``|L_i / Lh_i|`` over the chain -- the gain of the inner loop.

    Infinite when any joint is on-axle, which is exactly when a cascaded
    inversion controller ceases to exist.
    """
    lh = np.abs(np.asarray(model.Lh, dtype=float))
    if np.any(lh < 1e-12):
        return np.inf
    return float(np.max(np.abs(np.asarray(model.L, dtype=float)) / lh))


# --------------------------------------------------------------------------- #
#  Settled initial conditions
# --------------------------------------------------------------------------- #
def steady_state_betas(model: GNT, kappa: float, v: float = 1.0,
                       t_settle: float = 400.0) -> np.ndarray:
    """Equilibrium joint angles of the chain on an arc of curvature ``kappa``.

    Obtained by relaxing the chain under a constant input until the joint angles
    stop moving.  Starting a comparison from a posture that is *not* an
    equilibrium injects a transient whose size is set by the controller's gain,
    which on a cascaded controller is ``prod |L_i / Lh_i|`` and can be large
    enough to dominate everything the study is trying to measure.
    """
    if abs(kappa) < 1e-9:
        return np.zeros(model.N)
    m = GNT(N=model.N, Lh=model.Lh, L=model.L, Ts=0.002)
    q = m.state_from_pose(0.0, 0.0, np.zeros(model.N + 1))
    u = np.array([kappa * v, v])
    for _ in range(int(t_settle / m.Ts)):
        q = m.step(q, u)
    return wrap_to_pi(np.asarray(q)[m.i_beta])


def settled_state_on_path(model: GNT, path: Path, s0: float, v: float = 1.0,
                          anchor: int = 0) -> np.ndarray:
    """Segment ``anchor`` on the path at ``s0``, chain relaxed to the local curvature.

    The anchored segment is placed exactly on the path with the path heading and
    the trailers take their equilibrium joint angles, so the vehicle begins in a
    posture it would have reached anyway.

    Which segment is anchored matters, and it is not a cosmetic choice.  The
    chain cannot put two segments on the same curved path at once, so whichever
    segment is *not* anchored starts off the path by the geometric off-tracking.
    A controller that is trying to drive that segment onto the path therefore
    begins with an error it did not cause, and on a cascaded controller that
    error is amplified by ``prod |L_i / Lh_i|`` before it reaches the input.
    Anchoring the segment a method actually targets removes that bias, so each
    method is started in *its own* steady state.
    """
    if not 0 <= anchor <= model.N:
        raise IndexError(f"anchor {anchor} out of range for N={model.N}")
    ref = _PathRef(path)
    x, y, th_a, kap = ref.pose(s0)
    beta = steady_state_betas(model, kap, v=v)
    thetas = np.empty(model.N + 1)
    thetas[anchor] = th_a
    for k in range(anchor, 0, -1):            # forwards, towards the tractor
        thetas[k - 1] = thetas[k] + beta[k - 1]
    for k in range(anchor, model.N):          # backwards, down the chain
        thetas[k + 1] = thetas[k] - beta[k]
    # place the tractor so that the anchored segment lands on (x, y)
    q = model.state_from_pose(0.0, 0.0, thetas)
    shift = np.array([x, y]) - model.positions(q)[anchor]
    return model.state_from_pose(shift[0], shift[1], thetas)


# --------------------------------------------------------------------------- #
#  Path reference sampling
# --------------------------------------------------------------------------- #
class _PathRef:
    """Pose, tangent and signed curvature of a path at a given arc length.

    Headings are produced on a continuous branch so that the tracking error does
    not jump by ``2*pi`` between samples.
    """

    def __init__(self, path: Path, ds: float = 0.05):
        self.path, self.ds = path, ds
        self._prev = 0.0

    def seed(self, theta0: float):
        self._prev = float(theta0)

    def pose(self, s):
        """Return ``(x, y, theta, kappa)`` at arc length ``s``."""
        s = float(np.clip(s, 0.0, self.path.length))
        h = self.ds
        p = self.path.at(s)
        pm, pp = self.path.at(max(s - h, 0.0)), self.path.at(min(s + h, self.path.length))
        th = atan2c(pp[1] - pm[1], pp[0] - pm[0], self._prev)
        self._prev = th
        # signed curvature from a central difference of the heading
        prev = th
        thm = atan2c(p[1] - pm[1], p[0] - pm[0], prev)
        thp = atan2c(pp[1] - p[1], pp[0] - p[0], thm)
        kappa = wrap_to_pi(thp - thm) / max(h, 1e-9)
        return p[0], p[1], th, float(kappa)


# --------------------------------------------------------------------------- #
#  Baseline 1 -- tractor pure pursuit
# --------------------------------------------------------------------------- #
@dataclass
class PurePursuit:
    """Geometric look-ahead steering of the tractor.

    ``L_d`` is the look-ahead distance.  It is the only tuning knob and it trades
    corner-cutting against oscillation; the default scales with the vehicle so
    that the comparison is not decided by a badly chosen constant.
    """

    model: GNT
    speed: float = 1.0
    L_d: float | None = None
    u_lb: np.ndarray = None
    u_ub: np.ndarray = None

    def __post_init__(self):
        if self.L_d is None:
            self.L_d = max(1.0, 0.9 * self.model.total_length)
        self.u_lb = U_LB if self.u_lb is None else np.asarray(self.u_lb, float)
        self.u_ub = U_UB if self.u_ub is None else np.asarray(self.u_ub, float)
        self.name = "pure-pursuit"
        self.target_segment = 0

    def reset(self, q0, s0):
        pass

    def solve(self, q, path: Path, s_proj: float, ref: _PathRef, u_last):
        m = self.model
        p0 = q[m.i_pos(0)]
        th0 = float(q[m.N])
        # look-ahead point: fixed arc length ahead of the tractor's projection
        xt, yt, _, _ = ref.pose(s_proj + self.L_d)
        d = np.array([xt - p0[0], yt - p0[1]])
        ld = float(np.linalg.norm(d))
        if ld < 1e-6:
            return np.zeros(2), True
        alpha = wrap_to_pi(np.arctan2(d[1], d[0]) - th0)
        v = self.speed
        omega = 2.0 * v * np.sin(alpha) / ld          # kappa = 2 sin(alpha) / L_d
        u = scale_to_bounds(np.array([omega, v]), self.u_lb, self.u_ub)
        return u, True


# --------------------------------------------------------------------------- #
#  Baseline 2 -- Michałek's cascaded controller
# --------------------------------------------------------------------------- #
@dataclass
class MichalekCascade:
    """Cascaded last-trailer path follower with a disturbance observer.

    Port of ``robustnSNT_michalek.m``.  Structure:

    outer loop
        ``Phi(q_N, q_Nr, v_Nr, w_Nr)`` returns the velocity ``xi = [w_N, v_N]``
        the last trailer should have.  Two published forms are provided:
        ``law='M'`` (Morin and Samson) and ``law='S'`` (Canudas de Wit,
        Siciliano and Bastin).
    observer
        A linear observer on ``[q_N; F]`` with ``F`` a lumped perturbation,
        whose estimate is subtracted from ``Phi``.  Its contribution is ramped in
        over the first few seconds exactly as in the original.
    inner loop
        ``u_0 = (J_N ... J_1)^{-1} xi``, undefined when any ``Lh_i = 0``.

    Deviation from the MATLAB source: the original divides the final input by the
    sampling period and scales against differential-drive wheel speeds.  Here the
    input is left as a velocity and scaled against the same box bounds the NMPC is
    given, so that every controller in the comparison has identical actuator
    authority.  The *scaling* itself is kept exactly as in the original, for the
    reason given in :func:`scale_to_bounds`.
    """

    model: GNT
    law: str = "M"
    k1: float = 5.0
    k2: float = 2.0
    k3: float = 3.0
    k0: float = 10.0
    epsilon0_S: float = 1.0
    obs_gain: tuple = (20.0, 9.0)
    ramp_time: float = 5.0
    u_lb: np.ndarray = None
    u_ub: np.ndarray = None

    def __post_init__(self):
        if chain_condition(self.model) == np.inf:
            raise ValueError(
                "MichalekCascade requires Lh_i != 0 for every joint: the inner "
                "loop inverts J_i and det J_i = -Lh_i / L_i. This vehicle has an "
                "on-axle hitch, so the controller does not exist for it.")
        self.u_lb = U_LB if self.u_lb is None else np.asarray(self.u_lb, float)
        self.u_ub = U_UB if self.u_ub is None else np.asarray(self.u_ub, float)
        self.name = f"michalek-{self.law}"
        self.target_segment = self.model.N
        Ts = self.model.Ts
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])
        self.Ad = np.eye(6) + Ts * self.A
        self.C = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.K = np.vstack([self.obs_gain[0] * np.eye(3), self.obs_gain[1] * np.eye(3)])

    # ------------------------------------------------------------------ state
    def reset(self, q0, s0):
        m = self.model
        p = np.asarray(q0)[m.i_pos(m.N)]
        self.xhat = np.concatenate([[float(q0[m.N + m.N]), p[0], p[1]], np.zeros(3)])
        self.t = 0.0

    # ------------------------------------------------------------ outer loops
    def _phi(self, qN, qNr, vNr, wNr):
        eth = wrap_to_pi(qNr[0] - qN[0])
        ex, ey = qNr[1] - qN[1], qNr[2] - qN[2]
        if self.law == "M":
            c, s = np.cos(qNr[0]), np.sin(qNr[0])
            exh, eyh = ex * c + ey * s, -ex * s + ey * c
            # guard the 1/cos: the law is only valid for |e_theta| < pi/2
            eth = np.clip(eth, -0.45 * np.pi, 0.45 * np.pi)
            ce = np.cos(eth)
            w = (self.k2 * vNr * eyh + self.k3 * abs(vNr) * np.tan(eth)) * ce**2 + wNr
            v = (self.k1 * abs(vNr) * (exh - eyh * np.tan(eth)) + vNr) / ce
        else:
            c, s = np.cos(qN[0]), np.sin(qN[0])
            exh, eyh = ex * c + ey * s, -ex * s + ey * c
            k1u = 2.0 * self.epsilon0_S * np.sqrt(wNr**2 + self.k0 * vNr**2)
            sinc = 1.0 if abs(eth) < 1e-9 else np.sin(eth) / eth
            w = wNr + self.k0 * vNr * eyh * sinc + k1u * eth
            v = vNr * np.cos(eth) + k1u * exh
        return np.array([w, v])

    def _chain_inv(self, beta):
        """``(J_N ... J_1)^{-1}`` evaluated joint by joint."""
        M = np.eye(2)
        for i in range(self.model.N):
            Lh, L, b = self.model.Lh[i], self.model.L[i], beta[i]
            Ji = np.array([[-L * np.cos(b) / Lh, np.sin(b) / Lh],
                           [L * np.sin(b), np.cos(b)]])
            M = M @ Ji
        return M

    # ------------------------------------------------------------------ solve
    def solve(self, q, path: Path, s_proj: float, ref: _PathRef, u_last, s_ref=None, v_ref=1.0):
        m = self.model
        pN = np.asarray(q)[m.i_pos(m.N)]
        qN = np.array([float(q[m.N + m.N]), pN[0], pN[1]])
        xr, yr, thr, kap = ref.pose(s_ref if s_ref is not None else s_proj)
        qNr = np.array([thr, xr, yr])
        vNr, wNr = v_ref, kap * v_ref

        phi = self._phi(qN, qNr, vNr, wNr)
        # disturbance compensation, ramped in as in the original
        r = min(self.t / self.ramp_time, 1.0) if self.ramp_time > 0 else 1.0
        Fhat = self.xhat[3:]
        G = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(qN[0]), np.sin(qN[0])]])
        xi = phi - r * (G @ Fhat)

        u = self._chain_inv(np.asarray(q)[m.i_beta]) @ xi
        u = scale_to_bounds(u, self.u_lb, self.u_ub)

        # observer update, driven by the velocity that was actually commanded
        Ts = m.Ts
        B = Ts * np.array([[1.0, 0.0], [0.0, np.cos(qN[0])], [0.0, np.sin(qN[0])], ])
        B = np.vstack([B, np.zeros((3, 2))])
        self.xhat = self.Ad @ self.xhat + B @ xi + Ts * self.K @ (qN - self.C @ self.xhat)
        self.t += Ts
        ok = np.all(np.isfinite(u))
        return (u if ok else np.zeros(2)), bool(ok)


# --------------------------------------------------------------------------- #
#  Closed loop
# --------------------------------------------------------------------------- #
def simulate_baseline(
    model: GNT,
    path: Path,
    controller,
    cfg: SimConfig | None = None,
    sensors: SensorModel | None = None,
    estimator=None,
    s_start: float | None = None,
    q0: np.ndarray | None = None,
    use_estimate: bool = True,
    settled: bool = True,
) -> SimResult:
    """Run a baseline controller on the same plant, sensors and estimator.

    Everything outside the controller is identical to :func:`gntpf.sim.simulate`
    so that the comparison isolates the control law.  The recorded ``ref_dev`` is
    identically zero: these methods demand the nominal path itself, which is the
    property under examination.
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, Ts = model.N, model.Ts

    if sensors is None:
        sensors = SensorModel(
            sigma=np.concatenate([np.deg2rad(1.0) * np.ones(N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    # ``estimator=False`` skips state estimation entirely and feeds the true
    # state back.  Used only for tuning sweeps, where running an NLP per step
    # would dominate the cost and the estimator is not what is being varied.
    if estimator is False:
        estimator = None
        use_estimate = False
    elif estimator is None:
        estimator = NMHE(model)

    back = cfg.window_back_factor * model.total_length
    if s_start is None:
        s_start = back
    s_v = float(s_start)

    anchor = getattr(controller, "target_segment", 0)
    if q0 is None:
        q_true = (settled_state_on_path(model, path, s_v, v=cfg.sigma, anchor=anchor)
                  if settled else initial_state_on_path(model, path, s_v))
    else:
        q_true = np.asarray(q0, float).reshape(-1)
    if estimator is not None:
        estimator.initialise(q_true)

    ref = _PathRef(path)
    ref.seed(float(q_true[N]))
    controller.reset(q_true, s_v)

    # The last trailer's reference starts where the last trailer actually is, so
    # the run begins with zero tracking error rather than an offset that is an
    # artefact of how the vehicle was laid out.
    s_last = path.project(model.positions(q_true)[anchor], s_v, 3.0 * model.total_length)

    K = int(cfg.t_final / Ts)
    rec = {k: [] for k in ("q", "qhat", "qref", "u", "dev", "sdev", "refdev", "esterr",
                           "tref", "tmpc", "tmhe", "lam", "sproj")}
    fails = {"ref_fail": 0, "mpc_fail": 0, "mhe_fail": 0}
    u_last = np.zeros(model.nu)
    s_proj = np.full(N + 1, s_v)

    for k in range(K):
        if estimator is None:
            q_hat, t_mhe = q_true.copy(), 0.0
        else:
            y = sensors.corrupt(model.output(q_true), rng)
            estimator.push(y, u_last)
            t0 = time.perf_counter()
            q_hat, ok_e = estimator.solve()
            t_mhe = time.perf_counter() - t0
            fails["mhe_fail"] += (not ok_e)

        s_v = min(s_v + cfg.sigma * Ts, path.length - 1e-6)
        s_last = min(s_last + cfg.sigma * Ts, path.length - 1e-6)

        q_ctrl = q_hat if use_estimate else q_true
        t0 = time.perf_counter()
        try:
            if isinstance(controller, MichalekCascade):
                u, ok_m = controller.solve(q_ctrl, path, s_proj[N], ref, u_last,
                                           s_ref=s_last, v_ref=cfg.sigma)
            else:
                u, ok_m = controller.solve(q_ctrl, path, s_proj[0], ref, u_last)
        except (ValueError, np.linalg.LinAlgError):
            u, ok_m = np.zeros(model.nu), False
        t_mpc = time.perf_counter() - t0
        fails["mpc_fail"] += (not ok_m)

        u_app = np.asarray(u, float).reshape(-1) * np.asarray(cfg.slip, float)
        pos_true = model.positions(q_true)
        dev = np.array([path.deviation(pos_true[i], s_proj[i], 3.0 * model.total_length)
                        for i in range(N + 1)])
        sdev = np.array([path.signed_deviation(pos_true[i], s_proj[i], 3.0 * model.total_length)
                         for i in range(N + 1)])
        for i in range(N + 1):
            s_proj[i] = path.project(pos_true[i], s_proj[i], 3.0 * model.total_length)
        esterr = np.linalg.norm(model.positions(q_hat) - pos_true, axis=1)

        rec["q"].append(q_true.copy())
        rec["qhat"].append(np.asarray(q_hat).reshape(-1).copy())
        rec["qref"].append(np.full(model.nq, np.nan))
        rec["u"].append(u_app.copy())
        rec["dev"].append(dev)
        rec["sdev"].append(sdev)
        rec["refdev"].append(np.zeros(N + 1))
        rec["esterr"].append(esterr)
        rec["tref"].append(0.0)
        rec["tmpc"].append(t_mpc)
        rec["tmhe"].append(t_mhe)
        rec["lam"].append(np.zeros(N + 1))
        rec["sproj"].append(s_proj.copy())

        q_true = model.step(q_true, u_app)
        if cfg.process_noise:
            q_true = q_true + rng.uniform(-cfg.process_noise, cfg.process_noise, model.nq)
        q_true[model.i_beta] = wrap_to_pi(q_true[model.i_beta])
        u_last = np.asarray(u, float).reshape(-1)

        if not np.all(np.isfinite(q_true)):
            fails["mpc_fail"] += (K - k)
            break
        if cfg.stop_at_path_end and s_v >= path.length - 1e-3:
            break

    A = lambda key: np.array(rec[key]).T
    n = len(rec["q"])
    return SimResult(
        t=np.arange(n) * Ts,
        q=A("q"), qhat=A("qhat"), qref=A("qref"), u=A("u"),
        dev=A("dev"), sdev=A("sdev"), ref_dev=A("refdev"), est_err=A("esterr"),
        t_ref=np.array(rec["tref"]), t_mpc=np.array(rec["tmpc"]), t_mhe=np.array(rec["tmhe"]),
        lam=A("lam"), s_proj=A("sproj"),
        curved=path.curved_mask(np.array(rec["sproj"])[:, 0]),
        fails=fails,
        meta={"N": N, "path": path.name, "hitching": model.hitching,
              "controller": getattr(controller, "name", type(controller).__name__),
              "chain_condition": chain_condition(model),
              "n_var_ref": 0, "n_con_ref": 0, "n_var_mpc": 0, "n_con_mpc": 0,
              "n_var_mhe": getattr(estimator, "n_var", None) if estimator is not None else None,
              "Nc": 0, "Ts": Ts, "sigma": cfg.sigma},
    )


# --------------------------------------------------------------------------- #
#  Closed loop on a precomputed reference
# --------------------------------------------------------------------------- #
def simulate_with_reference(
    model: GNT,
    path: Path,
    s_ref: np.ndarray,
    Q_ref: np.ndarray,
    cfg: SimConfig | None = None,
    sensors: SensorModel | None = None,
    nmpc=None,
    estimator=None,
    s_start: float | None = None,
    label: str = "precomputed",
) -> SimResult:
    """Track a reference computed ahead of time, using the same tracking NMPC.

    This is the comparison that isolates the reference-generation stage: plant,
    sensors, estimator, controller, weights and bounds are all those of the
    proposed method, and the only thing replaced is where ``q_ref`` comes from.
    Any difference in the result is therefore attributable to the reference and
    to nothing else.

    ``Q_ref`` is ``(M, nq)`` sampled at arc lengths ``s_ref``.  Angles are
    interpolated on an unwrapped branch, since interpolating a wrapped angle
    across the +/- pi boundary produces a spurious full rotation.
    """
    from .nmpc import TrackingNMPC

    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, Ts = model.N, model.Ts

    if sensors is None:
        sensors = SensorModel(
            sigma=np.concatenate([np.deg2rad(1.0) * np.ones(N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    if nmpc is None:
        nmpc = TrackingNMPC(model)
    if estimator is False:
        estimator = None
    elif estimator is None:
        estimator = NMHE(model)

    s_ref = np.asarray(s_ref, float).reshape(-1)
    Q_ref = np.asarray(Q_ref, float)
    ang = list(range(0, 2 * N + 1))                 # betas and headings
    Qi = Q_ref.copy()
    for j in ang:
        Qi[:, j] = np.unwrap(Qi[:, j])

    def ref_at(s):
        return np.array([np.interp(s, s_ref, Qi[:, j]) for j in range(model.nq)])

    back = cfg.window_back_factor * model.total_length
    if s_start is None:
        s_start = back
    s_v = float(s_start)
    q_true = ref_at(s_v).copy()
    if estimator is not None:
        estimator.initialise(q_true)

    K = int(cfg.t_final / Ts)
    rec = {k: [] for k in ("q", "qhat", "qref", "u", "dev", "sdev", "refdev", "esterr",
                           "tref", "tmpc", "tmhe", "lam", "sproj")}
    fails = {"ref_fail": 0, "mpc_fail": 0, "mhe_fail": 0}
    u_last = np.zeros(model.nu)
    s_proj = np.full(N + 1, s_v)

    for k in range(K):
        if estimator is None:
            q_hat, t_mhe = q_true.copy(), 0.0
        else:
            y = sensors.corrupt(model.output(q_true), rng)
            estimator.push(y, u_last)
            t0 = time.perf_counter()
            q_hat, ok_e = estimator.solve()
            t_mhe = time.perf_counter() - t0
            fails["mhe_fail"] += (not ok_e)

        s_v = min(s_v + cfg.sigma * Ts, path.length - 1e-6)
        q_ref = ref_at(s_v)

        t0 = time.perf_counter()
        u, _, _, ok_m = nmpc.solve(q_hat, q_ref, u_last)
        t_mpc = time.perf_counter() - t0
        fails["mpc_fail"] += (not ok_m)

        u_app = np.asarray(u, float).reshape(-1) * np.asarray(cfg.slip, float)
        pos_true = model.positions(q_true)
        pos_ref = model.positions(q_ref)
        w = 3.0 * model.total_length
        dev = np.array([path.deviation(pos_true[i], s_proj[i], w) for i in range(N + 1)])
        sdev = np.array([path.signed_deviation(pos_true[i], s_proj[i], w) for i in range(N + 1)])
        for i in range(N + 1):
            s_proj[i] = path.project(pos_true[i], s_proj[i], w)
        refdev = np.array([path.deviation(pos_ref[i], s_proj[i], w) for i in range(N + 1)])

        rec["q"].append(q_true.copy())
        rec["qhat"].append(np.asarray(q_hat).reshape(-1).copy())
        rec["qref"].append(q_ref.copy())
        rec["u"].append(u_app.copy())
        rec["dev"].append(dev)
        rec["sdev"].append(sdev)
        rec["refdev"].append(refdev)
        rec["esterr"].append(np.linalg.norm(model.positions(q_hat) - pos_true, axis=1))
        rec["tref"].append(0.0)
        rec["tmpc"].append(t_mpc)
        rec["tmhe"].append(t_mhe)
        rec["lam"].append(np.zeros(N + 1))
        rec["sproj"].append(s_proj.copy())

        q_true = model.step(q_true, u_app)
        q_true[model.i_beta] = wrap_to_pi(q_true[model.i_beta])
        u_last = np.asarray(u, float).reshape(-1)
        if not np.all(np.isfinite(q_true)):
            fails["mpc_fail"] += (K - k)
            break
        if cfg.stop_at_path_end and s_v >= path.length - 1e-3:
            break

    A = lambda key: np.array(rec[key]).T
    n = len(rec["q"])
    return SimResult(
        t=np.arange(n) * Ts,
        q=A("q"), qhat=A("qhat"), qref=A("qref"), u=A("u"),
        dev=A("dev"), sdev=A("sdev"), ref_dev=A("refdev"), est_err=A("esterr"),
        t_ref=np.array(rec["tref"]), t_mpc=np.array(rec["tmpc"]), t_mhe=np.array(rec["tmhe"]),
        lam=A("lam"), s_proj=A("sproj"),
        curved=path.curved_mask(np.array(rec["sproj"])[:, 0]),
        fails=fails,
        meta={"N": N, "path": path.name, "hitching": model.hitching, "controller": label,
              "n_var_ref": 0, "n_con_ref": 0,
              "n_var_mpc": nmpc.n_var, "n_con_mpc": nmpc.n_con,
              "n_var_mhe": getattr(estimator, "n_var", None) if estimator is not None else None,
              "Nc": nmpc.Nc, "Ts": Ts, "sigma": cfg.sigma},
    )
