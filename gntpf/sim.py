"""Closed-loop simulation driver.

Puts the plant, the sensors, the estimator and the two-stage controller together
and records everything the paper needs to report: per-segment off-tracking,
control inputs against their bounds, estimation errors, and per-solve timings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import GNT, atan2c, wrap_to_pi
from .nmhe import NMHE
from .nmpc import TrackingNMPC
from .paths import Path
from .refgen import ReferenceGenerator, RefGenWeights

__all__ = ["SensorModel", "SimConfig", "SimResult", "simulate", "initial_state_on_path"]


# --------------------------------------------------------------------------- #
#  Sensors
# --------------------------------------------------------------------------- #
@dataclass
class SensorModel:
    """Measurement corruption applied to ``y = h(q)``.

    ``sigma`` has ``N + 3`` entries: one per joint angle, then the tractor
    heading, then x and y.  ``kind`` selects a uniform or a normal distribution;
    the paper's default is uniform, since the MHE assumes bounded noise rather
    than a particular density.
    """

    sigma: np.ndarray
    kind: str = "uniform"
    outlier_prob: float = 0.0
    outlier_gain: float = 10.0
    outlier_channels: tuple | None = None   # default: the joint angles

    def corrupt(self, y, rng) -> np.ndarray:
        y = np.asarray(y, dtype=float).reshape(-1)
        s = np.asarray(self.sigma, dtype=float).reshape(-1)
        if self.kind == "uniform":
            n = rng.uniform(-s, s)
        elif self.kind == "normal":
            n = rng.normal(0.0, s)
        else:
            raise ValueError(f"unknown noise kind {self.kind!r}")
        out = y + n
        if self.outlier_prob > 0.0:
            ch = self.outlier_channels
            if ch is None:
                ch = tuple(range(y.size - 3))
            for c in ch:
                if rng.random() < self.outlier_prob:
                    out[c] += self.outlier_gain * s[c] * rng.choice([-1.0, 1.0])
        return out


# --------------------------------------------------------------------------- #
#  Configuration and results
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    sigma: float = 1.3                 # virtual progression speed of the path parameter (m/s)
    t_final: float = 60.0
    window_back_factor: float = 1.6    # window extends this * total_length behind
    window_fwd_factor: float = 1.2     # ... and this * total_length ahead
    poly_order: int = 7
    process_noise: float = 0.0         # amplitude of an additive disturbance on q
    slip: tuple = (1.0, 1.0)           # multiplicative actuator gain (1 = none)
    seed: int = 0
    stop_at_path_end: bool = True
    detach_at: float | None = None     # time (s) at which a trailer is removed (S7)


@dataclass
class SimResult:
    t: np.ndarray
    q: np.ndarray                # true states, nq x K
    qhat: np.ndarray             # estimates
    qref: np.ndarray             # generated references
    u: np.ndarray                # applied inputs, 2 x K
    dev: np.ndarray              # per-segment off-tracking, (N+1) x K
    ref_dev: np.ndarray          # deviation of the generated reference from the path
    est_err: np.ndarray          # per-segment position estimation error
    t_ref: np.ndarray            # stage-1 solve times
    t_mpc: np.ndarray            # stage-2 solve times
    t_mhe: np.ndarray            # estimator solve times
    lam: np.ndarray              # path parameters, (N+1) x K
    s_proj: np.ndarray = None    # arc length of each segment's projection
    sdev: np.ndarray = None      # signed lateral offset, positive to the left
    curved: np.ndarray = None    # boolean mask: tractor in a curved section
    reach: np.ndarray = None     # reachability residual e_r of the generator, nq x K
    seg_mask: np.ndarray = None  # (N+1) x K: segment i inside the evaluation span
    seg_curved: np.ndarray = None  # (N+1) x K: segment i on a curved section of the path
    fails: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- masks
    def masks(self, skip: float = 0.15):
        """Which samples of each segment are evaluated, and which are curved.

        A path built with a lead-in carries an evaluation span; a sample of
        segment ``i`` is then evaluated when that segment's own projection lies
        inside the span, so every segment, the last trailer included, is judged
        over the same stretch of path.  Without a span the first ``skip`` of the
        run is discarded as transient and curvature is taken at the tractor.
        """
        n_seg, K = self.dev.shape
        span = self.meta.get("eval_span") if self.meta else None
        if self.seg_mask is not None and span is not None and span[0] > 0:
            M = self.seg_mask.copy()
            C = M & self.seg_curved
        else:
            k0 = int(skip * K)
            M = np.zeros((n_seg, K), bool)
            M[:, k0:] = True
            c = self.curved if self.curved is not None else np.zeros(K, bool)
            C = M & c[None, :]
        return M, C

    # --------------------------------------------------------------- corridor
    def corridor(self, skip: float = 0.15) -> dict:
        """Width of the lane the whole vehicle sweeps.

        The span between the left-most and right-most excursion of any segment,
        over the evaluated samples, overall and restricted to curved sections.
        """
        if self.sdev is None:
            return {}
        M, C = self.masks(skip)
        d = self.sdev
        dm = d[M] if M.any() else d.ravel()
        out = {"corridor_width": float(dm.max() - dm.min()),
               "corridor_left": float(dm.max()), "corridor_right": float(dm.min())}
        dc = d[C] if C.any() else dm
        out["corridor_width_curved"] = float(dc.max() - dc.min())
        return out

    # ---------------------------------------------------------------- summary
    def reach_summary(self, skip: float = 0.15) -> dict:
        """Size of the reachability residual e_r of Eq. (11) over the evaluated run.

        At each instant the residual is split into its position part, the
        largest Euclidean residual over the segment positions (m), and its angle
        part, the largest residual over the joint angles and headings (deg).
        Each is summarised overall, where some segment is on a curved section,
        and where none is.
        """
        if self.reach is None:
            return {}
        N = self.meta["N"]
        M, C = self.masks(skip)
        k_eval = M[0]
        curved = C.any(axis=0)
        R = self.reach
        pos = np.stack([np.hypot(R[2 * N + 1 + 2 * i], R[2 * N + 2 + 2 * i])
                        for i in range(N + 1)]).max(axis=0)
        ang = np.degrees(np.abs(R[:2 * N + 1]).max(axis=0))

        def stats(x):
            x = x[np.isfinite(x)]
            if x.size == 0:
                return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
            return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)),
                    "max": float(x.max())}

        a, c, st = k_eval, k_eval & curved, k_eval & ~curved
        return {"pos_m": {"all": stats(pos[a]), "curved": stats(pos[c]), "straight": stats(pos[st])},
                "ang_deg": {"all": stats(ang[a]), "curved": stats(ang[c]), "straight": stats(ang[st])}}

    def summary(self, skip: float = 0.15) -> dict:
        """Aggregate metrics over the evaluated samples of each segment."""
        M, C = self.masks(skip)
        n_seg = self.dev.shape[0]

        def per_seg(x, mask, fn):
            return np.array([fn(x[i, mask[i]]) if mask[i].any() else np.nan
                             for i in range(n_seg)])

        mean_dev = per_seg(self.dev, M, np.mean)
        max_dev = per_seg(self.dev, M, np.max)
        rms_dev = per_seg(self.dev, M, lambda v: np.sqrt(np.mean(v ** 2)))
        if C.any():
            mean_c = per_seg(self.dev, C, np.mean)
            max_c = per_seg(self.dev, C, np.max)
            frac = float(C[0].sum() / max(M[0].sum(), 1))
        else:
            mean_c, max_c, frac = mean_dev, max_dev, 0.0
        k = M[0]
        tt = self.t_ref + self.t_mpc + self.t_mhe
        est = self.est_err[M] if M.any() else self.est_err.ravel()
        return {
            "mean_dev": mean_dev,
            "max_dev": max_dev,
            "rms_dev": rms_dev,
            "mean_dev_all": float(self.dev[M].mean()) if M.any() else float("nan"),
            "mean_dev_last": float(mean_dev[-1]),
            "max_dev_last": float(max_dev[-1]),
            "worst_segment": float(np.nanmax(mean_dev)),
            "worst_segment_max": float(np.nanmax(max_dev)),
            "mean_dev_curved": mean_c,
            "max_dev_curved": max_c,
            "worst_curved": float(np.nanmax(mean_c)),
            "curved_fraction": frac,
            "t_ref_ms": float(np.nanmean(self.t_ref[k]) * 1e3),
            "t_mpc_ms": float(np.nanmean(self.t_mpc[k]) * 1e3),
            "t_mhe_ms": float(np.nanmean(self.t_mhe[k]) * 1e3),
            "t_total_ms": float(np.nanmean(tt[k]) * 1e3),
            "t_total_p95_ms": float(np.nanpercentile(tt[k], 95) * 1e3),
            "est_err_mean": float(np.nanmean(est)),
            "est_err_max": float(np.nanmax(est)),
            "completed": bool(not self.meta.get("diverged", False)),
            "progress": float(self.meta.get("progress", 1.0)),
            **self.corridor(skip),
            **self.fails,
        }


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
# A run is declared diverged when the tractor falls more than this many vehicle
# lengths behind the progress point and stays there for this long: the vehicle
# has lost its reference, and whatever deviation it accumulates afterwards is not
# a path-following result.
DIVERGE_LAG_LENGTHS = 2.0
DIVERGE_HOLD_S = 5.0


class Watchdog:
    """Detects a run in which the vehicle has lost its reference."""

    def __init__(self, model: GNT, Ts: float):
        self.limit = DIVERGE_LAG_LENGTHS * model.total_length
        self.hold = max(int(round(DIVERGE_HOLD_S / Ts)), 1)
        self.count = 0
        self.t_diverged = None

    def update(self, k: int, Ts: float, s_v: float, s_tractor: float) -> bool:
        self.count = self.count + 1 if (s_v - s_tractor) > self.limit else 0
        if self.count >= self.hold and self.t_diverged is None:
            self.t_diverged = (k - self.hold + 1) * Ts
        return self.t_diverged is not None


def evaluation_masks(path: Path, s_proj, k_threshold: float = 0.15):
    """Per-segment masks: inside the evaluation span, and on a curved section."""
    S = np.asarray(s_proj, dtype=float)
    mask = (S >= path.eval_start) & (S <= path.eval_end)
    curved = path.curved_mask(S.ravel(), k_threshold).reshape(S.shape)
    return mask, curved


def run_meta(path: Path, s_proj, watchdog: Watchdog) -> dict:
    """Evaluation span, divergence, and how much of the span the last trailer covered."""
    S = np.asarray(s_proj, dtype=float)
    span = path.eval_end - path.eval_start
    reached = float(np.nanmax(S[-1])) if S.size else path.eval_start
    prog = float(np.clip((reached - path.eval_start) / span, 0.0, 1.0)) if span > 0 else 1.0
    return {"eval_span": (path.eval_start, path.eval_end),
            "diverged": watchdog.t_diverged is not None,
            "t_diverged": watchdog.t_diverged, "progress": prog}


def initial_state_on_path(model: GNT, path: Path, s0: float, jitter=0.0, rng=None) -> np.ndarray:
    """A consistent initial configuration with the vehicle laid out along the path.

    Each segment is placed at the arc length its geometry implies, so the vehicle
    starts folded onto the path rather than in an arbitrary posture.
    """
    s = s0
    thetas = []
    stations = [s0]
    for k in range(model.N):
        s = s - (abs(model.Lh[k]) + abs(model.L[k]))
        stations.append(s)
    for st in stations:
        p1 = path.at(min(st + 0.05, path.length))
        p0 = path.at(max(st - 0.05, 0.0))
        prev = thetas[-1] if thetas else 0.0
        thetas.append(atan2c(p1[1] - p0[1], p1[0] - p0[0], prev))
    thetas = np.array(thetas)
    if jitter and rng is not None:
        thetas = thetas + rng.uniform(-jitter, jitter, thetas.size)
    p = path.at(s0)
    return model.state_from_pose(p[0], p[1], thetas)


def _sample_theta_path(fit, lam, prev) -> np.ndarray:
    out = np.empty(lam.size)
    p = prev
    for i, s in enumerate(lam):
        p = fit.heading(np.clip(s, 0.0, fit.s_len), p)
        out[i] = p
    return out


# --------------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------------- #
def simulate(
    model: GNT,
    path: Path,
    cfg: SimConfig | None = None,
    sensors: SensorModel | None = None,
    refgen: ReferenceGenerator | None = None,
    nmpc: TrackingNMPC | None = None,
    estimator=None,
    s_start: float | None = None,
    q0: np.ndarray | None = None,
    track_heading: bool = True,
    refgen_weights: RefGenWeights | None = None,
    progress: bool = False,
    lam_init: str = "staggered",
) -> SimResult:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, Ts = model.N, model.Ts

    if sensors is None:
        sensors = SensorModel(
            sigma=np.concatenate([np.deg2rad(1.0) * np.ones(N), [np.deg2rad(0.2)], [0.025, 0.025]])
        )
    if refgen is None:
        refgen = ReferenceGenerator(model, poly_order=cfg.poly_order,
                                    weights=refgen_weights or RefGenWeights(),
                                    track_heading=track_heading)
    elif refgen_weights is not None:
        refgen.set_weights(refgen_weights)
    if nmpc is None:
        nmpc = TrackingNMPC(model)
    if estimator is None:
        estimator = NMHE(model)

    back = cfg.window_back_factor * model.total_length
    fwd = cfg.window_fwd_factor * model.total_length

    if s_start is None:
        s_start = back
    s_v = float(s_start)

    q_true = initial_state_on_path(model, path, s_v) if q0 is None else np.asarray(q0, float).reshape(-1)
    estimator.initialise(q_true)
    refgen.set_reference_seed(q_true)

    fit = path.local_fit(s_v, back, fwd, cfg.poly_order)
    if lam_init == "equal":
        # every path parameter starts at the progress point; the ordering
        # constraint admits equality, and the generator separates them
        lam = np.full(N + 1, s_v - fit.s0_global)
    else:
        lam = np.array([s_v - fit.s0_global - sum(abs(model.Lh[:k]) + abs(model.L[:k])) if k
                        else s_v - fit.s0_global for k in range(N + 1)])
    lam = np.maximum(lam, 1e-3)
    theta_prev = float(q_true[model.N])

    K = int(cfg.t_final / Ts)
    rec = {k: [] for k in ("q", "qhat", "qref", "u", "dev", "sdev", "refdev", "esterr",
                           "tref", "tmpc", "tmhe", "lam", "sproj", "reach")}
    fails = {"ref_fail": 0, "mpc_fail": 0, "mhe_fail": 0}
    u_last = np.zeros(model.nu)
    s_proj = np.full(N + 1, s_v)
    watchdog = Watchdog(model, Ts)

    for k in range(K):
        # ---------------------------------------------------------- sensing
        y = sensors.corrupt(model.output(q_true), rng)
        estimator.push(y, u_last)
        t0 = time.perf_counter()
        q_hat, ok_e = estimator.solve()
        t_mhe = time.perf_counter() - t0
        fails["mhe_fail"] += (not ok_e)

        # ------------------------------------------- virtual target advances
        s_v = min(s_v + cfg.sigma * Ts, path.length - 1e-6)
        new_fit = path.local_fit(s_v, back, fwd, cfg.poly_order)
        # remap path parameters into the new window's coordinate
        lam = lam + (fit.s0_global - new_fit.s0_global)
        lam = np.clip(lam, 1e-3, new_fit.s_len)
        fit = new_fit
        s_target = float(np.clip(s_v - fit.s0_global, 1e-3, fit.s_len))

        theta_path = _sample_theta_path(fit, lam, theta_prev)
        theta_prev = float(theta_path[0])

        # ------------------------------------------------ stage 1: reference
        t0 = time.perf_counter()
        q_ref, lam, j_ref, ok_r = refgen.solve(fit, s_target, theta_path, s_guess=lam)
        t_ref = time.perf_counter() - t0
        fails["ref_fail"] += (not ok_r)

        # ------------------------------------------------ stage 2: tracking
        t0 = time.perf_counter()
        u, _, _, ok_m = nmpc.solve(q_hat, q_ref, u_last)
        t_mpc = time.perf_counter() - t0
        fails["mpc_fail"] += (not ok_m)

        # ------------------------------------------------------- apply & log
        u_app = np.asarray(u, float).reshape(-1) * np.asarray(cfg.slip, float)
        pos_true = model.positions(q_true)
        pos_ref = model.positions(q_ref)
        dev = np.array([path.deviation(pos_true[i], s_proj[i], 3.0 * model.total_length)
                        for i in range(N + 1)])
        sdev = np.array([path.signed_deviation(pos_true[i], s_proj[i], 3.0 * model.total_length)
                         for i in range(N + 1)])
        for i in range(N + 1):
            s_proj[i] = path.project(pos_true[i], s_proj[i], 3.0 * model.total_length)
        refdev = np.array([path.deviation(pos_ref[i], s_proj[i], 3.0 * model.total_length)
                           for i in range(N + 1)])
        esterr = np.linalg.norm(model.positions(q_hat) - pos_true, axis=1)

        rec["q"].append(q_true.copy())
        rec["qhat"].append(np.asarray(q_hat).reshape(-1).copy())
        rec["qref"].append(np.asarray(q_ref).reshape(-1).copy())
        rec["u"].append(u_app.copy())
        rec["dev"].append(dev)
        rec["sdev"].append(sdev)
        rec["refdev"].append(refdev)
        rec["esterr"].append(esterr)
        rec["tref"].append(t_ref)
        rec["tmpc"].append(t_mpc)
        rec["tmhe"].append(t_mhe)
        rec["lam"].append(lam.copy())
        rec["sproj"].append(s_proj.copy())
        if watchdog.update(k, Ts, s_v, s_proj[0]):
            e_r = getattr(refgen, "last_e_r", None)
            rec["reach"].append(np.full(model.nq, np.nan) if e_r is None else e_r.copy())
            break
        e_r = getattr(refgen, "last_e_r", None)
        rec["reach"].append(np.full(model.nq, np.nan) if e_r is None else e_r.copy())

        q_true = model.step(q_true, u_app)
        if cfg.process_noise:
            q_true = q_true + rng.uniform(-cfg.process_noise, cfg.process_noise, model.nq)
        q_true[model.i_beta] = wrap_to_pi(q_true[model.i_beta])
        u_last = np.asarray(u, float).reshape(-1)

        if progress and k % 100 == 0:
            print(f"  k={k:5d}/{K}  s_v={s_v:7.2f}  dev_last={dev[-1]:.3f}", flush=True)
        if cfg.stop_at_path_end and s_v >= path.length - 1e-3:
            break

    A = lambda key, ax=1: np.array(rec[key]).T if ax else np.array(rec[key])
    n = len(rec["q"])
    return SimResult(
        t=np.arange(n) * Ts,
        q=A("q"), qhat=A("qhat"), qref=A("qref"), u=A("u"),
        dev=A("dev"), sdev=A("sdev"), ref_dev=A("refdev"), est_err=A("esterr"),
        t_ref=np.array(rec["tref"]), t_mpc=np.array(rec["tmpc"]), t_mhe=np.array(rec["tmhe"]),
        lam=A("lam"), s_proj=A("sproj"),
        curved=path.curved_mask(np.array(rec["sproj"])[:, 0]),
        reach=A("reach"),
        seg_mask=evaluation_masks(path, A("sproj"))[0],
        seg_curved=evaluation_masks(path, A("sproj"))[1],
        fails=fails,
        meta={**run_meta(path, A("sproj"), watchdog),
              "N": N, "path": path.name, "hitching": model.hitching,
              "n_var_ref": refgen.n_var, "n_con_ref": refgen.n_con,
              "n_var_mpc": nmpc.n_var, "n_con_mpc": nmpc.n_con,
              "n_var_mhe": getattr(estimator, "n_var", None),
              "Nc": nmpc.Nc, "Ts": Ts, "sigma": cfg.sigma},
    )
