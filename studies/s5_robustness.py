"""S5 -- what the framework is and is not sensitive to.

The formulation assumes the kinematic parameters are known exactly. On a real
implement they are not: hitch offsets are measured with a tape, drawbars flex and
tyres slip. This study varies the three things a practitioner would worry about.

The result is not the one we expected, and it is reported as found. Off-tracking
is remarkably insensitive to all three: the tractor's position is measured
directly and the trailers follow it by rigid-body geometry, so the vehicle goes
very nearly where it would have gone anyway. What *is* sensitive is the
**estimate**. A wrong chain geometry puts the estimated trailer positions in the
wrong place while the real ones stay on the path, and the error grows roughly in
proportion to the parameter error.

That distinction matters operationally. If the estimate is used only to close the
loop, parameter error is close to harmless. If it is used to decide whether the
vehicle clears an obstacle -- which is the natural next use -- it is the dominant
error source, and the geometry must be calibrated rather than tape-measured.

Both quantities are therefore recorded for every sweep point, and the figure
plots them side by side.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import C_GREY, C_LAST, C_MID, C_TRACTOR, G2T, dump, save

from gntpf import (NMHE, GNT, ReferenceGenerator, RefGenWeights, SensorModel,
                   SimConfig, TrackingNMPC, make_path, simulate)
from parallel import pmap

NOISE_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
PARAM_ERRORS = [0.0, 0.05, 0.10, 0.15, 0.30]
SLIPS = [1.0, 0.95, 0.90, 0.85]
SEEDS = (0, 1, 2, 3)
W = RefGenWeights(w_path=20.0, w_theta=5.0)
PATH = "agricultural"
T_FINAL, SIGMA = 160.0, 1.0


def sensors_for(model, scale=1.0):
    return SensorModel(sigma=scale * np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))


def metrics(res):
    """Both quantities of interest: where the vehicle went, and where we thought
    it went."""
    s = res.summary()
    return s["worst_curved"], s["est_err_mean"], s["est_err_max"]


def perturbed_model(e, rng):
    """A controller/estimator model whose geometry is wrong by up to +/-e.

    The hitch offsets are perturbed by a fraction of the *nominal drawbar scale*
    rather than of their own value, since one of them is zero and a relative
    perturbation would leave it zero.
    """
    if e == 0.0:
        return GNT(N=2, Lh=G2T["Lh"], L=G2T["L"])
    scale = 0.342
    return GNT(N=2,
               Lh=np.asarray(G2T["Lh"]) + rng.uniform(-e, e, 2) * scale,
               L=np.asarray(G2T["L"]) * (1 + rng.uniform(-e, e, 2)))


def _run_one(plant, control_model, path, seed, sensors, slip=(1.0, 1.0)):
    same = control_model is plant
    kw = {}
    if not same:
        kw = dict(nmpc=TrackingNMPC(control_model, Nc=20),
                  estimator=NMHE(control_model),
                  refgen=ReferenceGenerator(control_model, weights=W))
    return simulate(plant, path, SimConfig(sigma=SIGMA, t_final=T_FINAL, seed=seed, slip=slip),
                    sensors=sensors, refgen_weights=W, **kw)


def _one(kind, value, seed, path_name):
    """One perturbed run.  Everything is rebuilt here so the job is picklable."""
    rng = np.random.default_rng(1000 * seed + int(1e4 * value))
    plant = GNT(N=2, Lh=G2T["Lh"], L=G2T["L"])
    cm = perturbed_model(value, rng) if kind == "hitch" else plant
    sen = sensors_for(plant, value if kind == "noise" else 1.0)
    sl = (value, value) if kind == "slip" else (1.0, 1.0)
    return metrics(_run_one(plant, cm, make_path(path_name), seed, sen, sl))


def sweep(kind, path, path_name=PATH):
    values = {"noise": NOISE_SCALES, "hitch": PARAM_ERRORS, "slip": SLIPS}[kind]
    jobs = [(kind, float(v), sd, path_name) for v in values for sd in SEEDS]
    done = pmap(_one, jobs)
    rows = []
    for i, v in enumerate(values):
        chunk = done[i * len(SEEDS):(i + 1) * len(SEEDS)]
        off = [c[0] for c in chunk]; est = [c[1] for c in chunk]; estmax = [c[2] for c in chunk]
        rows.append(dict(value=float(v),
                         off=float(np.mean(off)), off_sd=float(np.std(off)),
                         est=float(np.mean(est)), est_sd=float(np.std(est)),
                         est_max=float(np.max(estmax))))
        print(f"  {kind:6s} {v:<6} off-tracking {rows[-1]['off']:.3f} m   "
              f"estimation {rows[-1]['est']:.4f} m (max {rows[-1]['est_max']:.3f})", flush=True)
    return rows


def figure(data):
    labels = {"noise": "measurement noise\n(x nominal)",
              "hitch": "kinematic parameter error",
              "slip": "actuator gain\n(1 = no slip)"}
    cols = {"noise": C_MID, "hitch": C_TRACTOR, "slip": C_LAST}

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex="col")
    fig.subplots_adjust(hspace=0.22, wspace=0.34)
    for j, kind in enumerate(("noise", "hitch", "slip")):
        rows = data[kind]
        x = [r["value"] for r in rows]
        for i, (key, sd, lab) in enumerate(
                ((("off"), "off_sd", "off-tracking (m)"),
                 (("est"), "est_sd", "estimation error (m)"))):
            ax = axes[i, j]
            ax.errorbar(x, [r[key] for r in rows], yerr=[r[sd] for r in rows],
                        fmt="o-", ms=3.4, color=cols[kind], capsize=3, lw=1.1, elinewidth=0.7)
            ax.set_ylim(bottom=0)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 1:
                ax.set_xlabel(labels[kind])
        # a shared scale down each row makes the contrast between rows the point
    for i in range(2):
        hi = max(ax.get_ylim()[1] for ax in axes[i, :])
        for ax in axes[i, :]:
            ax.set_ylim(0, hi)
    axes[1, 1].xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[0, 0].set_title("what the vehicle does", fontsize=8.5, loc="left")
    axes[1, 0].set_title("what we think it does", fontsize=8.5, loc="left")
    save(fig, "s5_robustness")


if __name__ == "__main__":
    path = make_path(PATH)
    data = {}
    for kind in ("noise", "hitch", "slip"):
        print(f"[S5] {kind}", flush=True)
        data[kind] = sweep(kind, path)
    dump(data, "s5_robustness")
    figure(data)
