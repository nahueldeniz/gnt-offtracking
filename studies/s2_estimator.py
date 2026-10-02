"""S2 -- does the moving-horizon estimator earn its cost?

Reviewers of both submissions asked why an NMHE is used when the tractor pose and
all joint angles are measured, so that the remaining states follow algebraically.
The answer cannot be an assertion, so this study puts the NMHE against an EKF
given every advantage: the same model, the same measurements, and a covariance
matched to the true noise statistics.

Three regimes:

    clean    -- bounded measurement noise only.  The EKF should be competitive
                here, and if it is, that is reported.
    outliers -- occasional gross errors on the joint-angle encoders, which is
                what actually happens on the platform (slip in the potentiometer
                coupling, electrical pickup).  The noise is bounded but not
                Gaussian, which is the assumption the EKF rests on.
    limit    -- a manoeuvre driven close to the jackknife bound, where the
                unconstrained filter can return a physically impossible estimate.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from parallel import pmap  # noqa: E402
from common import (C_GREY, C_LAST, C_MID, C_PATH, dump, g2t, save, study_path, t_final_for,  # noqa: E402
                    seg_colours, seg_label)

from gntpf import (RefGenWeights, EKF, NMHE, SensorModel, SimConfig, make_path, simulate)  # noqa: E402

REGIMES = {
    "clean":    dict(outlier_prob=0.0),
    "outliers": dict(outlier_prob=0.06, outlier_gain=12.0),
    "limit":    dict(outlier_prob=0.0, tight=True),
}


def build_sensors(model, spec):
    sig = np.concatenate([np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]])
    return SensorModel(sigma=sig, kind="uniform",
                       outlier_prob=spec.get("outlier_prob", 0.0),
                       outlier_gain=spec.get("outlier_gain", 10.0))


def matched_ekf(model, sensors):
    """An EKF tuned to the true noise -- a uniform bound b has variance b^2/3."""
    v = (np.asarray(sensors.sigma) ** 2) / 3.0
    R = np.diag(np.maximum(v, 1e-10))
    Q = np.eye(model.nq) * 1e-5
    P0 = np.eye(model.nq) * 1e-3
    return EKF(model, P0=P0, Q=Q, R=R)


def _one(reg, est_name, seed):
    spec = REGIMES[reg]
    model = g2t()
    path = study_path("square" if spec.get("tight") else "lemniscate", model)
    sensors = build_sensors(model, spec)
    est = NMHE(model) if est_name == "nmhe" else matched_ekf(model, sensors)
    res = simulate(model, path, SimConfig(sigma=1.0, t_final=t_final_for(path, 1.0), seed=seed),
                   sensors=sensors, estimator=est,
                   refgen_weights=RefGenWeights(w_path=20.0, w_theta=5.0))
    return res


def run(seeds=(0, 1, 2)):
    out, traces = {}, {}
    jobs = [(reg, est, sd) for reg in REGIMES for est in ("nmhe", "ekf") for sd in seeds]
    done = pmap(_one, jobs)
    k = 0
    for reg in REGIMES:
        for est_name in ("nmhe", "ekf"):
            acc_all = done[k:k + len(seeds)]; k += len(seeds)
            acc = [r for r in acc_all if r.summary()["completed"]]
            n_div, fails = len(acc_all) - len(acc), int(sum(sum(r.fails.values()) for r in acc_all))
            model = g2t()
            traces[(reg, est_name)] = (acc[0] if acc else acc_all[0], model)
            if not acc:
                out[f"{reg}/{est_name}"] = {"n_diverged": n_div, "fails": fails}
                print(f"  {reg:9s} {est_name:5s}  no run completed", flush=True)
                continue
            errs = [np.where(r.masks()[0], r.est_err, np.nan) for r in acc]
            times = [r.summary()["t_mhe_ms"] for r in acc]
            betaviol = [100.0 * (np.abs(r.qhat[model.i_beta, :]) > np.deg2rad(85.0)).mean()
                        for r in acc]
            E = np.concatenate(errs, axis=1)
            out[f"{reg}/{est_name}"] = {
                "rmse": float(np.sqrt(np.nanmean(E ** 2))),
                "rmse_per_seg": np.sqrt(np.nanmean(E ** 2, axis=1)),
                "p95": float(np.nanpercentile(E, 95)),
                "max": float(np.nanmax(E)),
                "solve_ms": float(np.mean(times)),
                "infeasible_pct": float(np.mean(betaviol)),
                "n_diverged": n_div, "fails": fails,
            }
            r = out[f"{reg}/{est_name}"]
            print(f"  {reg:9s} {est_name:5s}  RMSE {r['rmse']:.4f} m   p95 {r['p95']:.4f}   "
                  f"max {r['max']:.3f}   infeasible {r['infeasible_pct']:.2f}%   "
                  f"{r['solve_ms']:.1f} ms", flush=True)
    return out, traces


def figure(out, traces):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    regs = list(REGIMES)
    titles = {"clean": "Bounded noise", "outliers": "Encoder outliers",
              "limit": "Near the jackknife bound"}

    for ax, reg in zip(axes, regs):
        for est, col, ls in (("ekf", C_GREY, "-"), ("nmhe", C_LAST, "-")):
            res, model = traces[(reg, est)]
            k0 = int(0.15 * res.t.size)
            e = np.linalg.norm(res.qhat[model.i_xy, :] - res.q[model.i_xy, :], axis=0)
            ax.plot(res.t[k0:], e[k0:], ls, color=col, lw=0.9,
                    label="EKF" if est == "ekf" else "NMHE")
        ax.set_title(titles[reg])
        ax.set_xlabel("time (s)")
        ax.set_yscale("log")
        if reg == regs[0]:
            ax.set_ylabel("position estimation error (m)")
            ax.legend()
    save(fig, "s2_estimator")

    # summary bars
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    x = np.arange(len(regs)); w = 0.36
    for j, (est, col) in enumerate((("ekf", C_GREY), ("nmhe", C_LAST))):
        axes[0].bar(x + (j - 0.5) * w, [out[f"{r}/{est}"]["rmse"] for r in regs],
                    width=w, color=col, label=est.upper())
        axes[1].bar(x + (j - 0.5) * w, [out[f"{r}/{est}"]["infeasible_pct"] for r in regs],
                    width=w, color=col, label=est.upper())
    for ax, lab in zip(axes, ["estimation RMSE (m)", "physically infeasible estimates (%)"]):
        ax.set_xticks(x); ax.set_xticklabels([titles[r] for r in regs], fontsize=7)
        ax.set_ylabel(lab); ax.legend()
    axes[0].set_yscale("log")
    save(fig, "s2_estimator_summary")


if __name__ == "__main__":
    print("[S2] estimator comparison", flush=True)
    out, traces = run()
    dump(out, "s2_estimator")
    figure(out, traces)
