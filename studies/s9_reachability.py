"""S9 -- how reachable are the generated references?

The generator does not impose reachability; it penalises the residual

    e_r = q_ref[k] - F(q_ref[k-1], u_ref),

Eq. (11) of the paper.  A penalty alone says nothing about how large e_r is in
practice, so this study measures it.  The residual is split into its position
part (the largest Euclidean residual over the segment positions, in metres) and
its angle part (the largest residual over joint angles and headings, in
degrees), and both are reported on the straight sections, where the path is
attainable, and on the curved ones, where it may not be.  The progress of the
reference in one sampling period, sigma * Ts, is the scale against which the
position residual should be read.

Run with the configuration of the simulation study on the agricultural path,
for the same duration as the other agricultural studies.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import C_LAST, C_MID, dump, g2t, save, study_path, t_final_for
from gntpf import RefGenWeights, SensorModel, SimConfig, make_path, simulate
from parallel import pmap

PATH = "agricultural"
SIGMA = 1.0
SEEDS = (0, 1, 2)


def _one(seed):
    m = g2t()
    sensors = SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(m.N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    path = study_path(PATH, m)
    return simulate(m, path, SimConfig(sigma=SIGMA, t_final=t_final_for(path, SIGMA), seed=seed),
                    sensors=sensors, refgen_weights=RefGenWeights(w_path=20.0, w_theta=5.0))


def _combine(summaries):
    """Mean of the means and of the 95th percentiles, maximum of the maxima."""
    out = {}
    for part in ("pos_m", "ang_deg"):
        out[part] = {}
        for region in ("all", "curved", "straight"):
            rows = [s[part][region] for s in summaries]
            out[part][region] = {
                "mean": float(np.nanmean([r["mean"] for r in rows])),
                "p95": float(np.nanmean([r["p95"] for r in rows])),
                "max": float(np.nanmax([r["max"] for r in rows])),
            }
    return out


def figure(res):
    N = res.meta["N"]
    M, C = res.masks()
    kk = M[0]
    t = res.t[kk]
    R = res.reach[:, kk]
    pos = np.stack([np.hypot(R[2 * N + 1 + 2 * i], R[2 * N + 2 + 2 * i])
                    for i in range(N + 1)]).max(axis=0)
    ang = np.degrees(np.abs(R[:2 * N + 1]).max(axis=0))
    c = C.any(axis=0)[kk]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.2), sharex=True)
    fig.subplots_adjust(hspace=0.12)
    for ax, y, col, lab in ((axes[0], 1000 * pos, C_MID, "position residual (mm)"),
                            (axes[1], ang, C_LAST, "angle residual (deg)")):
        ax.fill_between(t, 0, 1, where=c, transform=ax.get_xaxis_transform(),
                        color="0.9", lw=0, label="curved sections")
        ax.plot(t, y, color=col, lw=0.8)
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
    axes[0].axhline(1000 * SIGMA * res.meta.get("Ts", 0.05), color="0.4", lw=0.7, ls="--",
                    label=r"$\sigma T_s$")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[1].set_xlabel("time (s)")
    save(fig, "s9_reachability")


if __name__ == "__main__":
    print(f"[S9] reachability residual -- {PATH}, {len(SEEDS)} seeds", flush=True)
    runs_all = pmap(_one, [(sd,) for sd in SEEDS])
    runs = [r for r in runs_all if r.summary()["completed"]]
    out = _combine([r.reach_summary() for r in runs])
    out["n_runs"], out["n_diverged"] = len(runs_all), len(runs_all) - len(runs)
    out["fails"] = int(sum(sum(r.fails.values()) for r in runs_all))
    out["sigma_Ts_m"] = SIGMA * runs_all[0].meta.get("Ts", 0.05)
    out["path"] = PATH
    for part, unit in (("pos_m", "m"), ("ang_deg", "deg")):
        for region in ("straight", "curved"):
            r = out[part][region]
            print(f"  {part:7s} {region:8s} mean {r['mean']:.4f}  p95 {r['p95']:.4f}  "
                  f"max {r['max']:.4f} {unit}", flush=True)
    dump(out, "s9_reachability")
    figure(runs_all[0])
