"""S3 -- control inputs, saturation, and the generated reference.

Two reviewers asked to see the inputs: the tractor speed is a decision variable
and appears in the constraints, but never appeared in a plot.  This study reports
a single full run in detail -- inputs against their box bounds, input rates
against their bounds, the joint angles against the jackknife limit, and how far
the generated reference itself departs from the nominal path.

That last panel is the direct evidence for the feasibility claim: where the path
is incompatible with the chain, the reference is allowed to leave it rather than
the optimiser being handed an infeasible target.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import C_GREY, C_LAST, C_MID, C_PATH, C_REF, C_TRACTOR, dump, g2t, save, seg_colours, seg_label

from gntpf import RefGenWeights, SensorModel, SimConfig, TrackingNMPC, make_path, simulate

U_LB, U_UB = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
BETA_MAX = np.deg2rad(80.0)


def run(path_name="rounded_rect", t_final=48.0, sigma=1.2, seed=0):
    model = g2t()
    path = make_path(path_name)
    du = model.Ts * np.array([6.0, 3.0])
    nmpc = TrackingNMPC(model, Nc=20)
    sensors = SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    res = simulate(model, path, SimConfig(sigma=sigma, t_final=t_final, seed=seed),
                   sensors=sensors, nmpc=nmpc,
                   refgen_weights=RefGenWeights(w_path=20.0, w_theta=5.0))
    return model, path, res, du


def figure(model, path, res, du, name="s3_inputs"):
    N = model.N
    cols = seg_colours(N)
    t = res.t
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.6), sharex=True)

    ax = axes[0]
    ax.plot(t, res.u[0], color=C_TRACTOR, label=r"$\omega_0$ (rad/s)")
    ax.plot(t, res.u[1], color=C_MID, label=r"$v_0$ (m/s)")
    for b in (U_LB[0], U_UB[0]):
        ax.axhline(b, color=C_GREY, ls=":", lw=0.8)
    ax.set_ylabel("control input")
    ax.legend(ncol=2)
    sat = 100.0 * np.mean(np.any(np.abs(res.u) > 0.98 * U_UB[:, None], axis=0))
    ax.set_title(f"inputs saturated for {sat:.1f}% of the run", fontsize=8.5)

    ax = axes[1]
    dU = np.diff(res.u, axis=1)
    ax.plot(t[1:], dU[0], color=C_TRACTOR, lw=0.8, label=r"$\Delta\omega_0$")
    ax.plot(t[1:], dU[1], color=C_MID, lw=0.8, label=r"$\Delta v_0$")
    for s in (-1, 1):
        ax.axhline(s * du[0], color=C_GREY, ls=":", lw=0.8)
        ax.axhline(s * du[1], color=C_GREY, ls="--", lw=0.8)
    ax.set_ylabel("input rate per step")
    ax.legend(ncol=2)

    ax = axes[2]
    for j in range(N):
        ax.plot(t, np.rad2deg(res.q[j]), color=cols[j + 1], label=rf"$\beta_{{{j+1}}}$")
    for s in (-1, 1):
        ax.axhline(s * np.rad2deg(BETA_MAX), color=C_LAST, ls=":", lw=0.9)
    ax.set_ylabel("joint angle (deg)")
    ax.legend(ncol=N)

    ax = axes[3]
    for i in range(N + 1):
        ax.plot(t, res.dev[i], color=cols[i], label=seg_label(i, N))
    ax.plot(t, res.ref_dev.max(axis=0), color=C_REF, ls="--", lw=0.9,
            label="reference vs path (worst segment)")
    ax.set_ylabel("deviation (m)")
    ax.set_xlabel("time (s)")
    ax.legend(ncol=2, fontsize=7)
    save(fig, name)

    return {
        "saturation_pct": float(sat),
        "u_max": res.u.max(axis=1), "u_min": res.u.min(axis=1),
        "beta_max_deg": float(np.rad2deg(np.abs(res.q[model.i_beta]).max())),
        "ref_dev_max": float(res.ref_dev.max()),
        "ref_dev_mean": float(res.ref_dev.mean()),
        **{k: v for k, v in res.summary().items() if isinstance(v, (float, int))},
    }


if __name__ == "__main__":
    # The file names say which path each run used.  An earlier version called the
    # rounded-rectangle run "_headland", which is the agricultural path's feature,
    # and the numbers quoted in the paper come from the unsuffixed file.
    for pname, tf, sig, name in [("agricultural", 160.0, 1.0, "s3_inputs"),
                                 ("rounded_rect", 105.0, 1.2, "s3_inputs_rounded_rect")]:
        print(f"[S3] control inputs -- {pname}", flush=True)
        model, path, res, du = run(pname, t_final=tf, sigma=sig)
        dump(figure(model, path, res, du, name), name)
