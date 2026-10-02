"""S7 -- a trailer is detached mid-run.

The abstract and the introduction both claimed an advantage for operations in
which the number of trailers fluctuates, and a reviewer pointed out, correctly,
that no experiment ever varied it.  This study either substantiates the claim or
retires it.

The test: drive a four-trailer vehicle along a headland path, detach the last two
trailers at a chosen instant, and continue.  Nothing is re-derived.  The model,
the reference generator, the tracking controller and the estimator are all
rebuilt from the same code with a new ``N``; the surviving part of the state is
carried across.  What is measured is how long the loop takes to settle afterwards
and whether off-tracking returns to the level it held before.
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from common import C_GREY, C_LAST, C_PATH, dump, save, seg_colours, seg_label, vehicle

from gntpf import (NMHE, ReferenceGenerator, RefGenWeights, SensorModel,  # noqa: E402
                   SimConfig, TrackingNMPC, make_path, simulate)

W = RefGenWeights(w_path=20.0, w_theta=5.0)


def truncate_state(model_big, model_small, q):
    """Carry the surviving segments across when trailers are removed."""
    n = model_small.N
    th = np.asarray(q)[model_big.i_theta][: n + 1]
    x0, y0 = q[2 * model_big.N + 1], q[2 * model_big.N + 2]
    return model_small.state_from_pose(x0, y0, th)


def sensors_for(model, scale=1.0):
    return SensorModel(sigma=scale * np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))


def run(N_before=4, N_after=2, t_before=115.0, t_after=115.0, sigma=1.1, seed=0):
    path = make_path("rounded_rect")

    big = vehicle(N_before)
    res1 = simulate(big, path, SimConfig(sigma=sigma, t_final=t_before, seed=seed),
                    sensors=sensors_for(big), refgen_weights=W)
    print(f"  before: N={N_before}  worst {res1.summary()['worst_segment']:.3f} m", flush=True)

    # ---- the detachment: rebuild everything for the new N, no re-derivation
    t0 = time.perf_counter()
    small = vehicle(N_after)
    rg = ReferenceGenerator(small, weights=W)
    nmpc = TrackingNMPC(small, Nc=20)
    est = NMHE(small)
    rebuild_s = time.perf_counter() - t0
    print(f"  rebuild for N={N_after} took {rebuild_s:.2f} s", flush=True)

    q_hand = truncate_state(big, small, res1.q[:, -1])
    s_here = path.project(small.positions(q_hand)[0])
    res2 = simulate(small, path, SimConfig(sigma=sigma, t_final=t_after, seed=seed + 100),
                    sensors=sensors_for(small), refgen=rg, nmpc=nmpc, estimator=est,
                    refgen_weights=W, q0=q_hand, s_start=s_here)
    print(f"  after:  N={N_after}  worst {res2.summary()['worst_segment']:.3f} m", flush=True)

    # settling: time for the worst segment to fall back under its pre-event level
    pre = res1.summary()["worst_segment"]
    w_after = res2.dev.max(axis=0)
    band = 1.25 * pre
    below = np.where(w_after < band)[0]
    settle = float("nan")
    for i in below:
        if np.all(w_after[i:] < band * 1.6):
            settle = float(res2.t[i])
            break
    return path, big, small, res1, res2, dict(
        rebuild_s=float(rebuild_s), worst_before=float(pre),
        worst_after=float(res2.summary()["worst_segment"]),
        peak_after=float(w_after.max()), settle_s=settle,
        fails_before=int(sum(res1.fails.values())), fails_after=int(sum(res2.fails.values())))


def figure(path, big, small, res1, res2, info):
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(path.xy[0], path.xy[1], "--", color=C_PATH, lw=0.8, label="Nominal path")
    c1 = seg_colours(big.N)
    for i in range(big.N + 1):
        P = res1.q[2 * big.N + 1 + 2 * i:2 * big.N + 3 + 2 * i, :]
        ax.plot(P[0], P[1], color=c1[i], lw=0.8, alpha=0.55)
    c2 = seg_colours(small.N)
    for i in range(small.N + 1):
        P = res2.q[2 * small.N + 1 + 2 * i:2 * small.N + 3 + 2 * i, :]
        ax.plot(P[0], P[1], color=c2[i], lw=1.1)
    P0 = small.positions(res2.q[:, 0])
    ax.plot(P0[0, 0], P0[0, 1], "*", color="#111", ms=9, zorder=9, label="detachment")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"N = {big.N} (faint) then N = {small.N}", fontsize=8.5)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    t1, t2 = res1.t, res2.t + res1.t[-1]
    ax.plot(t1, res1.dev.max(axis=0), color=C_GREY, lw=1.0, label=f"N = {big.N}")
    ax.plot(t2, res2.dev.max(axis=0), color=C_LAST, lw=1.0, label=f"N = {small.N}")
    ax.axvline(res1.t[-1], color="#111", ls=":", lw=1.0)
    ax.axhline(info["worst_before"], color=C_GREY, ls="--", lw=0.8)
    if np.isfinite(info["settle_s"]):
        ax.axvline(res1.t[-1] + info["settle_s"], color=C_LAST, ls=":", lw=0.8)
        ax.annotate(f"settles in {info['settle_s']:.1f} s",
                    (res1.t[-1] + info["settle_s"], ax.get_ylim()[1] * 0.85),
                    fontsize=7, color=C_LAST)
    ax.set_xlabel("time (s)"); ax.set_ylabel("worst-segment off-tracking (m)")
    ax.legend(fontsize=7)
    save(fig, "s7_varying_n")


if __name__ == "__main__":
    print("[S7] trailer detached mid-run", flush=True)
    path, big, small, res1, res2, info = run()
    dump(info, "s7_varying_n")
    figure(path, big, small, res1, res2, info)
    print("  ", info, flush=True)
