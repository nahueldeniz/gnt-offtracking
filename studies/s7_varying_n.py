"""S7 -- a trailer is detached mid-run.

The abstract and the introduction both claimed an advantage for operations in
which the number of trailers fluctuates, and a reviewer pointed out, correctly,
that no experiment ever varied it.  This study either substantiates the claim or
retires it.

The test: drive a four-trailer vehicle along a headland path, detach the last two
trailers at a chosen instant, and continue.  Nothing is re-derived.  The model,
the reference generator, the tracking controller and the estimator are all
rebuilt from the same code with a new ``N``; the surviving part of the state is
carried across.

What the detached vehicle is compared with is a two-trailer vehicle driven along
the whole path from the start, at the same arc-length positions: the peak at the
first corner after the detachment against the peak of that vehicle at the same
corner, and the worst segment's mean deviation over the rest of the evaluated run
against that vehicle's over the same stretch.  The corners of the rounded
rectangle raise the deviation of either vehicle at every pass, so a threshold on
the deviation alone would measure the corners and not the detachment.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from common import (C_GREY, C_LAST, C_PATH, LAPS, OUT, _writable, dump, save, seg_colours, seg_label,
                    study_path, t_final_for, vehicle)

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


def run(N_before=4, N_after=2, sigma=1.1, seed=0):
    """N_before trailers for the first lap, N_after for the second."""
    big = vehicle(N_before)
    path = study_path("rounded_rect", big)          # lead-in, two laps, run-out
    lap = (path.eval_end - path.eval_start) / LAPS
    s0 = 1.6 * big.total_length                     # where the progress point starts
    t_before = (path.eval_start + lap - s0) / sigma  # the progress point completes lap 1
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
    # the two laps coincide in the plane, so the projection is searched around
    # the tractor's arc length at the hand-over, not over the whole path
    s_here = path.project(small.positions(q_hand)[0], res1.s_proj[0, -1], 3.0 * big.total_length)
    res2 = simulate(small, path, SimConfig(sigma=sigma, t_final=t_final_for(path, sigma),
                                           seed=seed + 100),
                    sensors=sensors_for(small), refgen=rg, nmpc=nmpc, estimator=est,
                    refgen_weights=W, q0=q_hand, s_start=s_here)
    print(f"  after:  N={N_after}  worst {res2.summary()['worst_segment']:.3f} m", flush=True)

    # The comparison run: two trailers from the start, same path, same noise seed
    # as the second phase.  Positions are matched by the tractor's arc length.
    ref_model = vehicle(N_after)
    res_ref = simulate(ref_model, path, SimConfig(sigma=sigma, t_final=t_final_for(path, sigma),
                                                  seed=seed + 100),
                       sensors=sensors_for(ref_model), refgen_weights=W)
    kk = res2.masks()[0][0]
    s2, w2 = res2.s_proj[0, kk], res2.dev[:, kk].max(axis=0)
    kr = np.where(res_ref.masks()[0][0])[0]
    sr, wr = res_ref.s_proj[0, kr], res_ref.dev[:, kr].max(axis=0)
    lap = (path.eval_end - path.eval_start) / LAPS
    i_pk = int(np.argmax(w2))
    near = np.abs(sr - s2[i_pk]) < 0.1 * lap
    same = (sr >= s2.min()) & (sr <= s2.max())

    def worst_mean(res, k):
        return float(res.dev[:, k].mean(axis=1).max())

    pre = res1.summary()["worst_segment"]
    return path, big, small, res1, res2, res_ref, dict(
        rebuild_s=float(rebuild_s), worst_before=float(pre),
        worst_after=float(res2.summary()["worst_segment"]),
        peak_after=float(w2[i_pk]), s_peak_after=float(s2[i_pk]),
        peak_ref_same_corner=float(wr[near].max()) if near.any() else float("nan"),
        worst_mean_after=worst_mean(res2, kk),
        worst_mean_ref_same=worst_mean(res_ref, kr[same]) if same.any() else float("nan"),
        completed_before=bool(res1.summary()["completed"]),
        completed_after=bool(res2.summary()["completed"]),
        completed_ref=bool(res_ref.summary()["completed"]),
        fails_before=int(sum(res1.fails.values())), fails_after=int(sum(res2.fails.values())),
        fails_ref=int(sum(res_ref.fails.values())))


def traces(big, small, res1, res2, res_ref):
    """What the figure draws, so that it can be redrawn without re-running."""
    pos = lambda m, r: np.stack([r.q[2 * m.N + 1 + 2 * i:2 * m.N + 3 + 2 * i, :] for i in range(m.N + 1)])
    return dict(P1=pos(big, res1), P2=pos(small, res2),
                s1=res1.s_proj[0], w1=res1.dev.max(axis=0),
                s2=res2.s_proj[0], w2=res2.dev.max(axis=0),
                sr=res_ref.s_proj[0], wr=res_ref.dev.max(axis=0),
                dmax=np.array([np.nanmax(res1.dev), np.nanmax(res2.dev), np.nanmax(res_ref.dev)]))


def figure(path, big, small, T):
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(path.xy[0], path.xy[1], "--", color=C_PATH, lw=0.8, label="Nominal path")
    c1 = seg_colours(big.N)
    for i in range(big.N + 1):
        ax.plot(T["P1"][i, 0], T["P1"][i, 1], color=c1[i], lw=0.8, alpha=0.55)
    c2 = seg_colours(small.N)
    for i in range(small.N + 1):
        ax.plot(T["P2"][i, 0], T["P2"][i, 1], color=c2[i], lw=1.1)
    ax.plot(T["P2"][0, 0, 0], T["P2"][0, 1, 0], "*", color="#111", ms=9, zorder=9,
            label="Detachment")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"N = {big.N} (faint) then N = {small.N}", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left")

    # Right: worst-segment deviation against the tractor's arc length, so that
    # the detached vehicle and the comparison run are read at the same corners.
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(T["sr"], T["wr"], color="#2B4C7E", lw=0.8, ls="--", label=f"N = {small.N} throughout")
    ax.plot(T["s1"], T["w1"], color=C_GREY, lw=1.0, label=f"N = {big.N}")
    ax.plot(T["s2"], T["w2"], color=C_LAST, lw=1.0, label=f"N = {small.N} after detachment")
    ax.axvline(T["s2"][0], color="#111", ls=":", lw=1.0)
    ax.set_xlim(path.eval_start - 2.0, path.eval_end + 2.0)
    ax.set_xlabel("tractor arc length (m)"); ax.set_ylabel("worst-segment off-tracking (m)")
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    ax.set_ylim(0.0, 1.3 * float(np.max(T["dmax"])))
    save(fig, "s7_varying_n")


if __name__ == "__main__":
    big, small = vehicle(4), vehicle(2)
    fn = os.path.join(OUT, "s7_varying_n_traces.npz")
    if "--replot" in sys.argv:                  # redraw from out/ without re-running
        z = np.load(fn)
        figure(study_path("rounded_rect", big), big, small, {k: z[k] for k in z.files})
        sys.exit(0)
    print("[S7] trailer detached mid-run", flush=True)
    path, big, small, res1, res2, res_ref, info = run()
    dump(info, "s7_varying_n")
    T = traces(big, small, res1, res2, res_ref)
    np.savez_compressed(_writable(fn), **T)
    print("    traces -> out/s7_varying_n_traces.npz", flush=True)
    figure(path, big, small, T)
    print("  ", info, flush=True)
