"""S1 -- where along the chain the reference is weighted (an ablation of the
proposed generator, not a comparison against other methods).

Four weightings of the *same* reference generator and tracking controller run
on identical vehicles, paths, noise realisations, horizons, estimator and solver
settings.  Only the distribution of the weight along the chain changes:

    tractor    : the tractor carries the weight; every other segment a small floor.
    last       : the last trailer carries the weight; the others a small floor.
    guidance   : the middle segment carries the weight; the others a small floor.
                 (Named "middle segment" in the figures; it is not the guidance
                 point of the literature, which s8_comparison.py implements.)
    proposed   : every segment carries the same weight.

The total position weight is the same in the four settings.  The segments that
are not emphasised keep EPS_FRAC of the weights they carry in the even setting
rather than zero, so that every weighting matrix stays positive definite and the problems
remain well posed; what is tested is the distribution of the weight, not a
degenerate cost.

All four are settings of the method of this paper.  None is an implementation
of a published method; the comparison against prior work is s8_comparison.py.

A run in which the vehicle loses its reference -- the tractor more than two
vehicle lengths behind the progress point for five seconds -- is reported as
not completed, with the fraction of the path the last trailer covered, and its
deviations are not averaged with completed runs.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from parallel import pmap  # noqa: E402
from common import (C_GREY, C_PATH, OUT, _writable, dump, save, seg_colours,  # noqa: E402
                    seg_label, study_path, t_final_for, vehicle, g2t)

from gntpf import (RefGenWeights, SensorModel, SimConfig, TrackingNMPC,  # noqa: E402
                   make_path, simulate)

STRATEGIES = ["tractor", "last", "guidance", "proposed"]
NICE = {"tractor": "Tractor emphasised", "last": "Last trailer emphasised",
        "guidance": "Middle segment emphasised", "proposed": "Even (proposed setting)"}
SHORT = {"tractor": "Tractor", "last": "Last", "guidance": "Middle", "proposed": "Even"}
LEG = {"tractor": "Tractor emph.", "last": "Last trailer emph.",
       "guidance": "Middle segment emph.", "proposed": "Even (proposed)"}
BARC = {"tractor": "#4A5560", "last": "#7C8B99", "guidance": "#4E7CB0", "proposed": "#A8372B"}

# Fraction of their even-setting weights kept by the segments that are not
# emphasised: small, but not zero, so that every weighting matrix stays
# positive definite.
EPS_FRAC = 0.01


def strategy_setup(model, name: str):
    """Return (RefGenWeights, Q) for a weighting of the proposed generator."""
    N = model.N
    w_seg, th_seg = 20.0, 5.0
    total = w_seg * (N + 1)                   # the same total position weight in every setting
    d_even = np.concatenate([np.ones(N), 0.5 * np.ones(N + 1), np.ones(2 * N), 5.0 * np.ones(2)])

    def emphasise(g):
        # the other segments keep EPS_FRAC of the weights they carry in the even
        # setting; the emphasised one takes the rest of the total position weight
        w = np.full(N + 1, EPS_FRAC * w_seg)
        w[g] = total - N * EPS_FRAC * w_seg
        th = np.full(N + 1, EPS_FRAC * th_seg)
        th[g] = th_seg
        d = EPS_FRAC * d_even
        d[:N] = d_even[:N]                             # joint angles: as in the even setting
        d[N + g] = d_even[N + g]                       # the emphasised heading keeps its weight
        pos = slice(2 * N + 1, 4 * N + 3)
        i_g = slice(2 * N + 1 + 2 * g, 2 * N + 3 + 2 * g)
        d[i_g] = 0.0
        d[i_g] = 0.5 * (d_even[pos].sum() - d[pos].sum())   # same total position weight in Q
        return w, th, d

    if name == "tractor":
        w, th, d = emphasise(0)
    elif name == "last":
        w, th, d = emphasise(N)
    elif name == "guidance":
        w, th, d = emphasise(N // 2)
    elif name == "proposed":
        w, th, d = np.full(N + 1, w_seg), np.full(N + 1, th_seg), d_even
    else:
        raise ValueError(name)
    return RefGenWeights(w_path=w, w_theta=th, w_prop=50.0, w_s1=1.0), np.diag(d)


def _one(path_name, N, sigma, strategy, seed):
    """One run of one weighting.  Built inside the worker: models are not picklable."""
    model = g2t() if N == 2 else vehicle(N)
    path = study_path(path_name, model)
    w, Q = strategy_setup(model, strategy)
    nmpc = TrackingNMPC(model, Nc=20, Q=Q)
    sensors = SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    return simulate(model, path, SimConfig(sigma=sigma, t_final=t_final_for(path, sigma), seed=seed),
                    sensors=sensors, nmpc=nmpc, refgen_weights=w)


def run(path_name="agricultural", N=2, seeds=(0, 1, 2), sigma=1.0):
    model = g2t() if N == 2 else vehicle(N)
    path = study_path(path_name, model)
    jobs = [(path_name, N, sigma, s, sd) for s in STRATEGIES for sd in seeds]
    runs = pmap(_one, jobs)
    results, traces = {}, {}
    for i, s in enumerate(STRATEGIES):
        acc = runs[i * len(seeds):(i + 1) * len(seeds)]
        r0 = acc[0]
        traces[s] = {"P": np.stack([r0.q[2 * N + 1 + 2 * i:2 * N + 3 + 2 * i, :]
                                    for i in range(N + 1)]),
                     "diverged": bool(r0.meta.get("diverged", False))}
        S = [r.summary() for r in acc]
        done = [x for x in S if x["completed"]]
        n_div = len(S) - len(done)
        base = {"n_runs": len(S), "n_diverged": n_div,
                "progress": [float(x["progress"]) for x in S],
                "t_diverged": [r.meta.get("t_diverged") for r in acc],
                "fails": int(sum(sum(r.fails.values()) for r in acc)),
                "t_total_ms": float(np.mean([x["t_total_ms"] for x in S]))}
        if done:
            mc = np.stack([x["mean_dev_curved"] for x in done])
            xc = np.stack([x["max_dev_curved"] for x in done])
            ma = np.stack([x["mean_dev"] for x in done])
            base.update({
                "mean_curved": mc.mean(0), "mean_curved_sd": mc.std(0),
                "max_curved": xc.mean(0), "mean_all": ma.mean(0),
                "worst_curved": float(mc.mean(0).max()),
                "worst_curved_max": float(xc.mean(0).max()),
                "last_curved": float(mc.mean(0)[-1]),
            })
        results[s] = base
        txt = (f"worst {base['worst_curved']:.3f} m" if done else "no run completed")
        print(f"  {NICE[s]:28s} completed {len(done)}/{len(S)}  {txt}  "
              f"progress {np.round(base['progress'], 2)}", flush=True)
    return path, results, traces


def save_traces(traces, tag):
    """The first realisation of each weighting, so the figure can be redrawn
    without re-running the study (``--replot``)."""
    fn = _writable(os.path.join(OUT, f"s1_baseline{tag}_traces.npz"))
    np.savez_compressed(fn, **{f"{s}_P": traces[s]["P"] for s in STRATEGIES},
                        **{f"{s}_diverged": traces[s]["diverged"] for s in STRATEGIES})
    print(f"    traces -> out/s1_baseline{tag}_traces.npz", flush=True)


def load_traces(tag):
    z = np.load(os.path.join(OUT, f"s1_baseline{tag}_traces.npz"))
    return {s: {"P": z[f"{s}_P"], "diverged": bool(z[f"{s}_diverged"])} for s in STRATEGIES}


def figure(path, results, traces, N, tag=""):
    """Trajectories of the four weightings of the proposed generator, and their deviations."""
    cols = seg_colours(N)
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.72, 0.26, 0.9], hspace=0.38, wspace=0.32)
    a0, a1 = path.eval_start, path.eval_end
    k0 = int(np.searchsorted(path.s, a0)); k1 = int(np.searchsorted(path.s, a1))

    handles = None
    for j, s in enumerate(STRATEGIES):
        ax = fig.add_subplot(gs[0, j])
        P = traces[s]["P"]
        ax.plot(path.xy[0, k0:k1], path.xy[1, k0:k1], "--", color=C_PATH, lw=0.8,
                label="Nominal path")
        for i in range(N + 1):
            ax.plot(P[i, 0], P[i, 1], color=cols[i], lw=0.9, label=seg_label(i, N))
        if traces[s]["diverged"]:
            ax.plot(P[0, 0, -1], P[0, 1, -1], "kx", ms=6, mew=1.4, label="Run stopped")
        ax.set_aspect("equal")
        ax.set_title("Ours: " + SHORT[s].lower() + ("" if s == "proposed" else " emph."),
                     fontsize=7.6)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("x (m)", fontsize=7)
        if j == 0:
            ax.set_ylabel("y (m)", fontsize=7)
        h, l = ax.get_legend_handles_labels()
        if handles is None or len(h) > len(handles[0]):
            handles = (h, l)

    # Both legends sit in their own row, clear of every panel.
    lax = fig.add_subplot(gs[1, :]); lax.axis("off")
    leg1 = lax.legend(*handles, loc="upper center", ncol=len(handles[0]), fontsize=7,
                      frameon=False, bbox_to_anchor=(0.5, 1.15), handlelength=1.8)
    lax.add_artist(leg1)
    bar_h = [plt.Rectangle((0, 0), 1, 1, color=BARC[s]) for s in STRATEGIES]
    bar_l = [LEG[s] + ("" if "worst_curved" in results[s] else " (n.c.)") for s in STRATEGIES]
    lax.legend(bar_h, bar_l, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.25), handlelength=1.2)

    ax = fig.add_subplot(gs[2, :3])
    w = 0.2
    for j, s in enumerate(STRATEGIES):
        r = results[s]
        if "mean_curved" in r:
            ax.bar(np.arange(N + 1) + (j - 1.5) * w, r["mean_curved"], width=w,
                   yerr=r["mean_curved_sd"], capsize=2, error_kw=dict(lw=0.6),
                   color=BARC[s])
    ax.set_xticks(np.arange(N + 1))
    ax.set_xticklabels([seg_label(i, N) for i in range(N + 1)], fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("mean off-tracking in\ncurved sections (m)", fontsize=7)

    ax = fig.add_subplot(gs[2, 3])
    x = np.arange(len(STRATEGIES))
    vals = [results[s].get("worst_curved", np.nan) for s in STRATEGIES]
    ax.bar(x, vals, color=[BARC[s] for s in STRATEGIES], width=0.62)
    for xi, s in zip(x, STRATEGIES):
        if "worst_curved" not in results[s]:
            ax.text(xi, 0.02, "n.c.", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in STRATEGIES], rotation=30, ha="right", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("worst segment (m)", fontsize=7)
    save(fig, f"s1_baseline{tag}")


if __name__ == "__main__":
    # The agricultural geometry at the nominal sigma, and the rounded rectangle.
    # ``--replot`` redraws the figures from out/ without re-running the study.
    CASES = [("agricultural", 2, "_agri", 1.0),
             ("rounded_rect", 2, "", 1.2)]
    replot = "--replot" in sys.argv
    for pname, N, tag, sig in CASES:
        model = g2t() if N == 2 else vehicle(N)
        if replot:
            with open(os.path.join(OUT, f"s1_baseline{tag}.json")) as fh:
                results = json.load(fh)
            figure(study_path(pname, model), results, load_traces(tag), N, tag)
            continue
        print(f"[S1] path={pname} N={N} sigma={sig}", flush=True)
        path, results, traces = run(pname, N=N, sigma=sig)
        dump(results, f"s1_baseline{tag}")
        save_traces(traces, tag)
        figure(path, results, traces, N, tag)
