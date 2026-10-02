"""S1 -- which segments receive a reference (an ablation, not a comparison).

Four weightings of the *same* reference generator run on identical vehicles,
paths, noise realisations, horizons, estimator and solver settings.  Only the
choice of which segments receive a reference changes:

    tractor    : only the tractor is referenced -- the classical single-vehicle
                 objective.
    last       : only the last trailer, the objective adopted by most prior work
                 on N-trailer off-tracking.
    guidance   : only the middle segment of the chain.  This is NOT a virtual
                 guidance point in the sense of the literature, which is a
                 weighted combination of segment postures and may lie between
                 segments; that objective is implemented properly and swept in
                 s8_comparison.py.  Here it is simply the middle segment, and it
                 is named "Middle segment only" in the figures for that reason.
    proposed   : every segment receives its own feasible reference.

These are NOT implementations of the published methods they echo, and must not
be reported as such.  An established controller designed around a single output
is engineered for that output; this generator weighted at a single point is just
an optimisation problem that has lost the ability to choose among postures.  When
only segment g is weighted the cost has a null space -- every posture placing
segment g identically scores the same -- and the unweighted segments drift.  That
degeneracy is the finding here, and it is why every weighting used elsewhere in
the paper keeps a non-zero floor on every segment.

The comparison against prior work is s8_comparison.py.

Off-tracking is reported over the curved sections of the path as well as
overall.  On a straight, every method is exact and averaging over the whole path
hides the behaviour that actually matters.

The principal case is the agricultural path: five crop rows joined by
omega-shaped headland turns of radius 2 m, against a vehicle 2.2 m long.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from parallel import pmap  # noqa: E402
from common import (C_GREY, C_PATH, dump, save, seg_colours, seg_label,  # noqa: E402
                    vehicle, g2t)

from gntpf import (RefGenWeights, SensorModel, SimConfig, TrackingNMPC,  # noqa: E402
                   make_path, simulate)

STRATEGIES = ["tractor", "last", "guidance", "proposed"]
NICE = {"tractor": "Tractor only", "last": "Last trailer only",
        "guidance": "Middle segment only", "proposed": "All segments (proposed)"}
SHORT = {"tractor": "Tractor", "last": "Last", "guidance": "Middle", "proposed": "Proposed"}
BARC = {"tractor": "#4A5560", "last": "#7C8B99", "guidance": "#4E7CB0", "proposed": "#A8372B"}


def strategy_setup(model, name: str):
    """Return (RefGenWeights, Q) for a reference strategy."""
    N = model.N
    # Every strategy carries the SAME total position weight: the single-point
    # settings concentrate on one segment what the proposed setting spreads over
    # all of them.  Giving the single-point settings a larger total would make
    # their regularisation relatively weaker and so exaggerate the very drift
    # this study is about -- the same confound that Section IX-F controls for.
    w_seg = 20.0
    big = w_seg * (N + 1)
    d = np.zeros(4 * N + 3)

    def single(g):
        w = np.zeros(N + 1); w[g] = big
        th = np.zeros(N + 1); th[g] = 5.0
        d[2 * N + 1 + 2 * g:2 * N + 3 + 2 * g] = 5.0
        return w, th

    if name == "tractor":
        w, th = single(0)
    elif name == "last":
        w, th = single(N)
    elif name == "guidance":
        w, th = single(N // 2)
    elif name == "proposed":
        w = np.full(N + 1, w_seg)
        th = np.full(N + 1, 5.0)
        d = np.concatenate([np.ones(N), 0.5 * np.ones(N + 1), np.ones(2 * N), 5.0 * np.ones(2)])
    else:
        raise ValueError(name)
    return RefGenWeights(w_path=w, w_theta=th, w_prop=50.0, w_s1=1.0), np.diag(d)


def _one(path_name, N, t_final, sigma, strategy, seed):
    """One run of one strategy.  Built inside the worker: models are not picklable."""
    path = make_path(path_name)
    model = g2t() if N == 2 else vehicle(N)
    w, Q = strategy_setup(model, strategy)
    nmpc = TrackingNMPC(model, Nc=20, Q=Q)
    sensors = SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))
    return simulate(model, path, SimConfig(sigma=sigma, t_final=t_final, seed=seed),
                    sensors=sensors, nmpc=nmpc, refgen_weights=w)


def run(path_name="rounded_rect", N=2, t_final=105.0, seeds=(0, 1, 2), sigma=1.2):
    path = make_path(path_name)
    jobs = [(path_name, N, t_final, sigma, s, sd) for s in STRATEGIES for sd in seeds]
    runs = pmap(_one, jobs)
    results, traces = {}, {}
    for i, s in enumerate(STRATEGIES):
        acc = runs[i * len(seeds):(i + 1) * len(seeds)]
        traces[s] = (acc[0], g2t() if N == 2 else vehicle(N))
        mc = np.stack([r.summary()["mean_dev_curved"] for r in acc])
        xc = np.stack([r.summary()["max_dev_curved"] for r in acc])
        ma = np.stack([r.summary()["mean_dev"] for r in acc])
        results[s] = {
            "mean_curved": mc.mean(0), "mean_curved_sd": mc.std(0),
            "max_curved": xc.mean(0),
            "mean_all": ma.mean(0),
            "worst_curved": float(mc.mean(0).max()),
            "worst_curved_max": float(xc.mean(0).max()),
            "last_curved": float(mc.mean(0)[-1]),
            "effort_w": float(np.mean([np.mean(r.u[0] ** 2) for r in acc])),
            "t_total_ms": float(np.mean([r.summary()["t_total_ms"] for r in acc])),
            "fails": int(sum(sum(r.fails.values()) for r in acc)),
        }
        print(f"  {NICE[s]:26s} curved mean/seg {np.round(results[s]['mean_curved'],3)} "
              f"worst {results[s]['worst_curved']:.3f} m  max {results[s]['worst_curved_max']:.3f} m",
              flush=True)
    return path, results, traces


def figure(path, results, traces, tag=""):
    res0, model = traces["proposed"]
    N = model.N
    cols = seg_colours(N)
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.40, wspace=0.34)

    for j, s in enumerate(["last", "guidance", "proposed"]):
        ax = fig.add_subplot(gs[0, j])
        r, m = traces[s]
        ax.plot(path.xy[0], path.xy[1], "--", color=C_PATH, lw=0.8, label="Nominal path")
        for i in range(N + 1):
            P = r.q[2 * N + 1 + 2 * i:2 * N + 3 + 2 * i, :]
            ax.plot(P[0], P[1], color=cols[i], lw=1.0, label=seg_label(i, N))
        ax.set_aspect("equal")
        ax.set_title(NICE[s], fontsize=8.5)
        ax.set_xlabel("x (m)")
        if j == 0:
            ax.set_ylabel("y (m)")
        if j == 2:
            ax.legend(loc="center", fontsize=6.2)

    ax = fig.add_subplot(gs[1, :2])
    w = 0.2
    for j, s in enumerate(STRATEGIES):
        ax.bar(np.arange(N + 1) + (j - 1.5) * w, results[s]["mean_curved"], width=w,
               yerr=results[s]["mean_curved_sd"], capsize=2, error_kw=dict(lw=0.6),
               color=BARC[s], label=NICE[s])
    ax.set_xticks(np.arange(N + 1))
    ax.set_xticklabels([seg_label(i, N) for i in range(N + 1)])
    ax.set_ylabel("mean off-tracking\nin curved sections (m)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(STRATEGIES))
    ax.bar(x, [results[s]["worst_curved_max"] for s in STRATEGIES],
           color=C_GREY, alpha=0.45, width=0.62, label="max")
    ax.bar(x, [results[s]["worst_curved"] for s in STRATEGIES],
           color=[BARC[s] for s in STRATEGIES], width=0.62, label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in STRATEGIES], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("worst segment (m)")
    ax.legend(fontsize=7)
    save(fig, f"s1_baseline{tag}")


if __name__ == "__main__":
    # Durations give the LAST trailer at least two laps of the path inside the
    # averaging window, i.e. after the first 15 % of the run is discarded.
    # Durations give the LAST trailer at least two laps of a closed path, or two
    # traversals of the open agricultural path, inside the averaging window.
    CASES = [("agricultural", 2, "_agri", 253.0),
             ("rounded_rect", 2, "", 105.0)]
    for pname, N, tag, tf in CASES:
        print(f"[S1] path={pname} N={N}", flush=True)
        path, results, traces = run(pname, N=N, t_final=tf)
        dump(results, f"s1_baseline{tag}")
        figure(path, results, traces, tag)
