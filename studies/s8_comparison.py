"""S8 -- comparison against prior work.

Three reference points, chosen so that each isolates something different.

``proposed``
    The two-stage framework: a reference generator that places every segment,
    followed by the tracking NMPC.

``mp2021``
    The reconstruction method of Michałek and Pazderski (EJC 2021), which solves
    the same problem -- admissible joint references from a prescribed output
    reference -- for the same vehicle class, nSNT and GNT alike.  It is driven
    into **the same tracking NMPC**, with the same weights, bounds, estimator,
    sensors and noise realisations.  The only thing that differs is where the
    reference comes from, so any difference in the result is attributable to the
    reference and to nothing else.  This is the substantive comparison.

``pure-pursuit``
    Geometric look-ahead steering of the tractor: the naive floor, and the
    method that requires no model of the chain at all.

A fourth, the cascaded tracking controller of Michałek (TCST 2017), is reported
separately by :mod:`s9_cascade` because it does not exist for this vehicle: its
inner loop inverts ``J_i``, and ``det J_i = -Lh_i / L_i`` vanishes at the
platform's on-axle second hitch.

What the comparison should show, and what it should not: ``mp2021`` prescribes
the path for the **last trailer only**, so it should place that segment at least
as well as the proposed method and the remaining segments worse.  The claim
being tested is not that the proposed method tracks better -- it is that it
distributes, and that it does so online.
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from common import (C_GREY, C_LAST, C_MID, C_PATH, C_TRACTOR, dump, g2t, save,  # noqa: E402
                    seg_colours, seg_label)

from gntpf import (RefGenWeights, SensorModel, SimConfig, TrackingNMPC,  # noqa: E402
                   make_path, simulate)
from gntpf.baselines import PurePursuit, simulate_baseline, simulate_with_reference
from gntpf.mp_refgen import MPReferenceGenerator

GUIDE_TS = [0.0, 0.5, 1.0, 1.5, 2.0]
W_GUIDE = 300.0

METHODS = ["pursuit", "mp2021", "proposed"]
NICE = {"pursuit": "Pure pursuit (tractor)",
        "mp2021": "Micha\u0142ek & Pazderski (2021)",
        "proposed": "Proposed (all segments)"}
SHORT = {"pursuit": "Pure pursuit", "mp2021": "Micha\u0142ek & Pazderski", "proposed": "Proposed"}
COL = {"pursuit": "#7C8B99", "mp2021": "#4E7CB0", "proposed": "#A8372B"}

# n_harm is set per path: the Fourier basis has to resolve the curvature
# profile, and a field path whose curvature is piecewise constant with jumps
# needs far more harmonics than a smooth closed curve.  The baseline is given
# the generous setting rather than the cheap one.
CASES = [("agricultural", 160.0, 64)]
SEEDS = (0, 1, 2)
SIGMA = 1.0
W = RefGenWeights(w_path=20.0, w_theta=5.0)


def sensors_for(model):
    return SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))


def mp_reference(model, path, n_harm=32, n_coll=600):
    """Precompute the Michałek-Pazderski reference, and time the computation.

    Reported separately from the closed-loop timings because it is an *offline*
    fit over the whole path: it is not a per-step cost, and presenting it as one
    would misrepresent the method.
    """
    g = MPReferenceGenerator(model, n_harm=n_harm, n_coll=n_coll, homotopy=5)
    t0 = time.perf_counter()
    s, Q = g.reference_states(path, n=n_coll)
    return s, Q, time.perf_counter() - t0, g.resid_rms


def run_case(path_name, t_final, n_harm=32):
    path = make_path(path_name)
    model = g2t()
    s_mp, Q_mp, t_fit, resid = mp_reference(model, path, n_harm=n_harm)
    print(f"  M&P offline fit: {t_fit:.1f} s, residual {resid:.3f}", flush=True)

    out, traces = {}, {}
    for meth in METHODS:
        acc = []
        for sd in SEEDS:
            m = g2t()
            cfg = SimConfig(sigma=SIGMA, t_final=t_final, seed=sd)
            if meth == "proposed":
                r = simulate(m, path, cfg, sensors=sensors_for(m), refgen_weights=W)
            elif meth == "mp2021":
                r = simulate_with_reference(m, path, s_mp, Q_mp, cfg,
                                            sensors=sensors_for(m),
                                            nmpc=TrackingNMPC(m, Nc=20), label="mp2021")
            else:
                r = simulate_baseline(m, path, PurePursuit(m, speed=SIGMA), cfg,
                                      sensors=sensors_for(m))
            acc.append(r)
            if sd == SEEDS[0]:
                traces[meth] = (r, m)
        S = [a.summary() for a in acc]
        mc = np.stack([s["mean_dev_curved"] for s in S])
        out[meth] = {
            "mean_curved": mc.mean(0), "mean_curved_sd": mc.std(0),
            "worst_curved": float(np.mean([s["worst_curved"] for s in S])),
            "last_curved": float(mc.mean(0)[-1]),
            "tractor_curved": float(mc.mean(0)[0]),
            "corridor": float(np.mean([s["corridor_width_curved"] for s in S])),
            "t_online_ms": float(np.mean([s["t_ref_ms"] + s["t_mpc_ms"] for s in S])),
            "fails": int(sum(sum(a.fails.values()) for a in acc)),
        }
        print(f"  {NICE[meth]:34s} per-seg {np.round(out[meth]['mean_curved'],3)} "
              f"worst {out[meth]['worst_curved']:.3f}  corridor {out[meth]['corridor']:.3f} m",
              flush=True)
    out["_mp_fit_s"] = t_fit
    out["_mp_resid"] = resid
    return path, out, traces


def guide_alpha(N, t):
    """Barycentric coefficients for a guidance point at chain position ``t``.

    ``t`` runs from 0 at the tractor to ``N`` at the last trailer and may fall
    between segments, which is the whole point: a guidance point is a virtual
    point, not a choice among the physical ones.
    """
    t = float(np.clip(t, 0.0, N))
    i = min(int(np.floor(t)), N - 1)
    f = t - i
    a = np.zeros(N + 1)
    a[i], a[i + 1] = 1.0 - f, f
    return a


def run_guidance_sweep(path, t_final, ts=GUIDE_TS, seeds=SEEDS):
    """Move a single guidance point along the chain and watch the reference drift.

    This is the guidance-point *objective* evaluated inside our own generator,
    not an implementation of any published controller: a cascaded guidance-point
    controller carries a stabilising inner loop that this formulation has no
    equivalent of, and labelling the curve with an author's name would credit
    them with a failure that is ours.  What it does show, and what no single-case
    comparison can, is that the objective is ill-posed as a function of where the
    point is put -- everywhere except the tractor it leaves the reference free to
    translate, and the reachability constraint slows that drift without removing
    it.

    The guidance point's own deviation is recorded alongside, because it is the
    control: if the point sits on the path while the vehicle does not, the
    objective has been met and is simply the wrong objective.
    """
    rows = []
    for t in ts:
        worst, corr, gdev = [], [], []
        for sd in seeds:
            m = g2t()
            a = guide_alpha(m.N, t)
            w = RefGenWeights(w_path=0.0, w_theta=5.0, w_guide=W_GUIDE, guide_alpha=a)
            r = simulate(m, path, SimConfig(sigma=SIGMA, t_final=t_final, seed=sd),
                         sensors=sensors_for(m), refgen_weights=w)
            s = r.summary()
            worst.append(s["worst_curved"])
            corr.append(s["corridor_width_curved"])
            k0 = int(0.15 * r.t.size)
            P = np.stack([r.qref[2 * m.N + 1 + 2 * i: 2 * m.N + 3 + 2 * i, k0:]
                          for i in range(m.N + 1)])
            g = np.tensordot(a, P, axes=(0, 0))
            gdev.append(float(np.mean([path.deviation(g[:, j])
                                       for j in range(0, g.shape[1], 40)])))
        rows.append(dict(t=float(t), worst=float(np.mean(worst)),
                         worst_sd=float(np.std(worst)),
                         corridor=float(np.mean(corr)), guide_dev=float(np.mean(gdev))))
        print(f"  guidance at t={t:.1f}  worst {rows[-1]['worst']:.3f} m  "
              f"corridor {rows[-1]['corridor']:.3f} m  "
              f"(the point itself: {rows[-1]['guide_dev']:.3f} m)", flush=True)
    return rows


def figure_guidance(rows, proposed_worst, tag=""):
    ts = [r["t"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.errorbar(ts, [r["worst"] for r in rows], yerr=[r["worst_sd"] for r in rows],
                fmt="o-", ms=3.6, color=C_LAST, lw=1.3, capsize=2.5, elinewidth=0.7,
                label="single guidance point")
    ax.axhline(proposed_worst, color=C_MID, ls="--", lw=1.1,
               label="every segment referenced")
    ax.set_xlabel("guidance point along the chain")
    ax.set_ylabel("worst-segment\noff-tracking (m)")
    ax.set_xticks(ts)
    ax.legend(fontsize=6.8)
    ax.set_title("where the point is put decides everything", fontsize=8, loc="left")

    ax = axes[1]
    ax.semilogy(ts, [max(r["guide_dev"], 1e-4) for r in rows], "o-", ms=3.6,
                color=C_TRACTOR, lw=1.3, label="the guidance point itself")
    ax.semilogy(ts, [r["worst"] for r in rows], "o-", ms=3.6, color=C_LAST, lw=1.3,
                label="the worst segment")
    ax.set_xlabel("guidance point along the chain")
    ax.set_ylabel("deviation from\nthe path (m)")
    ax.set_xticks(ts)
    ax.legend(fontsize=6.8)
    ax.set_title("the objective is met; the posture is still wrong", fontsize=8, loc="left")
    save(fig, f"s8_guidance{tag}")


def figure(path, out, traces, tag):
    r0, model = traces["proposed"]
    N = model.N
    cols = seg_colours(N)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.36)

    for j, meth in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, j])
        r, m = traces[meth]
        ax.plot(path.xy[0], path.xy[1], "--", color=C_PATH, lw=0.7, label="Nominal path")
        for i in range(N + 1):
            P = r.q[2 * N + 1 + 2 * i:2 * N + 3 + 2 * i, :]
            ax.plot(P[0], P[1], color=cols[i], lw=0.9, label=seg_label(i, N))
        ax.set_aspect("equal")
        ax.set_title(NICE[meth], fontsize=8)
        ax.set_xlabel("x (m)")
        if j == 0:
            ax.set_ylabel("y (m)")
        if j == len(METHODS) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.2)

    ax = fig.add_subplot(gs[1, :2])
    w = 0.26
    for j, meth in enumerate(METHODS):
        ax.bar(np.arange(N + 1) + (j - 1) * w, out[meth]["mean_curved"], width=w,
               yerr=out[meth]["mean_curved_sd"], capsize=2, error_kw=dict(lw=0.6),
               color=COL[meth], label=NICE[meth])
    ax.set_xticks(np.arange(N + 1))
    ax.set_xticklabels([seg_label(i, N) for i in range(N + 1)])
    ax.set_ylabel("mean off-tracking\nin curved sections (m)")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(METHODS))
    ax.bar(x, [out[m]["corridor"] for m in METHODS], color=[COL[m] for m in METHODS], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in METHODS],
                       rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("corridor width\nrequired (m)")
    save(fig, f"s8_comparison{tag}")


if __name__ == "__main__":
    for pname, tf, nh in CASES:
        tag = "" if pname == "agricultural" else f"_{pname}"
        print(f"[S8] comparison -- {pname} (n_harm={nh})", flush=True)
        path, out, traces = run_case(pname, tf, n_harm=nh)
        dump(out, f"s8_comparison{tag}")
        figure(path, out, traces, tag)
        if pname == "agricultural":
            print("[S8] guidance-point objective, swept along the chain", flush=True)
            grows = run_guidance_sweep(path, tf)
            figure_guidance(grows, out["proposed"]["worst_curved"])
            dump({"sweep": grows, "proposed_worst": out["proposed"]["worst_curved"],
                  "w_guide": W_GUIDE}, "s8_guidance")
