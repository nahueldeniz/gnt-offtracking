"""S6 -- the corridor swept by the vehicle against the distribution of the weight.

The quantity reported is the width of the lane the whole chain sweeps over the
curved sections: the span between the left-most and the right-most excursion of
any segment.  The position weights of the generator are tilted along the chain
by one parameter, kappa (towards the tractor for kappa < 0, towards the last
trailer for kappa > 0), at constant total weight.

Three views:

    corridor  -- swept width against kappa.
    profile   -- the mean off-tracking of each segment against kappa.
    frontier  -- corridor width against worst-segment off-tracking, with the
                 comparison methods plotted as the single points they occupy.

On this path the data show that weighting the last trailer more heavily does
not reduce its off-tracking; the study reports that as found and does not
explain it.

The vehicle and path are those of the field platform on the agricultural
geometry.  ``--replot`` redraws the figure from out/ without re-running.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import (C_GREY, C_LAST, C_MID, C_TRACTOR, OUT, dump, g2t, save, seg_colours, seg_label,
                    study_path, t_final_for)

from gntpf import RefGenWeights, SensorModel, SimConfig, make_path, simulate
from parallel import pmap
from gntpf.baselines import PurePursuit, simulate_baseline, simulate_with_reference
from gntpf.mp_refgen import MPReferenceGenerator

PATH = "agricultural"
SIGMA = 1.0
SEEDS = (0, 1, 2)
W_BASE = 20.0
KAPPAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


def sensors_for(model):
    return SensorModel(sigma=np.concatenate(
        [np.deg2rad(1.0) * np.ones(model.N), [np.deg2rad(0.2)], [0.025, 0.025]]))


def tilt_weights(N, kappa, w_base=W_BASE):
    """A weight profile tilted along the chain at *constant total weight*.

    ``kappa = 0`` weights every segment equally; positive kappa shifts emphasis
    towards the last trailer, negative towards the tractor.

    Two properties are deliberate and neither is cosmetic.

    First, the profile is **normalised so that the mean weight is ``w_base``
    regardless of kappa**.  Without this the knob is confounded: adding emphasis
    on top of a fixed floor raises the total weight on path proximity as |kappa|
    grows, so every tilted setting beats the untilted one simply because it
    presses harder on the same objective, and the study measures "more weight is
    better" rather than "where the weight goes".  The claim being tested is about
    distribution, so the magnitude must be held fixed.

    Second, the profile keeps a floor.  Letting any weight fall towards zero --
    in particular the tractor's -- leaves the reference generator with a null
    space: many vehicle postures place the weighted segments identically, nothing
    selects between them, and the whole reference drifts (Section S1).  The
    normalisation above bounds the smallest weight below by
    ``w_base / (1 + |kappa|)``, which for the range swept here stays well clear
    of the degenerate regime.
    """
    i = np.arange(N + 1) / N
    lean = i if kappa >= 0 else (1.0 - i)
    p = 1.0 + abs(kappa) * lean
    return w_base * (N + 1) * p / p.sum()


def _one_tilt(path_name, kappa, seed):
    m = g2t()
    path = study_path(path_name, m)
    res = simulate(m, path, SimConfig(sigma=SIGMA, t_final=t_final_for(path, SIGMA), seed=seed),
                   sensors=sensors_for(m),
                   refgen_weights=RefGenWeights(w_path=tilt_weights(m.N, kappa), w_theta=5.0))
    s = res.summary()
    return (s["mean_dev_curved"], s["corridor_width_curved"], s["worst_curved"], s["completed"],
            sum(res.fails.values()))


def run_sweep(path, kappas=KAPPAS, seeds=SEEDS, path_name=PATH):
    jobs = [(path_name, k, sd) for k in kappas for sd in seeds]
    done = pmap(_one_tilt, jobs)
    rows = []
    for i, k in enumerate(kappas):
        chunk = done[i * len(seeds):(i + 1) * len(seeds)]
        n_div = sum(not c[3] for c in chunk)
        fails = int(sum(c[4] for c in chunk))
        chunk = [c for c in chunk if c[3]]
        if not chunk:
            rows.append(dict(kappa=float(k), n_diverged=int(n_div), fails=fails))
            print(f"  kappa {k:+5.1f}  no run completed", flush=True)
            continue
        P = np.stack([c[0] for c in chunk])
        corr = [c[1] for c in chunk]
        worst = [c[2] for c in chunk]
        rows.append(dict(kappa=float(k), n_diverged=int(n_div), fails=fails,
                         per_seg=P.mean(0), per_seg_sd=P.std(0),
                         tractor=float(P.mean(0)[0]), last=float(P.mean(0)[-1]),
                         worst=float(np.mean(worst)), worst_sd=float(np.std(worst)),
                         corridor=float(np.mean(corr)), corridor_sd=float(np.std(corr))))
        print(f"  kappa {k:+5.1f}  per-seg {np.round(P.mean(0), 3)}  "
              f"worst {rows[-1]['worst']:.3f}  corridor {rows[-1]['corridor']:.3f} m", flush=True)
    return rows


def run_reference_points(path, seeds=SEEDS):
    """The comparison methods, as single points on the same axes."""
    out = {}
    m0 = g2t()
    # Same fit settings as S8 on this path, so that the corridor quoted for this
    # method is the same number in both studies.  A coarser basis gives a
    # different reference and therefore a different corridor, and two values for
    # one method on one path is the kind of inconsistency a careful reader
    # rightly distrusts.
    g = MPReferenceGenerator(m0, n_harm=64, n_coll=600, homotopy=5)
    s_mp, Q_mp = g.reference_states(path, n=600)
    for tag in ("pursuit", "mp2021"):
        corr, worst, n_div = [], [], 0
        for sd in seeds:
            m = g2t()
            cfg = SimConfig(sigma=SIGMA, t_final=t_final_for(path, SIGMA), seed=sd)
            if tag == "pursuit":
                r = simulate_baseline(m, path, PurePursuit(m, speed=SIGMA), cfg,
                                      sensors=sensors_for(m))
            else:
                r = simulate_with_reference(m, path, s_mp, Q_mp, cfg,
                                            sensors=sensors_for(m), label=tag)
            s = r.summary()
            if not s["completed"]:
                n_div += 1
                continue
            corr.append(s["corridor_width_curved"])
            worst.append(s["worst_curved"])
        out[tag] = dict(corridor=float(np.mean(corr)) if corr else float("nan"),
                        worst=float(np.mean(worst)) if worst else float("nan"),
                        n_diverged=n_div)
        print(f"  {tag:10s} corridor {out[tag]['corridor']:.3f} m  worst {out[tag]['worst']:.3f} m",
              flush=True)
    return out


def figure(rows, refs):
    N = 2
    cols = seg_colours(N)
    kap = np.array([r["kappa"] for r in rows])
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, wspace=0.42)

    # --- the headline: corridor width follows the knob
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(kap, [r["corridor"] for r in rows], yerr=[r["corridor_sd"] for r in rows],
                fmt="o-", ms=3.4, color=C_LAST, lw=1.3, capsize=2.5, elinewidth=0.7)
    ax.set_xlabel(r"weight tilt $\kappa$")
    ax.set_ylabel("corridor width\nrequired (m)")
    ax.set_title("corridor width", fontsize=8, loc="left")

    # --- the mechanism
    ax = fig.add_subplot(gs[0, 1])
    for i in range(N + 1):
        ax.errorbar(kap, [r["per_seg"][i] for r in rows], yerr=[r["per_seg_sd"][i] for r in rows],
                    fmt="o-", ms=3, color=cols[i], lw=1.1, capsize=2, elinewidth=0.6,
                    label=seg_label(i, N))
    ax.set_xlabel(r"weight tilt $\kappa$")
    ax.set_ylabel("off-tracking in\ncurved sections (m)")
    ax.legend(fontsize=6.5)
    ax.set_title("off-tracking per segment", fontsize=8, loc="left")

    # --- the reachable design trade-off
    ax = fig.add_subplot(gs[0, 2])
    w = [r["worst"] for r in rows]
    c = [r["corridor"] for r in rows]
    ax.plot(w, c, "-", color=C_GREY, lw=1.0, zorder=1)
    sc = ax.scatter(w, c, c=kap, cmap="viridis", s=24, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\kappa$", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    # The comparison methods are annotated rather than put in a legend: they sit
    # in the middle of the panel, where any legend box would cover one of them.
    mark = {"pursuit": ("s", "#7C8B99", "pure\npursuit", (-7, 2), "right"),
            "mp2021": ("^", "#4E7CB0", "Micha\u0142ek &\nPazderski", (-4, -8), "left")}
    for tag, (mk, col, lab, off, ha) in mark.items():
        if tag in refs:
            ax.plot(refs[tag]["worst"], refs[tag]["corridor"], mk, color=col, ms=6, zorder=4)
            ax.annotate(lab, (refs[tag]["worst"], refs[tag]["corridor"]),
                        textcoords="offset points", xytext=off, fontsize=6.2,
                        color=col, ha=ha, va="top")
    i0 = int(np.argmin(np.abs(kap)))           # kappa = 0: the even, proposed setting
    ax.annotate(r"even ($\kappa=0$)", (w[i0], c[i0]), textcoords="offset points",
                xytext=(16, -18), fontsize=6.2, color="#444444", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#444444", shrinkA=0, shrinkB=2))
    ax.set_xlabel("worst-segment\noff-tracking (m)")
    ax.set_ylabel("corridor width (m)")
    # Leave headroom: the comparison points sit well outside the reachable set,
    # and the default limits clip them against the axes.
    xs = w + [refs[t]["worst"] for t in refs]
    ys = c + [refs[t]["corridor"] for t in refs]
    mx, my = 0.10 * (max(xs) - min(xs)), 0.16 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - mx, max(xs) + mx)
    ax.set_ylim(min(ys) - my, max(ys) + my)
    ax.set_title("worst segment and corridor", fontsize=8, loc="left")
    save(fig, "s6_corridor_design")


if __name__ == "__main__":
    if "--replot" in sys.argv:                  # redraw from out/ without re-running
        with open(os.path.join(OUT, "s6_corridor_design.json")) as fh:
            d = json.load(fh)
        figure(d["tilt_sweep"], d["reference_points"])
        sys.exit(0)
    print("[S6] corridor as the design variable", flush=True)
    path = study_path(PATH, g2t())
    rows = run_sweep(path)
    refs = run_reference_points(path)
    dump({"tilt_sweep": rows, "reference_points": refs, "path": PATH, "w_base": W_BASE},
         "s6_corridor_design")
    figure(rows, refs)
