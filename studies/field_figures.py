"""The field figures: trajectories, distributions, timing, and the weighting sweep.

Replaces the two single-trial trajectory plots of the previous version.  A single
run shows what happened once; what a reader needs is the spread over the whole
session and the quantities that carry the argument, so every trial in an
experiment is drawn and the per-segment distributions are shown beside them.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from common import (C_GREY, C_LAST, C_MID, C_PATH, C_TRACTOR, OUT, save,
                    seg_colours, seg_label)
import field_analysis as FA
import field_report as FR


def trials_of(rows, spec, adopted=True):
    kw = dict(Nc=spec["Nc"], Ne=spec["Ne"], w=spec["w"]) if adopted else {}
    return FR.select(rows, spec["day"], spec["path"], spec["L"], **kw)


def _traj(fname):
    tr = FA.load_trial(os.path.join(FA.ROOT, fname))
    N, x = tr["N"], tr["xest"]
    k0 = int(FA.SKIP * x.shape[0])
    return tr, [x[k0:, 2 * N + 1 + 2 * i: 2 * N + 3 + 2 * i] for i in range(N + 1)]


def figure_trajectories(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(wspace=0.05)
    for ax, spec in zip(axes, FR.EXPERIMENTS):
        sel = trials_of(rows, spec)
        cols = seg_colours(2)
        first = True
        for t in sel:
            tr, P = _traj(t["file"])
            if first:
                ax.plot(tr["coords"][0], tr["coords"][1], "--", color=C_PATH, lw=1.0,
                        zorder=5, label="Nominal path")
            for i in range(3):
                ax.plot(P[i][:, 0], P[i][:, 1], color=cols[i], lw=0.55, alpha=0.5,
                        label=(seg_label(i, 2) if first else None), zorder=3)
            first = False
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(f"{spec['label']}: {len(sel)} trials", fontsize=8.5, loc="left")
        # Both panels are given the same span so that a deviation of a given size
        # looks the same size in each; equal aspect alone does not do that when
        # the two paths occupy different extents.
        ax.set_xlim(0.2, 12.8); ax.set_ylim(0.4, 9.0)
        if ax is axes[0]:
            ax.set_ylabel("y (m)")
            ax.legend(fontsize=6.2, loc="lower center", ncol=2, frameon=False,
                      borderpad=0.2, handletextpad=0.5, columnspacing=1.2)
        else:
            ax.tick_params(labelleft=False)
    save(fig, "field_trajectories")


def figure_summary(rows, report):
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    cols = seg_colours(2)

    # --- per-segment spread over the trials of each experiment
    ax = fig.add_subplot(gs[0, 0])
    pos, ticks, labels = 0, [], []
    for spec in FR.EXPERIMENTS:
        sel = trials_of(rows, spec)
        data = [np.array([t["mean_dev"][i] for t in sel]) for i in range(3)]
        bp = ax.boxplot(data, positions=[pos + 1, pos + 2, pos + 3], widths=0.62,
                        patch_artist=True, medianprops=dict(color="white", lw=1.1),
                        flierprops=dict(marker=".", ms=3, mfc=C_GREY, mec="none"))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("none")
        ticks.append(pos + 2); labels.append(spec["label"])
        pos += 4
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("off-tracking of the\nestimated posture (m)")
    ax.set_title("per segment, over trials", fontsize=8, loc="left")
    for i in range(3):
        ax.plot([], [], "s", color=cols[i], ms=5, label=seg_label(i, 2))
    ax.legend(fontsize=6.2, loc="upper left")

    # --- the weighting sweep, the field counterpart of S6
    ax = fig.add_subplot(gs[0, 1])
    pts = report["E3"]["points"]
    w = [p["w"] for p in pts]
    for i in range(3):
        ax.plot(w, [p["mean_dev"][i] for p in pts], "o-", ms=3.6, color=cols[i],
                lw=1.2, label=seg_label(i, 2))
    for p in pts:
        ax.annotate(f"n={p['n']}", (p["w"], max(p["mean_dev"])), fontsize=6,
                    textcoords="offset points", xytext=(0, 6), ha="center", color=C_GREY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(w); ax.set_xticklabels([f"{v:g}" for v in w])
    ax.set_xlabel(r"weight on the last trailer, $w_N/w_0$")
    ax.set_ylabel("off-tracking (m)")
    ax.set_title("weighting it more makes it worse", fontsize=8, loc="left")
    ax.legend(fontsize=6.2)

    # --- where the sampling period goes
    ax = fig.add_subplot(gs[0, 2])
    names = [s["label"] for s in FR.EXPERIMENTS]
    # The three optimisation stages do not account for the whole step: sensing,
    # the obstacle check and bookkeeping take the rest.  The remainder is shown
    # rather than dropped, so that the bar can honestly be compared with Ts.
    stages = [("t_pfa", "reference", C_TRACTOR), ("t_mhe", "estimator", C_MID),
              ("t_mpc", "tracking", C_LAST)]
    bottom = np.zeros(len(names))
    for key, lab, c in stages:
        vals = np.array([report[s["key"]][key] for s in FR.EXPERIMENTS])
        ax.bar(names, vals, bottom=bottom, color=c, width=0.55, label=lab)
        bottom += vals
    tot = np.array([report[s["key"]]["t_tot"] for s in FR.EXPERIMENTS])
    ax.bar(names, np.maximum(tot - bottom, 0.0), bottom=bottom, color=C_GREY,
           width=0.55, label="sensing, other", alpha=0.55)

    Ts = report[FR.EXPERIMENTS[0]["key"]]["Ts_ms"]
    ax.axhline(Ts, color=C_PATH, ls="--", lw=1.1)
    ax.annotate(f"sampling period {Ts:.0f} ms", (0.5, Ts), xycoords=("axes fraction", "data"),
                fontsize=6.4, ha="center", va="bottom", color=C_PATH)
    # Reference generation is far too small a slice to see; state it instead.
    pfa = np.mean([report[s["key"]]["t_pfa"] for s in FR.EXPERIMENTS])
    ax.annotate(f"reference generation\n{pfa:.1f} ms ({100*pfa/tot.mean():.1f}%)",
                (0.03, 0.44), xycoords="axes fraction", fontsize=6.2,
                color=C_TRACTOR, ha="left", va="center")
    ax.set_ylabel("solve time per step (ms)")
    ax.set_ylim(0, max(Ts, tot.max()) * 1.22)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.legend(fontsize=6.0, loc="upper right", ncol=2, columnspacing=0.9,
              handletextpad=0.4, borderpad=0.25)
    ax.set_title("measured on the vehicle", fontsize=8, loc="left")
    save(fig, "field_summary")


if __name__ == "__main__":
    rows = FR.load()
    report = json.load(open(os.path.join(OUT, "field_report.json")))
    figure_trajectories(rows)
    figure_summary(rows, report)
