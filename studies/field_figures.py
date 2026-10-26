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

from common import (C_GREY, C_LAST, C_MID, C_PATH, C_TRACTOR, COL_W, OUT, paper_style, save,
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


def figure_trajectories(rows, report):
    """One column wide, drawn at its printed size."""
    cols = seg_colours(2)
    with paper_style():
        fig = plt.figure(figsize=(COL_W, 2.0), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.01, wspace=0.03)
        axes = fig.subplots(1, 2, sharey=True)
        # The adopted configuration of each experiment is the one field_report.py
        # selected, so the specification is read from its output.
        for ax, spec in zip(axes, [report[e["key"]] for e in FR.EXPERIMENTS]):
            sel = trials_of(rows, spec)
            first = True
            for t in sel:
                tr, P = _traj(t["file"])
                if first:
                    ax.plot(tr["coords"][0], tr["coords"][1], "--", color=C_PATH, lw=0.8,
                            zorder=5, label="Nominal path")
                for i in range(3):
                    ax.plot(P[i][:, 0], P[i][:, 1], color=cols[i], lw=0.45, alpha=0.5,
                            label=(seg_label(i, 2) if first else None), zorder=3)
                first = False
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_title(f"{spec['label']}, {len(sel)} trials", loc="left")
            # Both panels are given the same span so that a deviation of a given size
            # looks the same size in each; equal aspect alone does not do that when
            # the two paths occupy different extents.
            ax.set_xlim(0.2, 12.8); ax.set_ylim(0.4, 9.0)
            ax.set_xticks([0, 4, 8, 12])
        axes[0].set_ylabel("y (m)")
        handles, labels = axes[0].get_legend_handles_labels()
        # opaque legend lines, whatever the alpha of the traces
        handles = [plt.Line2D([], [], color=h.get_color(), ls=h.get_linestyle(), lw=1.0)
                   for h in handles]
        fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False,
                   handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
        save(fig, "field_trajectories")


def figure_summary(rows, report):
    """One column wide, drawn at its printed size."""
    cols = seg_colours(2)
    exps = [report[e["key"]] for e in FR.EXPERIMENTS]
    with paper_style():
        fig = plt.figure(figsize=(COL_W, 3.0), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], width_ratios=[1.3, 1.0])

        # --- (a) per-segment spread over the trials of each experiment
        ax = fig.add_subplot(gs[0, 0])
        pos, ticks, labels = 0, [], []
        for spec in exps:
            sel = trials_of(rows, spec)
            data = [np.array([t["mean_dev"][i] for t in sel]) for i in range(3)]
            bp = ax.boxplot(data, positions=[pos + 1, pos + 2, pos + 3], widths=0.66,
                            patch_artist=True, medianprops=dict(color="white", lw=1.0),
                            whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                            flierprops=dict(marker=".", ms=3, mfc=C_GREY, mec="none"))
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c); patch.set_edgecolor("none")
            ticks.append(pos + 2); labels.append(spec["label"])
            pos += 4.4
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        ax.set_xlim(0.3, pos - 0.7)
        ax.set_ylabel("mean off-tracking (m)")
        ax.set_title("(a)", loc="left")
        ax.set_ylim(0, None)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75])
        ax_a = ax

        # --- (b) the weighting sweep, the field counterpart of S6
        ax = fig.add_subplot(gs[0, 1], sharey=ax_a)
        pts = report["E3"]["points"]
        w = [p["w"] for p in pts]
        for i in range(3):
            ax.plot(w, [p["mean_dev"][i] for p in pts], "o-", ms=3, color=cols[i], lw=1.0,
                    label=seg_label(i, 2))
        # The number of trials at each setting goes under its tick, clear of the data.
        ax.set_xscale("log", base=2)
        ax.set_xticks(w); ax.set_xticklabels([f"{p['w']:g}" for p in pts])
        ax.minorticks_off()
        for p in pts:
            ax.annotate(f"n={p['n']}", (p["w"], 0.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        color="#444444")
        ax.set_xlim(w[0] / 1.35, w[-1] * 1.35)
        ax.set_xlabel("weight of the last\ntrailer / tractor")
        ax.set_title("(b)", loc="left")
        ax.tick_params(labelleft=False)
        ax_a.set_ylim(0, 1.08 * max(ax_a.get_ylim()[1], max(max(p["mean_dev"]) for p in pts)))
        handles, labels = ax.get_legend_handles_labels()

        # --- (c) where the sampling period goes, as horizontal stacked bars
        ax = fig.add_subplot(gs[1, :])
        names = [s["label"] for s in exps]
        y = np.arange(len(names))[::-1]
        # The three optimisation stages do not account for the whole step: the ROS
        # interface and logging take the rest.  The remainder is shown rather than
        # dropped, so that the bar can be compared with Ts.  Reference generation is
        # too small a slice to see; the caption gives its cost.
        pfa = np.mean([s["t_pfa"] for s in exps])
        stages = [("t_pfa", f"reference ({pfa:.1f} ms)", C_TRACTOR), ("t_mhe", "estimator", C_MID),
                  ("t_mpc", "tracking", C_LAST)]
        left = np.zeros(len(names))
        bh, bl = [], []
        for key, lab, c in stages:
            vals = np.array([s[key] for s in exps])
            b = ax.barh(y, vals, left=left, color=c, height=0.55, label=lab)
            left += vals
            if key != "t_pfa":
                bh.append(b); bl.append(lab)
        tot = np.array([s["t_tot"] for s in exps])
        b = ax.barh(y, np.maximum(tot - left, 0.0), left=left, color=C_GREY, height=0.55,
                    alpha=0.55, label="other")
        bh.append(b); bl.append("other")
        Ts = exps[0]["Ts_ms"]
        ax.axvline(Ts, color=C_PATH, ls="--", lw=0.9)
        ax.annotate(f"$T_s$ = {Ts:.0f} ms", (Ts, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(-3, -1), textcoords="offset points", ha="right", va="top",
                    color=C_PATH)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlabel("time per step (ms)")
        ax.set_xlim(0, 1.04 * max(Ts, tot.max()))
        ax.set_ylim(-0.6, len(names) - 0.15)
        ax.grid(axis="y", visible=False)
        ax.set_title("(c)", loc="left")
        ax.legend(bh, bl, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                  handlelength=1.0, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)
        fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False,
                   handlelength=1.4, columnspacing=1.0)
        save(fig, "field_summary")


if __name__ == "__main__":
    rows = FR.load()
    report = json.load(open(os.path.join(OUT, "field_report.json")))
    figure_trajectories(rows, report)
    figure_summary(rows, report)
