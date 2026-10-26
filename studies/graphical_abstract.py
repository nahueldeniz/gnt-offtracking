"""The graphical abstract: 660 x 295 pixels at 300 dpi, as the IEEE asks.

Drawn from the comparison study (S8) on the agricultural path: the first
realisation of the reconstruction of Michalek and Pazderski and of the proposed
scheme, and the corridor each of the three methods requires.  Nothing in it is
computed here; run ``s8_comparison.py`` first.  Text is kept to labels, in a
sans-serif face at 16 pixels or more.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from common import C_PATH, FIGS, OUT, g2t, paper_style, seg_colours, seg_label, study_path
from s8_comparison import COL, load_traces

W_PX, H_PX, DPI = 660, 295, 300
PX = 72.0 / DPI                     # one pixel in points
INK = "#1B2027"


def main():
    with open(os.path.join(OUT, "s8_comparison.json")) as fh:
        out = json.load(fh)
    T = load_traces()
    path = study_path("agricultural", g2t())
    cols = seg_colours(2)
    fs = 18 * PX                    # 18 px labels; nothing below 16 px

    with paper_style(**{"font.size": fs, "axes.labelsize": fs, "axes.titlesize": 20 * PX,
                        "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": fs,
                        "axes.linewidth": 0.4, "xtick.major.width": 0.4,
                        "ytick.major.width": 0.4, "xtick.major.size": 1.5,
                        "ytick.major.size": 1.5, "xtick.major.pad": 1.0, "ytick.major.pad": 1.0,
                        "axes.labelpad": 1.0, "axes.titlepad": 2.0, "axes.edgecolor": INK,
                        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                        "ytick.color": INK}):
        fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)

        # --- left: the two references, tracked by the same controller
        k = path.xy
        for row, (meth, title) in enumerate([("mp2021", "Reconstruction of Micha\u0142ek & Pazderski"),
                                             ("proposed", "Proposed: every segment referenced")]):
            ax = fig.add_axes([0.005, 0.535 - 0.45 * row, 0.50, 0.355])
            ax.plot(k[0], k[1], "--", color=C_PATH, lw=0.35, dashes=(2.5, 1.5))
            P = T[meth]
            for i in range(P.shape[0]):
                ax.plot(P[i, 0], P[i, 1], color=cols[i], lw=0.45)
            ax.set_aspect("equal")
            ax.set_xlim(-6.5, 13.0); ax.set_ylim(-7.6, 1.6)
            ax.axis("off")
            ax.set_title(title, loc="left", fontsize=fs, color=INK, pad=1.5)

        # legend of the segments, under the two panels
        hs = [plt.Line2D([], [], color=c, lw=0.9) for c in cols]
        hs.append(plt.Line2D([], [], color=C_PATH, lw=0.5, ls="--", dashes=(2.5, 1.5)))
        fig.legend(hs, [seg_label(i, 2).replace(" (last)", "") for i in range(3)] + ["path"],
                   loc="lower left", bbox_to_anchor=(0.005, -0.01), ncol=4, frameon=False,
                   handlelength=1.1, handletextpad=0.3, columnspacing=0.7, borderaxespad=0.0)

        # --- right: the corridor each method requires, as horizontal bars so
        # that the method names can be written in full
        ax = fig.add_axes([0.715, 0.17, 0.205, 0.66])
        meths = ["pursuit", "mp2021", "proposed"]
        lab = ["Pure pursuit", "Micha\u0142ek &\nPazderski", "Proposed"]
        v = [out[m]["corridor"] for m in meths]
        y = np.arange(3)[::-1]
        ax.barh(y, v, color=[COL[m] for m in meths], height=0.62)
        for yy, val in zip(y[:2], v[:2]):
            ax.text(val + 0.03, yy, f"{val:.2f}", ha="left", va="center", fontsize=fs)
        gain = 100.0 * (v[1] - v[2]) / v[1]
        ax.text(v[2] + 0.03, y[2], f"{v[2]:.2f}", ha="left", va="center", fontsize=fs)
        ax.text(v[2] + 0.03, y[2] - 0.62, f"\u2212{gain:.0f} %", ha="left", va="center",
                fontsize=20 * PX, fontweight="bold", color=COL["proposed"])
        ax.set_yticks(y); ax.set_yticklabels(lab)
        ax.set_xlim(0, 1.45); ax.set_xticks([0, 0.5, 1.0])
        ax.set_ylim(-0.95, 2.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.set_xlabel("corridor swept in turns (m)")

        os.makedirs(FIGS, exist_ok=True)
        fn = os.path.join(FIGS, "gagraphic.png")
        fig.savefig(fn, dpi=DPI)
        plt.close(fig)
    # The IEEE recommends less than 45 KB: an adaptive palette of 64 colours
    # keeps the antialiasing and brings the file well under that.
    from PIL import Image
    im = Image.open(fn).convert("RGB")
    im.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        fn, optimize=True, dpi=(DPI, DPI))
    print(f"    graphical abstract -> figs/gagraphic.png ({os.path.getsize(fn) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
