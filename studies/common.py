"""Shared configuration, vehicle geometries and plotting style for the studies."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gntpf import GNT, make_path  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Vehicle geometries
# --------------------------------------------------------------------------- #
# The platform used in the field experiments: Husky A200 + two passive trailers.
G2T = dict(Lh=[0.342, 0.0], L=[1.08, 0.78])

# Extended chain for the scalability study.  The first two joints reproduce the
# field platform; beyond that the hitch offsets alternate in sign so that every
# configuration stays Generalised rather than degenerating to Standard.
LH_LONG = [0.342, 0.0, 0.15, -0.15, 0.0, -0.342, 0.342, 0.0, 0.15, -0.15, 0.0, -0.342, 0.2, -0.2, 0.0]
L_LONG = [1.08, 0.78, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]


def vehicle(N: int, Ts: float = 0.05) -> GNT:
    return GNT(N=N, Lh=LH_LONG[:N], L=L_LONG[:N], Ts=Ts)


def g2t(Ts: float = 0.05) -> GNT:
    return GNT(N=2, Lh=G2T["Lh"], L=G2T["L"], Ts=Ts)


# --------------------------------------------------------------------------- #
#  Paths as the studies use them
# --------------------------------------------------------------------------- #
# Closed paths are traversed this many times; open paths once.
LAPS = 2


def study_path(name: str, model: GNT, **kw):
    """A path with a straight lead-in, the nominal span, and a run-out.

    The lead-in absorbs the start-up transient before the span begins.  The
    span is ``LAPS`` laps of a closed path or one traversal of an open one.
    The run-out lets the last trailer finish the span before the progress point
    reaches the end of the path, which is where a run stops.  Every segment is
    evaluated only while its own projection lies inside the span.
    """
    from gntpf.paths import CLOSED_PATHS
    ell = model.total_length
    laps = LAPS if name in CLOSED_PATHS else 1
    return make_path(name, laps=laps, lead=1.6 * ell + 6.0, runout=ell + 2.0, **kw)


def t_final_for(path, sigma: float, margin: float = 20.0) -> float:
    """A time cap that the progress point cannot reach before the path end."""
    return path.length / sigma + margin


# --------------------------------------------------------------------------- #
#  Plot style -- consistent with the colour convention of the existing figures
# --------------------------------------------------------------------------- #
C_TRACTOR = "#B8800C"
C_MID = "#1F5FA8"
C_LAST = "#A8372B"
C_PATH = "#333333"
C_REF = "#6E39A0"
C_GREY = "#8A96A1"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.2,
    "legend.frameon": False,
})


def seg_colours(N: int):
    """Tractor amber, last trailer red, intermediate trailers graded blue."""
    if N == 0:
        return [C_TRACTOR]
    out = [C_TRACTOR]
    for i in range(1, N):
        f = i / max(N - 1, 1)
        out.append(plt.matplotlib.colors.to_hex(
            np.array(plt.matplotlib.colors.to_rgb(C_MID)) * (1 - 0.45 * f)
            + np.array([0.55, 0.68, 0.85]) * (0.45 * f)))
    out.append(C_LAST)
    return out


def seg_label(i: int, N: int) -> str:
    return "Tractor" if i == 0 else (f"Trailer {i}" if i < N else f"Trailer {i} (last)")


def _writable(path: str) -> str:
    """Make an existing output file writable before overwriting it.

    Files unpacked from an archive can arrive read-only, and a study that has
    just spent half an hour computing should not die on the write.
    """
    if os.path.exists(path) and not os.access(path, os.W_OK):
        os.chmod(path, 0o644)
    return path


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(_writable(os.path.join(FIGS, f"{name}.{ext}")))
    plt.close(fig)
    print(f"    figure -> figs/{name}.pdf", flush=True)


def dump(obj, name: str):
    path = _writable(os.path.join(OUT, f"{name}.json"))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    print(f"    data   -> out/{name}.json", flush=True)


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def draw_vehicle(ax, model, q, colour="#444444", lw=1.0, alpha=1.0, wheel=0.16):
    """Sketch the chain: hitch links, segment bodies and axles."""
    P = model.positions(q)
    th = np.asarray(q)[model.i_theta]
    ax.plot(P[:, 0], P[:, 1], "-", color=colour, lw=lw, alpha=alpha, zorder=5)
    for i in range(P.shape[0]):
        n = np.array([-np.sin(th[i]), np.cos(th[i])]) * wheel
        ax.plot([P[i, 0] - n[0], P[i, 0] + n[0]], [P[i, 1] - n[1], P[i, 1] + n[1]],
                "-", color=colour, lw=lw * 1.6, alpha=alpha, zorder=6)
    ax.plot(P[0, 0], P[0, 1], "o", color=colour, ms=2.6, alpha=alpha, zorder=7)
