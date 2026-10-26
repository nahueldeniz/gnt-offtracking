"""Schematic figures: the control architecture and the vehicle geometry."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc

from common import C_LAST, C_MID, C_TRACTOR, save, seg_colours

INK = "#1B2027"
SOFT = "#6B7883"
FAINT = "#AEB7BF"
BOX = "#F2F4F6"


# --------------------------------------------------------------------------- #
def architecture():
    """Block diagram of the two-stage architecture, drawn at the full text width
    and at its printed size: the data coordinates are inches, so every font
    prints at its nominal size.

    The boxes are sized from the rendered extent of their own text rather than
    from guessed widths, and the gaps between them are what remains of the text
    width, so a longer label widens its box instead of spilling out of it.
    """
    from common import TEXT_W, paper_style

    W, H = TEXT_W, 2.04
    FT, FS = 9.5, 9.0                    # box titles, everything else
    PAD_X, PAD_Y, MARGIN = 0.12, 0.085, 0.04
    ROW, EST = 1.35, 0.40

    with paper_style():
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        boxes = {}

        def texts(cx, cy, title, sub):
            t = ax.text(cx, cy, title, ha="center", va="bottom", fontsize=FT,
                        fontweight="bold", color=INK, zorder=4)
            u = ax.text(cx, cy - 0.05, sub, ha="center", va="top", fontsize=FS,
                        color=SOFT, zorder=4, linespacing=1.3)
            return t, u

        def extent(objs):
            fig.canvas.draw()
            inv = ax.transData.inverted()
            ext = [inv.transform(o.get_window_extent()) for o in objs]
            return (min(e[0][0] for e in ext), min(e[0][1] for e in ext),
                    max(e[1][0] for e in ext), max(e[1][1] for e in ext))

        def width(title, sub):
            objs = texts(0.0, 0.0, title, sub)
            x0, _, x1, _ = extent(objs)
            for o in objs:
                o.remove()
            return x1 - x0 + 2 * PAD_X

        def block(key, cx, cy, title, sub):
            """A titled box centred on (cx, cy), sized to its own text."""
            t, u = texts(cx, cy, title, sub)
            x0, y0, x1, y1 = extent((t, u))
            # centre the pair of lines vertically on cy
            dy = cy - 0.5 * (y0 + y1)
            for o in (t, u):
                o.set_y(o.get_position()[1] + dy)
            y0, y1 = y0 + dy, y1 + dy
            half = max(cx - x0, x1 - cx) + PAD_X
            x0, x1 = cx - half, cx + half
            y0, y1 = y0 - PAD_Y, y1 + PAD_Y
            ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                        boxstyle="round,pad=0.0,rounding_size=0.05",
                                        linewidth=1.0, edgecolor=INK, facecolor=BOX,
                                        zorder=3))
            boxes[key] = (x0, y0, x1, y1)
            return boxes[key]

        def arrow(p0, p1, col=INK, ls="-", lw=1.0, head=True):
            """A signal leg; only the last leg of a multi-segment route carries a head."""
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-",
                                         mutation_scale=9, linewidth=lw, color=col,
                                         linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))

        def label(x, y, txt, ha="center", va="bottom"):
            ax.text(x, y, txt, ha=ha, va=va, fontsize=FS, color=SOFT, zorder=5,
                    linespacing=1.2)

        top = [("path", "Nominal path", "local polynomial\nwindow, advancing at $\\sigma$"),
               ("ref", "Reference generator", "geometry exact,\nreachable in one step"),
               ("mpc", "Tracking NMPC", "horizon $N_c$,\nno terminal set"),
               ("veh", "GNT vehicle", "$N$ passive\ntrailers")]
        widths = [width(t, s) for _, t, s in top]
        # what is left of the text width goes to the three gaps, the first one,
        # which carries the widest signal label, taking the largest share
        share = np.array([1.25, 1.0, 0.8])
        gaps = share / share.sum() * (W - 2 * MARGIN - sum(widths))
        x = MARGIN
        for (key, t, s), w, g in zip(top, widths, list(gaps) + [0.0]):
            block(key, x + 0.5 * w, ROW, t, s)
            x += w + g

        # the estimator sits under the two problems it serves
        cx_est = 0.5 * (0.5 * (boxes["ref"][0] + boxes["ref"][2])
                        + 0.5 * (boxes["mpc"][0] + boxes["mpc"][2])) + 0.25
        block("est", cx_est, EST, "Moving horizon estimator", "window $N_e$, constrained")

        # the dashed group around the two problems solved every period
        gx0 = boxes["ref"][0] - 0.06
        gx1 = boxes["mpc"][2] + 0.06
        gy0 = min(boxes["ref"][1], boxes["mpc"][1]) - 0.10
        gy1 = max(boxes["ref"][3], boxes["mpc"][3]) + 0.10
        ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                    boxstyle="round,pad=0.0,rounding_size=0.05",
                                    linewidth=0.9, edgecolor=SOFT, facecolor="none",
                                    linestyle=(0, (4, 3)), zorder=1))
        ax.text(0.5 * (gx0 + gx1), gy1 + 0.04,
                "solved in sequence, once per sampling period",
                ha="center", va="bottom", fontsize=FS, color=SOFT, style="italic")

        # ------------------------------------------------------------ signals
        def mid(a, b):
            return 0.5 * (boxes[a][2] + boxes[b][0])

        # the labels of the legs that cross the dashed group are centred on the
        # part of the leg outside it
        arrow((boxes["path"][2], ROW), (boxes["ref"][0], ROW))
        label(0.5 * (boxes["path"][2] + gx0), ROW + 0.05, "$X(s)$,\n$Y(s)$")
        arrow((boxes["ref"][2], ROW), (boxes["mpc"][0], ROW))
        label(mid("ref", "mpc"), ROW + 0.05, "$q^{\\mathrm{ref}}$")
        arrow((boxes["mpc"][2], ROW), (boxes["veh"][0], ROW))
        label(0.5 * (gx1 + boxes["veh"][0]), ROW + 0.05, "$u_k$")

        # measurements: down the right-hand side and back into the estimator
        x_veh = 0.5 * (boxes["veh"][0] + boxes["veh"][2])
        arrow((x_veh, boxes["veh"][1]), (x_veh, EST), col=SOFT, head=False)
        arrow((x_veh, EST), (boxes["est"][2], EST), col=SOFT)
        ax.text(0.5 * (x_veh + boxes["est"][2]), EST - 0.06,
                "$r_k$: tractor pose,\njoint angles", ha="center", va="top",
                fontsize=FS, color=SOFT, linespacing=1.2)

        # the estimate, up into the tracking problem and across to the generator
        x_up = 0.5 * (boxes["mpc"][0] + boxes["mpc"][2])
        arrow((x_up, boxes["est"][3]), (x_up, boxes["mpc"][1]), col=SOFT)
        ax.text(x_up + 0.05, 0.5 * (boxes["est"][3] + gy0),
                "$\\hat q_{k|k}$", ha="left", va="center", fontsize=FS, color=SOFT)

        x_back = 0.5 * (boxes["ref"][0] + boxes["ref"][2])
        arrow((boxes["est"][0], EST), (x_back, EST), col=SOFT, ls=(0, (3, 2)),
              head=False)
        arrow((x_back, EST), (x_back, boxes["ref"][1]), col=SOFT, ls=(0, (3, 2)))
        save(fig, "architecture")


# --------------------------------------------------------------------------- #
def _unit(t):
    return np.array([np.cos(t), np.sin(t)])


def _norm(t):
    return np.array([-np.sin(t), np.cos(t)])


def geometry():
    """See ``_geometry``: drawn in the paper's typeface and cropped to the drawing,
    which is included at the column width (a scale of about 1.1, so the smallest
    label prints at about 9 pt)."""
    from common import paper_style
    with paper_style(**{"savefig.bbox": "tight", "savefig.pad_inches": 0.02}):
        _geometry()


def _geometry():
    """Rigid-body diagram of a generalised two-trailer vehicle.

    Drafting layout: the vehicle runs roughly horizontally with enough
    articulation for the joint angles to read, segment names sit above their
    axles, and every linear dimension sits on one baseline below.  The two short
    hitch dimensions are called out with leader lines, as a drawing would do, so
    that their labels do not crowd the long ones.  The hitch offsets have
    opposite signs, making the convention visible rather than asserted.
    """
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_aspect("equal"); ax.axis("off")

    Lh = [0.85, -0.60]
    L = [2.30, 1.95]
    th = [0.40, 0.03, -0.34]
    cols = [C_TRACTOR, C_MID, C_LAST]
    HALF = 0.44

    P = [np.array([8.3, 0.0])]
    joints = []
    for k in range(2):
        j = P[-1] - Lh[k] * _unit(th[k])
        joints.append(j)
        P.append(j - L[k] * _unit(th[k + 1]))

    def axle(p, t, c):
        n = _norm(t) * HALF
        ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]], "-",
                color=c, lw=1.7, solid_capstyle="round", zorder=5)
        d = _unit(t) * 0.23
        for sgn in (-1, 1):
            w = p + sgn * n
            ax.plot([w[0] - d[0], w[0] + d[0]], [w[1] - d[1], w[1] + d[1]], "-",
                    color=c, lw=4.4, solid_capstyle="round", zorder=6)
        ax.plot(*p, "o", color="white", ms=4.2, zorder=8,
                markeredgecolor=c, markeredgewidth=1.25)

    for k in range(3):
        d = _unit(th[k])
        a, b = P[k] - d * 1.25, P[k] + d * 1.7
        ax.plot([a[0], b[0]], [a[1], b[1]], ":", color=FAINT, lw=0.8, zorder=1)

    for k in range(2):
        ax.plot([P[k][0], joints[k][0]], [P[k][1], joints[k][1]], "-",
                color=SOFT, lw=2.0, zorder=3, solid_capstyle="round")
        ax.plot([joints[k][0], P[k + 1][0]], [joints[k][1], P[k + 1][1]], "-",
                color=cols[k + 1], lw=2.0, zorder=3, solid_capstyle="round")
        ax.plot(*joints[k], "o", color=INK, ms=5.0, zorder=8)

    for k in range(3):
        axle(P[k], th[k], cols[k])

    # ---- heading of the tractor against a horizontal datum
    ax.plot([P[0][0] - 0.2, P[0][0] + 2.2], [P[0][1], P[0][1]], "--",
            color=FAINT, lw=0.9, zorder=1)
    ax.add_patch(Arc(P[0], 2.6, 2.6, theta1=0, theta2=np.rad2deg(th[0]),
                     color=INK, lw=1.0, zorder=4))
    ax.text(P[0][0] + 1.44, P[0][1] + 0.32, r"$\theta_0$", fontsize=10, color=INK)

    # ---- joint angles between consecutive body axes
    for k in range(2):
        r = 1.9
        ax.add_patch(Arc(P[k + 1], r, r,
                         theta1=np.rad2deg(th[k + 1]), theta2=np.rad2deg(th[k]),
                         color=cols[k + 1], lw=1.4, zorder=4))
        mid = 0.5 * (th[k] + th[k + 1])
        # lift the label clear of the body it would otherwise sit on
        lab = P[k + 1] + _unit(mid) * (r / 2 + 0.20) + np.array([0.0, 0.62])
        ax.text(lab[0], lab[1], rf"$\beta_{{{k+1}}}$", fontsize=10,
                color=cols[k + 1], ha="center", va="bottom")

    # ---- one dimension baseline; short hitch dimensions get leader callouts
    y_dim = min(p[1] for p in P + joints) - 1.45
    spans = [(P[0], joints[0], r"$L_{h_1}>0$", True),
             (joints[0], P[1], r"$L_1$", False),
             (P[1], joints[1], r"$L_{h_2}<0$", True),
             (joints[1], P[2], r"$L_2$", False)]
    for a, b, _, _ in spans:
        for q in (a, b):
            ax.plot([q[0], q[0]], [q[1] - 0.30, y_dim + 0.10], "-",
                    color=FAINT, lw=0.6, zorder=1)
    for a, b, label, is_short in spans:
        ax.annotate("", xy=(b[0], y_dim), xytext=(a[0], y_dim),
                    arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.9,
                                    mutation_scale=7, shrinkA=0, shrinkB=0))
        xm = (a[0] + b[0]) / 2
        if is_short:
            # leader line down to a label placed clear of the baseline
            ax.plot([xm, xm, xm - 0.55], [y_dim - 0.06, y_dim - 0.62, y_dim - 0.62],
                    "-", color=INK, lw=0.7, zorder=2)
            ax.text(xm - 0.66, y_dim - 0.62, label, fontsize=8.8,
                    ha="right", va="center", color=INK)
        else:
            ax.text(xm, y_dim + 0.13, label, fontsize=9,
                    ha="center", va="bottom", color=INK)

    # ---- inputs at the tractor
    d0 = _unit(th[0])
    ax.add_patch(FancyArrowPatch(tuple(P[0] + d0 * 0.66), tuple(P[0] + d0 * 1.95),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.5,
                                 color=C_TRACTOR, zorder=9))
    lab = P[0] + d0 * 1.95 + np.array([0.22, 0.20])
    ax.text(lab[0], lab[1], r"$v_0$", fontsize=10, color=C_TRACTOR)
    ax.add_patch(Arc(P[0], 1.3, 1.3, theta1=118, theta2=212, color=C_TRACTOR, lw=1.5, zorder=9))
    ax.text(P[0][0] - 1.06, P[0][1] + 0.46, r"$\omega_0$", fontsize=10, color=C_TRACTOR)

    # ---- names above the axles, coordinates below them
    y_name = max(p[1] for p in P) + 1.55
    # stagger the two trailer names: their axles are close together in x
    for k, (nm, dy) in enumerate([("tractor", 0.0), ("trailer 1", 0.0), ("trailer 2", 0.52)]):
        ax.text(P[k][0], y_name + dy, nm, fontsize=8.2, color=cols[k],
                ha="center", va="bottom")
    ax.text(P[0][0] + 0.10, P[0][1] - 0.86, r"$(x_0,y_0)$", fontsize=8.4,
            color=SOFT, ha="center")
    ax.text(P[2][0] - 0.55, P[2][1] - 0.92, r"$(x_2,y_2)$", fontsize=8.4,
            color=SOFT, ha="center")

    xs = [p[0] for p in P] + [j[0] for j in joints]
    ys = [p[1] for p in P] + [j[1] for j in joints]
    ax.set_xlim(min(xs) - 1.9, max(xs) + 2.5)
    ax.set_ylim(y_dim - 1.15, y_name + 1.25)
    save(fig, "geometry")


# --------------------------------------------------------------------------- #
def hitching_classes():
    """See ``_hitching_classes``; drawn in the paper style, at its printed size."""
    from common import paper_style
    with paper_style():
        _hitching_classes()


def _hitching_classes():
    """The three hitching classes, drawn so the distinction is visible.

    This replaces a photograph.  A photograph of an articulated vehicle shows
    that such vehicles exist; it does not show where the hitch sits relative to
    the axle, which is the only thing separating the three classes and the
    quantity the whole paper turns on.  Each row is drawn with its joints
    articulated, because on a straight chain an on-axle and an off-axle hitch
    project onto nearly the same point and the difference does not read.

    The generalised row carries one joint of each kind -- behind the axle, on
    it, and ahead of it -- and is therefore one segment longer than the others.
    That asymmetry is the point rather than an inconsistency: the class is
    defined by admitting all three, and a row that showed only two of them would
    understate it.  Sign follows the convention of Fig. 3, where a positive
    offset places the hitch behind the axle of the segment in front and a
    negative one ahead of it, as a fifth wheel sits ahead of a tractor's axle.
    """
    # Drawn at the printed size, one column wide: the axes fill the figure and the
    # figure height follows from the data limits at equal aspect, so every font
    # prints at its nominal size.
    from common import COL_W
    HALF, ROW = 0.34, 2.45
    X0, X1, Y0, Y1 = -0.10, 9.30, -2 * ROW - 2.05, 1.00
    fig = plt.figure(figsize=(COL_W, COL_W * (Y1 - Y0) / (X1 - X0)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal"); ax.axis("off")
    cases = [
        ("SNT", "Standard:\nevery hitch\non the axle",
         [0.0, 0.0], [2.05, 1.75], [0.30, 0.02, -0.26]),
        ("nSNT", "Non-standard:\nevery hitch\noff the axle",
         [0.80, 0.62], [2.05, 1.75], [0.30, 0.02, -0.26]),
        # The last joint is articulated more than the others on purpose: with a
        # negative offset the drawbar runs forward past the axle and then back,
        # and at a small joint angle the two links lie almost on top of each
        # other and the arrangement cannot be read.
        ("GNT", "Generalised:\nany kind,\nany sign",
         [0.80, 0.0, -0.62], [1.85, 1.60, 1.50], [0.34, 0.06, -0.14, -0.62]),
    ]

    def axle(p, ang, c):
        """``ang`` is a heading, not a direction vector: both helpers take angles."""
        n = _norm(ang) * HALF
        ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]], "-",
                color=c, lw=1.6, solid_capstyle="round", zorder=5)
        d = _unit(ang) * 0.18
        for sgn in (-1, 1):
            w = p + sgn * n
            ax.plot([w[0] - d[0], w[0] + d[0]], [w[1] - d[1], w[1] + d[1]], "-",
                    color=c, lw=3.6, solid_capstyle="round", zorder=6)

    for r, (name, blurb, Lh, L, TH) in enumerate(cases):
        y0 = -r * ROW
        n = len(Lh)
        cols = seg_colours(n)
        P = [np.array([8.55, y0])]
        joints = []
        for k in range(n):
            j = P[-1] - Lh[k] * _unit(TH[k])
            joints.append(j)
            P.append(j - L[k] * _unit(TH[k + 1]))

        for k in range(n):
            ax.plot([P[k][0], joints[k][0]], [P[k][1], joints[k][1]], "-",
                    color=SOFT, lw=1.9, zorder=3, solid_capstyle="round")
            ax.plot([joints[k][0], P[k + 1][0]], [joints[k][1], P[k + 1][1]], "-",
                    color=SOFT, lw=1.9, zorder=3, solid_capstyle="round")
        for k in range(n + 1):
            axle(P[k], TH[k], cols[k])

        for k in range(n):
            on_axle = abs(Lh[k]) < 1e-9
            ax.plot(*joints[k], "o", color=(INK if on_axle else "white"), ms=5.4,
                    zorder=9, markeredgecolor=INK, markeredgewidth=1.3)
            rel = "=" if on_axle else (">" if Lh[k] > 0 else "<")
            lab = r"$L_{h%d}\!%s\!0$" % (k + 1, rel)
            # A label below the axle would collide with the wheels on the rows
            # where the hitch sits close to one; alternate above and below.
            dy = -18 if (Lh[k] >= 0) else 14
            ax.annotate(lab, joints[k], textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=9,
                        color=(INK if on_axle else SOFT), zorder=10)
            # The offset dimension is drawn only where the concept is being
            # defined.  Repeating it on the generalised row, which already
            # carries three joints and three labels, adds lines without adding
            # information: by then the hollow marker and the sign say it.
            if not on_axle and name == "nSNT":
                a, b = P[k], joints[k]
                off = _norm(TH[k]) * (0.55 if Lh[k] > 0 else -0.55)
                ax.annotate("", xy=tuple(a + off), xytext=tuple(b + off),
                            arrowprops=dict(arrowstyle="<->", color=FAINT, lw=0.8,
                                            shrinkA=0, shrinkB=0), zorder=4)
                for q in (a, b):
                    ax.plot([q[0], (q + off)[0]], [q[1], (q + off)[1]], "-",
                            color=FAINT, lw=0.6, zorder=2)

        ax.text(0.0, y0 + 0.62, name, fontsize=10.5, fontweight="bold",
                color=INK, ha="left", va="center")
        ax.text(0.0, y0 + 0.30, blurb, fontsize=9, color=SOFT,
                ha="left", va="top", linespacing=1.3)
        if name == "GNT":
            ax.text(0.0, y0 - 1.05, "the class this paper addresses;\nthe field platform is of this kind",
                    fontsize=9, color=C_LAST, ha="left", va="top",
                    style="italic", linespacing=1.3)

    ax.plot([], [], "o", color="white", markeredgecolor=INK, markeredgewidth=1.3,
            ms=5.4, label="hitch off the axle")
    ax.plot([], [], "o", color=INK, markeredgecolor=INK, markeredgewidth=1.3,
            ms=5.4, label="hitch on the axle")
    ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.005), ncol=1,
              fontsize=9, frameon=False, handletextpad=0.4, borderpad=0.0,
              borderaxespad=0.0, labelspacing=0.3)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    save(fig, "hitching_classes")


def paths_figure():
    """The three test geometries, one column wide and drawn at their printed size,
    each coloured by the magnitude of its curvature.  The agricultural path, the
    one most studies use, is given the widest panel."""
    from matplotlib.colors import ListedColormap, Normalize
    from scipy.ndimage import median_filter
    from gntpf import make_path
    from common import COL_W, paper_style

    # The caption names the paths; the panels carry their letters only.
    specs = [("agricultural", "(a)"), ("rounded_rect", "(b)"), ("lemniscate", "(c)")]
    paths = [make_path(n) for n, _ in specs]
    norm = Normalize(0.0, 1.0)
    # viridis without its palest yellow, which is faint against white
    cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.88, 256)))
    with paper_style():
        fig = plt.figure(figsize=(COL_W, 1.75), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, wspace=0.03)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.85, 1.0, 0.8], height_ratios=[1.0, 0.06])
        for j, ((name, title), p) in enumerate(zip(specs, paths)):
            # The finite-difference curvature has single-sample spikes where segments
            # sampled at different spacings meet (the corners of the rounded
            # rectangle); a five-sample median removes them and nothing else.
            k = median_filter(p.curvature(), size=5, mode="nearest")
            ax = fig.add_subplot(gs[0, j])
            # The paths are sampled every few millimetres.  The curvature is
            # quantised to 64 colour levels and each run of samples at one level is
            # drawn as a single line, which keeps the figure a small vector file;
            # 64 levels are finer than the eye resolves on the colour bar.
            lev = np.clip(np.round(norm(k) * 63), 0, 63).astype(int)
            cut = np.flatnonzero(np.diff(lev)) + 1
            for a, b in zip(np.r_[0, cut], np.r_[cut, lev.size]):
                seg = p.xy[:, a:min(b + 1, lev.size)]
                ax.plot(seg[0], seg[1], color=cmap(lev[a] / 63), lw=1.3,
                        solid_capstyle="round")
            ax.plot(p.xy[0, 0], p.xy[1, 0], "o", ms=4, mfc="white", mec="black", mew=0.8, zorder=5)
            ax.set_xlim(p.xy[0].min() - 0.8, p.xy[0].max() + 0.8)
            ax.set_ylim(p.xy[1].min() - 0.8, p.xy[1].max() + 0.8)
            ax.set_aspect("equal")
            ax.set_title(title, loc="left")
            ax.set_xlabel("x (m)")
            if j == 0:
                ax.set_ylabel("y (m)")
        cax = fig.add_subplot(gs[1, :])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal")
        cb.set_label(r"$|$curvature$|$ (m$^{-1}$)")
        save(fig, "paths")


if __name__ == "__main__":
    architecture()
    geometry()
    hitching_classes()
    paths_figure()
