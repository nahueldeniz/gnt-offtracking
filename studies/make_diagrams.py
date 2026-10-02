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
    """Block diagram of the two-stage architecture, drawn to the full text width.

    The boxes are sized from the rendered extent of their own text rather than
    from guessed widths, so a longer label widens its box instead of spilling
    out of it.
    """
    W, H = 16.0, 4.35
    fig, ax = plt.subplots(figsize=(7.16, 1.95))
    ax.set_xlim(-0.55, W); ax.set_ylim(-0.15, H); ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    PAD_X, PAD_Y = 0.42, 0.34
    boxes = {}

    def block(key, cx, cy, title, sub, fill=BOX, edge=INK):
        """Place a titled box centred on (cx, cy), sized to its own text."""
        t = ax.text(cx, cy + 0.17, title, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=INK, zorder=4)
        u = ax.text(cx, cy - 0.34, sub, ha="center", va="center",
                    fontsize=6.3, color=SOFT, zorder=4, linespacing=1.35)
        fig.canvas.draw()
        inv = ax.transData.inverted()
        ext = [inv.transform(o.get_window_extent()) for o in (t, u)]
        x0 = min(e[0][0] for e in ext) - PAD_X
        x1 = max(e[1][0] for e in ext) + PAD_X
        y0 = min(e[0][1] for e in ext) - PAD_Y
        y1 = max(e[1][1] for e in ext) + PAD_Y
        # keep the box centred on cx even if the two lines differ in width
        half = max(cx - x0, x1 - cx)
        x0, x1 = cx - half, cx + half
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.0,rounding_size=0.14",
                                    linewidth=1.05, edgecolor=edge, facecolor=fill,
                                    zorder=3))
        boxes[key] = (x0, y0, x1, y1)
        return boxes[key]

    def arrow(p0, p1, col=INK, ls="-", lw=1.0, head=True):
        """A signal leg; only the last leg of a multi-segment route carries a head."""
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-",
                                     mutation_scale=9, linewidth=lw, color=col,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=2))

    def label(x, y, txt, ha="center", va="bottom"):
        ax.text(x, y, txt, ha=ha, va=va, fontsize=6.4, color=SOFT, zorder=5)

    ROW, EST = 2.95, 0.80
    block("path", 1.45, ROW, "Nominal path",
          "local polynomial window,\nadvancing at $\\sigma$")
    block("ref", 6.75, ROW, "Reference generator",
          "geometry exact,\nreachable in one step")
    block("mpc", 10.85, ROW, "Tracking NMPC",
          "horizon $N_c$,\nno terminal set")
    block("veh", 14.45, ROW, "GNT vehicle", "$N$ passive trailers")
    block("est", 8.40, EST, "Moving horizon estimator",
          "window $N_e$, constrained")

    # the dashed group around the two problems solved every period
    gx0 = boxes["ref"][0] - 0.14
    gx1 = boxes["mpc"][2] + 0.14
    gy0 = boxes["ref"][1] - 0.28
    gy1 = boxes["ref"][3] + 0.28
    ax.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0,
                                boxstyle="round,pad=0.0,rounding_size=0.10",
                                linewidth=0.9, edgecolor=SOFT, facecolor="none",
                                linestyle=(0, (4, 3)), zorder=1))
    ax.text(0.5 * (gx0 + gx1), gy1 + 0.16,
            "solved in sequence, once per sampling period",
            ha="center", va="bottom", fontsize=6.5, color=SOFT, style="italic")

    # ---------------------------------------------------------------- signals
    arrow((boxes["path"][2], ROW), (boxes["ref"][0], ROW))
    label(0.5 * (boxes["path"][2] + boxes["ref"][0]), ROW + 0.16, "$X(s),\\,Y(s)$")

    arrow((boxes["ref"][2], ROW), (boxes["mpc"][0], ROW))
    label(0.5 * (boxes["ref"][2] + boxes["mpc"][0]), ROW + 0.16, "$q^{\\mathrm{ref}}$")

    arrow((boxes["mpc"][2], ROW), (boxes["veh"][0], ROW))
    label(0.5 * (boxes["mpc"][2] + boxes["veh"][0]), ROW + 0.16, "$u_k$")

    # measurements: down the right-hand side and back into the estimator
    y_meas = EST
    x_veh = 0.5 * (boxes["veh"][0] + boxes["veh"][2])
    arrow((x_veh, boxes["veh"][1]), (x_veh, y_meas), col=SOFT, head=False)
    arrow((x_veh, y_meas), (boxes["est"][2], y_meas), col=SOFT)
    ax.text(0.5 * (x_veh + boxes["est"][2]), y_meas - 0.30,
            "$r_k$: tractor pose, joint angles", ha="center", va="top",
            fontsize=6.6, color=SOFT)

    # the estimate, up into the tracking problem and across to the generator
    x_up = 0.5 * (boxes["mpc"][0] + boxes["mpc"][2]) - 0.55
    arrow((x_up, boxes["est"][3]), (x_up, boxes["mpc"][1]), col=SOFT)
    # centred in the clear band between the estimator and the dashed group
    ax.text(x_up + 0.16, 0.5 * (boxes["est"][3] + gy0),
            "$\\hat q_{k|k}$", ha="left", va="center", fontsize=6.6, color=SOFT)

    x_back = 0.5 * (boxes["ref"][0] + boxes["ref"][2]) - 0.55
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
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    ax.set_aspect("equal"); ax.axis("off")

    HALF, ROW = 0.34, 2.45
    cases = [
        ("SNT", "Standard:\nevery hitch on the axle",
         [0.0, 0.0], [2.05, 1.75], [0.30, 0.02, -0.26]),
        ("nSNT", "Non-standard:\nevery hitch off the axle",
         [0.80, 0.62], [2.05, 1.75], [0.30, 0.02, -0.26]),
        # The last joint is articulated more than the others on purpose: with a
        # negative offset the drawbar runs forward past the axle and then back,
        # and at a small joint angle the two links lie almost on top of each
        # other and the arrangement cannot be read.
        ("GNT", "Generalised:\nany kind, any sign",
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
                        xytext=(0, dy), ha="center", fontsize=7.2,
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

        ax.text(0.0, y0 + 0.50, name, fontsize=10.5, fontweight="bold",
                color=INK, ha="left", va="center")
        ax.text(0.0, y0 - 0.22, blurb, fontsize=7.4, color=SOFT,
                ha="left", va="center", linespacing=1.5)
        if name == "GNT":
            ax.text(0.0, y0 - 1.08, "the class this paper addresses;\nthe field platform is of this kind",
                    fontsize=7.2, color=C_LAST, ha="left", va="center",
                    style="italic", linespacing=1.5)

    ax.plot([], [], "o", color="white", markeredgecolor=INK, markeredgewidth=1.3,
            ms=5.4, label="hitch off the axle")
    ax.plot([], [], "o", color=INK, markeredgecolor=INK, markeredgewidth=1.3,
            ms=5.4, label="hitch on the axle")
    ax.legend(loc="lower center", bbox_to_anchor=(0.68, -0.055), ncol=2,
              fontsize=7.4, frameon=False, handletextpad=0.5, borderpad=0.2,
              columnspacing=1.6)
    ax.set_xlim(-0.15, 9.1); ax.set_ylim(-2 * ROW - 1.75, 1.05)
    save(fig, "hitching_classes")


def paths_figure():
    """The three test geometries, each with its curvature profile beneath."""
    from gntpf import make_path
    from common import C_GREY, C_PATH

    specs = [("agricultural", "(a) agricultural: omega headland turns"),
             ("rounded_rect", "(b) rounded rectangle"),
             ("lemniscate", "(c) lemniscate")]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8),
                             gridspec_kw=dict(height_ratios=[1.45, 1.0], hspace=0.70, wspace=0.34))
    for j, (name, title) in enumerate(specs):
        p = make_path(name)
        # The finite-difference curvature has single-sample spikes where segments
        # sampled at different spacings meet (the corners of the rounded
        # rectangle); a five-sample median removes them and nothing else.
        from scipy.ndimage import median_filter
        k = median_filter(p.curvature(), size=5, mode="nearest")
        kmax = np.percentile(k, 99.5)
        ax = axes[0, j]
        ax.plot(p.xy[0], p.xy[1], "-", color=C_PATH, lw=1.0)
        ax.plot(p.xy[0, 0], p.xy[1, 0], "o", color=C_TRACTOR, ms=4, zorder=5)
        ax.set_aspect("equal")
        # the quantities that matter live in the title, clear of the data
        ax.set_title(f"{title}\n{p.length:.0f} m long, min radius "
                     f"{1 / max(kmax, 1e-9):.1f} m", fontsize=7.6)
        ax.set_xlabel("x (m)", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("y (m)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)

        ax = axes[1, j]
        ax.plot(p.s, k, "-", color=C_GREY, lw=0.7)
        ax.set_ylim(0, max(kmax * 1.25, 0.1))
        ax.set_xlabel("arc length (m)", fontsize=7.5)
        if j == 0:
            ax.set_ylabel(r"$|$curvature$|$ (m$^{-1}$)", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    save(fig, "paths")


if __name__ == "__main__":
    architecture()
    geometry()
    hitching_classes()
    paths_figure()
