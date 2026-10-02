"""Measured model-verification residuals reported in Section VIII-C.

The paper quotes two numbers: the largest violation of the chain's algebraic
constraints accumulated over a long integration, and the error of the simulated
constant-curvature steady state against the closed form. The test suite asserts
that both stay below a threshold; this script measures what they actually are,
so that the numbers in the text are measurements rather than thresholds.

The geometries span the three hitching classes: Standard (every hitch on the
axle), non-Standard (every hitch off it) and Generalised (mixed).
"""

from __future__ import annotations

import numpy as np

from common import OUT, dump  # noqa: F401
from gntpf import GNT, make_path
from gntpf.mp_refgen import (MPReferenceGenerator, closed_form_betas,
                            joint_rhs)

# (label, class, Lh, L)
GEOMETRIES = [
    ("N=1 nSNT", "nSNT", [0.342], [1.08]),
    ("N=2 field", "GNT", [0.342, 0.0], [1.08, 0.78]),
    ("N=3 SNT", "SNT", [0.0, 0.0, 0.0], [1.0, 0.8, 0.7]),
    ("N=3 nSNT", "nSNT", [0.25, -0.2, 0.3], [1.0, 0.8, 0.7]),
    ("N=4 GNT", "GNT", [0.342, 0.0, 0.15, -0.15], [1.08, 0.78, 0.70, 0.70]),
    ("N=7 GNT", "GNT", [0.342, 0.0, 0.15, -0.15, 0.0, -0.342, 0.342],
     [0.38, 1.08, 0.70, 0.70, 0.70, 0.70, 0.70]),
]

N_STEPS = 2000
TS = 0.05          # the sampling period of every simulation in the paper


def chain_residual(m, q):
    """Largest violation of the rigid-body geometry, and of beta_i = th_{i-1} - th_i."""
    th = q[m.i_theta]
    x0, y0 = q[2 * m.N + 1], q[2 * m.N + 2]
    ax = ay = 0.0
    worst = 0.0
    for k in range(m.N):
        ax += m.Lh[k] * np.cos(th[k]) + m.L[k] * np.cos(th[k + 1])
        ay += m.Lh[k] * np.sin(th[k]) + m.L[k] * np.sin(th[k + 1])
        worst = max(worst, float(np.linalg.norm(
            q[m.i_pos(k + 1)] - np.array([x0 - ax, y0 - ay]))))
    return worst, float(np.max(np.abs(q[m.i_beta] - (th[:-1] - th[1:]))))


def drift():
    """Constraint violation accumulated over N_STEPS of the flow, per geometry."""
    rows = []
    for label, cls, Lh, L in GEOMETRIES:
        N = len(L)
        rng = np.random.default_rng(1)
        m = GNT(N=N, Lh=Lh, L=L, Ts=TS)
        th = np.cumsum(rng.uniform(-0.4, 0.4, N + 1))
        q = m.state_from_pose(*rng.uniform(-2, 2, 2), th)
        g_worst = b_worst = 0.0
        for k in range(N_STEPS):
            q = m.step(q, [0.6 * np.sin(k / 70), 1.2])
            g, b = chain_residual(m, q)
            g_worst, b_worst = max(g_worst, g), max(b_worst, b)
        rows.append({"label": label, "class": cls, "N": N,
                     "geom_max": g_worst, "beta_max": b_worst})
        print(f"  {label:12s} ({cls:4s})  geometry {g_worst:.2e}  beta {b_worst:.2e}")
    return rows


MP_GEOMETRIES = [([0.342, 0.25], [1.08, 0.78]),
                 ([0.342, 0.0], [1.08, 0.78]),
                 ([0.0, 0.0], [1.08, 0.78]),
                 ([1.497] * 3, [6.303] * 3)]


def mp_implementation():
    """Checks on the reconstruction of Michalek and Pazderski used in Section IX-H.

    Two things are measured, and they are different claims. First, that our
    transcription of their closed-form constant-curvature solution really is a
    solution: substituted into the joint dynamics of our own model it must
    annihilate them, and a mis-transcribed equation would not. Second, that the
    Fourier fit our implementation performs recovers that closed form on a path
    where the closed form applies.
    """
    worst = 0.0
    for Lh, L in MP_GEOMETRIES:
        for R in (3.0, 8.0, -5.0):
            m = GNT(N=len(Lh), Lh=Lh, L=L)
            b = closed_form_betas(m, 1.0 / R)
            d, _ = joint_rhs(m, b, np.array([1.0 / R, 1.0]), np.zeros(m.N))
            worst = max(worst, float(np.abs(d).max()))

    m = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78])
    g = MPReferenceGenerator(m, n_harm=6, n_coll=120, homotopy=3)
    path = make_path("circle", R=6.0, cx=0.0, cy=0.0)
    s = np.linspace(0.0, path.length, 120)
    kappa = g._signed_curvature(path, s)
    beta = g.fit(kappa, s)
    exact = closed_form_betas(m, float(np.median(kappa)))
    fit_deg = float(np.rad2deg(np.abs(beta - exact)).max())
    print(f"  closed form annihilates the joint dynamics to {worst:.1e}")
    print(f"  Fourier fit recovers it to {fit_deg:.2f} deg")
    return {"closed_form_residual": worst, "fit_err_deg": fit_deg,
            "n_harm": 6, "n_coll": 120}


def main():
    print("[V] chain-constraint drift over the flow", flush=True)
    rows = drift()
    gmax = max(r["geom_max"] for r in rows)
    bmax = max(r["beta_max"] for r in rows)

    print("[V] constant-curvature steady state against the closed form", flush=True)
    # Run the SNT on a circle and compare the settled inter-axle radii with
    # R_i^2 = R_{i-1}^2 - L_i^2.  The radii are measured about the centre of the
    # tractor's own circle, which is known exactly for a constant-curvature input.
    worst_R = 0.0
    radii_rows = []
    for R0 in (4.0, 8.0, 12.0):
        L = [1.0, 0.8, 0.7]
        m = GNT(N=3, Lh=[0.0, 0.0, 0.0], L=L, Ts=0.005)
        v, om = 1.0, 1.0 / R0
        q = m.state_from_pose(0.0, 0.0, np.zeros(4))   # tractor at origin, heading +x
        centre = np.array([0.0, R0])                   # left turn about (0, R0)
        for _ in range(int(120.0 / m.Ts)):
            q = m.step(q, [om, v])
        pos = m.positions(q)
        Rm = [float(np.linalg.norm(p - centre)) for p in pos]
        Rc = [Rm[0]]
        for Li in L:
            Rc.append(float(np.sqrt(max(Rc[-1] ** 2 - Li ** 2, 0.0))))
        err = max(abs(a - b) for a, b in zip(Rm, Rc))
        worst_R = max(worst_R, err)
        radii_rows.append({"R0": R0, "measured": Rm, "closed_form": Rc, "err": err})
        print(f"  R0={R0:4.1f}  worst radius error {err:.2e} m")

    print("[V] the Michalek-Pazderski reconstruction, as implemented here", flush=True)
    mp = mp_implementation()

    out = {"mp": mp, "n_steps": N_STEPS, "Ts": TS, "geometries": rows,
           "geom_max": gmax, "beta_max": bmax,
           "radii": radii_rows, "radius_err_max": worst_R,
           "classes": sorted({r["class"] for r in rows}),
           "N_max": max(r["N"] for r in rows)}
    dump(out, "verification")
    print(f"\nworst geometry drift {gmax:.2e} m, worst beta drift {bmax:.2e} rad, "
          f"worst radius error {worst_R:.2e} m")


if __name__ == "__main__":
    main()
