"""S4 -- computational scaling, and where the method stops being real time.

Reviewers asked two questions that share an answer: what does a solve cost, and
what happens beyond seven trailers?  The number of trailers is swept until the
total solve time per step exceeds the sampling period, and the problem sizes are
reported alongside, so the reader can scale the result to their own hardware
rather than trusting one machine's milliseconds.

The split between the two stages is the interesting part.  Reference generation
is a single-instant problem whose size grows linearly in N and stays cheap; the
tracking NMPC carries a horizon and dominates.  Decoupling them is what keeps the
reference-generation cost from being multiplied by the horizon length.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from common import C_GREY, C_LAST, C_MID, C_TRACTOR, dump, save, vehicle

from gntpf import RefGenWeights, SimConfig, TrackingNMPC, make_path, simulate

NS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
HORIZONS = [10, 15, 20, 30]

# Each configuration is solved REPEATS times and the *median* wall-clock is
# reported.  The simulation is deterministic given the seed, so the solver takes
# the same iterations every time and any spread between repeats is the machine,
# not the problem: CPU frequency scaling, cache pressure, another process.  A
# single timing run of this study produced a 3x spike at one value of N that did
# not reproduce, which is exactly the artefact a median over repeats removes.
# Timings are the one thing in this paper that cannot be made deterministic, so
# they are the one thing that needs repetition.
REPEATS = 3


def _timed(model, path, t_final, sigma, Nc=20, repeats=REPEATS):
    """Solve the same configuration ``repeats`` times; return the median timings.

    The accuracy metrics are identical across repeats by determinism, so only
    the timings are aggregated; the last result is returned for everything else.
    """
    runs = []
    for _ in range(repeats):
        res = simulate(model, path, SimConfig(sigma=sigma, t_final=t_final, seed=0),
                       nmpc=TrackingNMPC(model, Nc=Nc),
                       refgen_weights=RefGenWeights(w_path=20.0, w_theta=5.0))
        runs.append((res, res.summary(skip=0.35)))
    med = {k: float(np.median([s[k] for _, s in runs]))
           for k in ("t_ref_ms", "t_mpc_ms", "t_mhe_ms", "t_total_ms", "t_total_p95_ms")}
    res, s = runs[-1]
    spread = max(s2["t_total_ms"] for _, s2 in runs) / max(
        min(s2["t_total_ms"] for _, s2 in runs), 1e-9)
    return res, {**s, **med}, spread


def run_scaling(t_final=12.0, sigma=1.0):
    path = make_path("circle", R=8.0, cx=9.0, cy=9.0)
    rows = []
    for N in NS:
        model = vehicle(N)
        res, s, spread = _timed(model, path, t_final, sigma)
        rows.append(dict(N=N, nq=model.nq, t_spread=spread,
                         n_var_ref=res.meta["n_var_ref"], n_con_ref=res.meta["n_con_ref"],
                         n_var_mpc=res.meta["n_var_mpc"], n_con_mpc=res.meta["n_con_mpc"],
                         n_var_mhe=res.meta["n_var_mhe"],
                         t_ref=s["t_ref_ms"], t_mpc=s["t_mpc_ms"], t_mhe=s["t_mhe_ms"],
                         t_tot=s["t_total_ms"], t_p95=s["t_total_p95_ms"],
                         worst=s["worst_segment"], last=s["mean_dev_last"],
                         fails=int(sum(res.fails.values()))))
        print(f"  N={N:2d} nq={model.nq:3d}  ref {s['t_ref_ms']:6.1f}  mpc {s['t_mpc_ms']:7.1f}  "
              f"mhe {s['t_mhe_ms']:6.1f}  total {s['t_total_ms']:7.1f} ms  "
              f"(spread {spread:.2f}x)  worst {s['worst_segment']:.3f} m", flush=True)
    return rows


def run_horizon(t_final=12.0, sigma=1.0, N=4):
    path = make_path("circle", R=8.0, cx=9.0, cy=9.0)
    rows = []
    for Nc in HORIZONS:
        model = vehicle(N)
        res, s, _ = _timed(model, path, t_final, sigma, Nc=Nc)
        rows.append(dict(Nc=Nc, t_mpc=s["t_mpc_ms"], t_tot=s["t_total_ms"],
                         worst=s["worst_segment"]))
        print(f"  Nc={Nc:3d}  mpc {s['t_mpc_ms']:7.1f} ms  worst {s['worst_segment']:.3f} m",
              flush=True)
    return rows


def figure(rows, hrows, Ts=0.05):
    N = np.array([r["N"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    ax.plot(N, [r["t_ref"] for r in rows], "o-", color=C_TRACTOR, ms=3, label="reference gen.")
    ax.plot(N, [r["t_mhe"] for r in rows], "s-", color=C_MID, ms=3, label="estimator")
    ax.plot(N, [r["t_mpc"] for r in rows], "^-", color=C_LAST, ms=3, label="tracking NMPC")
    ax.plot(N, [r["t_tot"] for r in rows], "-", color="#222", lw=1.4, label="total")
    ax.axhline(Ts * 1e3, color=C_GREY, ls="--", lw=0.9)
    ax.annotate(f"$T_s$ = {Ts*1e3:.0f} ms", (N[-1], Ts * 1e3 * 0.62), fontsize=7,
                color=C_GREY, ha="right")
    ax.set_yscale("log"); ax.set_xlabel("number of trailers $N$")
    ax.set_ylabel("solve time per step (ms)"); ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    ax.plot(N, [r["n_var_ref"] for r in rows], "o-", color=C_TRACTOR, ms=3, label="ref. gen.")
    ax.plot(N, [r["n_var_mhe"] for r in rows], "s-", color=C_MID, ms=3, label="estimator")
    ax.plot(N, [r["n_var_mpc"] for r in rows], "^-", color=C_LAST, ms=3, label="NMPC")
    ax.set_xlabel("number of trailers $N$"); ax.set_ylabel("decision variables")
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[2]
    ax.plot([r["Nc"] for r in hrows], [r["t_mpc"] for r in hrows], "^-", color=C_LAST, ms=3,
            label="tracking NMPC")
    ax.plot([r["Nc"] for r in hrows], [r["t_tot"] for r in hrows], "-", color="#222", lw=1.4,
            label="total")
    ax.axhline(Ts * 1e3, color=C_GREY, ls="--", lw=0.9)
    ax.set_xlabel("control horizon $N_c$  ($N=4$)")
    ax.set_ylabel("solve time (ms)"); ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "s4_scaling")


if __name__ == "__main__":
    print("[S4] scaling in N", flush=True)
    rows = run_scaling()
    print("[S4] scaling in the horizon", flush=True)
    hrows = run_horizon()
    dump({"N_sweep": rows, "horizon_sweep": hrows}, "s4_scaling")
    figure(rows, hrows)
