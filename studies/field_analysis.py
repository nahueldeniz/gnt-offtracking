"""Field-experiment analysis: what the logs actually support.

Reads the MATLAB v7.3 logs from the November trials and reports, per trial and
per experiment, the off-tracking of every segment, the joint angles against the
mechanical limit, and the measured solve times.

Three facts about the data decided the structure of this analysis, and each of
them contradicts something the previous version of the manuscript asserted.

*The sampling period is 0.65 s, not the 50 ms of the simulation study.* The field
controller ran an order of magnitude slower, with a much shorter horizon
(``Nc`` between 5 and 13 against 20). Field and simulation timings are therefore
not comparable and are never pooled here.

*Two different vehicles were used.* The trials of 13-15 November have
``L = (1.08, 0.78)``; those of 20 November have ``L = (0.38, 1.08)``. Off-tracking
depends on the geometry, so these cannot be averaged together. The 20 November
trials are reported separately and are not used for the headline numbers.

*No trial referenced the last trailer alone.* Every one of the logs weights the
position of every segment in the reference generator; what varies between trials
is how heavily the last trailer is weighted relative to the others, from parity
up to a factor of about sixteen. There is therefore no field counterpart of the
"last trailer only" strategy, and the claim that there was one has been removed.
What the spread in weights does provide is a field counterpart of S6, which is
reported instead because it is what the data supports.

Positions of the trailers are *estimates*, not measurements: only the tractor
pose and the joint angles were instrumented. Everything reported here is
therefore the off-tracking of the estimated posture, and is labelled as such.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

from common import C_GREY, C_LAST, C_MID, C_PATH, C_TRACTOR, OUT, dump, save, seg_colours, seg_label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gntpf import Path  # noqa: E402

def _log_root():
    """Where the November experiment logs live.

    The logs are part of the released data set rather than of this repository,
    so their location is supplied from outside: set ``GNT_FIELD_LOGS`` to the
    folder holding the per-session subfolders of ``.mat`` files.
    """
    root = os.environ.get("GNT_FIELD_LOGS")
    if root:
        return os.path.abspath(os.path.expanduser(root))
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "field_logs")
    return local


ROOT = _log_root()
BETA_MAX_DEG = 80.0
SKIP = 0.15          # discard the initial transient, as in the simulation study
NOMINAL_L = (1.08, 0.78)


def _str(ds):
    return "".join(chr(c) for c in np.asarray(ds).ravel())


def load_trial(fp):
    """Everything one log has to say, in the paper's own conventions."""
    with h5py.File(fp, "r") as h:
        S = h["S"]
        g = lambda p: float(np.asarray(S[p]).ravel()[0])
        xest = np.asarray(S["data/xest"])                 # (K, 11): the estimate
        ysim = np.asarray(S["data/ysim"])                 # (K-1, 5): measurements
        coords = np.asarray(S["path/coordinates"]).T      # (2, M)
        N = int(g("config/N"))
        t = {k: np.asarray(S[f"exec_time/{k}"]).ravel()
             for k in ("t_tot", "t_mpc", "t_mhe", "t_pfa")}
        return dict(
            file=os.path.relpath(fp, ROOT),
            day=os.path.basename(os.path.dirname(fp)),
            path_kind=_str(S["path/path"]),
            N=N, Nc=int(g("config/Nc")), Ne=int(g("config/Ne")),
            Ts=g("config/Ts"), vNr=g("config/vNr"),
            Lh=np.asarray(S["system/Lhi"]).ravel()[:N],
            L=np.asarray(S["system/Li"]).ravel()[:N],
            wref=np.diag(np.asarray(S["Mtxs/QrefOpt"]))[2 * N + 1:4 * N + 3],
            xest=xest, ysim=ysim, coords=coords, times=t,
        )


def analyse(tr):
    """Per-segment off-tracking of the estimated posture, and the joint angles."""
    N, xest = tr["N"], tr["xest"]
    path = Path(tr["coords"], name=tr["path_kind"])
    K = xest.shape[0]
    k0 = int(SKIP * K)
    pos = [xest[:, 2 * N + 1 + 2 * i: 2 * N + 3 + 2 * i] for i in range(N + 1)]

    dev = np.zeros((N + 1, K - k0))
    for i in range(N + 1):
        for j, k in enumerate(range(k0, K)):
            dev[i, j] = path.deviation(pos[i][k])

    # No curved/straight split here.  The field paths are a rectangle and a
    # lemniscate: on the rectangle the "curved section" is a corner of zero
    # length, so the fraction of samples inside it is near zero and the split
    # that works in simulation reports nothing.  Field numbers are whole-run.
    s_tr = np.array([path.project(pos[0][k]) for k in range(k0, K)])

    beta = np.rad2deg(tr["ysim"][:, :N])          # measured joint angles
    tt = {k: float(np.nanmean(v[k0:]) * 1e3) for k, v in tr["times"].items()}
    tt["t_p95"] = float(np.nanpercentile(tr["times"]["t_tot"][k0:], 95) * 1e3)

    out = dict(
        **{k: tr[k] for k in ("file", "day", "path_kind", "N", "Nc", "Ne", "Ts", "vNr")},
        L=list(np.round(tr["L"], 3)), Lh=list(np.round(tr["Lh"], 3)),
        w_last_ratio=float(tr["wref"][-1] / max(tr["wref"][0], 1e-9)),
        steps=K, duration_s=float(K * tr["Ts"]),
        mean_dev=dev.mean(axis=1), max_dev=dev.max(axis=1),
        p95_dev=np.percentile(dev, 95, axis=1),
        worst_segment=float(dev.mean(axis=1).max()),
        worst_segment_max=float(dev.max()),
        beta_max_deg=float(np.nanmax(np.abs(beta))) if beta.size else float("nan"),
        beta_over_limit_pct=float(100.0 * np.mean(np.abs(beta) > BETA_MAX_DEG)) if beta.size else 0.0,
        **tt,
    )
    return out, path, pos, k0


def main():
    files = sorted(glob.glob(f"{ROOT}/*/*.mat"))
    if not files:
        print(f"!! no logs found under {ROOT}\n"
              f"   set GNT_FIELD_LOGS to the folder holding the experiment "
              f"subfolders, or place them in <repo>/field_logs")
        return
    rows = []
    for fp in files:
        try:
            r, _, _, _ = analyse(load_trial(fp))
            rows.append(r)
            print(f"  {r['file']:26s} {r['path_kind']:14s} Nc={r['Nc']:2d} "
                  f"seg dev {np.round(r['mean_dev'], 3)}  worst {r['worst_segment']:.3f} m  "
                  f"|beta|max {r['beta_max_deg']:5.1f} deg  t_tot {r['t_tot']:6.1f} ms", flush=True)
        except Exception as e:
            print(f"  !! {os.path.relpath(fp, ROOT)}: {type(e).__name__}: {e}", flush=True)
    dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
          for r in rows], "field_trials")
    print(f"\n{len(rows)} trials analysed -> out/field_trials.json")


if __name__ == "__main__":
    main()
