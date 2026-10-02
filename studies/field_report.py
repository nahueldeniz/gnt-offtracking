"""Aggregate the field trials into the three experiments the data actually contains.

Trials were not labelled, so the experiments are recovered from the logs by a
rule that never looks at how well a trial tracked.  The rule has two parts.

*Which configuration a session adopted.*  The modal configuration: the
``(Nc, Ne, w_last/w_0)`` triple carrying more of that session's trials than any
other.  It is computed here rather than asserted, and the numbers it selects are
reported alongside the numbers for the session as a whole, so that a reader can
see directly how much of the result the selection is responsible for.  On the
data it is 18 of 23 trials on 14 November and 13 of 21 on 15 November, and it
changes the headline deviation by 1--4 per cent.

*Which sessions are experiments at all.*  The 20 November session swept the
weighting deliberately, on a vehicle with different trailer lengths, and is
reported as the sweep it is.  The other three sessions give two experiments,
because two of them adopted the *same* configuration: the modal configuration of
13 November and that of 15 November are the same triple on the same path and the
same vehicle.  Rather than set one of them aside -- a judgement the data cannot
support -- the earlier session is reported as an independent repetition of
Experiment 2 on a different day, which is a repeatability result and not a
selection.  Trials at neither session's modal configuration are counted and
reported, not silently dropped.

No filter anywhere looks at the outcome.  Dropping the trials that tracked badly
would guarantee a good-looking table and destroy its meaning.
"""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from common import (C_GREY, C_LAST, C_MID, C_PATH, C_TRACTOR, OUT, dump, save,
                    seg_colours, seg_label)

BETA_LIMIT_DEG = 80.0

EXPERIMENTS = [
    dict(key="E1", label="Lemniscate", day="exps_14_nov", path="flat_infinity",
         L=(1.08, 0.78),
         blurb="a smooth closed path at the adopted horizon"),
    dict(key="E2", label="Rectangle", day="exps_15_nov", path="rectangular",
         L=(1.08, 0.78),
         blurb="corners the chain cannot negotiate without leaving the path"),
]
SWEEP = dict(key="E3", label="Weighting sweep", day="exps_20_nov",
             path="rectangular", L=(0.38, 1.08))
# The remaining session.  Its modal configuration turns out to be Experiment 2's,
# so it is reported as a repetition of it rather than as a separate experiment.
OTHER_DAY = "exps_13_nov"


def load():
    with open(os.path.join(OUT, "field_trials.json")) as fh:
        return json.load(fh)


def select(rows, day, path, L, Nc=None, Ne=None, w=None):
    out = []
    for r in rows:
        if r["day"] != day or r["path_kind"] != path:
            continue
        if tuple(np.round(r["L"], 3)) != tuple(np.round(L, 3)):
            continue
        if Nc is not None and r["Nc"] != Nc:
            continue
        if Ne is not None and r["Ne"] != Ne:
            continue
        if w is not None and abs(r["w_last_ratio"] - w) > 1e-6:
            continue
        out.append(r)
    return out


def config(trial):
    """The configuration triple that identifies a trial, to the logged precision."""
    return (trial["Nc"], trial["Ne"], round(trial["w_last_ratio"], 2))


def modal_config(trials):
    """The configuration carrying more of these trials than any other.

    Ties are broken by the lexicographic order of the triple, so the rule is
    deterministic; on this data there are none.  Nothing here looks at how well
    a trial tracked.
    """
    counts = {}
    for t in trials:
        counts[config(t)] = counts.get(config(t), 0) + 1
    return max(sorted(counts), key=lambda k: counts[k])


def stats(trials):
    md = np.stack([t["mean_dev"] for t in trials])
    xd = np.stack([t["max_dev"] for t in trials])
    return dict(
        n=len(trials),
        mean_dev=md.mean(axis=0), mean_dev_sd=md.std(axis=0),
        max_dev=xd.mean(axis=0),
        worst=float(md.mean(axis=0).max()),
        worst_max=float(xd.mean(axis=0).max()),
        t_tot=float(np.mean([t["t_tot"] for t in trials])),
        t_p95=float(np.mean([t["t_p95"] for t in trials])),
        t_mpc=float(np.mean([t["t_mpc"] for t in trials])),
        t_mhe=float(np.mean([t["t_mhe"] for t in trials])),
        t_pfa=float(np.mean([t["t_pfa"] for t in trials])),
        Ts_ms=float(np.mean([t["Ts"] for t in trials]) * 1e3),
        beta_max=float(np.mean([t["beta_max_deg"] for t in trials])),
        beta_over=float(np.mean([t["beta_over_limit_pct"] for t in trials])),
        duration=float(np.mean([t["duration_s"] for t in trials])),
    )


def main():
    rows = load()
    report = {"n_logs": len(rows)}

    for spec in EXPERIMENTS:
        allday = select(rows, spec["day"], spec["path"], spec["L"])
        Nc, Ne, w = modal_config(allday)
        sel = select(rows, spec["day"], spec["path"], spec["L"], Nc=Nc, Ne=Ne, w=w)
        s = stats(sel)
        s_all = stats(allday)
        report[spec["key"]] = {**spec, "Nc": Nc, "Ne": Ne, "w": w, **s,
                               "n_session": len(allday),
                               "n_configs": len({config(t) for t in allday}),
                               "modal_share": len(sel) / len(allday),
                               "worst_session": s_all["worst"]}
        print(f"{spec['key']} {spec['label']:12s} {len(sel):2d}/{len(allday):2d} trials  "
              f"per-seg {np.round(s['mean_dev'],3)}  worst {s['worst']:.3f} m  "
              f"(whole session {s_all['worst']:.3f})  t_tot {s['t_tot']:.0f} ms / Ts {s['Ts_ms']:.0f} ms")

    # The earlier session at Experiment 2's configuration: a repetition on a
    # different day, found by asking for that configuration rather than by
    # nominating the session.
    E2 = report["E2"]
    rep = [r for r in rows
           if r["day"] == OTHER_DAY and r["path_kind"] == E2["path"]
           and tuple(np.round(r["L"], 3)) == tuple(np.round(E2["L"], 3))
           and config(r) == (E2["Nc"], E2["Ne"], round(E2["w"], 2))]
    day_all = [r for r in rows if r["day"] == OTHER_DAY]
    if rep:
        pooled = rep + select(rows, E2["day"], E2["path"], E2["L"],
                              Nc=E2["Nc"], Ne=E2["Ne"], w=E2["w"])
        report["repeat"] = {**stats(rep), "day": OTHER_DAY,
                            "n_day": len(day_all),
                            "n_configs_day": len({config(t) for t in day_all}),
                            "same_as": "E2",
                            "pooled": stats(pooled)}
        r_ = report["repeat"]
        print(f"R  repetition of E2 on {OTHER_DAY}: {r_['n']} of {r_['n_day']} trials "
              f"at the same configuration, per-seg {np.round(r_['mean_dev'],3)}  "
              f"worst {r_['worst']:.3f} m (E2 {E2['worst']:.3f}); "
              f"pooled over both days {np.round(r_['pooled']['mean_dev'],3)}")

    sw = select(rows, SWEEP["day"], SWEEP["path"], SWEEP["L"])
    by_w = {}
    for t in sw:
        by_w.setdefault(round(t["w_last_ratio"], 2), []).append(t)
    report["E3"] = {**SWEEP, "n": len(sw), "points": [
        dict(w=w, n=len(v), mean_dev=list(np.stack([x["mean_dev"] for x in v]).mean(axis=0)),
             worst=float(np.stack([x["mean_dev"] for x in v]).mean(axis=0).max()))
        for w, v in sorted(by_w.items())]}
    print(f"E3 weighting sweep {len(sw)} trials, w_last/w_0 in "
          f"{sorted(by_w)} on the second vehicle")
    for p in report["E3"]["points"]:
        print(f"     w={p['w']:5.2f} n={p['n']:2d}  per-seg {np.round(p['mean_dev'],3)}")

    dump(report, "field_report")
    return report


if __name__ == "__main__":
    main()
