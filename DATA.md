# Field data set

The field experiments of Section IX were logged on the generalised two-trailer
platform at AC3E, Universidad Técnica Federico Santa María, in November 2022.
The logs are released as a data set of their own rather than inside this
repository, because they are measurements and not code.

## Getting them

Download the data set, then point the studies at it:

```bash
export GNT_FIELD_LOGS=/path/to/field_logs      # the folder holding the per-session subfolders
cd studies && python3 field_analysis.py && python3 field_report.py \
           && python3 field_figures.py && python3 duro_quality.py
```

Without `GNT_FIELD_LOGS` the four field studies are skipped, the field macros in
`paper/results.tex` expand to visible placeholders, and every simulation result
is unaffected.

## Layout

One subfolder per session, named by date, each holding one MATLAB v7.3 (HDF5)
file per trial. Each file has a single structure `S`:

| Field | Contents |
|---|---|
| `S/config/N`, `Nc`, `Ne`, `Ts`, `vNr` | trailers, control horizon, estimation window, sampling period, reference speed |
| `S/system/Lhi`, `Li` | hitch offsets and trailer lengths, in metres |
| `S/path/path`, `S/path/coordinates` | the geometry's name and its sampled centre-line |
| `S/data/xest` | the estimated state, one row per instant, in the ordering of `README.md` |
| `S/data/ysim` | the measurements: joint angles, tractor heading, tractor position |
| `S/Mtxs/QrefOpt` | the weighting matrix of the reference generator, whose position entries give the per-segment weights |
| `S/exec_time/t_tot`, `t_mpc`, `t_mhe`, `t_pfa` | measured cost of a step and of each problem, in seconds |

The trailers carried no positioning instrument, so every deviation computed from
these logs is that of the *estimated* posture. A Swift Duro receiver on the last
trailer is logged but is not a reference: it never reached a real-time-kinematic
fix during the campaign, and `duro_quality.py` reproduces that check from the
logs themselves.
