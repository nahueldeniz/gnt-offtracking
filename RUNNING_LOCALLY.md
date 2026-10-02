# Running the studies locally

## What changed in this update

**Run durations.** Every study now runs long enough for the *last trailer* to
cover at least two laps of a closed path, or one full traversal of the open
agricultural path, *inside the averaging window* -- that is, after the first
15 % of the run is discarded as transient. The earlier durations gave about one
lap before the discard and less than one after it, so part of what was reported
as steady behaviour was still the start-up transient. The durations are:

| study | path | sigma (m/s) | t_final (s) |
|---|---|---|---|
| s1_baseline | rounded rectangle | 1.2 | 105 |
| s2_estimator | lemniscate / square | 1.0 | 118 / 45 |
| s3_inputs | agricultural / rounded rect | 1.0 / 1.2 | 160 / 105 |
| s5_robustness | agricultural | 1.0 | 160 |
| s6_corridor_design | agricultural | 1.0 | 160 |
| s7_varying_n | rounded rectangle | 1.1 | 115 + 115 |
| s8_comparison | agricultural | 1.0 | 160 |

**Parallel execution.** `studies/parallel.py` spreads the runs of a sweep over
`GNT_JOBS` worker processes. Each worker rebuilds its own model, because a
`GNT` carries CasADi objects and is not picklable. Only `s5_robustness` and
`s6_corridor_design` are run in parallel by the runner: every other study
reports a measured solve time, and a time measured on an oversubscribed machine
is not a measurement.

```bash
GNT_JOBS=10 ./studies/run_all_parallel.sh     # about 1.5 h on 10 cores
```

Expect the sweeps to use one core per worker and about 400 MB of memory each.

## Setup

The studies do not need the package to be installed -- the runners put the
repository root on `PYTHONPATH`. Only the dependencies are required:

```bash
pip install --user "casadi>=3.6" "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"
```

An editable install is optional, and needs pip 21.3 or newer for PEP 660:

```bash
cd gnt
pip install -e .            # setup.py is a shim for older pip
PYTHONPATH= pytest -q       # 56 tests, ~1 min
```

`PYTHONPATH=` clears a ROS environment leak that otherwise breaks pytest
collection. `pyproject.toml` already disables the offending plugins.

## Running

```bash
cd studies
./run_all.sh                # everything
./run_all.sh s8_comparison  # one study
```

Both runners finish by regenerating `paper/results.tex` and the result tables
(`make_tables.py`), which is where every number quoted in the manuscript comes
from. Nothing is typed into the text by
hand; if a study has not been run its macro expands to a visible placeholder.

### Running in parallel

```bash
GNT_JOBS=10 ./studies/run_all_parallel.sh
```

The runner spreads `s5_robustness` and `s6_corridor_design` over `GNT_JOBS`
workers and runs every other study serially, for the reason given above. To
parallelise a single study by hand:

```bash
cd studies
export PYTHONPATH="$(cd .. && pwd)" OMP_NUM_THREADS=1
GNT_JOBS=10 python3 s5_robustness.py
```

Keep `OMP_NUM_THREADS=1`: IPOPT threading fights with process-level parallelism
and makes the whole set slower.

## Expected timings

With the durations above, on one core of a 2.1 GHz Xeon (about 1.2 s of
wall-clock per simulated second):

| study | runs | serial | 10 workers |
|---|---|---|---|
| `verification` | 6 geometries x 2000 steps | 2 min | -- |
| `s5_robustness` | 13 values x 4 seeds x 160 s | 3.5 h | 25 min |
| `s6_corridor_design` | (7 tilts + 2 references) x 3 seeds x 160 s | 2.2 h | 15 min |
| `s1_baseline` | 4 weightings x 3 seeds x 105 s | 30 min | serial by design |
| `s2_estimator` | 3 regimes x 2 estimators x 3 seeds | 40 min | serial by design |
| `s3_inputs` | 2 runs | 6 min | serial by design |
| `s7_varying_n` | 115 s at N = 4 then 115 s at N = 2 | 15 min | serial by design |
| `s8_comparison` | 3 methods x 3 seeds x 160 s + the offline fit | 50 min | serial by design |
| `s4_scaling` | N = 1..12 and Nc = 10..30 | 20 min | serial by design |

A faster machine scales these roughly with single-core clock. On ten cores of a
recent desktop the whole set takes about 1.5 h.

The field studies (`field_analysis`, `field_report`, `field_figures`,
`duro_quality`) read the November experiment logs, which are not in the
repository. `run_all.sh` skips them when the logs are absent; point them at the
experiment folder to re-run them.

`s5` is the long one because it is the only study that sweeps three axes.

## Already run here

`s8_comparison` has been run on all three paths and its results are in `out/`
and `figs/`. It does not need re-running unless you change something.

## After running

```bash
cp figs/*.pdf paper/Figs/
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`Figs/exp1.eps`, `exp2.eps` and `exp3.eps` need `pdflatex --shell-escape` (or
convert them to PDF once); this is pre-existing and unrelated to the studies.

## What to check in the output

Each study prints its own numbers as it goes. Two things are worth a glance:

- `fails` should be 0 everywhere. A non-zero `ref_fail` or `mpc_fail` means a
  solver did not converge, and the affected run should not be reported.
- In `s8`, the Michalek-Pazderski *fit residual* is printed before the
  closed-loop runs. On the lemniscate it is about 0.04 (converged); on the
  agricultural path about 0.66, because their method assumes a periodic guiding
  velocity and a boustrophedon path's curvature is piecewise constant with
  jumps. That number belongs in the paper -- it is the honest measure of how
  well their method applies to this geometry, and it should not be quietly
  dropped.
