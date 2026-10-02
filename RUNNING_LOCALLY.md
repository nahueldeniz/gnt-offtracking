# Running the studies locally

## Evaluation protocol

Every study builds its path with `common.study_path`:

* a straight **lead-in** of length `1.6 l + 6 m` (with `l` the vehicle length), on
  which the vehicle starts, so that the start-up transient is not evaluated;
* the **nominal span**: two laps of a closed path (rounded rectangle, lemniscate,
  rhombus) or one traversal of the open agricultural path;
* a **run-out** of length `l + 2 m`, so that the last trailer completes the span
  before the progress point reaches the end of the path, which is where a run stops.

A sample of segment `i` is evaluated only while that segment's own projection lies
inside the span (`SimResult.masks`), so every segment, the last trailer included,
is judged over the same stretch of path. A run in which the tractor falls more than
two vehicle lengths behind the progress point for five seconds is declared **not
completed** (`SimResult.meta["diverged"]`): it is reported as such, with the
fraction of the span the last trailer covered, and never averaged with completed
runs.

**Parallel execution.** `studies/parallel.py` spreads the runs of a sweep over
`GNT_JOBS` worker processes; each worker rebuilds its own model, because a `GNT`
carries CasADi objects and is not picklable. The runner parallelises the studies
that report no solve time (`s1`, `s5`, `s6`, `s9`) and runs the others serially,
because a time measured on an oversubscribed machine is not a measurement. The
machine is recorded in `out/machine.json` and quoted in the paper.

```bash
GNT_JOBS=10 ./studies/run_all_parallel.sh
```

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

Roughly 1.2 s of wall-clock per simulated second on one core of a 2.1 GHz Xeon;
a recent laptop is about twice as fast. With the protocol above a run lasts
about 115 s on the agricultural path and 80-110 s on the closed paths.

| study | runs | 10 workers |
|---|---|---|
| `s1_baseline` | 2 paths x 4 weightings x 3 seeds | 15 min |
| `s5_robustness` | 14 values x 4 seeds | 25 min |
| `s6_corridor_design` | (7 tilts + 2 references) x 3 seeds | 15 min |
| `s9_reachability` | 3 seeds | 5 min |
| `s2_estimator` | 3 regimes x 2 estimators x 3 seeds | 45 min, serial |
| `s3_inputs` | 2 runs | 5 min, serial |
| `s7_varying_n` | one lap at N = 4, one at N = 2, and a two-trailer comparison run | 12 min, serial |
| `s8_comparison` | 3 methods x 3 seeds + guidance sweep | 60 min, serial |
| `s4_scaling` | N = 1..12 and Nc = 10..30 | 20 min, serial |

The field studies (`field_analysis`, `field_report`, `field_figures`,
`duro_quality`) read the November experiment logs, which are not in the
repository. `run_all.sh` skips them when the logs are absent; point them at the
experiment folder to re-run them.

`s5` is the long one because it is the only study that sweeps three axes.

`s1_baseline.py`, `s6_corridor_design.py` and `s7_varying_n.py` accept
`--replot`, which redraws their figures from `out/` without re-running them;
`s1` and `s7` keep the trajectories they plot in `out/*_traces.npz`.

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
