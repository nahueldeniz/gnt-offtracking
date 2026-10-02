# Off-tracking minimisation for Generalised N-Trailer vehicles

Reference implementation for *Reachable Reference Generation and Predictive Path
Following for Off-tracking Minimisation of Generalised N-Trailer Vehicles*
(Deniz and Auat Cheein).

Every simulation figure and every number quoted in the paper is produced by a
script in `studies/`. Nothing in the paper is hand-entered.

## What it does

A tractor pulling `N` passive trailers cannot place every segment on a curved
path at once, and a nominal path only says where *one* point should go. The
framework solves both halves of that problem in sequence at each sampling
instant:

1. **Reference generation** (`gntpf/refgen.py`) — a small nonlinear programme
   returns one complete vehicle posture in which the rigid-body geometry of the
   chain holds *exactly* and which is reachable in one step from the previous
   posture under admissible inputs. Only proximity to the path is relaxed, so
   the reference stays attainable even where the path is not.
2. **Tracking** (`gntpf/nmpc.py`) — a conventional NMPC steers the estimated
   state to that posture. Because the reference is reachable by construction, no
   terminal set or terminal control law is needed.

States that are not measured are reconstructed by a moving horizon estimator
(`gntpf/nmhe.py`); an EKF (`gntpf/ekf.py`) is included as the baseline the paper
compares against.

## Install

```bash
pip install --user "casadi>=3.6" "numpy>=1.24" "scipy>=1.10" "matplotlib>=3.7"
pip install -e .         # optional; needs pip >= 21.3 for PEP 660
PYTHONPATH= pytest -q    # 56 tests, ~1 min
```

Requires Python ≥ 3.10, CasADi (with IPOPT), NumPy, SciPy and Matplotlib.

## Quick start

```python
from gntpf import GNT, make_path, simulate, SimConfig

# the field platform: Husky A200 + two passive trailers, generalised hitching
model = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78], Ts=0.05)
path = make_path("rounded_rect")        # a headland geometry

res = simulate(model, path, SimConfig(sigma=1.2, t_final=48.0))
s = res.summary()
print(s["mean_dev"])          # mean off-tracking, one entry per segment
print(s["mean_dev_curved"])   # ... restricted to the curved sections
print(s["t_total_ms"])        # mean solve time per step, all three problems
```

## Reproducing the paper

```bash
cd studies && ./run_all.sh                   # serial, about 9 h
GNT_JOBS=10 ./studies/run_all_parallel.sh    # the two large sweeps in parallel, about 1.5 h
```

Either runner writes figures to `figs/`, raw results to `out/*.json`, and the
LaTeX macros and tables the manuscript reads into `paper/`. Every number and
every simulation figure in the paper comes from those files; nothing is typed in
by hand, and a study that has not been run leaves a visible placeholder rather
than a stale number. The manuscript source itself is not part of this
repository; `out/*.json` and `figs/` hold the results it reports, so they can be
inspected, and compared against a fresh run, without it.

What is and is not reproducible from this repository:

* **Simulation results: fully.** Noise realisations are drawn from seeded
  generators and each run is independent of the order in which the runs are
  executed, so the serial and the parallel runners give the same numbers. A
  different CasADi, IPOPT or linear-solver build can move the last digits,
  because the solver takes a different interior-point path.
* **Solve times: machine-dependent by nature.** The times in the paper were
  measured on one core of a 2.1 GHz Intel Xeon with MUMPS, with every timing
  study run serially. Running those studies in parallel inflates them.
* **Field results: not reproducible without the logs**, which are part of the
  released data set rather than of this repository (see below). Without them the
  field macros expand to placeholders and the field studies are skipped.

Every path is preceded by a lead-in, on which the start-up transient is not
evaluated, and followed by a run-out; closed paths are traversed twice and the
open agricultural path once, and each segment is evaluated only while its own
projection lies inside those traversals. A run whose tractor falls more than two
vehicle lengths behind the progress point for five seconds is reported as not
completed and excluded from the averages. `RUNNING_LOCALLY.md` gives the details
and the expected timings.

| Script | Question it answers |
|---|---|
| `s1_baseline.py` | How does the proposed generator behave when its weight is spread evenly along the chain, or concentrated on the tractor, the middle segment or the last trailer? |
| `s2_estimator.py` | Does the moving horizon estimator earn its cost against an EKF with matched covariances? |
| `s3_inputs.py` | Do the inputs saturate, do the joint angles approach the jackknife limit, and where does the generated reference leave the path? |
| `s4_scaling.py` | What does a solve cost, and beyond how many trailers is it no longer real time? |
| `s5_robustness.py` | How does off-tracking degrade under measurement noise, kinematic parameter error and actuator slip? |
| `s6_corridor_design.py` | How wide a corridor does the vehicle need, and how far can the weighting move it? |
| `s7_varying_n.py` | What happens when trailers are detached mid-run, compared with a vehicle that has two trailers throughout? |
| `s8_comparison.py` | How does the method compare with the reconstruction of Michalek & Pazderski (EJC 2021), and with pure pursuit? |
| `s9_reachability.py` | How large is the reachability residual $e_r$ of Eq. (11) -- how close is each reference to a configuration reachable from the previous one? |
| `verification.py` | What are the measured model-verification residuals quoted in the paper? |

The field studies (`field_analysis.py`, `field_report.py`, `field_figures.py`,
`duro_quality.py`) read the November experiment logs, which are released as a
separate data set: see `DATA.md` for the download, the layout of a log and the
`GNT_FIELD_LOGS` variable. Both runners skip these studies when the logs are
absent.

## Model

State ordering, with `nq = 4N + 3`:

```
q = [ beta_1 .. beta_N , theta_0 .. theta_N , x_0, y_0 , x_1, y_1 .. x_N, y_N ]
u = [ omega_0 , v_0 ]
```

`Lh[i]` is the signed hitch offset of joint `i+1` from the preceding axle and
`L[i]` the length of trailer `i+1`. All hitch offsets zero gives a Standard
N-Trailer, all non-zero a non-Standard one, and a mixture a Generalised one.

The velocity transmission between consecutive segments satisfies
`det J_i = -Lh[i] / L[i]`, so it is singular for every joint angle exactly when
the hitch is on-axle — which is why the generalised class is treated directly
rather than as a perturbation of the standard one.

## Verification

`tests/test_model.py` checks the kinematics against properties established
independently of the implementation, since the same model serves as plant,
predictor and estimator model and an error in it would otherwise be invisible:

- the chain's algebraic constraints are invariants of the flow (violation
  < 1e-7 after 2000 steps, for `N` up to 7, across SNT/nSNT/GNT);
- driving straight monotonically straightens a folded chain, and leaves a
  straight one straight;
- with on-axle hitching, turning on the spot leaves the trailers stationary;
  with an off-axle hitch it does not;
- at constant curvature an SNT settles on `R_i^2 = R_{i-1}^2 - L_i^2`, matched
  to better than 1e-9 m.

## Licence

MIT. If you use this in academic work, please cite the paper.
