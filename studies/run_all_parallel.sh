#!/bin/bash
# Re-run every study, with the sweeps spread over several cores.
#
#   ./studies/run_all_parallel.sh            # 10 workers for the sweeps
#   GNT_JOBS=4 ./studies/run_all_parallel.sh # 4 workers
#
# Durations give the LAST trailer at least two laps of a closed path, or one
# full traversal of the open agricultural path, inside the averaging window --
# that is, after the first 15 % of each run is discarded as transient.
#
# Studies whose results include a measured solve time run serially and alone,
# because a time measured on an oversubscribed machine means nothing:
#   s1 (cost of a step per weighting), s2 (cost of the estimator),
#   s4 (the scaling study), s8 (online cost), s3, s7.
# The two large sweeps, s5 and s6, report no timing and run in parallel.
set -u
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)"
export OMP_NUM_THREADS=1
JOBS="${GNT_JOBS:-10}"

run () {  # run <script> <jobs>
  echo "=============== $1  (jobs=$2, $(date +%H:%M:%S)) ==============="
  GNT_JOBS=$2 python3 "$1.py" || echo "!!! $1 FAILED"
}

run verification 1
run s5_robustness "$JOBS"
run s6_corridor_design "$JOBS"
run s9_reachability "$JOBS"
run s1_baseline 1
run s2_estimator 1
run s3_inputs 1
run s7_varying_n 1
run s8_comparison 1
run s4_scaling 1

# The field studies read the experiment logs, which are not in the repository.
for s in field_analysis field_report field_figures duro_quality; do
  python3 ${s}.py 2>/dev/null || echo "--- $s skipped (field logs not present)"
done

python3 make_diagrams.py
python3 make_results_tex.py
python3 make_tables.py
python3 sync_figs.py
echo "ALL DONE $(date +%H:%M:%S)"
