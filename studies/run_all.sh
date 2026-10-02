#!/bin/bash
# Run every study in sequence, then regenerate the paper's numbers.
# See RUNNING_LOCALLY.md for a parallel variant and expected timings.
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)"
export OMP_NUM_THREADS=1

STUDIES="${@:-verification s4_scaling s3_inputs s1_baseline s2_estimator s6_corridor_design s7_varying_n s8_comparison s5_robustness}"

for s in $STUDIES; do
  echo "=============== $s  ($(date +%H:%M:%S)) ==============="
  python3 ${s}.py || echo "!!! $s FAILED"
done
# The field studies read the experiment logs, which are not in the repository;
# they are skipped without them.  duro_quality.py needs the same logs.
for s in field_analysis field_report field_figures duro_quality; do
  python3 ${s}.py 2>/dev/null || echo "--- $s skipped (field logs not present)"
done
python3 make_diagrams.py
python3 make_results_tex.py
python3 make_tables.py
python3 sync_figs.py
echo "ALL DONE $(date +%H:%M:%S)"
