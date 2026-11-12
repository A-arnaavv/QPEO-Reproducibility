#!/usr/bin/env bash
# Carbon-v2 small pilot: 13 algorithms × 1 seed × P=4 × T=2.
# Does NOT run the locked 30-seed / 500-iteration experiment.
# Does NOT trigger GitHub Actions or overwrite original __final.json shards.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q \
  tests/test_time_aligned_carbon_accounting.py \
  tests/test_time_aligned_carbon_neso_window.py \
  tests/test_time_aligned_carbon_full_suite.py \
  tests/test_carbon_v2_fixed_planning_full_suite.py \
  tests/test_temporal_cpu_capacity_v2.py \
  tests/test_cpu_carbon_joint_v2.py

python scripts/run_final_experiment_shard.py \
  --scenario microsoft_elasticflow_itp_cluster08 \
  --seed-index 0 \
  --population-size 4 \
  --max-iterations 2 \
  --carbon-time-accounting repeat \
  --output-dir results/carbon_v2_pilot

output="results/carbon_v2_pilot/microsoft_elasticflow_itp_cluster08__seed00_104729__test-P4-T2-N64-carbon-time-v2-repeat-fixedmean.json"
if [[ ! -f "$output" ]]; then
  echo "ERROR: expected carbon-v2 pilot shard not found: $output" >&2
  exit 1
fi
python scripts/validate_carbon_v2_pilot.py "$output" \
  --report-file results/carbon_v2_pilot/pilot_validation_report.json

echo "Carbon-v2 small pilot completed and validated. NO primary benchmark was rerun."
