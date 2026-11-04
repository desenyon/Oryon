#!/bin/bash
# P5s under protocol A2 (D015): seeds 0-2, validation only; then seed aggregation.
set -euo pipefail
cd "$(dirname "$0")/.."
for S in 0 1 2; do
  RES=$(uv run python scripts/pick_run.py resonator osc $S)
  REC=$(uv run python scripts/pick_run.py recall zerofreq $S)
  echo "seed $S: $RES $REC"
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$RES
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$REC
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$RES input_noise=0.01
done
uv run python -m physics_ml.aggregate
