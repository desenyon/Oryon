#!/bin/bash
# P5s under protocol A2 (D015): seeds 0-2, validation only; then seed aggregation.
set -u
cd "$(dirname "$0")/.."
for S in 0 1 2; do
  RES=$(ls -d runs/p4-resonator-osc-s$S-* | grep -v 20261104T1724 | grep -v 20261104T1725 | head -1)
  REC=$(ls -d runs/p4-recall-zerofreq-s$S-* | head -1)
  echo "seed $S: $RES $REC"
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$RES
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$REC
  uv run python -m physics_ml.evaluate --config configs/policy_eval_v2.yaml --set model_run=$RES input_noise=0.01
done
uv run python -m physics_ml.aggregate
