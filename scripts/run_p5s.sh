#!/bin/bash
# P5s: robustness of D011/D012 on seed 1-2 models (validation only), then seed aggregation.
set -u
cd "$(dirname "$0")/.."
for S in 1 2; do
  RES=$(ls -d runs/p4-resonator-osc-s$S-* | head -1)
  REC=$(ls -d runs/p4-recall-zerofreq-s$S-* | head -1)
  uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$RES
  uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$REC
  uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$RES input_noise=0.01
done
uv run python -m physics_ml.aggregate
