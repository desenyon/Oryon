#!/bin/bash
# P5 evaluations (sequential). Validation only.
set -u
cd "$(dirname "$0")/.."
RES=runs/p4-resonator-osc-s0-20261104T172656
REC=$(ls -d runs/p4-recall-zerofreq-s0-* | head -1)
REC_OSC=$(ls -d runs/p4-recall-osc-s0-* | head -1)
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$RES
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$REC
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$RES input_noise=0.01
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$RES input_noise=0.05
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$REC input_noise=0.05
uv run python -m physics_ml.evaluate --config configs/policy_eval.yaml --set model_run=$REC_OSC time_ops=false
