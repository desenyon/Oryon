#!/bin/bash
# P4 pilot: sequential (one training process at a time). Usage: scripts/run_pilot.sh SEED
set -u
SEED=${1:-0}
cd "$(dirname "$0")/.."
for TASK in resonator recall; do
  for MODEL in osc zerofreq gru; do
    echo "=== $(date -u +%FT%TZ) start $TASK $MODEL seed $SEED"
    uv run python -m physics_ml.train --config configs/pilot.yaml --set task=$TASK model=$MODEL seed=$SEED
    rc=$?  # capture before $(date) overwrites $? (the earlier logs' rc=0 lines are not trustworthy)
    echo "=== $(date -u +%FT%TZ) end $TASK $MODEL seed $SEED rc=$rc"
  done
done
