#!/bin/bash
# P4 seeds 1..2 for uncertainty on the physics-specific comparison (sequential).
set -u
cd "$(dirname "$0")/.."
for SEED in 1 2; do scripts/run_pilot.sh $SEED; done
