#!/usr/bin/env bash
# Run R, Stage 2 (Stage 2 addendum): wait until no Stage 1 job is running, then run the phases in order, up to 76
# processes each (the mask extraction 9).  Stops at the first phase with a failed job (the launcher exits 1).
# Launch approved by the researcher on 12 November 2026.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
while pgrep -f "runR_stage1.py job" > /dev/null; do sleep 60; done
echo "Stage 1 jobs ended: $(date)"
for phase in stage2_bundle stage2_build stage2_eval_main stage2_mc stage2_eval_mc; do
  echo "start $phase: $(date)"
  "$HERE/runR_launch.sh" "$phase" 76
done
echo "Stage 2 finished: $(date)"
