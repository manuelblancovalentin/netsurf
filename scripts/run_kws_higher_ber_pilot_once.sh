#!/usr/bin/env bash
set -Eeuo pipefail

repo=/projects/netsurf/manuelbv/workspace/implementation-worktrees/kws-ber-transition-pilot
run_root="$repo/workspace/runs/kws_higher_ber_pilot_v1"
manifest="$run_root/campaign.json"
analysis="$repo/workspace/runs/kws_higher_ber_pilot_analysis_v1"
inference_py=/projects/kona-assets/manuelbv/.venv-mlperf/bin/python
analysis_py=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python

cd "$repo"
test -f "$manifest"
test -x "$inference_py"
test -x "$analysis_py"
test ! -e "$analysis"
printf '%s\n' "$$" > "$run_root/runner.pid"
exec > >(tee -a "$run_root/runner.log") 2>&1

stage() {
  printf '%s|%s\n' "$(date -u +%FT%TZ)" "$1" > "$run_root/runner.stage"
  printf '%s  %s\n' "$(date -u +%FT%TZ)" "$1"
}
failed() {
  code=$?
  stage "FAILED exit=$code; inspect runner.log and incomplete attempt records"
  exit "$code"
}
trap failed ERR

stage "VALIDATING frozen campaign and pinned assets"
"$inference_py" -m netsurf multiber validate "$manifest"
stage "SELECTION exact 64-state tables, all BERs and pools"
"$inference_py" -m netsurf multiber run "$manifest" --phase selection --no-progress
if test ! -f "$run_root/SEALED"; then
  stage "SEALING all independently selected masks"
  "$inference_py" -m netsurf multiber seal "$manifest"
fi
stage "FINAL paired fault-field evaluation, all BERs and pools"
"$inference_py" -m netsurf multiber run "$manifest" --phase final --no-progress
stage "AGGREGATING saved results; no model inference"
"$analysis_py" -m netsurf multiber aggregate "$manifest" --output "$analysis"
stage "COMPLETED; finalized analysis at $analysis"
