#!/usr/bin/env bash
set -Eeuo pipefail

repo=/projects/netsurf/manuelbv/workspace/implementation-worktrees/kws-transition-validation
inference=/projects/kona-assets/manuelbv/.venv-mlperf/bin/python
analysis=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
root="$repo/workspace/runs/kws_transition_independent_pools_p05_v1"
derived="$repo/workspace/runs/kws_transition_independent_pools_p05_analysis_v1"
state="$repo/workspace/runs/kws_transition_validation.state"
cd "$repo"
test -x "$inference"
test -x "$analysis"
mkdir -p workspace/runs
stage() { printf '%s|%s\n' "$(date -u +%FT%TZ)" "$1" > "$state"; echo "$1"; }
failed() { code=$?; stage "FAILED exit=$code; inspect preserved attempts and log"; exit "$code"; }
trap failed ERR

if test ! -f "$root/campaign.json"; then
    stage "FREEZING 32 new candidate pools and untouched 384-clip final cohort"
    "$analysis" -m netsurf campaign freeze \
        config/experiments/kws_transition_independent_pools_p05_v1.toml --output "$root"
fi
stage "VALIDATING frozen KWS plan"
"$analysis" -m netsurf campaign validate "$root/campaign.json"
stage "SELECTION: exact 64-state tables on calibration inputs"
"$inference" -m netsurf campaign run "$root/campaign.json" --phase selection --no-progress
if test ! -f "$root/final-plan/SEALED"; then
    stage "SEALING all selected masks before any final outcome"
    "$analysis" -m netsurf campaign seal "$root/campaign.json"
fi
stage "FINAL: 32 paired fields per pool on 384 untouched clips"
"$inference" -m netsurf campaign run "$root/campaign.json" --phase final --no-progress
if test ! -f "$derived/FINALIZED"; then
    stage "OFFLINE AGGREGATION: no model inference"
    "$analysis" -m netsurf campaign aggregate "$root/campaign.json" --output "$derived"
fi
stage "COMPLETED: finalized KWS validation analysis"
