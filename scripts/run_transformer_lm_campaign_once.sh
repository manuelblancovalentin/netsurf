#!/usr/bin/env bash
# Foreground targeted 96-bit LM study. Run only after researcher authorization.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 TARGET /absolute/frozen-plan.json /absolute/frozen-READY.json /absolute/run-dir [user@kona-host]" >&2
  exit 2
fi
readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly registry="$source_root/config/transformer_runtime_registry.json"
readonly target=$1
readonly plan=$2
readonly readiness=$3
readonly run_root=$4
readonly scp_host=${5:-kona}
if [[ $plan != /* || $readiness != /* || $run_root != /* || ! -f $pinned_python ]]; then
  echo "absolute frozen plan, READY snapshot and output path plus pinned interpreter required" >&2
  exit 2
fi
cd "$source_root"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$pinned_python" -m netsurf.transformer_lm_campaign_cli validate "$target" "$plan" "$readiness" "$registry" "$run_root"
"$pinned_python" -m netsurf.transformer_lm_campaign_cli selection "$target" "$plan" "$readiness" "$registry" "$run_root"
"$pinned_python" -m netsurf.transformer_lm_campaign_cli final "$target" "$plan" "$readiness" "$registry" "$run_root"
"$pinned_python" -m netsurf.transformer_lm_campaign_cli aggregate "$target" "$plan" "$readiness" "$registry" "$run_root"
"$pinned_python" -m netsurf.transformer_lm_campaign_cli export "$target" "$plan" "$readiness" "$registry" "$run_root" --scp-host "$scp_host"
cat "${run_root}_export/mac_scp_commands.txt"
