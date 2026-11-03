#!/usr/bin/env bash
# Foreground prospective replication; selection is sealed before final fields.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 /absolute/plan.json /absolute/frozen-READY.json /absolute/run-dir [user@kona-host]" >&2
  exit 2
fi

readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly plan=$1
readonly readiness=$2
readonly run_root=$3
readonly scp_host=${4:-kona}

if [[ $plan != /* || $readiness != /* || $run_root != /* || ! -f $pinned_python ]]; then
  echo "absolute frozen plan, READY snapshot, output and pinned Kona interpreter required" >&2
  exit 2
fi
cd "$source_root"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$pinned_python" -m netsurf.transformer_campaign_cli validate "$plan" "$readiness" "$run_root"
"$pinned_python" -m netsurf.transformer_campaign_cli selection "$plan" "$readiness" "$run_root"
"$pinned_python" -m netsurf.transformer_campaign_cli final "$plan" "$readiness" "$run_root"
"$pinned_python" -m netsurf.transformer_campaign_cli aggregate "$plan" "$readiness" "$run_root"
"$pinned_python" -m netsurf.transformer_campaign_cli export "$plan" "$readiness" "$run_root" --scp-host "$scp_host"
cat "${run_root}_export/mac_scp_commands.txt"
