#!/usr/bin/env bash
# One foreground engineering receipt; no production Bernoulli campaign.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 TARGET /absolute/output-dir [user@kona-host]" >&2
  exit 2
fi

readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly pinned_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly readiness=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
readonly registry="$source_root/config/transformer_runtime_registry.json"
readonly target=$1
readonly output=$2
readonly scp_host=${3:-kona}

if [[ $output != /* || ! -f $pinned_python || ! -f $readiness ]]; then
  echo "absolute output and provisioned Kona interpreter/READY manifest required" >&2
  exit 2
fi
cd "$source_root"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$pinned_python" -m netsurf.transformer_lm_smoke_cli "$target" "$readiness" "$registry" "$output" --scp-host "$scp_host"
cat "$output/mac_scp_commands.txt"
