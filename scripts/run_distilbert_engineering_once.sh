#!/usr/bin/env bash
# Foreground, write-once engineering conformance; no production effectiveness claim.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /absolute/NETSURF_ASSET_READINESS.json /absolute/output-dir [user@kona-host]" >&2
  exit 2
fi

readonly source_root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly verifier_python=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
readonly readiness_manifest=$1
readonly output_dir=$2
readonly scp_host=${3:-kona}

if [[ $readiness_manifest != /* || $output_dir != /* ]]; then
  echo "manifest and output must be absolute paths" >&2
  exit 2
fi

cd "$source_root"
export PYTHONPATH="$source_root"
ready_json=$($verifier_python -m netsurf.transformer_cli verify distilbert_sst2_fp32 "$readiness_manifest")
pinned_python=$(printf '%s' "$ready_json" | "$verifier_python" -c 'import json,sys; print(json.load(sys.stdin)["interpreter"])')
if [[ $pinned_python != /* || ! -f $pinned_python ]]; then
  echo "READY handoff does not identify an absolute interpreter" >&2
  exit 2
fi

"$pinned_python" -m netsurf.transformer_cli smoke-distilbert "$readiness_manifest" "$output_dir"
"$pinned_python" -m netsurf.transformer_cli aggregate-smoke "$output_dir"
"$pinned_python" -m netsurf.transformer_cli export-smoke "$output_dir" "$readiness_manifest" --scp-host "$scp_host"
