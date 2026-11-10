#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

inference_python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
analysis_python=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
ready_manifest=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
result_root="$repo_root/workspace/runs/pythia_w8a8_high_ber_clean_gated_v2"
mkdir -p "$repo_root/workspace/runs"
exec > >(tee -a "$repo_root/workspace/runs/pythia_w8a8_high_ber_clean_gated_v2.log") 2>&1

if [[ -f "$result_root/FINALIZED" ]]; then
    "$analysis_python" -m netsurf.w8a8_pythia_wide_confirm status "$result_root"
    exit 0
fi

if [[ ! -f "$result_root/plan.json" ]]; then
    "$inference_python" -m netsurf.w8a8_pythia_wide_confirm freeze \
        "$ready_manifest" "$result_root"
fi

# Each invocation is a separate process. No fault field is created until all
# four workers' repeated exact clean outputs agree with the finalized source.
printf '0\n1\n2\n3\n' | xargs -P 4 -I '{}' \
    "$inference_python" -m netsurf.w8a8_pythia_wide_confirm preflight-worker \
    "$ready_manifest" "$result_root" '{}'
"$analysis_python" -m netsurf.w8a8_pythia_wide_confirm seal-preflight "$result_root"

# The same four deterministic shards own their fields on every resume. A
# partial field without its clean gate fails closed instead of being retried.
printf '0\n1\n2\n3\n' | xargs -P 4 -I '{}' \
    "$inference_python" -m netsurf.w8a8_pythia_wide_confirm run-shard \
    "$ready_manifest" "$result_root" '{}'

# This phase cannot initialize the model or generate any fault fields.
"$analysis_python" -m netsurf.w8a8_pythia_wide_confirm finalize \
    "$ready_manifest" "$result_root"
