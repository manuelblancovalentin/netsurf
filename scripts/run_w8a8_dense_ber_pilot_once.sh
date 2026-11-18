#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
ready_manifest=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
result_root="$repo_root/workspace/runs/w8a8_dense_ber_exposure_pilot_v1"
mkdir -p "$result_root"

for target in distilbert_sst2_int8 pythia160m_int8 qwen3_0_6b_int8; do
    "$python_bin" -m netsurf.w8a8_dense_ber_pilot \
        "$target" "$ready_manifest" "$result_root/$target"
done
