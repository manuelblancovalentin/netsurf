#!/usr/bin/env bash
set -euo pipefail

target="${1:?expected frozen W8A8 target}"
case "$target" in
    distilbert_sst2_int8|pythia160m_int8|qwen3_0_6b_int8) ;;
    *) echo "unsupported W8A8 target: $target" >&2; exit 2 ;;
esac
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
analysis_python=/mnt/raid0/ml/conda_envs/manuelbv/envs/netsurf/bin/python
ready_manifest=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
result_root="$repo_root/workspace/runs/w8a8_wide_ber_matched_v1/$target"

if [[ ! -f "$result_root/plan.json" ]]; then
    "$python_bin" -m netsurf.w8a8_wide_ber_campaign freeze \
        "$ready_manifest" "$target" "$result_root"
fi

# Four independent deterministic field shards. xargs remains foreground in tmux.
printf '0\n1\n2\n3\n' | xargs -P 4 -I '{}' \
    "$python_bin" -m netsurf.w8a8_wide_ber_campaign run-shard \
    "$ready_manifest" "$result_root" '{}' 4

"$analysis_python" -m netsurf.w8a8_wide_ber_campaign finalize \
    "$ready_manifest" "$result_root"
