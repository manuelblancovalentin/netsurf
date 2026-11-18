#!/usr/bin/env bash
set -euo pipefail
target="${1:?expected target}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p workspace/runs
set -o pipefail
bash scripts/run_w8a8_wide_ber_model.sh "$target" 2>&1 | \
    tee "workspace/runs/w8a8_wide_ber_matched_v1_${target}.log"
