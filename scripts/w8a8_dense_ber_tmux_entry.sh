#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p workspace/runs
set -o pipefail
bash scripts/run_w8a8_dense_ber_pilot_once.sh 2>&1 | tee workspace/runs/w8a8_dense_ber_exposure_pilot_v1.log
