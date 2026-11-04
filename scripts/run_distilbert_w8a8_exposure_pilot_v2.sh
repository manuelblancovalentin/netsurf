#!/usr/bin/env bash
# One foreground command, after researcher approval of this separately frozen pilot.
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/transformer-integration
readonly python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly ready=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
readonly plan="$root/workspace/plans/distilbert_sst2_w8a8_exposure_pilot_v2"
readonly run="$root/workspace/runs/distilbert_sst2_w8a8_exposure_pilot_v2_run1"
cd "$root"
export PYTHONPATH="$root" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
"$python" -m netsurf.w8a8_exposure_pilot validate "$ready" "$plan"
cpus=$("$python" scripts/w8a8_exposure_pilot_gate_v2.py "$ready" "$plan" "$run")
taskset -c "$cpus" /usr/bin/time -v "$python" -m netsurf.w8a8_exposure_pilot run "$ready" "$plan" "$run"
"$python" -m netsurf.w8a8_exposure_pilot aggregate "$ready" "$plan" "$run"
"$python" -m netsurf.w8a8_exposure_pilot export "$ready" "$plan" "$run"
cat "${run}_export/mac_scp_commands.txt"
