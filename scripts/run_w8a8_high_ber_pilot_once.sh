#!/usr/bin/env bash
set -euo pipefail
readonly root=/projects/netsurf/manuelbv/workspace/implementation-worktrees/w8a8-ber-stress
readonly python=/projects/kona-assets/manuelbv/envs/torch-2.9.1-cpu-py311/bin/python
readonly ready=/projects/kona-assets/manuelbv/NETSURF_ASSET_READINESS.json
readonly outputs="$root/workspace/runs/w8a8_high_ber_exposure_pilot_v1"
cd "$root"
export PYTHONPATH="$root" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
for target in distilbert_sst2_int8 pythia160m_int8 qwen3_0_6b_int8; do
  "$python" -m netsurf.w8a8_high_ber_pilot "$target" "$ready" "$outputs/$target"
done
