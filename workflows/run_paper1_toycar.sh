#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=-1
exec /projects/kona-assets/manuelbv/.venv-mlperf/bin/python -m workflows.paper1_campaign toycar "$@"
