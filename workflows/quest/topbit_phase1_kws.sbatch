#!/bin/bash
# EXPLORATORY, top-bit analysis plan Phase 1 (review/TOPBIT_ANALYSIS_PLAN_2026-11-14.md), one GPU:
#   logits      every bit-6/7 single flip again, keeping the INT8 output code of all 12 classes per clip (about 5 min,
#               about 0.7 GB), compared clip by clip with the margins run paper1_topbit_flips_kws_v1;
#   background  288 Stage 4 cells x (16 Gaussian low-bit passes + 8 real low-bit passes + 8 real full passes) = 9,216
#               network passes (a few minutes).  Needs paper1_topbit_cells_kws_v1 copied from kona.
# Submit from the Quest login node:   sbatch $HOME/netsurf_site/wt/paper1-runR-v2-validated/workflows/quest/topbit_phase1_kws.sbatch
# Follow progress:                    tail -F $HOME/netsurf_site/logs/topbit_phase1_<jobid>.log
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=topbit_p1
#SBATCH --output=/home/mbv6231/netsurf_site/logs/topbit_phase1_%j.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 NVIDIA_TF32_OVERRIDE=0
NV=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
mkdir -p $SITE/libshim && ln -sfn $NV/cuda_nvrtc/lib/libnvrtc.so.11.2 $SITE/libshim/libnvrtc.so
export LD_LIBRARY_PATH=$SITE/libshim:$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); assert g, 'no GPU'; y = tf.nn.conv2d(tf.ones((1, 8, 8, 4)), tf.ones((3, 3, 4, 2)), 1, 'SAME'); print('GPU convolution ok', float(tf.reduce_sum(y)))"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_topbit_flips.py site_paths.py runR_kws.py quest/topbit_phase1_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
test -f $RUNS/paper1_topbit_cells_kws_v1/cells.npz || { echo "cells missing: copy paper1_topbit_cells_kws_v1 from kona"; exit 1; }
python runR_topbit_flips.py logits --out $RUNS/paper1_topbit_logits_kws_v1 --compare $RUNS/paper1_topbit_flips_kws_v1 --threads 4 --log-every 4000
python runR_topbit_flips.py background --out $RUNS/paper1_topbit_background_kws_v1 --cells $RUNS/paper1_topbit_cells_kws_v1/cells.npz --threads 4
echo "finished at $(date)"
