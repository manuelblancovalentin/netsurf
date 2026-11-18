#!/bin/bash
# EXPLORATORY, group-search plan v2 Phase A (review/GROUP_SEARCH_PLAN_v2_2026-11-14.md): six search paths, one A100 each.
#   task 0-2: real-flip search (mc) at 1e-3 (K = 128), 3e-3 and 1e-2 (K = 32); task 3-5: Gaussian search, K = 16.
# Cost guard: a task stops if one round takes more than 3x its estimate (passes x 7 ms x 3).
# Submit:  sbatch group_search_kws.sbatch ; then the evaluation with --dependency=afterok:<this array id>
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=group_srch
#SBATCH --output=/home/mbv6231/netsurf_site/logs/group_search_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 NVIDIA_TF32_OVERRIDE=0
NV=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
mkdir -p $SITE/libshim && ln -sfn $NV/cuda_nvrtc/lib/libnvrtc.so.11.2 $SITE/libshim/libnvrtc.so
export LD_LIBRARY_PATH=$SITE/libshim:$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
KINDS=(mc mc mc gauss gauss gauss); RATES=(0.001 0.003 0.01 0.001 0.003 0.01); KS=(128 32 32 16 16 16)
T=$SLURM_ARRAY_TASK_ID; KIND=${KINDS[$T]}; RATE=${RATES[$T]}; K=${KS[$T]}
GUARD=$(( 81 * K * 7 * 3 / 1000 + 30 ))
echo "job $SLURM_ARRAY_JOB_ID task $T on $(hostname) at $(date): kind $KIND rate $RATE K $K round guard ${GUARD}s"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'); y = tf.nn.conv2d(tf.ones((1, 8, 8, 4)), tf.ones((3, 3, 4, 2)), 1, 'SAME'); print('GPU convolution ok', float(tf.reduce_sum(y)))"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_group_search.py ../netsurf/group_search.py quest/group_search_kws.sbatch
python runR_group_search.py search --out $SITE/wt/paper1-tecs-revision-v1/workspace/runs/paper1_group_search_kws_v1 \
  --kind $KIND --rate $RATE --K $K --threads 4 --max-round-seconds $GUARD
echo "finished at $(date)"
