#!/bin/bash
# Partial-search follow-up (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md): greedy slice search at rate 1e-2,
# 4 slices per group, K from the environment (32 or 128).  Array task t -> replication 1 + t (t = 0-4).
# Estimate about 1.8 h per path at K = 32 and 7.5 h at K = 128 (16 ms per pass); cost guard 4 s x K per round.
#   K=32  sbatch --array=0-4 --time=04:00:00 psearch_d_greedy_kws.sbatch
#   K=128 sbatch --array=0-4 --time=12:00:00 psearch_d_greedy_kws.sbatch
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=psd_greedy
#SBATCH --output=/home/mbv6231/netsurf_site/logs/psearch_d_greedy_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 NVIDIA_TF32_OVERRIDE=0
NV=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
mkdir -p $SITE/libshim && ln -sfn $NV/cuda_nvrtc/lib/libnvrtc.so.11.2 $SITE/libshim/libnvrtc.so
export LD_LIBRARY_PATH=$SITE/libshim:$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'); y = tf.nn.conv2d(tf.ones((1, 8, 8, 4)), tf.ones((3, 3, 4, 2)), 1, 'SAME'); print('GPU convolution ok', float(tf.reduce_sum(y)))"
cd $SITE/wt/paper1-runR-v2-validated/workflows
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
K=${K:?set K to 32 or 128}; REP=$((1 + SLURM_ARRAY_TASK_ID))
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date): replication $REP rate 0.01 parts 4 K $K"
sha256sum runR_partial_search.py runR_group_search.py ../netsurf/group_search.py quest/psearch_d_greedy_kws.sbatch
python runR_partial_search.py search --out $RUNS/paper1_psearch_d_k${K}_v1 --rep $REP --rate 0.01 --parts 4 \
  --K $K --threads 4 --max-round-seconds $((4 * K))
echo "finished at $(date)"
