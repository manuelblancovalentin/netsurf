#!/bin/bash
# Partial-search follow-up (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md): swap search from abs1024 at rate 1e-2.
# Array task t (0-19) -> replication 1 + t / 4, budget (0.125 0.1875 0.25 0.375)[t % 4].  Move 256 bits, K 32 to
# propose, 128 fresh fields to verify, stop after 3 rejections in a row or 40 rounds.  About 3,300 passes per round
# (16 ms each): at most about 40 min per search; cost guard 240 s per round.
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --job-name=psd_swap
#SBATCH --output=/home/mbv6231/netsurf_site/logs/psearch_d_swap_%A_%a.log
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
T=$SLURM_ARRAY_TASK_ID; REP=$((1 + T / 4)); BUDGETS=(0.125 0.1875 0.25 0.375); B=${BUDGETS[$((T % 4))]}
echo "job $SLURM_ARRAY_JOB_ID task $T on $(hostname) at $(date): replication $REP rate 0.01 budget $B"
sha256sum runR_partial_search.py runR_group_search.py ../netsurf/group_search.py quest/psearch_d_swap_kws.sbatch
python runR_partial_search.py swap --out $RUNS/paper1_psearch_d_swap_v1 --rep $REP --rate 0.01 --budget $B \
  --move 256 --K 32 --K-verify 128 --stop-after 3 --max-rounds 40 --threads 4 --max-round-seconds 240
echo "finished at $(date)"
