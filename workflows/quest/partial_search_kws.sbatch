#!/bin/bash
# Partial-group search plan (review/PARTIAL_GROUP_SEARCH_PLAN_2026-11-14.md), Phase C1/C2 search: one A100 per path.
# Array task t -> rate RATES[t % 3] of replication REPS[t / 3]; REPS from the environment (default "0"); PARTS (default 4).
# K = 32.  Estimate 1.2-2.2 h per path (<= 2,600 passes x 15 ms per round, 110-200 rounds); cost guard 120 s per round.
#   sbatch --array=0-2 partial_search_kws.sbatch                       (C1, replication 0)
#   REPS="1 2 3 4 5" sbatch --array=0-14 partial_search_kws.sbatch     (C2)
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=partial_srch
#SBATCH --output=/home/mbv6231/netsurf_site/logs/partial_search_%A_%a.log
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
read -a RS <<< "${REPS:-0}"; RATES=(0.003 0.01 0.03)
T=$SLURM_ARRAY_TASK_ID; REP=${RS[$((T / 3))]}; RATE=${RATES[$((T % 3))]}; PARTS=${PARTS:-4}
echo "job $SLURM_ARRAY_JOB_ID task $T on $(hostname) at $(date): replication $REP rate $RATE parts $PARTS K 32"
sha256sum runR_partial_search.py runR_group_search.py ../netsurf/group_search.py quest/partial_search_kws.sbatch
python runR_partial_search.py search --out $RUNS/paper1_partial_search_kws_v1 --rep $REP --rate $RATE --parts $PARTS \
  --K 32 --threads 4 --max-round-seconds 120
echo "finished at $(date)"
