#!/bin/bash
# Partial-group search plan: abs scores of replication = array task id, on CPU (the stored abs1024 masks were built on
# CPU; GPU gradients move 85-100 near-tied bits).  Must rebuild the stored abs1024 masks exactly.  About 5 min per task.
#   sbatch --array=0-5 partial_search_scores_kws.sbatch
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=partial_scr
#SBATCH --output=/home/mbv6231/netsurf_site/logs/partial_search_scores_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_partial_search.py quest/partial_search_scores_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runR_partial_search.py scores --rep $SLURM_ARRAY_TASK_ID --out $RUNS/paper1_partial_search_scores_v1 \
  --threads $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
