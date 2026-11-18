#!/bin/bash
# EXPLORATORY, group-search plan v2 Phase A evaluation: every finished search mask in the Stage 4 protocol (TFLite),
# one array task per fault field.  Submit with --dependency=afterok:<search array id>.
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --job-name=group_eval
#SBATCH --output=/home/mbv6231/netsurf_site/logs/group_search_eval_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_group_search.py quest/group_search_eval_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runR_group_search.py evaluate --out $RUNS/paper1_group_search_kws_eval_v1 --search $RUNS/paper1_group_search_kws_v1 \
  --field $SLURM_ARRAY_TASK_ID --threads $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
