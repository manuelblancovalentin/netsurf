#!/bin/bash
# Partial-search follow-up evaluation (review/PARTIAL_SEARCH_FOLLOWUP_PREREG_2026-11-14.md): the Stage 4 protocol
# (TFLite) for arm ARM (partial_k32, partial_k128 or swap) from search folder NAME (paper1_psearch_d_k32,
# paper1_psearch_d_k128 or paper1_psearch_d_swap).  Array task t (0-39) -> replication 1 + t / 8, field t % 8.
# About 2 min per task.  Submit with --dependency=afterok:<search>.
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-39
#SBATCH --job-name=psd_eval
#SBATCH --output=/home/mbv6231/netsurf_site/logs/psearch_d_eval_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
T=$SLURM_ARRAY_TASK_ID; REP=$((1 + T / 8)); F=$((T % 8))
echo "job $SLURM_ARRAY_JOB_ID task $T on $(hostname) at $(date): arm ${ARM:?} from ${NAME:?}, replication $REP field $F"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_partial_search.py quest/psearch_d_eval_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runR_partial_search.py evaluate --rep $REP --out $RUNS/${NAME}_eval_v1 --search $RUNS/${NAME}_v1 \
  --arm $ARM --field $F --threads $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
