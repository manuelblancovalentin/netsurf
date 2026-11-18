#!/bin/bash
# EXPLORATORY, top-bit analysis plan Phase 3b: the 8 measured-damage masks in the Stage 4 protocol (TFLite replay,
# engine check, test window 0, field seeds [20261119, 0, f], 12 rates), one array task per field = 96 evaluations each.
# Submit after Phase 3a:  sbatch --dependency=afterok:<masks job> topbit_phase3_eval_kws.sbatch
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-7
#SBATCH --job-name=topbit_p3e
#SBATCH --output=/home/mbv6231/netsurf_site/logs/topbit_phase3_eval_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_topbit_analysis.py quest/topbit_phase3_eval_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runR_topbit_analysis.py evaluate --out $RUNS/paper1_topbit_mask_eval_kws_v1 --masks $RUNS/paper1_topbit_masks_kws_v1 \
  --field $SLURM_ARRAY_TASK_ID --threads $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
