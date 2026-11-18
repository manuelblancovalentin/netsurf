#!/bin/bash
# EXPLORATORY, top-bit analysis plan Phase 3a: measured-damage masks from the calibration clips, and Q4 (gradient scores
# against measured top-bit damage).  CPU.  Phase 3b (topbit_phase3_eval_kws.sbatch) depends on it.
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=topbit_p3m
#SBATCH --output=/home/mbv6231/netsurf_site/logs/topbit_phase3_masks_%j.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_topbit_analysis.py quest/topbit_phase3_masks_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runR_topbit_analysis.py masks --out $RUNS/paper1_topbit_masks_kws_v1 --threads $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
