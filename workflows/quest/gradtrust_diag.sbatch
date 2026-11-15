#!/bin/bash
# Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md), Phase 1 diagnostics for one checkpoint card (CPU):
# per-example gradients on 6 calibration draws of 1,024, D1-D3, the t histogram, the calibration-size study.
# VWW (208k weights): about 0.07 s per example on kona, about 15 min in all.   CARD=<card.json> sbatch gradtrust_diag.sbatch
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=gt_diag
#SBATCH --output=/home/mbv6231/netsurf_site/logs/gradtrust_diag_%j.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
echo "job $SLURM_JOB_ID on $(hostname) at $(date): card ${CARD:?}"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runG_gradtrust.py ../netsurf/tflite_graph_companion.py quest/gradtrust_diag.sbatch "$CARD"
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runG_gradtrust.py diag $RUNS/paper1_gradtrust_diag_v1 --card "$CARD" --threads 8 --grad-batch ${GRAD_BATCH:-256}
echo "finished at $(date)"
