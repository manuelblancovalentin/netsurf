#!/bin/bash
# Gradient-trust plan (review/GRADIENT_TRUST_PLAN_2026-11-15.md), Phase 2 fault runs for one checkpoint card (CPU,
# TFLite reference kernels).  Array task t (0-47) -> replication t / 8, field t % 8.  78 evaluations per task (13 rates
# x 2 budgets x 3 arms) on the replication's evaluation window.  MLPs: seconds per task; VWW: time the smoke task first.
#   CARD=<card.json> sbatch --array=0-47 gradtrust_faults.sbatch
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=gt_faults
#SBATCH --output=/home/mbv6231/netsurf_site/logs/gradtrust_faults_%A_%a.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
T=$SLURM_ARRAY_TASK_ID; REP=$((T / 8)); F=$((T % 8))
echo "job $SLURM_ARRAY_JOB_ID task $T on $(hostname) at $(date): card ${CARD:?} replication $REP field $F"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runG_gradtrust.py ../netsurf/tflite_graph_companion.py quest/gradtrust_faults.sbatch "$CARD"
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
python runG_gradtrust.py faults $RUNS/paper1_gradtrust_faults_v1 --card "$CARD" --diag $RUNS/paper1_gradtrust_diag_v1 \
  --rep $REP --field $F --threads 2
echo "finished at $(date)"
