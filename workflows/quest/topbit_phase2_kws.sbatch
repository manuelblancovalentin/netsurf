#!/bin/bash
# EXPLORATORY, top-bit analysis plan Phase 2 (review/TOPBIT_ANALYSIS_PLAN_2026-11-14.md), CPU only:
# predictions of the 288 Stage 4 cells (additive events with and without background, bell-curve and exact lattice
# readouts) against the Stage 4 TFLite accuracy.  First a 12-cell smoke run into scratch (fails fast), then the full run.
# Submit from the Quest login node:   sbatch $HOME/netsurf_site/wt/paper1-runR-v2-validated/workflows/quest/topbit_phase2_kws.sbatch
# Follow progress:                    tail -F $HOME/netsurf_site/logs/topbit_phase2_<jobid>.log
#SBATCH --account=p33112
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=topbit_p2
#SBATCH --output=/home/mbv6231/netsurf_site/logs/topbit_phase2_%j.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "job $SLURM_JOB_ID on $(hostname) at $(date), $SLURM_CPUS_PER_TASK cores"
cd $SITE/wt/paper1-runR-v2-validated/workflows
sha256sum runR_topbit_analysis.py ../netsurf/event_readout.py quest/topbit_phase2_kws.sbatch
RUNS=$SITE/wt/paper1-tecs-revision-v1/workspace/runs
for d in paper1_topbit_cells_kws_v1 paper1_topbit_logits_kws_v1 paper1_topbit_background_kws_v1 paper1_runR_stage4_kws_eval_v1/rep0; do
  test -e $RUNS/$d || { echo "missing $RUNS/$d"; exit 1; }
done
SMOKE=$SITE/scratch/topbit_phase2_smoke_$SLURM_JOB_ID
echo "== smoke: first 12 cells"
python runR_topbit_analysis.py analyze --out $SMOKE --max-cells 12 --workers $SLURM_CPUS_PER_TASK
python -c "import json; d = json.load(open('$SMOKE/summary.json')); print(json.dumps(d['q1_cells'], indent=0)[:1500])"
echo "== full: 288 cells"
python runR_topbit_analysis.py analyze --out $RUNS/paper1_topbit_analysis_kws_v1 --workers $SLURM_CPUS_PER_TASK
echo "finished at $(date)"
