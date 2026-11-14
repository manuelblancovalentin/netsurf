#!/bin/bash
# EXPLORATORY: every top-bit (6, 7) single flip of the DS-CNN, per clip, on one GPU (runR_topbit_flips.py).
# Phase 1 checks the GPU against the CPU reference (the blur pilot's 800 measured flips) and times one flip;
# phase 2 measures all 44,032 flips, one output file per layer, restartable.
# Submit from the Quest login node:   sbatch $HOME/netsurf_site/wt/paper1-runR-v2-validated/workflows/quest/topbit_flips_kws.sbatch
# Follow progress:                    tail -f $HOME/netsurf_site/logs/topbit_kws_<jobid>.log      and   squeue -u $USER
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=topbit_kws
#SBATCH --output=/home/mbv6231/netsurf_site/logs/topbit_kws_%j.log
set -euo pipefail
SITE=$HOME/netsurf_site
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate netsurf-gpu
export NETSURF_SITE_DATA=$SITE/data NETSURF_SITE_WT=$SITE/wt NETSURF_SITE_LOCKS=$SITE/locks
export TF_DETERMINISTIC_OPS=1 TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
# CUDA 11.8 libraries come from the nvidia-* pip packages of the environment
NV=$(python -c "import nvidia; print(list(nvidia.__path__)[0])")
export LD_LIBRARY_PATH=$(ls -d $NV/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import tensorflow as tf; g = tf.config.list_physical_devices('GPU'); print('TensorFlow GPUs:', g); assert g, 'no GPU visible to TensorFlow'"
cd $SITE/wt/paper1-runR-v2-validated/workflows
git log --oneline -1
OUT=$SITE/wt/paper1-tecs-revision-v1/workspace/runs/paper1_topbit_flips_kws_v1
python runR_topbit_flips.py check --out $OUT --threads 4
python runR_topbit_flips.py measure --out $OUT --threads 4 --log-every 1000
echo "finished at $(date)"
