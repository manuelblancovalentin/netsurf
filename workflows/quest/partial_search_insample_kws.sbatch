#!/bin/bash
# Partial-group search, diagnostic after C1: search masks vs abs1024 and msb on the search clips and on the unseen
# calibration clips, 64 fresh real-flip fields per cell; about 4,600 passes (2-5 min on an A100).
#SBATCH --account=p33112
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=partial_insmp
#SBATCH --output=/home/mbv6231/netsurf_site/logs/partial_search_insample_%j.log
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
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
sha256sum runR_partial_search_insample.py quest/partial_search_insample_kws.sbatch
python runR_partial_search_insample.py $RUNS/paper1_partial_search_insample_c1_v1 --rep 0 --K 64 --threads 4
echo "finished at $(date)"
