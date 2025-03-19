#!/bin/bash

#Run wsbmr example
benchmark='mnist_hls4ml'
benchmarks_dir='/Users/mbvalentin/scripts/wsbmr/benchmarks'
datasets_dir='/Users/mbvalentin/scripts/wsbmr/datasets'

# conda env bin
# Define conda env
CONDA_ENV=wsbmr

# Get bin path for CONDA_ENV
#CONDA_BIN=$(conda info --base)/envs/${CONDA_ENV}/bin/python
CONDA_BIN=/Users/mbvalentin/miniconda3/envs/${CONDA_ENV}/bin/python

# wsbmr path 
WSBMR_PATH='/Users/mbvalentin/scripts/wsbmr/dev/wsbmr'

# Arguments:
# GENERIC:
# --benchmark: Benchmark name
# --benchmarks_dir: Directory with benchmarks
# --datasets_dir: Directory with datasets
# --plot: Plot results
# --save_weights_checkpoint: Save weights every epoch
# --bits_config: configuration of bits (e.g., "num_bits=6 integer=0")

# METHOD SPECIFICAETION:
# --method: Method name 
# --method_suffix: Suffix for method name
# --method_kws: Method keyword arguments (e.g., "ascending=True times_weights=True")

# FOR MODEL/TRAINING:
# --prune: Prune percentage (0.0 to 1.0)
# --model_prefix: Prefix for model name
# --load_weights: Load weights from previous training
# --train_model: Force retraining

# FOR FAULT INJECTION:
# --overwrite_ranking: Overwrite ranking
# --reload_ranking: Reload ranking file
# --num_reps: Number of repetitions for each combination of (protection, BER)
# --protection_range: Protection range (e.g., "0.0,0.1,0.2")
# --ber_range: BER range (e.g., "0.0,0.1,0.2")
# --normalize: Normalize ranking (0.0 to 1.0)

# Run wsbmr just to train the model (prune = 0.0)
${CONDA_BIN} ${WSBMR_PATH} \
    --benchmark $benchmark \
    --benchmarks_dir $benchmarks_dir \
    --datasets_dir $datasets_dir \
    --plot \
    --load_weights

# Now run wsbmr with injection 
${CONDA_BIN} ${WSBMR_PATH} \
    --benchmark $benchmark \
    --benchmarks_dir $benchmarks_dir \
    --datasets_dir $datasets_dir \
    --load_weights \
    --method random \
    --method_suffix None \
    --method_kws ascending=False 

