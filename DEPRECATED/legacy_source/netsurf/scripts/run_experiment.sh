#!/bin/bash

# Read input arguments (usage should be like this: ./run_experiment.sh --benchmarks cifar10_hls4ml mnist_squeezenet ...)

# --benchmarks is an array
BENCHMARKS=()
prune=0.0
methods=""
load_weights=false
train_model=false
reload_ranking=false
save_weights_checkpoint=true
tmr_range=""
ber_range=""
method_kws=""
overwrite_ranking=false
run_name=""

# dirs 
datasets_dir="~/wsbmr/datasets"
benchmarks_dir="~/wsbmr/benchmarks"

# Vars to "LOCK" in the current flag 
in_benchmarks=false
in_methods=false
in_tmr=false
in_ber=false
in_method_kws=false

while [ "$1" != "" ]; do
    case $1 in
        --benchmarks | -b )          
                                # shift the "-b" argument 
                                shift
                                # Append to the array of benchmarks
                                BENCHMARKS+=($1)
                                in_benchmarks=true
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                shift
                                ;;
        --benchmarks_dir )          
                                # shift the "-b" argument 
                                shift
                                # Append to the array of benchmarks
                                benchmarks_dir=$1
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                shift
                                ;;
        --datasets_dir )
                                # shift the "-b" argument 
                                shift
                                # Append to the array of benchmarks
                                datasets_dir=$1
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                shift
                                ;;
        --prune | -p )          
                                # shift the "-p" argument
                                shift
                                # Get the actual value
                                prune=$1
                                # shift the value
                                shift
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --methods | -m | --method )         
                                shift
                                methods="${methods} $1"
                                shift
                                in_methods=true
                                in_benchmarks=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --run_name )       
                                shift
                                run_name=$1
                                in_method_kws=false
                                in_tmr=false
                                in_ber=false
                                in_benchmarks=false
                                in_methods=false
                                shift
                                ;;
        --load_weights )
                                shift
                                load_weights=true
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --overwrite_ranking )
                                shift
                                overwrite_ranking=true
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --train_model )
                                shift
                                train_model=true
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --reload_ranking )
                                shift
                                reload_ranking=true
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --no_save_model_weights_every_epoch )
                                shift
                                save_weights_checkpoint=false
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        --tmr_range )
                                shift
                                tmr_range="$1"
                                in_tmr=true
                                in_ber=false
                                in_benchmarks=false
                                in_methods=false
                                in_method_kws=false
                                shift
                                ;;
        --ber_range)
                                shift
                                ber_range="$1"
                                in_tmr=false
                                in_ber=true
                                in_benchmarks=false
                                in_methods=false
                                in_method_kws=false
                                shift
                                ;;
        --method_kws )        
                                shift
                                method_kws="$1"
                                in_method_kws=true
                                in_tmr=false
                                in_ber=false
                                in_benchmarks=false
                                in_methods=false
                                shift
                                ;;
        --help | -h )
                                # Print message with all the flags and options
                                echo "Usage: ./run_experiment.sh --benchmarks <benchmark1> <benchmark2> ... --methods <method1> <method2> ... --prune --load_weights --train_model --reload_ranking --no_save_model_weights_every_epoch"
                                in_benchmarks=false
                                in_methods=false
                                in_tmr=false
                                in_ber=false
                                in_method_kws=false
                                ;;
        * )                     
                                # Check if this starts with "-"
                                if [ "${1:0:1}" == "-"  ]; then
                                    in_benchmarks=false
                                    in_methods=false
                                    in_tmr=false
                                    in_ber=false
                                    in_method_kws=false
                                else
                                    if [ "$in_benchmarks" = true ]; then
                                        # Append to the array of benchmarks
                                        BENCHMARKS+=($1)
                                    fi
                                    if [ "$in_methods" = true ]; then
                                        # Append to the array of benchmarks
                                        methods="${methods} $1"
                                    fi
                                    if [ "$in_tmr" = true ]; then
                                        # Append to the array of benchmarks
                                        tmr_range="${tmr_range} $1"
                                    fi
                                    if [ "$in_method_kws" = true ]; then
                                        # Append to the array of benchmarks
                                        method_kws="${method_kws} $1"
                                    fi
                                    if [ "$in_ber" = true ]; then
                                        # Append to the array of benchmarks
                                        ber_range="${ber_range} $1"
                                    fi
                                fi
                                shift
                                ;;
        
    esac
    #shift
done


# If methods is empty, set it to none
if [ -z "${methods}" ]; then
    methods="none"
fi

#printf 'Benchmarks: %s\n' "${BENCHMARKS[@]}"
#printf 'Methods: %s\n' "${methods}"

# Define custom working directory
#ROOT_DIR=/home/${USER}/WSBMR/workspace
#DATASETS_DIR=${ROOT_DIR}/datasets
DATASETS_DIR=${datasets_dir}
BENCHMARKS_DIR=${benchmarks_dir}

# Python file is sibling to this script, so let's get this script's path 
# and then append the python file
SCRIPT_PATH=$(dirname $(realpath $0))
PYTHON_FILE=${SCRIPT_PATH}/run_experiment.py

# Define conda env
CONDA_ENV=qkeras

# Get bin path for CONDA_ENV
CONDA_BIN=$(conda info --base)/envs/${CONDA_ENV}/bin/python


# -----------------------------------------------------------
# EXP ARGS 
# -----------------------------------------------------------
plot=true
show_plots=false
bits_config="num_bits=6 integer=0"
normalize=true

# Make sure prune is a float between 0 and 1
if [ $(echo "$prune < 0" | bc) -eq 1 ] || [ $(echo "$prune > 1" | bc) -eq 1 ]; then
    # Raise error
    echo "Prune value must be between 0 and 1"
    exit 1
fi

# If tmr_range is empty, initialize to [0.0, 0.2, 0.4, 0.6, 0.8]
if [ -z "${tmr_range}" ]; then
    tmr_range="0.0 0.2 0.4 0.6 0.8"
fi

# Make sure every value in tmr_range is a float between 0 and 1
for value in $tmr_range; do
    if [ $(echo "$value < 0" | bc) -eq 1 ] || [ $(echo "$value > 1" | bc) -eq 1 ]; then
        # Raise error
        echo "TMR range values must be between 0 and 1, however ${value} is not"
        exit 1
    fi
done


# Get prune prefix
model_prefix="pruned_${prune}_"

# Build the list of generic arguments for all experiments
gen_args="--bits_config ${bits_config} --model_prefix ${model_prefix}"
if [ "$run_name" != "" ] ; then
    gen_args="${gen_args} --run_name ${run_name}"
fi
if [ "$load_weights" = true ] ; then
    gen_args="${gen_args} --load_weights"
fi
if [ "$plot" = true ] ; then
    gen_args="${gen_args} --plot"
fi
if [ "$show_plots" = true ] ; then
    gen_args="${gen_args} --show_plots"
fi
if [ "$train_model" = true ] ; then
    gen_args="${gen_args} --train_model"
fi
if [ "$normalize" = true ] ; then
    gen_args="${gen_args} --normalize"
fi
if [ "$overwrite_ranking" = true ] ; then
    gen_args="${gen_args} --overwrite_ranking"
fi
gen_args="${gen_args} --prune ${prune}"
if [ "$reload_ranking" = true ] ; then
    gen_args="${gen_args} --reload_ranking"
fi
if [ "$save_weights_checkpoint" = true ] ; then
    gen_args="${gen_args} --save_weights_checkpoint"
fi
if [ "$tmr_range" != "" ] ; then
    gen_args="${gen_args} --tmr_range ${tmr_range}"
fi
if [ "$ber_range" != "" ] ; then
    gen_args="${gen_args} --ber_range ${ber_range}"
fi

# -----------------------------------------------------------
# GENERIC PARAMETERS 
# -----------------------------------------------------------

# Define args per method
declare -A METHOD_PARAMS
#METHOD_PARAMS['recursive_uneven']="--method recursive_uneven --method_suffix None --method_kws ascending=False"
METHOD_PARAMS['random']="--method random --method_suffix None --method_kws ascending=False"
#METHOD_PARAMS['diffbitperweight']="--method diffbitperweight --method_suffix None --method_kws ascending=False"

METHOD_PARAMS['layerwise']="--method layerwise --method_suffix first_to_last --method_kws ascending=True"
METHOD_PARAMS['layerwise_last']="--method layerwise --method_suffix last_to_first --method_kws ascending=False"
METHOD_PARAMS['layerwise_first']="--method layerwise --method_suffix first_to_last --method_kws ascending=True"

METHOD_PARAMS['bitwise']="--method bitwise --method_suffix msb_to_lsb --method_kws ascending=True"
METHOD_PARAMS['bitwise_msb']="--method bitwise --method_suffix msb_to_lsb --method_kws ascending=True"
METHOD_PARAMS['bitwise_lsb']="--method bitwise --method_suffix lsb_to_msb --method_kws ascending=False"

METHOD_PARAMS['weight_abs_val']="--method weight_abs_val --method_suffix None --method_kws ascending=False batch_size=100"

METHOD_PARAMS['hirescam']="--method hirescam --method_suffix None --method_kws times_weights=False ascending=False normalize_score=False batch_size=100"
METHOD_PARAMS['hirescam_norm']="--method hirescam --method_suffix None --method_kws times_weights=False ascending=False normalize_score=True batch_size=100"
METHOD_PARAMS['hessian']='--method hessian --method_suffix ranking_msb --method_kws ranking_method=msb batch_size=96'

METHOD_PARAMS['hiresdelta']="--method hiresdelta --method_suffix None --method_kws times_weights=False ascending=False normalize_score=False batch_size=100"
METHOD_PARAMS['hiresdelta_norm']="--method hiresdelta --method_suffix None --method_kws times_weights=False ascending=False normalize_score=True batch_size=100"
METHOD_PARAMS['hessiandelta']='--method hessiandelta --method_suffix ranking_msb --method_kws ranking_method=msb batch_size=96'

# Times weights 
METHOD_PARAMS['hirescam_times_weights_norm']="--method hirescam --method_suffix None --method_kws times_weights=True ascending=False normalize_score=True batch_size=100"
METHOD_PARAMS['hiresdelta_times_weights_norm']="--method hiresdelta --method_suffix None --method_kws times_weights=True ascending=False normalize_score=True batch_size=100"
METHOD_PARAMS['hirescam_times_weights']="--method hirescam --method_suffix None --method_kws times_weights=True ascending=False normalize_score=False batch_size=100"
METHOD_PARAMS['hiresdelta_times_weights']="--method hiresdelta --method_suffix None --method_kws times_weights=True ascending=False normalize_score=False batch_size=100"

#METHOD_PARAMS['hessian_hierarchical']="--method hessian --method_suffix ranking_hierarchical --method_kws ranking_method=hierarchical batch_size=96"

# Define array of arrays for each benchmark
declare -A BENCHMARK_METHODS
BENCHMARK_METHODS[cifar10_hls4ml]="all"
BENCHMARK_METHODS[mnist_squeezenet]="all"
BENCHMARK_METHODS[tinyml_anomaly_detection]="all"
BENCHMARK_METHODS[ECONT_AE]="all"
BENCHMARK_METHODS[mnist_hls4ml]="all"
BENCHMARK_METHODS[tinyml_person_detection]="all"
BENCHMARK_METHODS[mnist_lenet5]="all"
BENCHMARK_METHODS[cifar10_resnet18]="all"
BENCHMARK_METHODS[keyword_spotting]="all"

#recursive_uneven random diffbitperweight layerwise_last weight_abs_val hirescam_norm hirescam_nonorm hessian hiresdelta_norm hessiandelta hirescam_times_weights_norm hiresdelta_times_weights_norm hirescam_times_weights hiresdelta_times_weights hessian_hierarchical

# all are: hirescam_nonorm hirescam_norm weight_abs_val bitwise_msb layerwise_first layerwise_last diffbitperweight random recursive_uneven hessian_same hessian_hierarchical   

# -----------------------------------------------------------


#-------------------------------------------------------------
# EXPERIMENTS 
#-------------------------------------------------------------

# Define BENCHMARKS to run
#BENCHMARKS=('cifar10_hls4ml')
#BENCHMARKS=('mnist_squeezenet')

# Loop thru benchmarks
for BENCHMARK in "${BENCHMARKS[@]}"; do
    printf "+------------------------------------------------------------------------------------------------------------+\n"
    printf '| Running benchmark: %s%s|\n' "${BENCHMARK}" "$(printf '%*s' $((88 - ${#BENCHMARK})) '')"
    printf "+-+----------------------------------------------------------------------------------------------------------+\n"
    printf "| |                                                                                                           \n"

    # Create workdir 
    WORKDIR=${BENCHMARKS_DIR}/${BENCHMARK}

    # Concatenate all the flags for each method in the benchmark
    args="--workdir ${WORKDIR} --datasets_dir ${DATASETS_DIR}"
    printf '| | %s %s\n' "${CONDA_BIN}" "${PYTHON_FILE}" 
    printf '| | \t--benchmark %s\n' "${BENCHMARK}"
    printf '| | \t%s\n' "${gen_args}"
    printf '| | \t--workdir %s\n' "${WORKDIR}"
    printf '| | \t--datasets_dir %s\n' "${DATASETS_DIR}"

    # Check if method is all
    if [ "${methods}" == "all" ]; then
        #  Get all keys from METHOD_PARAMS
        submethods=(${!METHOD_PARAMS[@]})
    else
        submethods=(${methods})
    fi

    # Loop thru methods
    for METHOD in "${submethods[@]}"; do
        methods_kws="${METHOD_PARAMS[${METHOD}]}"
        if [ ! -z "${method_kws}" ]; then
            if [[ $methods_kws == *"--method_kws"* ]]; then
                methods_kws="${methods_kws} ${method_kws}"
            else
                methods_kws="${methods_kws} --method_kws ${method_kws}"
            fi
        fi
        printf '| |  \t%s\n' "${methods_kws}"
        args="${args} ${methods_kws}"
        # if method_kws is not empty, add it to the args. 
        # However, if the word "--method_kws" is already in args, just append the values, without the word "--method_kws".
        # if it's not there, then add "--method_kws" and the values
        
    done

    # Add generic args
    args="${args} ${gen_args}"

    printf "+------------------------------------------------------------------------------------------------------------+\n\n"

    # Run the experiment
    echo "${CONDA_BIN} ${PYTHON_FILE} --benchmark ${BENCHMARK} ${args}"
    ${CONDA_BIN} ${PYTHON_FILE} --benchmark ${BENCHMARK} ${args}
done