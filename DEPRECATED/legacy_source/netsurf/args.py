
# Imports
import argparse

# netsurf
import netsurf

# numpy
import numpy as np

# Define function
def parse_arguments(*args, **kwargs):

    # Parse arguments 
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('-b', '--benchmark', type=str, required = False, help=f'Benchmark to run, available options: {", ".join(netsurf.config.AVAILABLE_BENCHMARKS)}')
    # Methods is a list of methods to run
    parser.add_argument('--method', type=str, action='append', required = False, help='Methods to run')
    parser.add_argument('--method_suffix', type=str, action='append', required = False, help='Methods to run')
    parser.add_argument('--method_kws', type=str, nargs='+', action='append', required = False, help='Methods to run')
    # Directories 
    parser.add_argument('--benchmarks_dir', type=str, required = False, help='Directory to store results')
    parser.add_argument('--datasets_dir', type=str, required = False, help='Directory where data is stored')
    # Arguments for the experiment
    parser.add_argument('--no-load_weights', action='store_true', help='Disable loading weights from previous run')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--show_plots', action='store_true', help='Display plots or just save them')
    parser.add_argument('--train_model', action='store_true', help='Retrain the model (regardless of --load_weights)')
    parser.add_argument('--save_weights_checkpoint', action='store_true', help='Save weights every epoch while training')
    
    # Quantization
    parser.add_argument('--quantization', type=str, required=False, default="q<6,0,1>", help='Configuration of quantization in format q<m,n,s>: m=number of bits, n=integer bits, s=signed(1)/unsigned(0)')
    
    # Other params
    parser.add_argument('--normalize', action='store_true', help='Normalize netsurf parameters in ranking')
    parser.add_argument('--overwrite_ranking', action='store_true', help='Overwrite ranking file')
    parser.add_argument('--no-reload_ranking', action='store_true', help='Disable reloading ranking file')
    parser.add_argument('--prune', type=float, default = 0.0, help='Pruning rate (default 0.0)')
    parser.add_argument('--model_prefix', type=str, required = False, default = "", help='Prefix to be added to the name of the folder where the results are stored.')
    # Bit-flip exp arguments 
    parser.add_argument('--num_reps', type=int, default = None, help='Number of repetitions for the bit-flip experiment')
    # Tmr range
    parser.add_argument('--protection_range', type=float, nargs='+', default = None, help='Range of protection values to test')
    # BER range
    parser.add_argument('--ber_range', type=float, nargs='+', default = None, help='Range of bit error rates to test')
    # Injection batch size 
    parser.add_argument('--injection_batch_size', type=int, default = 1000, help='Batch size for the injection experiment')
    # Save benchmark object 
    parser.add_argument('--no-save_benchmark', action='store_true', help='Save benchmark object')
    # Save experiment object
    parser.add_argument('--no-save_experiment', action='store_true', help='Save experiment object')
    # Add a "GUI" flag to know if we are running in command line (default) or in a GUI
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')

    # Parse arguments
    args = parser.parse_args()
    benchmark = args.benchmark
    # Check if the benchmark is available
    if benchmark is not None:
        assert benchmark in netsurf.config.AVAILABLE_BENCHMARKS, f'Benchmark {benchmark} not available. Available options: {", ".join(netsurf.config.AVAILABLE_BENCHMARKS)}'

    # Get the methods we need to run 
    methods = args.method
    method_suffixes = args.method_suffix
    method_kws = args.method_kws
    config_per_methods = []
    # Make sure all vars have same length
    if methods is not None and method_suffixes is not None and method_kws is not None:
        assert len(methods) == len(method_suffixes) == len(method_kws), f'Parameters --method, --method_suffix and --method_kws should have the same length, but received lengths: {len(methods)}, {len(method_suffixes)}, {len(method_kws)}'
    
        # Build configuration of parameters per test to be run
        for method, suffix, kws in zip(methods, method_suffixes, method_kws):
            # Parse kws and turn into dictionary
            _kws = {}
            for i in range(len(kws)):
                kw = kws[i].split('=')
                # Interpret kw[1]
                try:
                    _kws[kw[0]] = eval(kw[1])
                except:
                    _kws[kw[0]] = kw[1]
            subconfig = (method, {'suffix': suffix, 'kws': _kws})
            config_per_methods.append(subconfig)

    """ Directories """
    # Datasets dir
    datasets_dir = args.datasets_dir
    # Workdir
    benchmarks_dir = args.benchmarks_dir

    # If benchmarks dir is None, set the default
    if benchmarks_dir is None: benchmarks_dir = netsurf.config.DEFAULT_BENCHMARKS_DIR
    # If datasets dir is None, set the default
    if datasets_dir is None: datasets_dir = netsurf.config.DEFAULT_DATASETS_DIR

    """ Flags """
    load_weights = not args.no_load_weights
    plot = args.plot
    show_plots = args.show_plots
    train_model = args.train_model
    save_weights_checkpoint = args.save_weights_checkpoint
    normalize = args.normalize
    overwrite_ranking = args.overwrite_ranking
    prune = args.prune
    model_prefix = args.model_prefix
    reload_ranking = not args.no_reload_ranking
    # parse quantization 
    quantization = netsurf.QuantizationScheme(args.quantization)
    
    # Rerun experiment flag 
    #rerun_experiment = overwrite_ranking or train_model
    
    # Experiment args
    num_reps = args.num_reps
    if num_reps is None: num_reps = netsurf.config.DEFAULT_NUM_REPS
    if num_reps is not None:
        assert num_reps > 0, f'Number of repetitions should be greater than 0, but received {num_reps}'

    # TMr range
    protection_range = args.protection_range
    # check that protection_range is valid (0 <= tmr <= 1) and sorted
    if protection_range is None: protection_range = netsurf.config.DEFAULT_PROTECTION
    if protection_range is not None:
        if isinstance(protection_range, str):
            if protection_range == 'all' or protection_range == 'default': protection_range = netsurf.config.DEFAULT_PROTECTION
        if not isinstance(protection_range, list) and not isinstance(protection_range, tuple) and not isinstance(protection_range, np.ndarray):
            protection_range = [protection_range]
        protection_range = np.sort(protection_range)
        assert all([0 <= tmr <= 1 for tmr in protection_range]), f'Protection range should be between 0 and 1, but received {protection_range}'

    # BER range 
    ber_range = args.ber_range
    # check that ber_range is valid (0 <= ber <= 1) and sorted
    if ber_range is None: ber_range = netsurf.config.DEFAULT_BER
    if ber_range is not None:
        if isinstance(ber_range, str):
            if ber_range == 'default': ber_range = netsurf.config.DEFAULT_BER
        if not isinstance(ber_range, list) and not isinstance(ber_range, tuple) and not isinstance(ber_range, np.ndarray):
            ber_range = [ber_range]
        ber_range = np.sort(ber_range)
        assert all([0 <= ber <= 1 for ber in ber_range]), f'BER range should be between 0 and 1, but received {ber_range}'


    # Save benchmark object
    save_benchmark = not args.no_save_benchmark
    # Save experiment object
    save_experiment = not args.no_save_experiment
    # Injection batch size
    injection_batch_size = args.injection_batch_size
    # Gui 
    gui = args.gui

    # build arguments as dict 
    args_dict = dict(benchmarks_dir = benchmarks_dir, datasets_dir = datasets_dir,
                  benchmark = benchmark, config_per_methods = config_per_methods,
                  load_weights = load_weights, plot = plot, show = show_plots, train_model = train_model, 
                  quantization = quantization, normalize = normalize, overwrite_ranking = overwrite_ranking,
                  reload_ranking = reload_ranking, prune = prune, model_prefix = model_prefix, num_reps = num_reps,
                  save_weights_checkpoint = save_weights_checkpoint,
                  protection_range = protection_range, ber_range = ber_range, #run_name = run_name,
                  injection_batch_size = injection_batch_size, #rerun_experiment = rerun_experiment, 
                  save_benchmark = save_benchmark, save_experiment = save_experiment,
                  gui = gui)

    return args_dict
