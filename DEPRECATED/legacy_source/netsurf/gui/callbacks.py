# Custom imports
import netsurf

""" Callback for processing all the experiments given a set of configs """
def process_experiments(config, pbar = lambda x: None):
    # Let's print the config 
    netsurf.utils.log._log("Processing experiments with the following config:")
    
    # Print config
    for i, (key, value) in enumerate(config.items()):
        print(f"\t{key}: {value}")

    # Progress bar function
    progress_fcn = lambda i: pbar((i+1)/len(config.keys()) * 100)

    # Get config vars
    benchmarks_dir = config['benchmarks_dir']
    benchmarks_selected = config['benchmarks']
    datasets_dir = config['datasets_dir']
    reps = config['num_reps']
    protection_values = config['protections']
    ber_values = config['bers']
    pruning_values = config['pruning']
    methods_selected = config['methods']
    quantizations_selected = config['quantizations']

    # Instantiate first bucket. This will be the root bucket that will start at benchmarks_dir. 
    # The structure we expect is: 
    #   (0) - benchmarks_dir
    #   (1)   - quantization (e.g., 6bits_0int)
    #   (2)     - model (e.g., pruned_0.25_1095_hls4ml_cnn)
    #   (3)       - experiments (folder containing the results for each method)
    #   (4)         - method (e.g., bitwise_msb)
    #   (5)           - run (e.g., run_20241215_103739)
    #   (6)             - results.csv (file containing the results of the run)
    #                   - config.json (file with the configuration of the run)
    #                   - ranking.csv (file with the ranking)
    #                   - global_metrics.json (file with the global metrics for this run)
    #                   - job.log* (log file for the run)
    #                   - job.sh* (job file for this run)
    #                   - job.progress* (text file with the progress of the run)
    #   (3)      - models (folder containing the models trained)
    #   (3)      - sessions (folder containing the training sessions)
    #   (4)         - training session (e.g. training_session.20241215_010614)
    #   (5)           - config_training_session.*.json (config file with the training session parameters)
    #   (5)           - training_session.*.log (log file with the training summary)
    #   (5)           - training_session.*.pkl (binary file with the progress of the training session (can be loaded into netsurf))
    # So let's start with the first level (0) and go from there.
    root_bucket = netsurf.explorer.create_bucket(benchmarks_dir, 'root', verbose = True, level = 0, 
                                               benchmarks = benchmarks_selected,
                                                protection = protection_values, ber = ber_values, num_reps = reps, 
                                                pruning = pruning_values, methods = methods_selected,
                                                quantizations = quantizations_selected,
                                                benchmarks_dir = benchmarks_dir, datasets_dir = datasets_dir,
                                                pbar = pbar)

    return root_bucket