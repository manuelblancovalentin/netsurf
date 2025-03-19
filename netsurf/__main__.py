# Basic modules 
import sys, os
# Add current path to sys.path for wsbmr
# Get path for parent of current file 
wsbmr_dir = os.path.abspath(__file__)
wsbmr_dir = os.path.dirname(os.path.dirname(wsbmr_dir))
sys.path.append(wsbmr_dir)
print(f'Adding {wsbmr_dir} to sys.path')

# Add fkeras and qkeras (expected to be ../)
scripts_dir = os.path.dirname(os.path.dirname(wsbmr_dir))
sys.path.append(os.path.join(scripts_dir, "fkeras"))
sys.path.append(os.path.join(scripts_dir, "qkeras"))
sys.path.append(os.path.join(scripts_dir, "nodus"))

""" Numpy """
import numpy as np

# Import wsbmr
import wsbmr




""" Get training session (load or train) """
def get_training_session(bmk, train_model = False, prune = 0.0, show_plots = False):

    # Get benchmark config
    benchmarks_config = wsbmr.config.BENCHMARKS_CONFIG

    # Parse config
    session_params = wsbmr.utils.parse_config(benchmarks_config[bmk.name])

    # Try to find the latest session and load 
    sess = wsbmr.load_session(bmk.sessions_dir, latest = True)

    """ Fit params """
    if train_model or sess is None:

        # Loop thru sessions
        for sp in session_params:
            
            """ Optimizer params """
            opt_params = sp['optimizer_params']
            batch_size = sp['batch_size']
            epochs = sp['epochs']
            pruning_params = {}
            if prune > 0.0:
                pruning_params = sp['pruning_params']
                pruning_params['final_sparsity'] = prune

            
            # Print info 
            wsbmr.utils.log._custom('MDL', f'Running session with batch_size = {batch_size}, epochs = {epochs}, opt_params = {opt_params}, pruning_params = {pruning_params}')

            # Parse callbacks 
            callbacks = wsbmr.dnn.callbacks.parse_callbacks(sp['callbacks'], prune = prune)

            # Compile model 
            bmk.compile(opt_params = opt_params, pruning_params = pruning_params, batch_size = batch_size)

            # Run fitting
            sess, logs = bmk.fit(batch_size = batch_size, epochs = epochs, callbacks = callbacks, prune = prune) #callbacks=[pruning_callbacks.UpdatePruningStep()]

            # Save session config and object
            sess.save()

            # Save weights into file
            bmk.save_weights(prefix = bmk.model_full_name)   

    else:

        # Parse 
        opt_params = {}
        pruning_params = {}
        batch_size = 32
        if len(session_params) > 0:
            # Last session
            sp = session_params[-1]

            """ Optimizer params """
            opt_params = sp['optimizer_params']
            batch_size = sp['batch_size']
            epochs = sp['epochs']
            pruning_params = sp['pruning_params'] if prune else {}
            
            # Print info 
            wsbmr.utils.log._custom('MDL', f'Loading session with batch_size = {batch_size}, epochs = {epochs}, opt_params = {opt_params}, pruning_params = {pruning_params}')

            # Parse callbacks 
            callbacks = wsbmr.dnn.callbacks.parse_callbacks(sp['callbacks'], prune = prune)
            
        # Compile model 
        bmk.compile(opt_params = opt_params, pruning_params = pruning_params, batch_size = batch_size)

        # Get logs 
        logs = sess.logs

    """ Plots before leaving """
    # Plot training history
    sess.plot_training_history(logs, to_file = True, show = show_plots)

    # Plot weight dist 
    bmk.plot_weight_pie(to_file = True, show = show_plots)

    # Plot sparsity
    bmk.plot_sparsity(sess, to_file = True, show = show_plots)
    bmk.plot_sparsity(sess, separated = True, to_file = True, show = show_plots)

    # Evaluate model 
    bmk.evaluate(sess, to_file = True, show = show_plots)
    
    return sess


def run_injection_experiment(bmk: 'Benchmark', quantization: 'QuantizationScheme', 
                             batch_size = 1000,
                             protection_range = [], ber_range = [], num_reps = 1,
                             config_per_methods = {},
                             normalize = False, reload_ranking = True, 
                             overwrite_ranking = False,
                             save_experiment = True,
                             **kwargs):
    
    # Convert ranges into numpy arrays
    protection_range = np.array(protection_range)
    ber_range = np.array(ber_range)

    # if no config per methods, no experiment. Return
    if len(config_per_methods) == 0:
        wsbmr.utils.log._warn('No config per methods specified. No experiment. Returning now.')
        return None

    """ Get the data """   
    # First we need to prepare the dataset. 
    # If this is the ECON dataset, we need to downsample it to 48 samples batch, otherwise it's 
    # crazy big. 
    nsample_mod = 48 if 'ECON' in bmk.name else -1

    # Now let's prepare the data
    XYTrain = wsbmr.utils.prepare_data(bmk, subset = 'train', nsample_mod = nsample_mod)

    # Loop thru methods
    exps = {}
    for method, c in config_per_methods:

        #################################################################
        # 1. Initialize experiment object
        #################################################################
        # Extend config dict
        c_ext = dict(**c, **{'normalize': normalize})

        # Get kws from config
        kws = c.pop('kws') if 'kws' in c else {}

        #################################################################
        # 1. Create experiment object
        #################################################################
        exp = wsbmr.Experiment(method, bmk, quantization, c_ext, reload_ranking = reload_ranking, verbose = True, 
                               ber_range = ber_range, protection_range = protection_range, **kws)
    
        # Print experiment info 
        print(exp)

        #################################################################
        # 2. Perform ranking according to method
        #################################################################
        # Rank weights 
        df = exp.rank(bmk.model, *XYTrain, verbose = True, **kws)

        # Save rank to csv file 
        exp.save_ranking(df, overwrite = overwrite_ranking)

        #################################################################
        # 3. Run experiment with given ranking and for whatever 
        #       range of protection and rad 
        #################################################################
        exp.run_experiment(bmk, batch_size = batch_size, num_reps = num_reps, 
                           ber_range = ber_range, 
                           protection_range = protection_range, 
                           rerun = False)
        
        # Save experiment object
        if save_experiment:
            exp.save()

        # Add to dict
        exps[method] = exp

    return exps


# Define run experiment 
def run_experiment(benchmark: str, 
                   quantization: wsbmr.QuantizationScheme, 
                   benchmarks_dir = None, 
                   datasets_dir = None,  
                   load_weights = True, model_prefix = "",
                   plot = False, show_plots = False,
                   train_model = False, prune = 0.0,
                   protection_range = [], ber_range = [],
                   injection_batch_size = 1000,
                   normalize = False, reload_ranking = True, 
                   config_per_methods = {}, 
                   overwrite_ranking = False, 
                   save_benchmark = True, save_experiment = True,
                   **kwargs):
    
    """ 
        # BENCHMARK
    """
    # Add model_prefix to prune
    model_prefix = model_prefix + f"_pruned_{prune}_" if model_prefix != "" else f"pruned_{prune}_"
    bmk = wsbmr.get_benchmark(benchmark, quantization, 
                                benchmarks_dir = benchmarks_dir, 
                                datasets_dir = datasets_dir, 
                                load_weights = load_weights, 
                                model_prefix = model_prefix)
    if bmk is None: 
        wsbmr.utils.log._error('Benchmark object is None. Returning now.')
        return None

    # At this point we might want to plot the benchmark
    if plot:
        bmk.plot_dataset(subset = 'train', title = 'Training', show = show_plots)
        bmk.plot_dataset(subset = 'validation', title = 'Validation', show = show_plots)
        bmk.dataset_summary()
        bmk.model_summary(to_file = True)
        bmk.model_summary(to_file = True, quantized = True)
        #bmk.plot_model(verbose = False)
    
    # Save benchmark object
    if save_benchmark:
        bmk.save()

    """
        # SESSION LOADING / MODEL TRAINING
    """
    sess = get_training_session(bmk, train_model = train_model, prune = prune, show_plots = show_plots)

    # if sess is None, return 
    if sess is None: 
        wsbmr.utils.log._error('Session object is None. Returning now.')
        return None

    """
        # RUN THE INJECTION EXPERIMENT (IF APPLICABLE)
    """
    exps = run_injection_experiment(bmk, quantization,
                                   protection_range = protection_range, 
                                   ber_range = ber_range, 
                                   batch_size = injection_batch_size,
                                   normalize = normalize,
                                   reload_ranking = reload_ranking,
                                   config_per_methods = config_per_methods,
                                   prune = prune, overwrite_ranking = overwrite_ranking,
                                   save_experiment = save_experiment,
                                   **kwargs)

    return bmk, sess, exps
    
# Main
if __name__ == "__main__":

    # Parse args
    args = wsbmr.args.parse_arguments()
    gui = args.pop('gui', False)
    benchmark = args.pop('benchmark')
    quantization = args.pop('quantization')

    # If args.gui, build GUI
    if gui:
        wsbmr.gui.build_gui(**args)
    else:
        # Run experiment
        run_experiment(benchmark, quantization, **args)

