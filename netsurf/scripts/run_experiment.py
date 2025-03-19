#!/usr/bin/env python3
import sys 
sys.path.append('/home/manuelbv/WSBMR/workspace/dev')
sys.path.append('/home/manuelbv/fkeras') #https://github.com/KastnerRG/fkeras

""" Basic modules """
import os 
import time
import yaml

""" Data exploration & manipulation """
import numpy as np

""" Let's add our custom wsbmr code """
import wsbmr

""" Custom fkeras for hessian ranker """
import fkeras

""" Qkeras """
import qkeras

""" Visual """
from tqdm import tqdm

""" Tensorflow """
import tensorflow as tf

# Available benchmarks are entries in config
AVAILABLE_BENCHMARKS = wsbmr.AVAILABLE_BENCHMARKS
BENCHMARKS_CONFIG = wsbmr.BENCHMARKS_CONFIG


""" Run experiment function """
def run_experiment(workdir, datasets_dir, benchmark, config_per_methods, load_weights, 
                    plot, show, train_model, bits_config, normalize, overwrite_ranking, reload_ranking, 
                    prune, model_prefix, num_reps, rerun_experiment, save_weights_checkpoint, tmr_range, ber_range, run_name):

    """ Build benchmark object """
    bmk = wsbmr.get_benchmark(benchmark, bits_config = bits_config, datasets_dir = datasets_dir, workdir = workdir, 
                                load_weights = load_weights, model_prefix = model_prefix, **BENCHMARKS_CONFIG[benchmark])

    # Print dataset and model summary
    if plot and False:
        bmk.plot_dataset(num_samples = 12, subset = 'train', title = 'Training', show = False)
        bmk.plot_dataset(num_samples = 12, subset = 'validation', title = 'Validation', show = False)
        bmk.dataset_summary()
        bmk.model_summary(to_file = True)
        bmk.model_summary(to_file = True, quantized = True)
        #if benchmark != 'ECONT_AE': bmk.plot_model(verbose = False)


    # Parse config
    session_params = wsbmr.utils.parse_config(BENCHMARKS_CONFIG[benchmark])

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
            print(f'[INFO] - Running session with batch_size = {batch_size}, epochs = {epochs}, opt_params = {opt_params}, pruning_params = {pruning_params}')

            callbacks = [
                #tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            ]

            if prune > 0.0:
                callbacks.append(wsbmr.pruning_callbacks.UpdatePruningStep())

            # Compile model 
            bmk.compile(opt_params = opt_params, pruning_params = pruning_params, batch_size = batch_size)

            # Run fitting
            sess, logs = bmk.fit(batch_size = batch_size, epochs = epochs, callbacks = callbacks, save_weights_checkpoint = save_weights_checkpoint) #callbacks=[pruning_callbacks.UpdatePruningStep()]

            # Save session config and object
            sess.save()

            # Save weights into file
            bmk.save_weights(prefix = bmk.model_name)   

        # Plot training history
        sess.plot_training_history(logs, show = False, to_file = True)

        # Plot weight dist 
        bmk.plot_weight_pie(to_file = True, show = False)

        # Plot sparsity
        bmk.plot_sparsity(sess, show = False, to_file = True)
        bmk.plot_sparsity(sess, separated = True, show = False, to_file = True)

        # Evaluate model 
        bmk.evaluate(sess, show = False, to_file = True)

    else:

        # Get logs 
        logs = sess.logs

        # Plot training history
        sess.plot_training_history(logs, show = False, to_file = False)

        # Plot sparsity
        bmk.plot_sparsity(sess, show = False, to_file = False)
        bmk.plot_sparsity(sess, separated = True, show = False, to_file = True)

        # Evaluate model 
        bmk.evaluate(sess, show = show, to_file = False)


    """ Define params and configs for all tests """
    batch_size = 1000
    #rad_range = np.arange(0.005, 0.055, step=0.005)
    #rad_range = np.logspace(-4, -1, num=10, base = 10)
    #rad_range = np.logspace(-3, -1, num=10, base = 10)
    #tmr_range = np.arange(0.0, 1.0, step = 0.2)

    # Convert ranges into numpy arrays
    tmr_range = np.array(tmr_range)
    rad_range = np.array(ber_range)

    """ Get the data """
    # prepare data 
    nsample_mod = 48 if 'ECON' in benchmark else -1
    XYTrain = wsbmr.utils.prepare_data(bmk, subset = 'train', nsample_mod = nsample_mod)

    # Loop thru methods
    for method, c in config_per_methods:

        #################################################################
        # 1. Initialize experiment object
        #################################################################
        # Extend config dict
        c_ext = dict(**c, **{'normalize': normalize}, **{'bits_config': bits_config})

        # Get kws from config
        kws = c['kws'] if 'kws' in c else {}

        #################################################################
        # 1. Create experiment object
        #################################################################
        exp = wsbmr.Experiment(method, c_ext, bmk, reload_ranking = reload_ranking, name = run_name, verbose = True, **kws)
    
        # Print experiment info 
        exp.print_info()

        #################################################################
        # 2. Perform ranking according to method
        #################################################################
        # Rank weights 
        df = exp.rank(bmk.model, *XYTrain, verbose = True, **kws)

        # Save rank to csv file 
        exp.save_ranking(df, overwrite = overwrite_ranking)

        #################################################################
        # 3. Run experiment with given ranking and for whatever 
        #       range of tmr and rad 
        #################################################################
        exp.run_experiment(bmk, batch_size = batch_size, num_reps = num_reps, rad_range = rad_range, tmr_range = tmr_range, rerun = rerun_experiment)


""" Entry point """
if __name__ == '__main__':

    # Parse arguments 
    args = wsbmr.parse_arguments(*sys.argv[1:])

    # Print configuration
    #print(f'Running benchmark {benchmark} with method {method} and config {subconfig}')
    run_experiment(**args)