# Basic modules 
import re

# Custom print
from . import log

""" Get the pruning factor from the name """
def get_pruning_factor(name):
    # Get the pruning factor from the name
    # The name is expected to be something like pruned_0.25_1095_hls4ml_cnn
    # It might also be "non_pruned_..." meaning pruning is 0.0
    # Use regex
    m = re.search(r'(non_)?pruned_(0\.\d+)?\_?(.*)', name)
    # Get groups 
    if m is None:
        return 0.0, name
    groups = m.groups()
    # If there is no match, then return 0.0
    if groups is None:
        return 0.0, name
    
    # If there is a match, then return the prune factor
    prune_factor = 0.0
    if groups[0] == "non_":
        prune_factor = 0.0
    elif groups[1] is not None:
        prune_factor = float(groups[1])
    
    model_name = groups[2]
    return prune_factor, model_name


""" Parse benchmark config file """
def parse_config(config):

    # Get session params
    if 'sessions' in config:
        session_params = config['sessions']
    else:
        log._info('No session params found in config file. Using default values: batch_size = 32, epochs = 10')
        session_params = [{'batch_size': 32, 'epochs': 10}]

    # Loop thru sessions
    for isp, sp in enumerate(session_params):
        
        """ Optimizer params """
        opt_params = {}
        if 'optimizer_params' in sp:
            opt_params = sp['optimizer_params']
        session_params[isp]['optimizer_params'] = opt_params

        """ Batch size """
        if 'batch_size' not in sp:
            session_params[isp]['batch_size'] = 32
        
        """ Epochs """
        if 'epochs' not in sp:
            session_params[isp]['epochs'] = 10
        
        """ Callbacks """
        if 'callbacks' not in sp:
            session_params[isp]['callbacks'] = []

        """ Pruning params """
        pruning_params = {}
        if 'pruning_params' in sp:
            pruning_params = sp['pruning_params']
        if 'final_sparsity' not in pruning_params:
            pruning_params['final_sparsity'] = 0.5
        if 'step' not in pruning_params:
            pruning_params['step'] = 2
        if 'end_epoch' not in pruning_params:
            pruning_params['end_epoch'] = 10
        session_params[isp]['pruning_params'] = pruning_params
    
    return session_params